//! Decoding JPEG2000 code streams.
//!
//! This is the "core" module of the crate that orchestrates all
//! stages in such a way that a given codestream is decoded into its
//! component channels.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::bitplane::{BitPlaneDecodeBuffers, BitPlaneDecodeContext};
use super::build::{CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand, SubBandType};
use super::codestream::{ComponentInfo, Header, QuantizationStyle, WaveletTransform};
use super::idwt::IDWTOutput;
use super::limits::BufferSizes;
use super::progression::{ProgressionData, progression_iterator};
use super::tag_tree::TagNode;
use super::tile::Tile;
use super::{ComponentData, bitplane, build, idwt, mct, segment, tile};
use crate::error::{DecodingError, Result, TileError, bail};
use crate::j2c::segment::MAX_BITPLANE_COUNT;
use crate::math::SimdBuffer;
use crate::reader::BitReader;
use core::ops::Range;

pub(crate) fn decode<'a>(
    data: &'a [u8],
    header: &'a Header<'a>,
    ctx: &mut DecoderContext<'a>,
    settings: &crate::DecodeSettings,
    palette_components: usize,
) -> Result<()> {
    super::limits::check_image(header, settings)?;
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header)?;

    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }

    let sizes = BufferSizes::new(header, &tiles, settings, palette_components)?;
    ctx.reset(header, &sizes);

    for tile in &tiles {
        let progression_iterator = progression_iterator(tile, header)?;

        decode_tile(
            tile,
            header,
            progression_iterator,
            &mut ctx.tile_decode_context,
            &mut ctx.channel_data,
            &mut ctx.storage,
        )?;
    }

    apply_sign_shift(&mut ctx.channel_data, &header.component_infos);

    Ok(())
}

/// A decoder context for decoding JPEG2000 images.
#[derive(Default)]
pub struct DecoderContext<'a> {
    tile_decode_context: TileDecodeContext,
    /// The raw, decoded samples for each channel.
    pub(crate) channel_data: Vec<ComponentData>,
    storage: DecompositionStorage<'a>,
}

impl DecoderContext<'_> {
    fn reset(&mut self, header: &Header<'_>, sizes: &BufferSizes) {
        // Drop retained buffers from earlier images before applying this
        // image's budget. Buffers remain reusable across its tiles.
        self.tile_decode_context = TileDecodeContext::default();
        self.storage = DecompositionStorage::default();
        self.channel_data.clear();
        self.storage.coefficients = alloc::vec![0.0; sizes.coefficients];
        self.tile_decode_context.idwt_output.coefficients = Vec::with_capacity(sizes.output);
        self.tile_decode_context.idwt_scratch_buffer = Vec::with_capacity(sizes.scratch);

        for info in &header.component_infos {
            self.channel_data.push(ComponentData {
                container: SimdBuffer::zeros(sizes.samples),
                bit_depth: info.size_info.precision,
                is_signed: info.size_info.is_signed,
            });
        }
    }
}

fn decode_tile<'a, 'b>(
    tile: &'b Tile<'a>,
    header: &Header<'_>,
    progression_iterator: Box<dyn Iterator<Item = ProgressionData> + '_>,
    tile_ctx: &mut TileDecodeContext,
    channel_data: &mut [ComponentData],
    storage: &mut DecompositionStorage<'a>,
) -> Result<()> {
    storage.reset();

    // This is the method that orchestrates all steps.

    // First, we build the decompositions, including their sub-bands, precincts
    // and code blocks.
    build::build(tile, storage, header.skipped_resolution_levels)?;
    // Next, we parse the layers/segments for each code block.
    segment::parse(tile, progression_iterator, header, storage)?;

    // Interleave bitplane decoding, IDWT, and storage for each component tile
    // so we can reuse the coefficient and IDWT buffers.
    for (idx, component_info) in tile.component_infos.iter().enumerate() {
        if idx > 0 {
            let count = storage.tile_decompositions[idx].coefficient_count;
            storage.coefficients[..count].fill(0.0);
        }
        // We then decode the bitplanes of each code block, yielding the
        // (possibly dequantized) coefficients of each code block.
        decode_component_tile_bit_planes(idx, tile, tile_ctx, storage, header)?;
        // Next, we apply the inverse discrete wavelet transform.
        idwt::apply(
            storage,
            tile_ctx,
            idx,
            header,
            component_info.wavelet_transform(),
        );
        // Store each component before reusing the IDWT buffers. Once all
        // components are stored, MCT can operate on this tile's output region.
        store(
            tile,
            header,
            tile_ctx,
            &mut channel_data[idx],
            component_info,
        );
    }

    if tile.mct {
        mct::apply_inverse(channel_data, tile, header)?;
    }

    Ok(())
}

/// All decompositions for a single tile.
#[derive(Clone)]
pub(crate) struct TileDecompositions {
    pub(crate) first_ll_sub_band: usize,
    pub(crate) decompositions: Range<usize>,
    pub(crate) coefficient_count: usize,
}

impl TileDecompositions {
    pub(crate) fn sub_band_iter(
        &self,
        resolution: u8,
        decompositions: &[Decomposition],
    ) -> SubBandIter {
        let indices = if resolution == 0 {
            [
                self.first_ll_sub_band,
                self.first_ll_sub_band,
                self.first_ll_sub_band,
            ]
        } else {
            decompositions[self.decompositions.clone()][resolution as usize - 1].sub_bands
        };

        SubBandIter {
            next_idx: 0,
            indices,
            resolution,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SubBandIter {
    resolution: u8,
    next_idx: usize,
    indices: [usize; 3],
}

impl Iterator for SubBandIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.resolution == 0 {
            if self.next_idx > 0 {
                None
            } else {
                Some(self.indices[0])
            }
        } else if self.next_idx >= self.indices.len() {
            None
        } else {
            Some(self.indices[self.next_idx])
        };

        self.next_idx += 1;

        value
    }
}

/// A buffer so that we can reuse allocations for layers/code blocks/etc.
/// across different tiles.
#[derive(Default)]
pub(crate) struct DecompositionStorage<'a> {
    pub(crate) segments: Vec<Segment<'a>>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) code_blocks: Vec<CodeBlock>,
    pub(crate) precincts: Vec<Precinct>,
    pub(crate) tag_tree_nodes: Vec<TagNode>,
    pub(crate) coefficients: Vec<f32>,
    pub(crate) sub_bands: Vec<SubBand>,
    pub(crate) decompositions: Vec<Decomposition>,
    pub(crate) tile_decompositions: Vec<TileDecompositions>,
}

impl DecompositionStorage<'_> {
    fn reset(&mut self) {
        self.segments.clear();
        self.layers.clear();
        self.code_blocks.clear();
        // No need to clear the coefficients, as they will be resized
        // and then overridden.
        // self.coefficients.clear();
        self.precincts.clear();
        self.sub_bands.clear();
        self.decompositions.clear();
        self.tile_decompositions.clear();
        self.tag_tree_nodes.clear();
    }
}

/// A reusable context used during the decoding of a single tile.
///
/// Some of the fields are temporary in nature and reset after moving on to the
/// next tile, some contain global state.
#[derive(Default)]
pub(crate) struct TileDecodeContext {
    /// A reusable buffer for the IDWT output.
    pub(crate) idwt_output: IDWTOutput,
    /// A scratch buffer used during IDWT.
    pub(crate) idwt_scratch_buffer: Vec<f32>,
    /// A reusable context for decoding code blocks.
    pub(crate) bit_plane_decode_context: BitPlaneDecodeContext,
    /// Reusable buffers for decoding bitplanes.
    pub(crate) bit_plane_decode_buffers: BitPlaneDecodeBuffers,
}

fn decode_component_tile_bit_planes<'a>(
    component_idx: usize,
    tile: &Tile<'a>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
) -> Result<()> {
    let component_info = &tile.component_infos[component_idx];

    // Only decode the resolution levels we actually care about.
    for resolution in 0..component_info.num_resolution_levels() - header.skipped_resolution_levels {
        let tile_composition = &storage.tile_decompositions[component_idx];
        let sub_band_iter = tile_composition.sub_band_iter(resolution, &storage.decompositions);

        for sub_band_idx in sub_band_iter {
            decode_sub_band_bitplanes(
                sub_band_idx,
                resolution,
                component_info,
                tile_ctx,
                storage,
                header,
            )?;
        }
    }

    Ok(())
}

fn decode_sub_band_bitplanes(
    sub_band_idx: usize,
    resolution: u8,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
) -> Result<()> {
    let sub_band = &storage.sub_bands[sub_band_idx];

    let quantised =
        component_info.quantization_info.quantization_style != QuantizationStyle::NoQuantization;
    let irreversible = component_info.wavelet_transform() == WaveletTransform::Irreversible97;
    let dequantization_step = {
        if !quantised {
            1.0
        } else {
            let (exponent, mantissa) =
                component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;

            let r_b = {
                let log_gain = match sub_band.sub_band_type {
                    SubBandType::LowLow => 0,
                    SubBandType::LowHigh => 1,
                    SubBandType::HighLow => 1,
                    SubBandType::HighHigh => 2,
                };

                component_info.size_info.precision as u16 + log_gain
            };

            crate::math::pow2i(r_b as i32 - exponent as i32) * (1.0 + (mantissa as f32) / 2048.0)
        }
    };

    let num_bitplanes = {
        let (exponent, _) = component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;
        // Equations (E-2) and (H-3).
        let num_bitplanes = (component_info.quantization_info.guard_bits as u16)
            .checked_add(exponent)
            .and_then(|x| x.checked_sub(1))
            .and_then(|x| x.checked_add(component_info.roi_shift as u16))
            .ok_or(DecodingError::InvalidBitplaneCount)?;

        if num_bitplanes > MAX_BITPLANE_COUNT as u16 {
            bail!(DecodingError::TooManyBitplanes);
        }

        num_bitplanes as u8
    };

    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            bitplane::decode(
                code_block,
                sub_band.sub_band_type,
                num_bitplanes,
                &component_info.coding_style.parameters.code_block_style,
                tile_ctx,
                storage,
                header.strict,
            )?;

            // Turn the signs and magnitudes into singular coefficients and
            // copy them into the sub-band.

            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;

            let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
            let mut base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;

            for (coefficients, coefficient_states) in
                tile_ctx.bit_plane_decode_context.coefficient_rows()
            {
                let out_row = &mut base_store[base_idx..];

                for ((output, coefficient), coefficient_state) in
                    out_row.iter_mut().zip(coefficients).zip(coefficient_states)
                {
                    *output = coefficient.reconstructed(
                        coefficient_state,
                        irreversible,
                        component_info.roi_shift,
                    );
                    *output *= dequantization_step;
                }

                base_idx += sub_band.rect.width() as usize;
            }
        }
    }

    Ok(())
}

fn apply_sign_shift(channel_data: &mut [ComponentData], component_infos: &[ComponentInfo]) {
    use crate::math::{Level, dispatch, f32x8};

    for (channel, component_info) in channel_data.iter_mut().zip(component_infos.iter()) {
        if component_info.size_info.is_signed {
            continue;
        }
        let offset = (1_u32 << (component_info.size_info.precision - 1)) as f32;
        dispatch!(Level::new(), simd => {
            let offset_v = f32x8::splat(simd, offset);
            for chunk in channel.container.chunks_exact_mut(8) {
                let v = f32x8::from_slice(simd, chunk);
                (v + offset_v).store(chunk);
            }
        });
    }
}

fn store(
    tile: &Tile<'_>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    channel_data: &mut ComponentData,
    component_info: &ComponentInfo,
) {
    let output = &tile_ctx.idwt_output;
    if output.rect.width() == 0 || output.rect.height() == 0 {
        return;
    }
    let rect = tile.output_rect(header);
    let size = &header.size_data;
    let width = size.image_width() as usize;
    let source_width = output.rect.width() as usize;
    let sx = u64::from(size.x_shrink_factor);
    let sy = u64::from(size.y_shrink_factor);
    let ox = u64::from(size.image_area_x_offset)
        .div_ceil(sx * u64::from(size.x_resolution_shrink_factor));
    let oy = u64::from(size.image_area_y_offset)
        .div_ceil(sy * u64::from(size.y_resolution_shrink_factor));
    let cx = u64::from(component_info.size_info.horizontal_resolution);
    let cy = u64::from(component_info.size_info.vertical_resolution);

    for y in rect.y0..rect.y1 {
        let source_y = ((u64::from(y) + oy) * sy / cy)
            .clamp(u64::from(output.rect.y0), u64::from(output.rect.y1 - 1));
        let source_row = (source_y as usize - output.rect.y0 as usize) * source_width;
        let output_row = y as usize * width;
        if sx == cx {
            let source_x = (u64::from(rect.x0) + ox)
                .saturating_sub(u64::from(output.rect.x0)) as usize;
            let count = rect.width() as usize;
            if source_x + count <= source_width {
                channel_data.container[output_row + rect.x0 as usize..][..count]
                    .copy_from_slice(&output.coefficients[source_row + source_x..][..count]);
                continue;
            }
        }
        // Expand subsampled components only inside this tile's output region.
        // This prevents a neighbouring tile's samples being transformed twice.
        for x in rect.x0..rect.x1 {
            let source_x = ((u64::from(x) + ox) * sx / cx)
                .clamp(u64::from(output.rect.x0), u64::from(output.rect.x1 - 1));
            channel_data.container[output_row + x as usize] =
                output.coefficients[source_row + source_x as usize - output.rect.x0 as usize];
        }
    }
}

