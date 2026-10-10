/*!
A memory-safe, pure-Rust JPEG 2000 decoder.

`hayro-jpeg2000` can decode both raw JPEG 2000 codestreams (`.j2c`) and images wrapped
inside the JP2 container format. The decoder supports the vast majority of features
defined in the JPEG2000 core coding system (ISO/IEC 15444-1) as well as some color
spaces from the extensions (ISO/IEC 15444-2). There are still some missing pieces
for some "obscure" features(like for example support for progression order
changes in tile-parts), but all features that actually commonly appear in real-life
images should be supported (if not, please open an issue!).

The decoder abstracts away most of the internal complexity of JPEG2000
and yields decoded image components, which can be inspected directly or packed
into a simple 8-bit image with either greyscale, RGB, CMYK or an ICC-based color
space.

# Example
```rust,no_run
use hayro_jpeg2000::{DecodeSettings, DecoderContext, Image};

let data = std::fs::read("image.jp2").unwrap();
let image = Image::new(&data, &DecodeSettings::default()).unwrap();

println!(
    "{}x{} image in {:?} with alpha={}",
    image.width(),
    image.height(),
    image.color_space(),
    image.has_alpha(),
);

let mut ctx = DecoderContext::default();
let decoded = image.decode(&mut ctx).unwrap();
let bitmap = decoded.data_u8();
```

If you want to see a more comprehensive example, please take a look
at the example in [GitHub](https://github.com/LaurenzV/hayro/blob/main/hayro-jpeg2000/examples/png.rs),
which shows you the main steps needed to convert a JPEG2000 image into PNG for example.

# Testing
The decoder has been tested against 20.000+ images scraped from random PDFs
on the internet and also passes a large part of the `OpenJPEG` test suite. So you
can expect the crate to perform decently in terms of decoding correctness.

# Performance
A decent amount of effort has already been put into optimizing this crate
(both in terms of raw performance but also memory allocations). However, there
are some more important optimizations that have not been implemented yet, so
there is definitely still room for improvement (and I am planning on implementing
them eventually).

Overall, you should expect this crate to have worse performance than `OpenJPEG`,
but the difference gap should not be too large.

# Safety
The default `std` feature adds no dependencies. With
`default-features = false`, the decoder uses only `core` and `alloc`.
The optional `simd` feature enables `fearless_simd`. Unsafe code is forbidden
in this crate via a crate-level attribute.

The crate is `no_std` compatible but requires an allocator to be available.
*/

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![forbid(missing_docs)]

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::{bail, err};
use crate::j2c::Header;
use crate::jp2::ImageBoxes;

use crate::jp2::cmap::ComponentMappingType;
use crate::jp2::colr::{CieLab, EnumeratedColorspace};
use crate::jp2::icc::ICCMetadata;

pub mod error;
pub(crate) mod math;

use crate::math::{Level, SIMD_WIDTH, Simd, dispatch, f32x8};
pub use error::{
    ColorError, DecodeError, DecodingError, FormatError, MarkerError, Result, TileError,
    ValidationError,
};
pub use j2c::{ComponentData, DecoderContext};
pub use jp2::DecodedImage;

mod j2c;
mod jp2;
pub(crate) mod reader;

/// JP2 signature box: 00 00 00 0C 6A 50 20 20
pub(crate) const JP2_MAGIC: &[u8] = b"\x00\x00\x00\x0C\x6A\x50\x20\x20";
/// Codestream signature: FF 4F FF 51 (SOC + SIZ markers)
pub(crate) const CODESTREAM_MAGIC: &[u8] = b"\xFF\x4F\xFF\x51";

/// Settings to apply during decoding.
#[derive(Debug, Copy, Clone)]
pub struct DecodeSettings {
    /// Whether palette indices should be resolved.
    ///
    /// JPEG2000 images can be stored in two different ways. First, by storing
    /// RGB values (depending on the color space) for each pixel. Secondly, by
    /// only storing a single index for each channel, and then resolving the
    /// actual color using the index.
    ///
    /// If you disable this option, in case you have an image with palette
    /// indices, they will not be resolved, but instead a grayscale image
    /// will be returned, with each pixel value corresponding to the palette
    /// index of the location.
    pub resolve_palette_indices: bool,
    /// Whether strict mode should be enabled when decoding.
    ///
    /// It is recommended to leave this flag disabled, unless you have a
    /// specific reason not to.
    pub strict: bool,
    /// A hint for the target resolution that the image should be decoded at.
    pub target_resolution: Option<(u32, u32)>,
    /// Maximum number of decoded image pixels.
    pub max_pixels: u64,
    /// Maximum bytes for component planes, coefficient storage, IDWT output
    /// and scratch, including the peak planes needed for palette expansion.
    ///
    /// Compressed input, packet metadata, code-block scratch and caller-owned
    /// packed output are excluded.
    pub max_bytes: u64,
}

impl Default for DecodeSettings {
    fn default() -> Self {
        Self {
            resolve_palette_indices: true,
            strict: false,
            target_resolution: None,
            max_pixels: 1 << 28,
            max_bytes: 1 << 31,
        }
    }
}

/// A JPEG2000 image or codestream.
pub struct Image<'a> {
    /// The codestream containing the data to decode.
    pub(crate) codestream: &'a [u8],
    /// The header of the J2C codestream.
    pub(crate) header: Header<'a>,
    /// The JP2 boxes of the image. In the case of a raw codestream, we
    /// will synthesize the necessary boxes.
    pub(crate) boxes: ImageBoxes,
    /// Settings that should be applied during decoding.
    pub(crate) settings: DecodeSettings,
    /// Whether the image has an alpha channel.
    pub(crate) has_alpha: bool,
    /// The color space of the image.
    pub(crate) color_space: ColorSpace,
}

impl<'a> Image<'a> {
    /// Try to create a new JPEG2000 image from the given data.
    pub fn new(data: &'a [u8], settings: &DecodeSettings) -> Result<Self> {
        if data.starts_with(JP2_MAGIC) {
            jp2::parse(data, *settings)
        } else if data.starts_with(CODESTREAM_MAGIC) {
            j2c::parse(data, settings)
        } else {
            err!(FormatError::InvalidSignature)
        }
    }

    /// Whether the image has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Whether the alpha channel describes premultiplied colour samples.
    ///
    /// The decoder reports this representation without unpremultiplying it.
    pub fn has_premultiplied_alpha(&self) -> bool {
        self.has_alpha && self.boxes.channel_definition.as_ref()
            .is_some_and(|cdef| cdef.has_premultiplied_alpha())
    }

    /// The color space of the image.
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// The width of the image.
    pub fn width(&self) -> u32 {
        self.header.size_data.image_width()
    }

    /// The height of the image.
    pub fn height(&self) -> u32 {
        self.header.size_data.image_height()
    }

    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    pub fn original_bit_depth(&self) -> u8 {
        // Note that this only works if all components have the same precision.
        self.header.component_infos[0].size_info.precision
    }

    /// Decode the image and return its decoded components.
    pub fn decode<'b>(
        &'a self,
        decoder_context: &'b mut DecoderContext<'a>,
    ) -> Result<DecodedImage<'b>> {
        let settings = &self.settings;
        let palette_components = if settings.resolve_palette_indices && self.boxes.palette.is_some() {
            self.boxes.component_mapping.as_ref().map_or(0, |mapping| mapping.entries.len())
        } else {
            0
        };
        j2c::decode(
            self.codestream, &self.header, decoder_context, settings, palette_components,
        )?;
        let mut decoded_image = DecodedImage {
            decoded_components: &mut decoder_context.channel_data,
            boxes: self.boxes.clone(),
            has_premultiplied_alpha: self.has_premultiplied_alpha(),
        };

        // Resolve palette indices.
        if settings.resolve_palette_indices {
            let components = core::mem::take(decoded_image.decoded_components);
            *decoded_image.decoded_components =
                resolve_palette_indices(components, &decoded_image.boxes)?;
        }

        if (settings.resolve_palette_indices || decoded_image.boxes.palette.is_none())
            && let Some(cdef) = &decoded_image.boxes.channel_definition
        {
            cdef.reorder(decoded_image.decoded_components)?;
        }

        // Note that this is only valid if all images have the same bit depth.
        let bit_depth = decoded_image.decoded_components[0].bit_depth;
        convert_color_space(&mut decoded_image, bit_depth)?;

        Ok(decoded_image)
    }
}

pub(crate) fn resolve_alpha_and_color_space(
    boxes: &ImageBoxes,
    header: &Header<'_>,
    settings: &DecodeSettings,
) -> Result<(ColorSpace, bool)> {
    let actual_num_components = header.component_infos.len();
    let defined_components = if boxes.palette.is_some() {
        boxes.component_mapping.as_ref()
            .ok_or(ColorError::PaletteResolutionFailed)?.entries.len()
    } else {
        actual_num_components
    };
    if let Some(cdef) = &boxes.channel_definition {
        cdef.permutation(defined_components)?;
    }

    let num_components = if settings.resolve_palette_indices && boxes.palette.is_some() {
        defined_components
    } else {
        actual_num_components
    };
    if num_components == 0 || num_components > u8::MAX as usize {
        bail!(ValidationError::TooManyChannels);
    }
    let mut has_alpha = boxes.channel_definition.as_ref().is_some_and(|cdef| cdef.has_alpha());
    if !settings.resolve_palette_indices && boxes.palette.is_some() {
        has_alpha = false;
        return Ok((ColorSpace::Gray, has_alpha));
    }

    let colour_components = num_components - usize::from(has_alpha);
    let mut color_space = get_color_space(boxes, colour_components)?;
    if color_space.num_channels() as usize != colour_components {
        // Opacity is authoritative only when a CDEF marks it.
        color_space = match colour_components {
            1 => ColorSpace::Gray,
            3 => ColorSpace::RGB,
            4 => ColorSpace::CMYK,
            _ => ColorSpace::Unknown {
                num_channels: colour_components as u8,
            },
        };
    }

    Ok((color_space, has_alpha))
}

/// The color space of the image.
#[derive(Debug, Clone)]
pub enum ColorSpace {
    /// A grayscale image.
    Gray,
    /// An RGB image.
    RGB,
    /// A CMYK image.
    CMYK,
    /// An unknown color space.
    Unknown {
        /// The number of channels of the color space.
        num_channels: u8,
    },
    /// An image based on an ICC profile.
    Icc {
        /// The raw data of the ICC profile.
        profile: Arc<[u8]>,
        /// The number of channels used by the ICC profile.
        num_channels: u8,
    },
}

impl ColorSpace {
    /// Return the number of expected channels for the color space.
    pub fn num_channels(&self) -> u8 {
        match self {
            Self::Gray => 1,
            Self::RGB => 3,
            Self::CMYK => 4,
            Self::Unknown { num_channels } => *num_channels,
            Self::Icc {
                num_channels: num_components,
                ..
            } => *num_components,
        }
    }
}

/// A bitmap storing the decoded result of the image.
pub struct Bitmap {
    /// The color space of the image.
    pub color_space: ColorSpace,
    /// The raw pixel data of the image. The result will always be in
    /// 8-bit (in case the original image had a different bit-depth,
    /// hayro-jpeg2000 always scales to 8-bit).
    ///
    /// The size is guaranteed to equal
    /// `width * height * (num_channels + (if has_alpha { 1 } else { 0 })`.
    /// Pixels are interleaved on a per-channel basis, the alpha channel always
    /// appearing as the last channel, if available.
    pub data: Vec<u8>,
    /// Whether the image has an alpha channel.
    pub has_alpha: bool,
    /// The width of the image.
    pub width: u32,
    /// The height of the image.
    pub height: u32,
    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    pub original_bit_depth: u8,
}

impl DecodedImage<'_> {
    /// The decoded components of the image.
    pub fn components(&self) -> &[ComponentData] {
        self.decoded_components
    }

    /// Whether the alpha channel describes premultiplied colour samples.
    pub fn has_premultiplied_alpha(&self) -> bool {
        self.has_premultiplied_alpha
    }

    /// Return interleaved unsigned 16-bit samples, scaled per component,
    /// rounded to nearest and clamped. Signed ranges are biased to unsigned.
    pub fn data_u16(&self) -> Vec<u16> {
        let components = self.components();
        let buffer_size = components[0].samples().len() * components.len();
        let mut buf = vec![0; buffer_size];
        self.store_u16_into(&mut buf);
        buf
    }

    /// Store scaled unsigned 16-bit samples into an exactly sized buffer.
    /// Signed ranges are biased to unsigned; planar samples remain signed.
    pub fn store_u16_into(&self, buf: &mut [u16]) {
        let components = self.components();
        let count = components[0].samples().len();
        assert_eq!(buf.len(), count * components.len());
        for (index, component) in components.iter().enumerate() {
            let (offset, scale) = component_conversion(component, 65535.0);
            for (sample, &value) in component.samples().iter().enumerate() {
                buf[sample * components.len() + index] =
                    math::round_f32(((value + offset) * scale).clamp(0.0, 65535.0)) as u16;
            }
        }
    }

    /// Return the decoded image as interleaved unsigned 8-bit sample data.
    /// Signed ranges are biased to unsigned; planar samples remain signed.
    pub fn data_u8(&self) -> Vec<u8> {
        let components = self.components();
        let buffer_size = components[0].samples().len() * components.len();
        let mut buf = vec![0; buffer_size];
        self.store_u8_into(&mut buf);
        buf
    }

    /// Store the decoded image as interleaved unsigned 8-bit sample data into `buf`.
    ///
    /// The buffer must have the correct size.
    pub fn store_u8_into(&self, buf: &mut [u8]) {
        interleave_and_convert(self, buf);
    }
}

fn interleave_and_convert(image: &DecodedImage<'_>, buf: &mut [u8]) {
    let components = &*image.decoded_components;
    let num_components = components.len();

    let mut all_same_bit_depth = Some(components[0].bit_depth);

    for component in components.iter().skip(1) {
        if Some(component.bit_depth) != all_same_bit_depth {
            all_same_bit_depth = None;
        }
    }

    let max_len = components[0].container.truncated().len();

    assert_eq!(buf.len(), max_len * num_components);
    let mut output_iter = buf.iter_mut();

    if all_same_bit_depth == Some(8) && num_components <= 4
        && components.iter().all(|component| !component.is_signed)
    {
        // Fast path for the common case.
        match num_components {
            // Gray-scale.
            1 => {
                for (output, input) in output_iter.zip(
                    components[0]
                        .container
                        .iter()
                        .map(|v| math::round_f32(*v) as u8),
                ) {
                    *output = input;
                }
            }
            // Gray-scale with alpha.
            2 => {
                let c0 = &components[0];
                let c1 = &components[1];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                }
            }
            // RGB
            3 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];
                let c2 = &c2.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c2[i]) as u8;
                }
            }
            // RGBA or CMYK.
            4 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];
                let c3 = &components[3];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];
                let c2 = &c2.container[..max_len];
                let c3 = &c3.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c2[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c3[i]) as u8;
                }
            }
            _ => unreachable!(),
        }
    } else {
        // Compute each component's conversion once, then write its samples.
        for (index, component) in components.iter().enumerate() {
            let (offset, scale) = component_conversion(component, 255.0);
            for (sample, &value) in component.samples().iter().enumerate() {
                buf[sample * num_components + index] =
                    math::round_f32(((value + offset) * scale).clamp(0.0, 255.0)) as u8;
            }
        }
    }
}

fn component_conversion(component: &ComponentData, maximum: f32) -> (f32, f32) {
    let offset = if component.is_signed {
        (1_u64 << (component.bit_depth - 1)) as f32
    } else {
        0.0
    };
    let scale = maximum / ((1_u64 << component.bit_depth) - 1) as f32;
    (offset, scale)
}

fn convert_color_space(image: &mut DecodedImage<'_>, bit_depth: u8) -> Result<()> {
    if let Some(jp2::colr::ColorSpace::Enumerated(e)) = &image
        .boxes
        .color_specification
        .as_ref()
        .map(|i| &i.color_space)
    {
        match e {
            EnumeratedColorspace::Sycc => {
                dispatch!(Level::new(), simd => {
                    sycc_to_rgb(simd, image.decoded_components, bit_depth)
                })?;
            }
            EnumeratedColorspace::CieLab(cielab) => {
                dispatch!(Level::new(), simd => {
                    cielab_to_rgb(simd, image.decoded_components, bit_depth, cielab)
                })?;
            }
            EnumeratedColorspace::Ycck => {
                dispatch!(Level::new(), simd => {
                    ycck_to_cmyk(simd, image.decoded_components, bit_depth)
                })?;
            }
            _ => {}
        }
    }

    Ok(())
}

fn get_color_space(boxes: &ImageBoxes, num_components: usize) -> Result<ColorSpace> {
    let cs = match boxes
        .color_specification
        .as_ref()
        .map(|c| &c.color_space)
        .unwrap_or(&jp2::colr::ColorSpace::Unknown)
    {
        jp2::colr::ColorSpace::Enumerated(e) => {
            match e {
                EnumeratedColorspace::Cmyk | EnumeratedColorspace::Ycck => ColorSpace::CMYK,
                EnumeratedColorspace::Srgb => ColorSpace::RGB,
                EnumeratedColorspace::RommRgb => {
                    // Use an ICC profile to process the RommRGB color space.
                    ColorSpace::Icc {
                        profile: Arc::from(include_bytes!("../assets/ProPhoto-v2-micro.icc").as_slice()),
                        num_channels: 3,
                    }
                }
                EnumeratedColorspace::EsRgb => ColorSpace::RGB,
                EnumeratedColorspace::Greyscale => ColorSpace::Gray,
                EnumeratedColorspace::Sycc => ColorSpace::RGB,
                EnumeratedColorspace::CieLab(_) => ColorSpace::Icc {
                    profile: Arc::from(include_bytes!("../assets/LAB.icc").as_slice()),
                    num_channels: 3,
                },
                _ => bail!(FormatError::Unsupported),
            }
        }
        jp2::colr::ColorSpace::Icc(icc) => {
            if let Some(metadata) = ICCMetadata::from_data(icc) {
                ColorSpace::Icc {
                    profile: icc.clone(),
                    num_channels: metadata.color_space.num_components(),
                }
            } else {
                // See OPENJPEG test orb-blue10-lin-jp2.jp2. They seem to
                // assume RGB in this case (even though the image has 4
                // components with no opacity channel, they assume RGBA instead
                // of CMYK).
                ColorSpace::RGB
            }
        }
        jp2::colr::ColorSpace::Unknown => match num_components {
            1 => ColorSpace::Gray,
            3 => ColorSpace::RGB,
            4 => ColorSpace::CMYK,
            _ => ColorSpace::Unknown {
                num_channels: num_components as u8,
            },
        },
    };

    Ok(cs)
}

fn resolve_palette_indices(
    components: Vec<ComponentData>,
    boxes: &ImageBoxes,
) -> Result<Vec<ComponentData>> {
    let Some(palette) = boxes.palette.as_ref() else {
        // Nothing to resolve.
        return Ok(components);
    };

    let mapping = boxes.component_mapping.as_ref().unwrap();
    let max_index = palette.entries.len() as i64 - 1;
    let mut resolved = Vec::with_capacity(mapping.entries.len());

    for entry in &mapping.entries {
        let component_idx = entry.component_index as usize;
        let component = components
            .get(component_idx)
            .ok_or(ColorError::PaletteResolutionFailed)?;

        match entry.mapping_type {
            ComponentMappingType::Direct => resolved.push(component.clone()),
            ComponentMappingType::Palette { column } => {
                let column_idx = column as usize;
                let column_info = palette
                    .columns
                    .get(column_idx)
                    .ok_or(ColorError::PaletteResolutionFailed)?;

                if column_info.bit_depth > 31 {
                    bail!(ColorError::PaletteResolutionFailed);
                }
                let mut mapped =
                    Vec::with_capacity(component.container.truncated().len() + SIMD_WIDTH);

                for &sample in component.container.truncated() {
                    // Lossy decoding can reconstruct indices outside the palette.
                    let index = (math::round_f32(sample) as i64).clamp(0, max_index);
                    let value = palette
                        .map(index as usize, column_idx)
                        .ok_or(ColorError::PaletteResolutionFailed)?;
                    mapped.push(value as f32);
                }

                resolved.push(ComponentData {
                    container: math::SimdBuffer::new(mapped),
                    bit_depth: column_info.bit_depth,
                    is_signed: false,
                });
            }
        }
    }

    Ok(resolved)
}

#[inline(always)]
fn cielab_to_rgb<S: Simd>(
    simd: S,
    components: &mut [ComponentData],
    bit_depth: u8,
    lab: &CieLab,
) -> Result<()> {
    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::LabConversionFailed)?;

    let [l, a, b] = head else {
        unreachable!();
    };

    let prec0 = l.bit_depth;
    let prec1 = a.bit_depth;
    let prec2 = b.bit_depth;

    // Prevent underflows/divisions by zero further below.
    if prec0 < 4 || prec1 < 4 || prec2 < 4 {
        bail!(ColorError::LabConversionFailed);
    }

    let rl = lab.rl.unwrap_or(100);
    let ra = lab.ra.unwrap_or(170);
    let rb = lab.rb.unwrap_or(200);
    let ol = lab.ol.unwrap_or(0);
    let oa = lab.oa.unwrap_or(1 << (bit_depth - 1));
    let ob = lab
        .ob
        .unwrap_or((1 << (bit_depth - 2)) + (1 << (bit_depth - 3)));

    // Copied from OpenJPEG.
    let min_l = -(rl as f32 * ol as f32) / ((1 << prec0) - 1) as f32;
    let max_l = min_l + rl as f32;
    let min_a = -(ra as f32 * oa as f32) / ((1 << prec1) - 1) as f32;
    let max_a = min_a + ra as f32;
    let min_b = -(rb as f32 * ob as f32) / ((1 << prec2) - 1) as f32;
    let max_b = min_b + rb as f32;

    let bit_max = (1_u32 << bit_depth) - 1;

    // Note that we are not doing the actual conversion with the ICC profile yet,
    // just decoding the raw LAB values.
    // We leave applying the ICC profile to the user.
    let divisor_l = ((1 << prec0) - 1) as f32;
    let divisor_a = ((1 << prec1) - 1) as f32;
    let divisor_b = ((1 << prec2) - 1) as f32;

    let scale_l_final = bit_max as f32 / 100.0;
    let scale_ab_final = bit_max as f32 / 255.0;

    let l_offset = min_l * scale_l_final;
    let l_scale = (max_l - min_l) / divisor_l * scale_l_final;
    let a_offset = (min_a + 128.0) * scale_ab_final;
    let a_scale = (max_a - min_a) / divisor_a * scale_ab_final;
    let b_offset = (min_b + 128.0) * scale_ab_final;
    let b_scale = (max_b - min_b) / divisor_b * scale_ab_final;

    let l_offset_v = f32x8::splat(simd, l_offset);
    let l_scale_v = f32x8::splat(simd, l_scale);
    let a_offset_v = f32x8::splat(simd, a_offset);
    let a_scale_v = f32x8::splat(simd, a_scale);
    let b_offset_v = f32x8::splat(simd, b_offset);
    let b_scale_v = f32x8::splat(simd, b_scale);

    // Note that we are not doing the actual conversion with the ICC profile yet,
    // just decoding the raw LAB values.
    // We leave applying the ICC profile to the user.
    for ((l_chunk, a_chunk), b_chunk) in l
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(a.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(b.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let l_v = f32x8::from_slice(simd, l_chunk);
        let a_v = f32x8::from_slice(simd, a_chunk);
        let b_v = f32x8::from_slice(simd, b_chunk);

        l_v.mul_add(l_scale_v, l_offset_v).store(l_chunk);
        a_v.mul_add(a_scale_v, a_offset_v).store(a_chunk);
        b_v.mul_add(b_scale_v, b_offset_v).store(b_chunk);
    }

    Ok(())
}

#[inline(always)]
fn ycck_to_cmyk<S: Simd>(simd: S, components: &mut [ComponentData], bit_depth: u8) -> Result<()> {
    // Convert YCbCr to CMY, preserving K (T.801, Table M.25).
    sycc_to_rgb(simd, components, bit_depth)?;

    let max_value = ((1_u32 << bit_depth) - 1) as f32;
    let max_v = f32x8::splat(simd, max_value);

    for component in components.iter_mut().take(3) {
        let mut chunks = component.container.chunks_exact_mut(SIMD_WIDTH);
        for chunk in chunks.by_ref() {
            (max_v - f32x8::from_slice(simd, chunk)).store(chunk);
        }
        for value in chunks.into_remainder() {
            *value = max_value - *value;
        }
    }

    Ok(())
}

fn sycc_to_rgb<S: Simd>(simd: S, components: &mut [ComponentData], bit_depth: u8) -> Result<()> {
    let offset = (1_u32 << (bit_depth as u32 - 1)) as f32;
    let max_value = ((1_u32 << bit_depth as u32) - 1) as f32;

    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::SyccConversionFailed)?;

    let [y, cb, cr] = head else {
        unreachable!();
    };

    let offset_v = f32x8::splat(simd, offset);
    let max_v = f32x8::splat(simd, max_value);
    let zero_v = f32x8::splat(simd, 0.0);
    let cr_to_r = f32x8::splat(simd, 1.402);
    let cb_to_g = f32x8::splat(simd, -0.344136);
    let cr_to_g = f32x8::splat(simd, -0.714136);
    let cb_to_b = f32x8::splat(simd, 1.772);

    for ((y_chunk, cb_chunk), cr_chunk) in y
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(cb.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(cr.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let y_v = f32x8::from_slice(simd, y_chunk);
        let cb_v = f32x8::from_slice(simd, cb_chunk) - offset_v;
        let cr_v = f32x8::from_slice(simd, cr_chunk) - offset_v;

        // r = y + 1.402 * cr
        let r = cr_v.mul_add(cr_to_r, y_v);
        // g = y - 0.344136 * cb - 0.714136 * cr
        let g = cr_v.mul_add(cr_to_g, cb_v.mul_add(cb_to_g, y_v));
        // b = y + 1.772 * cb
        let b = cb_v.mul_add(cb_to_b, y_v);

        r.min(max_v).max(zero_v).store(y_chunk);
        g.min(max_v).max(zero_v).store(cb_chunk);
        b.min(max_v).max(zero_v).store(cr_chunk);
    }

    Ok(())
}
