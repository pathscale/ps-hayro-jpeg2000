//! Limits for component planes and reusable reconstruction buffers.

use super::tile::{ResolutionTile, Tile};
use super::Header;
use crate::error::{Result, ValidationError};
use crate::DecodeSettings;

pub(crate) struct BufferSizes {
    pub(crate) samples: usize,
    pub(crate) coefficients: usize,
    pub(crate) output: usize,
    pub(crate) scratch: usize,
}

fn memory_error() -> ValidationError {
    ValidationError::MemoryLimitExceeded
}

fn checked_area(width: u32, height: u32) -> Result<usize> {
    usize::try_from(u64::from(width) * u64::from(height)).map_err(|_| memory_error().into())
}

fn plane_size(samples: usize) -> Result<usize> {
    samples
        .checked_add(crate::math::SIMD_WIDTH - samples % crate::math::SIMD_WIDTH)
        .ok_or_else(|| memory_error().into())
}

pub(crate) fn raw_plane_bytes(header: &Header<'_>, samples: usize) -> Result<usize> {
    let plane = plane_size(samples)?
        .checked_mul(header.component_infos.len())
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(memory_error)?;
    check_allocation_bytes(plane)?;
    Ok(plane)
}

fn check_allocation_bytes(bytes: usize) -> Result<()> {
    if bytes > isize::MAX as usize {
        return Err(memory_error().into());
    }
    Ok(())
}

pub(crate) fn check_total_bytes(bytes: usize, settings: &DecodeSettings) -> Result<()> {
    check_allocation_bytes(bytes)?;
    if u64::try_from(bytes).map_err(|_| memory_error())? > settings.max_bytes {
        return Err(memory_error().into());
    }
    Ok(())
}

pub(crate) fn tile_table_bytes(header: &Header<'_>, tile_count: usize) -> Result<usize> {
    let mut per_tile = size_of::<Tile<'_>>();
    for info in &header.component_infos {
        let precincts = info
            .coding_style
            .parameters
            .precinct_exponents
            .len()
            .checked_mul(size_of::<(u8, u8)>())
            .ok_or_else(memory_error)?;
        let quantization = info
            .quantization_info
            .step_sizes
            .len()
            .checked_mul(size_of::<super::codestream::StepSize>())
            .ok_or_else(memory_error)?;
        let component = size_of_val(info)
            .checked_add(precincts)
            .and_then(|n| n.checked_add(quantization))
            .ok_or_else(memory_error)?;
        per_tile = per_tile.checked_add(component).ok_or_else(memory_error)?;
    }
    check_allocation_bytes(per_tile)?;
    let bytes = tile_count.checked_mul(per_tile).ok_or_else(memory_error)?;
    check_allocation_bytes(bytes)?;
    Ok(bytes)
}

pub(crate) fn check_image(header: &Header<'_>, settings: &DecodeSettings) -> Result<usize> {
    let size = &header.size_data;
    let pixels = u64::from(size.image_width()) * u64::from(size.image_height());
    if pixels > settings.max_pixels {
        return Err(ValidationError::PixelLimitExceeded.into());
    }
    let samples = checked_area(size.image_width(), size.image_height())?;
    let bytes = raw_plane_bytes(header, samples)?;
    check_total_bytes(bytes, settings)?;
    Ok(samples)
}

impl BufferSizes {
    pub(crate) fn new(
        header: &Header<'_>,
        tiles: &[Tile<'_>],
        settings: &DecodeSettings,
        palette_components: usize,
        tile_bytes: usize,
    ) -> Result<Self> {
        let samples = check_image(header, settings)?;
        let mut sizes = Self {
            samples,
            coefficients: 0,
            output: 0,
            scratch: 0,
        };
        for tile in tiles {
            for component_tile in tile.component_tiles() {
                let levels = component_tile
                    .component_info
                    .num_resolution_levels()
                    .checked_sub(header.skipped_resolution_levels)
                    .filter(|&n| n > 0)
                    .ok_or(ValidationError::InvalidComponentMetadata)?;
                let top = ResolutionTile::new(component_tile, levels - 1);
                let count = checked_area(top.rect.width(), top.rect.height())?;
                sizes.coefficients = sizes.coefficients.max(count);
                let padded_area = |r: &ResolutionTile<'_>| -> Result<usize> {
                    let width = r.rect.width().checked_add(1).ok_or_else(memory_error)?;
                    let height = r.rect.height().checked_add(1).ok_or_else(memory_error)?;
                    checked_area(width, height)
                };
                sizes.output = sizes.output.max(if levels == 1 {
                    count
                } else {
                    padded_area(&top)?
                });
                if levels > 2 {
                    let previous = ResolutionTile::new(component_tile, levels - 2);
                    sizes.scratch = sizes.scratch.max(padded_area(&previous)?);
                }
            }
        }
        let planes = raw_plane_bytes(header, samples)?;
        let palette_floats = samples
            .checked_add(crate::math::SIMD_WIDTH)
            .and_then(|n| n.checked_mul(palette_components))
            .ok_or_else(memory_error)?;
        let reconstruction = sizes
            .coefficients
            .checked_add(sizes.output)
            .and_then(|n| n.checked_add(sizes.scratch))
            .ok_or_else(memory_error)?;
        let reconstruction_bytes = reconstruction
            .checked_mul(size_of::<f32>())
            .ok_or_else(memory_error)?;
        let palette_bytes = palette_floats
            .checked_mul(size_of::<f32>())
            .ok_or_else(memory_error)?;
        let total = tile_bytes
            .checked_add(planes)
            .and_then(|n| n.checked_add(palette_bytes))
            .and_then(|n| n.checked_add(reconstruction_bytes))
            .ok_or_else(memory_error)?;
        check_total_bytes(total, settings)?;
        Ok(sizes)
    }
}
