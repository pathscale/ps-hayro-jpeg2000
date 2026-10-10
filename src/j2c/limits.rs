//! Limits for component planes and reusable reconstruction buffers.

use super::Header;
use super::tile::{ResolutionTile, Tile};
use crate::DecodeSettings;
use crate::error::{Result, ValidationError};

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
    samples.checked_add(crate::math::SIMD_WIDTH - samples % crate::math::SIMD_WIDTH)
        .ok_or_else(|| memory_error().into())
}

fn check_bytes(samples: usize, settings: &DecodeSettings) -> Result<()> {
    let bytes = samples.checked_mul(size_of::<f32>())
        .ok_or_else(memory_error)?;
    if bytes > isize::MAX as usize || bytes as u64 > settings.max_bytes {
        return Err(memory_error().into());
    }
    Ok(())
}

pub(crate) fn check_image(header: &Header<'_>, settings: &DecodeSettings) -> Result<usize> {
    let size = &header.size_data;
    let pixels = u64::from(size.image_width()) * u64::from(size.image_height());
    if pixels > settings.max_pixels {
        return Err(ValidationError::PixelLimitExceeded.into());
    }
    let samples = checked_area(size.image_width(), size.image_height())?;
    let planes = plane_size(samples)?.checked_mul(header.component_infos.len())
        .ok_or_else(memory_error)?;
    check_bytes(planes, settings)?;
    Ok(samples)
}

impl BufferSizes {
    pub(crate) fn new(
        header: &Header<'_>,
        tiles: &[Tile<'_>],
        settings: &DecodeSettings,
        palette_components: usize,
    ) -> Result<Self> {
        let samples = check_image(header, settings)?;
        let mut sizes = Self { samples, coefficients: 0, output: 0, scratch: 0 };
        for tile in tiles {
            for component_tile in tile.component_tiles() {
                let levels = component_tile.component_info.num_resolution_levels()
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

        // Palette expansion retains the source planes until the mapped planes
        // have been built. Reserve their peak alongside reconstruction scratch.
        let planes = plane_size(samples)?.checked_mul(header.component_infos.len())
            .ok_or_else(memory_error)?;
        let palette_planes = samples.checked_add(crate::math::SIMD_WIDTH)
            .and_then(|n| n.checked_mul(palette_components))
            .ok_or_else(memory_error)?;
        let total = planes.checked_add(palette_planes)
            .and_then(|n| n.checked_add(sizes.coefficients))
            .and_then(|n| n.checked_add(sizes.output))
            .and_then(|n| n.checked_add(sizes.scratch))
            .ok_or_else(memory_error)?;
        check_bytes(total, settings)?;
        Ok(sizes)
    }
}

