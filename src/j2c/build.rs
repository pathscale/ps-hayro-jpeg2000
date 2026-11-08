//! Building and setting up decompositions, sub-bands, precincts and code-blocks.

use super::decode::{DecompositionStorage, TileDecompositions};
use super::limits::check_total_bytes;
use super::rect::IntRect;
use super::tag_tree::TagTree;
use super::tile::{ResolutionTile, Tile};
use crate::DecodeSettings;
use crate::error::{DecodingError, Result, ValidationError};
use alloc::vec::Vec;
use core::iter;
use core::ops::Range;

/// Tracks the estimated live payload bytes for a decoding phase.
#[derive(Clone, Copy)]
pub(crate) struct BuildBudget<'a> {
    total_bytes: usize,
    settings: &'a DecodeSettings,
}

impl<'a> BuildBudget<'a> {
    pub(crate) fn new(fixed_bytes: usize, settings: &'a DecodeSettings) -> Self {
        Self {
            total_bytes: fixed_bytes,
            settings,
        }
    }

    pub(crate) fn reserve<T>(&mut self, values: &mut Vec<T>, additional: usize) -> Result<()> {
        self.reserve_entries(values, additional, false)
    }

    pub(crate) fn reserve_exact<T>(&mut self, values: &mut Vec<T>, additional: usize) -> Result<()> {
        self.reserve_entries(values, additional, true)
    }

    fn reserve_entries<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
        exact: bool,
    ) -> Result<()> {
        let error = ValidationError::MemoryLimitExceeded;
        let mut next = *self;
        next.charge::<T>(additional)?;
        let reserved = if exact {
            values.try_reserve_exact(additional)
        } else {
            values.try_reserve(additional)
        };
        reserved.map_err(|_| error)?;
        *self = next;
        Ok(())
    }

    pub(crate) fn charge<T>(&mut self, count: usize) -> Result<()> {
        self.charge_bytes(Self::bytes::<T>(count)?)
    }

    pub(crate) fn charge_bytes(&mut self, bytes: usize) -> Result<()> {
        let total = self.total_bytes
            .checked_add(bytes)
            .ok_or(ValidationError::MemoryLimitExceeded)?;
        check_total_bytes(total, self.settings)?;
        self.total_bytes = total;
        Ok(())
    }

    pub(crate) fn bytes<T>(count: usize) -> Result<usize> {
        count
            .checked_mul(size_of::<T>())
            .ok_or_else(|| ValidationError::MemoryLimitExceeded.into())
    }

    pub(crate) fn total(self) -> usize {
        self.total_bytes
    }

    pub(crate) fn release_bytes(&mut self, bytes: usize) -> Result<()> {
        self.total_bytes = self.total_bytes
            .checked_sub(bytes)
            .ok_or(ValidationError::MemoryLimitExceeded)?;
        Ok(())
    }

    pub(crate) fn copy_slice<T: Copy>(&mut self, values: &[T]) -> Result<Vec<T>> {
        let mut copy = Vec::new();
        self.reserve_exact(&mut copy, values.len())?;
        copy.extend_from_slice(values);
        Ok(copy)
    }
}

#[cfg(test)]
mod construction_budget_tests {
    use super::*;

    #[test]
    fn reused_capacity_still_obeys_the_tile_budget() {
        let settings = DecodeSettings { max_bytes: 16, ..DecodeSettings::default() };
        let mut entries = Vec::<u64>::with_capacity(64);
        let mut budget = BuildBudget::new(8, &settings);
        budget.reserve(&mut entries, 1).unwrap();
        entries.push(7);
        assert!(budget.reserve(&mut entries, 1).is_err());
        assert_eq!(entries.as_slice(), &[7]);
        entries.clear();
        let mut next_tile = BuildBudget::new(8, &settings);
        next_tile.reserve(&mut entries, 1).unwrap();
    }

    #[test]
    fn overflow_is_rejected_before_vector_growth() {
        let settings = DecodeSettings { max_bytes: u64::MAX, ..DecodeSettings::default() };
        let mut entries = Vec::<u64>::new();
        let mut budget = BuildBudget::new(0, &settings);
        assert!(budget.reserve(&mut entries, usize::MAX).is_err());
        assert_eq!(entries.capacity(), 0);
        budget.reserve(&mut entries, 1).unwrap();
    }
}

/// Build the structures needed to process a tile's code-blocks.
pub(crate) fn build(
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
    skipped_resolution_levels: u8,
    budget: &mut BuildBudget<'_>,
) -> Result<()> {
    let mut max_coefficients = 0;
    for component_tile in tile.component_tiles() {
        let decoded_resolutions =
            component_tile.component_info.num_resolution_levels() - skipped_resolution_levels;
        let top_resolution = ResolutionTile::new(component_tile, decoded_resolutions - 1);
        let samples = usize::try_from(top_resolution.rect.area())
            .map_err(|_| ValidationError::ImageTooLarge)?;
        max_coefficients = max_coefficients.max(samples);
    }

    // BufferSizes preflights and allocates the largest tile before construction.
    storage.coefficients.resize(max_coefficients, 0.0);

    for component_tile in tile.component_tiles() {
        let mut coefficient_counter = 0_usize;
        let decoded_resolutions =
            component_tile.component_info.num_resolution_levels() - skipped_resolution_levels;
        let d_start = storage.decompositions.len();
        let mut resolution_tiles = component_tile.resolution_tiles();

        let mut build_sub_band = |sub_band_type: SubBandType,
                                  resolution_tile: &ResolutionTile<'_>,
                                  storage: &mut DecompositionStorage<'_>,
                                  budget: &mut BuildBudget<'_>|
         -> Result<usize> {
            let sub_band_rect = resolution_tile.sub_band_rect(sub_band_type);
            let precincts = build_precincts(resolution_tile, sub_band_rect, storage, budget)?;

            // No need to allocate coefficients for skipped resolution levels.
            let added_coefficients = if resolution_tile.resolution < decoded_resolutions {
                usize::try_from(sub_band_rect.area()).map_err(|_| ValidationError::ImageTooLarge)?
            } else {
                0
            };
            let next_coefficient_counter = coefficient_counter
                .checked_add(added_coefficients)
                .ok_or(ValidationError::ImageTooLarge)?;
            let coefficients = coefficient_counter..next_coefficient_counter;
            coefficient_counter = next_coefficient_counter;

            let idx = storage.sub_bands.len();
            budget.reserve(&mut storage.sub_bands, 1)?;
            storage.sub_bands.push(SubBand {
                sub_band_type,
                rect: sub_band_rect,
                precincts,
                coefficients,
            });

            Ok(idx)
        };

        // Resolution 0 always is the LL sub-band.
        let ll_resolution_tile = resolution_tiles.next().unwrap();
        let first_ll_sub_band =
            build_sub_band(SubBandType::LowLow, &ll_resolution_tile, storage, budget)?;

        for resolution_tile in resolution_tiles {
            let decomposition = Decomposition {
                sub_bands: [
                    build_sub_band(SubBandType::HighLow, &resolution_tile, storage, budget)?,
                    build_sub_band(SubBandType::LowHigh, &resolution_tile, storage, budget)?,
                    build_sub_band(SubBandType::HighHigh, &resolution_tile, storage, budget)?,
                ],
                rect: resolution_tile.rect,
            };

            budget.reserve(&mut storage.decompositions, 1)?;
            storage.decompositions.push(decomposition);
        }

        let d_end = storage.decompositions.len();

        budget.reserve(&mut storage.tile_decompositions, 1)?;
        storage.tile_decompositions.push(TileDecompositions {
            decompositions: d_start..d_end,
            first_ll_sub_band,
            coefficient_count: coefficient_counter,
        });

        let top_resolution = ResolutionTile::new(component_tile, decoded_resolutions - 1);
        let samples = usize::try_from(top_resolution.rect.area())
            .map_err(|_| ValidationError::ImageTooLarge)?;
        assert_eq!(coefficient_counter, samples);
    }

    Ok(())
}

fn build_precincts(
    resolution_tile: &ResolutionTile<'_>,
    sub_band_rect: IntRect,
    storage: &mut DecompositionStorage<'_>,
    budget: &mut BuildBudget<'_>,
) -> Result<Range<usize>> {
    let start = storage.precincts.len();

    for precinct_data in resolution_tile
        .precincts()
        .ok_or(DecodingError::InvalidPrecinct)?
    {
        let precinct_rect = precinct_data.rect;

        let cb_width = resolution_tile.code_block_width();
        let cb_height = resolution_tile.code_block_height();

        // See Figure B.9. Conceptually, the area of code-blocks is aligned
        // to the width/height of a code block.
        // Keep the aligned bounds in block units. The exclusive end in
        // coefficient units can exceed u32::MAX for the final partial block.
        let cb_x0 = precinct_rect.x0.max(sub_band_rect.x0) / cb_width;
        let cb_y0 = precinct_rect.y0.max(sub_band_rect.y0) / cb_height;
        let cb_x1 = precinct_rect.x1.min(sub_band_rect.x1).div_ceil(cb_width);
        let cb_y1 = precinct_rect.y1.min(sub_band_rect.y1).div_ceil(cb_height);

        // If the sub-band is empty, there are no code-blocks, but due to our
        // flooring/ceiling above, we would get 1 code-block in each direction.
        // Because of this, we need to special-case this.
        let code_blocks_x = if sub_band_rect.width() == 0 {
            0
        } else {
            cb_x1.saturating_sub(cb_x0)
        };

        let code_blocks_y = if sub_band_rect.height() == 0 {
            0
        } else {
            cb_y1.saturating_sub(cb_y0)
        };

        let blocks = build_code_blocks(
            (cb_x0, cb_y0),
            sub_band_rect,
            resolution_tile,
            code_blocks_x,
            code_blocks_y,
            storage,
            budget,
        )?;

        let code_inclusion_tree = TagTree::new(
            code_blocks_x,
            code_blocks_y,
            &mut storage.tag_tree_nodes,
            budget,
        )?;
        let zero_bitplane_tree = TagTree::new(
            code_blocks_x,
            code_blocks_y,
            &mut storage.tag_tree_nodes,
            budget,
        )?;

        budget.reserve(&mut storage.precincts, 1)?;
        storage.precincts.push(Precinct {
            code_blocks: blocks,
            code_inclusion_tree,
            zero_bitplane_tree,
            visited_layers: 0,
        });
    }

    let end = storage.precincts.len();

    Ok(start..end)
}

fn build_code_blocks(
    (block_x0, block_y0): (u32, u32),
    sub_band_rect: IntRect,
    tile_instance: &ResolutionTile<'_>,
    code_blocks_x: u32,
    code_blocks_y: u32,
    storage: &mut DecompositionStorage<'_>,
    budget: &mut BuildBudget<'_>,
) -> Result<Range<usize>> {
    let num_layers = usize::from(tile_instance.component_tile.tile.num_layers);
    let block_count = usize::try_from(u64::from(code_blocks_x) * u64::from(code_blocks_y))
        .map_err(|_| ValidationError::MemoryLimitExceeded)?;
    let layer_count = block_count
        .checked_mul(num_layers)
        .ok_or(ValidationError::MemoryLimitExceeded)?;
    budget.reserve(&mut storage.code_blocks, block_count)?;
    budget.reserve(&mut storage.layers, layer_count)?;

    let code_block_width = tile_instance.code_block_width();
    let code_block_height = tile_instance.code_block_height();

    let start = storage.code_blocks.len();

    for y_idx in 0..code_blocks_y {
        let y = (block_y0 + y_idx) * code_block_height;

        for x_idx in 0..code_blocks_x {
            let x = (block_x0 + x_idx) * code_block_width;
            // "Code-blocks in the partition may extend beyond the boundaries of
            // the sub-band coefficients. When this happens, only the
            // coefficients lying within the sub-band are coded using the method
            // described in Annex D."
            let area = IntRect::from_ltrb(
                x,
                y,
                x.saturating_add(code_block_width),
                y.saturating_add(code_block_height),
            )
            .intersect(sub_band_rect);

            let start = storage.layers.len();
            storage.layers.extend(iter::repeat_n(
                Layer {
                    // This will be updated once we actually read the
                    // layer segments.
                    segments: None,
                },
                num_layers,
            ));
            let end = storage.layers.len();

            storage.code_blocks.push(CodeBlock {
                x_idx,
                y_idx,
                rect: area,
                has_been_included: false,
                missing_bit_planes: 0,
                l_block: 3,
                number_of_coding_passes: 0,
                layers: start..end,
                non_empty_layer_count: 0,
            });
        }
    }

    let end = storage.code_blocks.len();

    Ok(start..end)
}

pub(crate) struct Decomposition {
    /// In the order low-high, high-low and high-high.
    pub(crate) sub_bands: [usize; 3],
    /// The rectangle of the decomposition.
    pub(crate) rect: IntRect,
}

#[derive(Clone)]
pub(crate) struct SubBand {
    pub(crate) sub_band_type: SubBandType,
    pub(crate) rect: IntRect,
    pub(crate) precincts: Range<usize>,
    pub(crate) coefficients: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubBandType {
    LowLow = 0,
    HighLow = 1,
    LowHigh = 2,
    HighHigh = 3,
}

#[derive(Clone)]
pub(crate) struct Precinct {
    pub(crate) code_blocks: Range<usize>,
    pub(crate) code_inclusion_tree: TagTree,
    pub(crate) zero_bitplane_tree: TagTree,
    /// POC duplicate suppression uses the first sub-band's precinct.
    pub(crate) visited_layers: u32,
}

pub(crate) struct PrecinctData {
    /// The x coordinate mapped back to the reference grid.
    pub(crate) r_x: u64,
    /// The y coordinate mapped back to the reference grid.
    pub(crate) r_y: u64,
    /// The actual rectangle of the precinct (in the sub-band coordinate
    /// system).
    pub(crate) rect: IntRect,
    /// The index of the precinct in the sub-band.
    pub(crate) idx: u64,
}

#[derive(Clone)]
pub(crate) struct CodeBlock {
    pub(crate) rect: IntRect,
    pub(crate) x_idx: u32,
    pub(crate) y_idx: u32,
    pub(crate) layers: Range<usize>,
    pub(crate) has_been_included: bool,
    pub(crate) missing_bit_planes: u8,
    pub(crate) number_of_coding_passes: u8,
    pub(crate) l_block: u32,
    pub(crate) non_empty_layer_count: u8,
}

pub(crate) struct Segment<'a> {
    pub(crate) idx: u8,
    pub(crate) coding_pases: u8,
    pub(crate) data_length: u32,
    pub(crate) data: &'a [u8],
}

#[derive(Clone)]
pub(crate) struct Layer {
    pub(crate) segments: Option<Range<usize>>,
}

