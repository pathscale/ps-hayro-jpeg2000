//! Progression iterators, defined in Section B.12.
//!
//! A progression iterator essentially yields tuples of
//! (`layer_num`, resolution, component, precinct) in a specific order that
//! determines in which order the data appears in the codestream.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::build::BuildBudget;
use super::codestream::{Header, ProgressionOrder};
use super::tile::{ComponentTile, ResolutionTile, Tile};
use crate::error::{DecodingError, Result};
use core::cmp::Ordering;
use core::iter;

#[derive(Default, Copy, Clone, Debug, PartialEq, Hash, Eq, PartialOrd, Ord)]
pub(crate) struct ProgressionData {
    pub(crate) layer_num: u8,
    pub(crate) resolution: u8,
    pub(crate) component: u8,
    pub(crate) precinct: u64,
}

pub(crate) struct IteratorInput<'a> {
    layers: (u8, u8),
    tile: &'a Tile<'a>,
    resolutions: (u8, u8),
    components: (u8, u8),
}

impl<'a> IteratorInput<'a> {
    pub(crate) fn new(tile: &'a Tile<'a>) -> Self {
        Self::new_with_custom_bounds(
            tile,
            // Will be clamped automatically.
            (0, u8::MAX),
            (0, u8::MAX),
            (0, u8::MAX),
        )
    }

    pub(crate) fn new_with_custom_bounds(
        tile: &'a Tile<'a>,
        mut resolutions: (u8, u8),
        mut layers: (u8, u8),
        mut components: (u8, u8),
    ) -> Self {
        let max_resolution = tile
            .component_infos
            .iter()
            .map(|c| c.coding_style.parameters.num_resolution_levels)
            .max()
            .unwrap_or(0);
        let max_layer = tile.num_layers;
        let max_component = tile.component_infos.len() as u8;

        // Make sure we don't exceed what's actually possible
        resolutions.1 = resolutions.1.min(max_resolution);
        layers.1 = layers.1.min(max_layer);
        components.1 = components.1.min(max_component);

        Self {
            layers,
            tile,
            resolutions,
            components,
        }
    }

    fn min_layer(&self) -> u8 {
        self.layers.0
    }

    fn max_layer(&self) -> u8 {
        self.layers.1
    }

    fn min_resolution(&self) -> u8 {
        self.resolutions.0
    }

    fn total_max_resolution(&self) -> u8 {
        self.resolutions.1
    }

    fn max_resolution(&self, component_idx: u8) -> u8 {
        self.total_max_resolution()
            // It's possible that the different component tiles have different resolution levels
            // (self.resolutions.1 stores the maximum across all component tiles), so
            // take the minimum of both.
            .min(self.tile.component_infos[component_idx as usize].num_resolution_levels())
    }

    fn min_comp(&self) -> u8 {
        self.components.0
    }

    fn max_comp(&self) -> u8 {
        self.components.1
    }

    fn is_empty(&self) -> bool {
        self.layers.0 >= self.layers.1
            || self.resolutions.0 >= self.resolutions.1
            || self.components.0 >= self.components.1
    }

    fn component_tile(&self, component_idx: u8) -> ComponentTile<'a> {
        ComponentTile::new(self.tile, &self.tile.component_infos[component_idx as usize])
    }

    /// The number of precincts of the given component at the given
    /// resolution, or 0 if the component doesn't have that resolution.
    fn num_precincts(&self, component_idx: u8, resolution: u8) -> u64 {
        if component_idx >= self.max_comp() || resolution >= self.max_resolution(component_idx) {
            return 0;
        }

        ResolutionTile::new(self.component_tile(component_idx), resolution).num_precincts()
    }
}

/// Create the iterator that yields the packets of the tile in codestream order,
/// taking progression order changes into account (B.12.3).
pub(crate) fn progression_iterator<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    budget: &mut BuildBudget<'_>,
) -> Result<(PacketIterator<'a>, bool)> {
    let progression_changes = if tile.progression_changes.is_empty() {
        &header.progression_changes
    } else {
        &tile.progression_changes
    };

    if progression_changes.is_empty() {
        let iterator = progression_order_iterator(
            IteratorInput::new(tile),
            tile.progression_order,
            budget,
        )?;
        return Ok((iterator, false));
    }

    let clamp = |value: u16| u8::try_from(value).unwrap_or(u8::MAX);

    let mut iterators = Vec::new();
    budget.reserve_exact(&mut iterators, progression_changes.len())?;
    for change in progression_changes {
        let input = IteratorInput::new_with_custom_bounds(
            tile,
            (change.resolution_start, change.resolution_end),
            (0, clamp(change.layer_end)),
            (clamp(change.component_start), clamp(change.component_end)),
        );
        iterators.push(progression_order_iterator(input, change.progression_order, budget)?);
    }
    // Keep iterator storage live until packet parsing ends, including exhausted
    // entries; the charged amount is shared with segment-record allocations.
    let mut current = 0_usize;
    let flattened = iter::from_fn(move || {
        while let Some(iterator) = iterators.get_mut(current) {
            if let Some(packet) = iterator.next() {
                return Some(packet);
            }
            current += 1;
        }
        None
    });
    Ok((boxed_iterator(flattened, budget)?, true))
}

pub(crate) type PacketIterator<'a> = Box<dyn Iterator<Item = ProgressionData> + 'a>;

fn boxed_iterator<'a>(
    iterator: impl Iterator<Item = ProgressionData> + 'a,
    budget: &mut BuildBudget<'_>,
) -> Result<PacketIterator<'a>> {
    budget.charge_bytes(size_of_val(&iterator))?;
    Ok(Box::new(iterator))
}

fn progression_order_iterator<'a>(
    input: IteratorInput<'a>,
    progression_order: ProgressionOrder,
    budget: &mut BuildBudget<'_>,
) -> Result<PacketIterator<'a>> {
    match progression_order {
        ProgressionOrder::LayerResolutionComponentPosition => {
            boxed_iterator(layer_resolution_component_position_progression(input), budget)
        }
        ProgressionOrder::ResolutionLayerComponentPosition => {
            boxed_iterator(resolution_layer_component_position_progression(input), budget)
        }
        ProgressionOrder::ResolutionPositionComponentLayer => {
            let iterator = resolution_position_component_layer_progression(input, budget)?;
            boxed_iterator(iterator, budget)
        }
        ProgressionOrder::PositionComponentResolutionLayer => {
            let iterator = position_component_resolution_layer_progression(input, budget)?;
            boxed_iterator(iterator, budget)
        }
        ProgressionOrder::ComponentPositionResolutionLayer => {
            let iterator = component_position_resolution_layer_progression(input, budget)?;
            boxed_iterator(iterator, budget)
        }
    }
}

/// B.12.1.1 Layer-resolution level-component-position progression.
pub(crate) fn layer_resolution_component_position_progression<'a>(
    input: IteratorInput<'a>,
) -> impl Iterator<Item = ProgressionData> + 'a {
    let mut layer = input.min_layer();
    let mut resolution = input.min_resolution();
    let mut component_idx = input.min_comp();
    let mut num_precincts = input.num_precincts(component_idx, resolution);
    let mut precinct = 0;

    iter::from_fn(move || {
        if input.is_empty() || layer >= input.max_layer() {
            return None;
        }

        while precinct == num_precincts {
            precinct = 0;
            component_idx += 1;

            if component_idx == input.max_comp() {
                component_idx = input.min_comp();
                resolution += 1;

                if resolution == input.total_max_resolution() {
                    resolution = input.min_resolution();
                    layer += 1;

                    if layer == input.max_layer() {
                        return None;
                    }
                }
            }

            num_precincts = input.num_precincts(component_idx, resolution);
        }

        let data = ProgressionData {
            layer_num: layer,
            resolution,
            component: component_idx,
            precinct,
        };

        precinct += 1;

        Some(data)
    })
}

/// B.12.1.2 Resolution level-layer-component-position progression.
pub(crate) fn resolution_layer_component_position_progression<'a>(
    input: IteratorInput<'a>,
) -> impl Iterator<Item = ProgressionData> + 'a {
    let mut layer = input.min_layer();
    let mut resolution = input.min_resolution();
    let mut component_idx = input.min_comp();
    let mut num_precincts = input.num_precincts(component_idx, resolution);
    let mut precinct = 0;

    iter::from_fn(move || {
        if input.is_empty() || resolution >= input.total_max_resolution() {
            return None;
        }

        while precinct == num_precincts {
            precinct = 0;
            component_idx += 1;

            if component_idx == input.max_comp() {
                component_idx = input.min_comp();
                layer += 1;

                if layer == input.max_layer() {
                    layer = input.min_layer();
                    resolution += 1;

                    if resolution == input.total_max_resolution() {
                        return None;
                    }
                }
            }

            num_precincts = input.num_precincts(component_idx, resolution);
        }

        let data = ProgressionData {
            layer_num: layer,
            resolution,
            component: component_idx,
            precinct,
        };

        precinct += 1;

        Some(data)
    })
}

// The formula for the remaining three progressions looks very intimidating.
// But really, all they boil down to is that we need to determine all precinct
// indices for each component/resolution combination and sort them by ascending
// y/x coordinate on the reference grid. Other than that, they can be treated
// exactly the same, except that the sort order precedence of the fields change.

// Note that the order of fields here is important!
struct PrecinctStore {
    resolution: u8,
    precinct_y: u64,
    precinct_x: u64,
    component_idx: u8,
    precinct_idx: u64,
}

fn position_progression_common<'a>(
    input: IteratorInput<'a>,
    budget: &mut BuildBudget<'_>,
    sort: fn(&PrecinctStore, &PrecinctStore) -> Ordering,
) -> Result<impl Iterator<Item = ProgressionData> + use<'a>> {
    let mut elements = Vec::new();

    for component_idx in input.min_comp()..input.max_comp() {
        let component_tile = input.component_tile(component_idx);

        for resolution in input.min_resolution()..input.max_resolution(component_idx) {
            let resolution_tile = ResolutionTile::new(component_tile, resolution);

            let precincts = resolution_tile.precincts()
                .ok_or(DecodingError::InvalidProgressionIterator)?;
            let count = usize::try_from(resolution_tile.num_precincts())
                .map_err(|_| crate::ValidationError::MemoryLimitExceeded)?;
            budget.reserve(&mut elements, count)?;
            elements.extend(precincts.map(|d| PrecinctStore {
                precinct_y: d.r_y,
                precinct_x: d.r_x,
                component_idx,
                resolution,
                precinct_idx: d.idx,
            }));
        }
    }

    // The key includes the unique (component, resolution, precinct) tuple.
    elements.sort_unstable_by(sort);

    Ok(elements.into_iter().flat_map(move |e| {
        (input.min_layer()..input.max_layer()).map(move |layer| ProgressionData {
            layer_num: layer,
            resolution: e.resolution,
            component: e.component_idx,
            precinct: e.precinct_idx,
        })
    }))
}

/// B.12.1.3 Resolution level-position-component-layer progression.
pub(crate) fn resolution_position_component_layer_progression<'a>(
    input: IteratorInput<'a>,
    budget: &mut BuildBudget<'_>,
) -> Result<impl Iterator<Item = ProgressionData> + use<'a>> {
    position_progression_common(input, budget, |p, s| {
        p.resolution
            .cmp(&s.resolution)
            .then_with(|| p.precinct_y.cmp(&s.precinct_y))
            .then_with(|| p.precinct_x.cmp(&s.precinct_x))
            .then_with(|| p.component_idx.cmp(&s.component_idx))
            .then_with(|| p.precinct_idx.cmp(&s.precinct_idx))
    })
}

/// B.12.1.4 Position-component-resolution level-layer progression.
pub(crate) fn position_component_resolution_layer_progression<'a>(
    input: IteratorInput<'a>,
    budget: &mut BuildBudget<'_>,
) -> Result<impl Iterator<Item = ProgressionData> + use<'a>> {
    position_progression_common(input, budget, |p, s| {
        p.precinct_y
            .cmp(&s.precinct_y)
            .then_with(|| p.precinct_x.cmp(&s.precinct_x))
            .then_with(|| p.component_idx.cmp(&s.component_idx))
            .then_with(|| p.resolution.cmp(&s.resolution))
            .then_with(|| p.precinct_idx.cmp(&s.precinct_idx))
    })
}

/// B.12.1.5 Component-position-resolution level-layer progression.
pub(crate) fn component_position_resolution_layer_progression<'a>(
    input: IteratorInput<'a>,
    budget: &mut BuildBudget<'_>,
) -> Result<impl Iterator<Item = ProgressionData> + use<'a>> {
    position_progression_common(input, budget, |p, s| {
        p.component_idx
            .cmp(&s.component_idx)
            .then_with(|| p.precinct_y.cmp(&s.precinct_y))
            .then_with(|| p.precinct_x.cmp(&s.precinct_x))
            .then_with(|| p.resolution.cmp(&s.resolution))
            .then_with(|| p.precinct_idx.cmp(&s.precinct_idx))
    })
}
