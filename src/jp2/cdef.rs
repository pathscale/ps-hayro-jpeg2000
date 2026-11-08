//! The channel definition box (cdef), defined in I.5.3.6.

use alloc::vec::Vec;

use crate::error::{FormatError, Result, bail};
use crate::j2c::{BuildBudget, ComponentData};
use crate::jp2::ImageBoxes;
use crate::reader::BitReader;

pub(crate) fn parse(boxes: &mut ImageBoxes, data: &[u8], budget: &mut BuildBudget<'_>) -> Result<()> {
    let mut reader = BitReader::new(data);
    let count = reader.read_u16().ok_or(FormatError::InvalidBox)? as usize;
    if count == 0 || data.len() != 2 + count * 6 {
        bail!(FormatError::InvalidBox);
    }
    let mut definitions = Vec::new();
    budget.reserve_exact(&mut definitions, count)?;

    for _ in 0..count {
        let channel_index = reader.read_u16().ok_or(FormatError::InvalidBox)?;
        let channel_type = reader.read_u16().ok_or(FormatError::InvalidBox)?;
        let association = reader.read_u16().ok_or(FormatError::InvalidBox)?;

        definitions.push(ChannelDefinition {
            channel_index,
            channel_type: ChannelType::from_raw(channel_type).ok_or(FormatError::InvalidBox)?,
            association: ChannelAssociation::from_raw(association)
                .ok_or(FormatError::InvalidBox)?,
        });
    }

    definitions.sort_unstable_by_key(|definition| definition.channel_index);

    // Ensure channel indices increases in steps of 1, starting from 0.
    for (idx, def) in definitions.iter().enumerate() {
        if def.channel_index as usize != idx {
            bail!(FormatError::InvalidBox);
        }
    }

    if let Some(old) = boxes.channel_definition.replace(ChannelDefinitionBox {
        channel_definitions: definitions,
    }) {
        budget.release_bytes(BuildBudget::bytes::<ChannelDefinition>(
            old.channel_definitions.len(),
        )?)?;
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct ChannelDefinitionBox {
    pub(crate) channel_definitions: Vec<ChannelDefinition>,
}

impl ChannelDefinitionBox {
    pub(crate) fn has_alpha(&self) -> bool {
        self.channel_definitions.iter().any(|d| d.channel_type != ChannelType::Colour)
    }

    pub(crate) fn has_premultiplied_alpha(&self) -> bool {
        self.channel_definitions.iter()
            .any(|d| d.channel_type == ChannelType::PremultipliedOpacity)
    }

    // Output index -> input channel index. Require a complete bijection,
    // with one whole-image opacity channel at most, placed after colours.
    pub(crate) fn permutation(
        &self, component_count: usize, mut budget: BuildBudget<'_>,
    ) -> Result<Vec<usize>> {
        if self.channel_definitions.len() != component_count {
            bail!(FormatError::InvalidBox);
        }
        let alpha_count = self.channel_definitions.iter()
            .filter(|d| d.channel_type != ChannelType::Colour).count();
        if alpha_count > 1 || component_count <= alpha_count {
            bail!(FormatError::InvalidBox);
        }
        let colours = component_count - alpha_count;
        let mut permutation = Vec::new();
        budget.reserve_exact(&mut permutation, component_count)?;
        permutation.resize(component_count, usize::MAX);
        for definition in &self.channel_definitions {
            let source = definition.channel_index as usize;
            let destination = match definition.channel_type {
                ChannelType::Colour => match definition.association {
                    ChannelAssociation::Colour(c) => c as usize - 1,
                    ChannelAssociation::WholeImage => source,
                },
                ChannelType::Opacity | ChannelType::PremultipliedOpacity => {
                    if definition.association != ChannelAssociation::WholeImage {
                        bail!(FormatError::Unsupported);
                    }
                    colours
                }
            };
            if source >= component_count
                || (definition.channel_type == ChannelType::Colour && destination >= colours)
                || destination >= component_count
                || permutation[destination] != usize::MAX
            {
                bail!(FormatError::InvalidBox);
            }
            permutation[destination] = source;
        }
        if permutation.contains(&usize::MAX) {
            bail!(FormatError::InvalidBox);
        }
        Ok(permutation)
    }

    pub(crate) fn reorder(
        &self, components: &mut [ComponentData], budget: BuildBudget<'_>,
    ) -> Result<()> {
        let mut permutation = self.permutation(components.len(), budget)?;
        for start in 0..permutation.len() {
            let mut current = start;
            while permutation[current] != current {
                let source = core::mem::replace(&mut permutation[current], current);
                if source == start {
                    break;
                }
                components.swap(current, source);
                current = source;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ChannelDefinition {
    pub(crate) channel_index: u16,
    pub(crate) channel_type: ChannelType,
    pub(crate) association: ChannelAssociation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelType {
    Colour,
    Opacity,
    PremultipliedOpacity,
}

impl ChannelType {
    fn from_raw(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Colour),
            1 => Some(Self::Opacity),
            2 => Some(Self::PremultipliedOpacity),
            // We don't support the others.
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelAssociation {
    WholeImage,
    Colour(u16),
}

impl ChannelAssociation {
    fn from_raw(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::WholeImage),
            // Unspecified.
            u16::MAX => None,
            v => Some(Self::Colour(v)),
        }
    }
}

