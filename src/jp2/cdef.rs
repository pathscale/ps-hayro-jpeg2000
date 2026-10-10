//! The channel definition box (cdef), defined in I.5.3.6.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{FormatError, Result, bail};
use crate::j2c::ComponentData;
use crate::jp2::ImageBoxes;
use crate::reader::BitReader;

pub(crate) fn parse(boxes: &mut ImageBoxes, data: &[u8]) -> Result<()> {
    let mut reader = BitReader::new(data);
    let count = reader.read_u16().ok_or(FormatError::InvalidBox)? as usize;
    if count == 0 || data.len() != 2 + count * 6 {
        bail!(FormatError::InvalidBox);
    }
    let mut definitions = Vec::with_capacity(count);

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

    definitions.sort_by(|a, b| a.channel_index.cmp(&b.channel_index));

    // Ensure channel indices increases in steps of 1, starting from 0.
    for (idx, def) in definitions.iter().enumerate() {
        if def.channel_index as usize != idx {
            bail!(FormatError::InvalidBox);
        }
    }

    boxes.channel_definition = Some(ChannelDefinitionBox {
        channel_definitions: definitions,
    });

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
    pub(crate) fn permutation(&self, component_count: usize) -> Result<Vec<usize>> {
        if self.channel_definitions.len() != component_count {
            bail!(FormatError::InvalidBox);
        }
        let alpha_count = self.channel_definitions.iter()
            .filter(|d| d.channel_type != ChannelType::Colour).count();
        if alpha_count > 1 || component_count <= alpha_count {
            bail!(FormatError::InvalidBox);
        }
        let colours = component_count - alpha_count;
        let mut permutation = vec![usize::MAX; component_count];
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

    pub(crate) fn reorder(&self, components: &mut Vec<ComponentData>) -> Result<()> {
        let permutation = self.permutation(components.len())?;
        let mut sources: Vec<_> = core::mem::take(components).into_iter().map(Some).collect();
        *components = permutation.into_iter().map(|source| {
            // The validated permutation references every source exactly once.
            sources[source].take().unwrap()
        }).collect();
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

