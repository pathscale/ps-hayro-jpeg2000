mod arithmetic_decoder;
mod bitplane;
mod build;
mod codestream;
mod decode;
mod idwt;
mod limits;
mod mct;
mod progression;
mod rect;
mod segment;
mod tag_tree;
mod tile;

use super::jp2::ImageBoxes;
use crate::error::{FormatError, MarkerError, Result, bail};
use crate::j2c::codestream::markers;
use crate::reader::BitReader;
use crate::{ColorSpace, DecodeSettings, Image};

use crate::math::{SIMD_WIDTH, SimdBuffer};
pub(crate) use codestream::Header;
pub use decode::DecoderContext;
pub(crate) use decode::decode;

pub(crate) struct ParsedCodestream<'a> {
    pub(crate) header: Header<'a>,
    pub(crate) data: &'a [u8],
}

/// Decoded data for one JPEG2000 component.
#[derive(Debug, Clone)]
pub struct ComponentData {
    pub(crate) container: SimdBuffer<{ SIMD_WIDTH }>,
    pub(crate) bit_depth: u8,
    pub(crate) is_signed: bool,
}

impl ComponentData {
    /// The bit depth of this component.
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Whether this component contains signed samples.
    pub fn is_signed(&self) -> bool {
        self.is_signed
    }

    /// The decoded samples of this component.
    pub fn samples(&self) -> &[f32] {
        self.container.truncated()
    }
}

pub(crate) fn parse<'a>(stream: &'a [u8], settings: &DecodeSettings) -> Result<Image<'a>> {
    let parsed_codestream = parse_raw(stream, settings)?;
    let header = &parsed_codestream.header;
    let boxes = ImageBoxes::default();
    let num_channels = u8::try_from(header.component_infos.len())
        .map_err(|_| crate::ValidationError::TooManyChannels)?;
    // A raw codestream carries no colour space or opacity definitions.
    // Preserve all four channels for embedders such as PDF.
    let color_space = match num_channels {
        1 => ColorSpace::Gray,
        3 => ColorSpace::RGB,
        _ => ColorSpace::Unknown { num_channels },
    };
    let has_alpha = false;

    Ok(Image {
        codestream: parsed_codestream.data,
        header: parsed_codestream.header,
        boxes,
        settings: *settings,
        color_space,
        has_alpha,
    })
}

pub(crate) fn parse_raw<'a>(
    stream: &'a [u8],
    settings: &DecodeSettings,
) -> Result<ParsedCodestream<'a>> {
    let mut reader = BitReader::new(stream);

    let marker = reader.read_marker()?;
    if marker != markers::SOC {
        bail!(MarkerError::Expected("SOC"));
    }

    let header = codestream::read_header(&mut reader, settings)?;
    let code_stream_data = reader.tail().ok_or(FormatError::MissingCodestream)?;

    Ok(ParsedCodestream {
        header,
        data: code_stream_data,
    })
}

