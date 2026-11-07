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
use crate::error::{bail, FormatError, MarkerError, Result};
use crate::j2c::codestream::markers;
use crate::reader::BitReader;
use crate::{ColorSpace, DecodeSettings, Image};

use crate::math::{SimdBuffer, SIMD_WIDTH};
pub(crate) use codestream::Header;
pub(crate) use decode::decode;
pub use decode::DecoderContext;

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

#[cfg(test)]
mod recovery_budget_tests {
    use super::super::{DecodeSettings, DecoderContext, Image, ValidationError};
    use alloc::vec::Vec;

    fn unit_fixture(width: u32, tile_body: &[u8]) -> Vec<u8> {
        let mut data = alloc::vec![
            0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x29, 0x00, 0x00, 0, 0, 0, 0, // Xsiz patched below
            0, 0, 0, 1, // Ysiz
            0, 0, 0, 0, // XOsiz
            0, 0, 0, 0, // YOsiz
            0, 0, 0, 1, // XTsiz
            0, 0, 0, 1, // YTsiz
            0, 0, 0, 0, // XTOsiz
            0, 0, 0, 0, // YTOsiz
            0, 1, 7, 1, 1, // one unsigned 8-bit component
            0xFF, 0x52, 0, 12, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0xFF, 0x5C, 0, 4, 0x40, 0x40,
        ];
        data[8..12].copy_from_slice(&width.to_be_bytes());
        data.extend_from_slice(tile_body);
        data
    }

    #[test]
    fn tile_grid_error_precedes_pixel_budget_and_byte_gate_precedes_sot_parse() {
        for strict in [false, true] {
            let settings = DecodeSettings {
                strict,
                max_bytes: 0,
                ..DecodeSettings::default()
            };
            let bytes = unit_fixture(1, &[0xFF, 0x90]);
            let image = Image::new(&bytes, &settings).unwrap();
            let mut context = DecoderContext::default();
            assert!(matches!(
                image.decode(&mut context),
                Err(crate::DecodeError::Validation(
                    ValidationError::MemoryLimitExceeded
                ))
            ));
        }

        let settings = DecodeSettings::default();
        let bytes = unit_fixture(65_536, &[0xFF, 0x90]);
        let probe = Image::new(&bytes, &settings).unwrap();
        assert_eq!(probe.header.size_data.num_tiles().unwrap(), 65_536);
        let planes = super::limits::raw_plane_bytes(&probe.header, 65_536).unwrap();
        let capped = DecodeSettings {
            max_bytes: planes as u64,
            ..settings
        };
        let image = Image::new(&bytes, &capped).unwrap();
        let mut context = DecoderContext::default();
        assert!(matches!(
            image.decode(&mut context),
            Err(crate::DecodeError::Validation(
                ValidationError::MemoryLimitExceeded
            ))
        ));

        let settings = DecodeSettings {
            max_pixels: 1,
            ..DecodeSettings::default()
        };
        let bytes = unit_fixture(65_537, &[0xFF, 0x90]);
        let image = Image::new(&bytes, &settings).unwrap();
        assert_eq!(image.header.size_data.num_tiles().unwrap(), 65_537);
        let mut context = DecoderContext::default();
        assert!(matches!(
            image.decode(&mut context),
            Err(crate::DecodeError::Validation(
                ValidationError::ImageTooLarge
            ))
        ));
    }

    #[test]
    fn truncated_sot_obeys_combined_tile_and_plane_cap_in_both_modes() {
        for strict in [false, true] {
            let bytes = unit_fixture(1, &[0xFF, 0x90]);
            let probe = Image::new(&bytes, &DecodeSettings::default()).unwrap();
            let samples =
                super::limits::check_image(&probe.header, &DecodeSettings::default()).unwrap();
            let planes = super::limits::raw_plane_bytes(&probe.header, samples).unwrap();
            let tiles = super::limits::tile_table_bytes(&probe.header, 1).unwrap();
            assert!(planes > 0 && tiles > 0);

            let cap = planes.max(tiles) as u64;
            let settings = DecodeSettings {
                strict,
                max_bytes: cap,
                ..DecodeSettings::default()
            };
            let image = Image::new(&bytes, &settings).unwrap();

            let valid_body = [
                0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 15, 0, 1, 0xFF, 0x93, 0, 0xFF, 0xD9,
            ];
            let seed_bytes = unit_fixture(1, &valid_body);
            let seed_settings = DecodeSettings {
                strict,
                max_bytes: u64::MAX,
                ..DecodeSettings::default()
            };
            let seed_image = Image::new(&seed_bytes, &seed_settings).unwrap();
            let mut context = DecoderContext::default();
            let seeded = seed_image.decode(&mut context).unwrap();
            assert_eq!(seeded.components()[0].samples(), &[128.0_f32]);
            drop(seeded);
            assert!(matches!(
                image.decode(&mut context),
                Err(crate::DecodeError::Validation(
                    ValidationError::MemoryLimitExceeded
                ))
            ));
            assert_eq!(context.channel_data[0].samples(), &[128.0_f32]);

            let cap = (planes + tiles) as u64;
            let settings = DecodeSettings {
                strict,
                max_bytes: cap,
                ..DecodeSettings::default()
            };
            let image = Image::new(&bytes, &settings).unwrap();
            assert!(matches!(
                image.decode(&mut context),
                Err(crate::DecodeError::Marker(
                    crate::MarkerError::ParseFailure("SOT")
                ))
            ));

            let valid_probe = Image::new(&seed_bytes, &seed_settings).unwrap();
            let valid_samples =
                super::limits::check_image(&valid_probe.header, &seed_settings).unwrap();
            let valid_tiles = super::limits::tile_table_bytes(&valid_probe.header, 1).unwrap();
            let exact = valid_tiles
                + super::limits::raw_plane_bytes(&valid_probe.header, valid_samples).unwrap()
                + size_of::<super::tile::TilePart<'_>>()
                + 2 * size_of::<f32>();
            let cap = u64::try_from(exact - 1).unwrap();
            let post_parse_settings = DecodeSettings {
                strict,
                max_bytes: cap,
                ..DecodeSettings::default()
            };
            let post_parse_image = Image::new(&seed_bytes, &post_parse_settings).unwrap();
            assert!(matches!(
                post_parse_image.decode(&mut context),
                Err(crate::DecodeError::Validation(
                    ValidationError::MemoryLimitExceeded
                ))
            ));
            assert_eq!(context.channel_data[0].samples(), &[128.0_f32]);
        }
    }

    #[test]
    fn valid_empty_packet_exact_boundary_and_generous_cap_succeed_in_both_modes() {
        let body = [
            0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 15, 0, 1, 0xFF, 0x93, 0, 0xFF, 0xD9,
        ];
        let bytes = unit_fixture(1, &body);
        for strict in [false, true] {
            let generous = DecodeSettings {
                strict,
                max_bytes: u64::MAX,
                ..DecodeSettings::default()
            };
            let probe = Image::new(&bytes, &generous).unwrap();
            let samples = super::limits::check_image(&probe.header, &generous).unwrap();
            let tile_count = usize::try_from(probe.header.size_data.num_tiles().unwrap()).unwrap();
            let planes = super::limits::raw_plane_bytes(&probe.header, samples).unwrap();
            let tiles = super::limits::tile_table_bytes(&probe.header, tile_count).unwrap();
            let exact =
                tiles + planes + size_of::<super::tile::TilePart<'_>>() + 2 * size_of::<f32>();

            let exact_settings = DecodeSettings {
                strict,
                max_bytes: u64::try_from(exact).unwrap(),
                ..DecodeSettings::default()
            };
            let exact_image = Image::new(&bytes, &exact_settings).unwrap();
            let mut exact_context = DecoderContext::default();
            let exact_decoded = exact_image.decode(&mut exact_context).unwrap();
            assert_eq!(exact_decoded.components()[0].bit_depth(), 8);
            assert!(!exact_decoded.components()[0].is_signed());
            assert_eq!(exact_decoded.components()[0].samples(), &[128.0_f32]);
            let exact_samples = exact_decoded.components()[0].samples().to_vec();
            drop(exact_decoded);

            let generous_image = Image::new(&bytes, &generous).unwrap();
            let mut generous_context = DecoderContext::default();
            let generous_decoded = generous_image.decode(&mut generous_context).unwrap();
            assert_eq!(
                generous_decoded.components()[0].samples(),
                exact_samples.as_slice()
            );
        }
    }

    #[test]
    fn qcd_overrides_share_the_retained_tile_budget() {
        const QCD: &[u8] = &[0xFF, 0x5C, 0, 11, 0x42, 0x40, 0, 0x40, 0, 0x40, 0, 0x40, 0];
        const QCC: &[u8] = &[0xFF, 0x5D, 0, 6, 0, 0x41, 0x40, 0];

        let fixture = |first: &[&[u8]], second: &[&[u8]]| {
            let mut bytes = unit_fixture(4, &[]);
            // Two 2x2 tiles with one decomposition and irreversible quantization.
            bytes[15] = 2; // Ysiz
            bytes[27] = 2; // XTsiz
            bytes[31] = 2; // YTsiz
            bytes[54] = 1; // decomposition levels
            bytes[58] = 0; // irreversible transform
            bytes.truncate(bytes.len() - 6);
            bytes.extend_from_slice(&[0xFF, 0x5C, 0, 5, 0x41, 0x40, 0]);

            for (index, tile_markers) in [first, second].into_iter().enumerate() {
                let marker_bytes: usize = tile_markers.iter().map(|marker| marker.len()).sum();
                let part_len = u32::try_from(16 + marker_bytes).unwrap();
                bytes.extend_from_slice(&[0xFF, 0x90, 0, 10, 0, index as u8]);
                bytes.extend_from_slice(&part_len.to_be_bytes());
                bytes.extend_from_slice(&[0, 1]);
                for marker in tile_markers {
                    bytes.extend_from_slice(marker);
                }
                // SOD and one empty packet per resolution.
                bytes.extend_from_slice(&[0xFF, 0x93, 0, 0]);
            }
            bytes.extend_from_slice(&[0xFF, 0xD9]);
            bytes
        };

        let both = fixture(&[QCD], &[QCD]);
        let first_only = fixture(&[QCD], &[]);
        let second_only = fixture(&[], &[QCD]);
        let replaced = fixture(&[QCD, QCC], &[QCD]);
        for strict in [false, true] {
            let settings = DecodeSettings {
                strict,
                ..DecodeSettings::default()
            };
            let probe = Image::new(&both, &settings).unwrap();
            let inherited = super::limits::tile_table_bytes(&probe.header, 2).unwrap();
            let planes = super::limits::raw_plane_bytes(&probe.header, 8).unwrap();
            let baseline = inherited + 2 * size_of::<super::tile::TilePart<'_>>();
            let growth = 3 * size_of::<super::codestream::StepSize>();
            let exact = planes + baseline + 2 * growth;
            let parse = |bytes: &[u8], cap: usize| {
                let settings = DecodeSettings {
                    max_bytes: u64::try_from(cap).unwrap(),
                    ..settings
                };
                let image = Image::new(bytes, &settings).unwrap();
                let mut reader = super::BitReader::new(image.codestream);
                super::tile::parse(&mut reader, &image.header, inherited, planes, &settings)
                    .map(|(_, retained)| retained)
            };

            assert!(matches!(
                parse(&both, exact - 1),
                Err(crate::DecodeError::Validation(
                    ValidationError::MemoryLimitExceeded
                ))
            ));
            assert_eq!(parse(&both, exact).unwrap(), baseline + 2 * growth);
            for bytes in [&first_only, &second_only, &replaced] {
                assert_eq!(parse(bytes, exact - growth).unwrap(), baseline + growth);
            }

            // Four coefficients and a 3x3 padded IDWT output per tile.
            let decode_cap = exact + (4 + 9) * size_of::<f32>();
            for cap in [decode_cap - 1, decode_cap] {
                let settings = DecodeSettings {
                    max_bytes: u64::try_from(cap).unwrap(),
                    ..settings
                };
                let image = Image::new(&both, &settings).unwrap();
                let mut context = DecoderContext::default();
                let result = image.decode(&mut context);
                if cap == decode_cap {
                    assert_eq!(result.unwrap().components()[0].samples(), &[128.0; 8]);
                } else {
                    assert!(matches!(
                        result,
                        Err(crate::DecodeError::Validation(
                            ValidationError::MemoryLimitExceeded
                        ))
                    ));
                }
            }
        }
    }

    #[test]
    fn discarded_ppt_does_not_consume_the_retained_budget() {
        // Psot covers only SOT and SOD. The intervening PPT makes the tile
        // part invalid, so permissive parsing discards it.
        let body = [
            0xFF, 0x90, 0, 10, 0, 0, 0, 0, 0, 14, 0, 1, 0xFF, 0x61, 0, 4, 0, 0, 0xFF, 0x93,
            0xFF, 0xD9,
        ];
        let bytes = unit_fixture(1, &body);
        for strict in [false, true] {
            let settings = DecodeSettings {
                strict,
                ..DecodeSettings::default()
            };
            let image = Image::new(&bytes, &settings).unwrap();
            let inherited = super::limits::tile_table_bytes(&image.header, 1).unwrap();
            let planes = super::limits::raw_plane_bytes(&image.header, 1).unwrap();
            let settings = DecodeSettings {
                max_bytes: u64::try_from(inherited + planes).unwrap(),
                ..settings
            };
            let mut reader = super::BitReader::new(image.codestream);
            let result = super::tile::parse(&mut reader, &image.header, inherited, planes, &settings);
            if strict {
                assert!(matches!(
                    result,
                    Err(crate::DecodeError::Tile(crate::TileError::Invalid))
                ));
            } else {
                let (tiles, retained) = result.unwrap();
                assert!(tiles[0].tile_parts.is_empty());
                assert_eq!(retained, inherited);
            }
        }
    }

    #[test]
    fn checked_resource_helpers_report_exact_overflow_and_cap_errors() {
        let bytes = unit_fixture(1, &[0xFF, 0x90]);
        let probe = Image::new(&bytes, &DecodeSettings::default()).unwrap();
        let settings = DecodeSettings {
            max_bytes: 10,
            ..DecodeSettings::default()
        };

        assert!(super::limits::check_total_bytes(10, &settings).is_ok());
        assert!(matches!(
            super::limits::check_total_bytes(11, &settings),
            Err(crate::DecodeError::Validation(
                ValidationError::MemoryLimitExceeded
            ))
        ));
        assert!(matches!(
            super::limits::check_total_bytes(usize::MAX, &settings),
            Err(crate::DecodeError::Validation(
                ValidationError::MemoryLimitExceeded
            ))
        ));
        assert!(matches!(
            super::limits::raw_plane_bytes(&probe.header, usize::MAX),
            Err(crate::DecodeError::Validation(
                ValidationError::MemoryLimitExceeded
            ))
        ));
        let plane_multiply_overflow = usize::MAX / size_of::<f32>() - 1;
        assert!(matches!(
            super::limits::raw_plane_bytes(&probe.header, plane_multiply_overflow),
            Err(crate::DecodeError::Validation(
                ValidationError::MemoryLimitExceeded
            ))
        ));
        assert!(matches!(
            super::limits::tile_table_bytes(&probe.header, usize::MAX),
            Err(crate::DecodeError::Validation(
                ValidationError::MemoryLimitExceeded
            ))
        ));

        let generous = DecodeSettings {
            max_bytes: u64::MAX,
            ..DecodeSettings::default()
        };
        let samples = super::limits::check_image(&probe.header, &generous).unwrap();
        let tile_bytes = super::limits::tile_table_bytes(&probe.header, 1).unwrap();
        let palette_stride = (samples + crate::math::SIMD_WIDTH) * size_of::<f32>();
        let palette_components = usize::MAX / palette_stride;
        assert!(matches!(
            super::limits::BufferSizes::new(
                &probe.header,
                &[],
                &generous,
                palette_components,
                tile_bytes
            ),
            Err(crate::DecodeError::Validation(
                ValidationError::MemoryLimitExceeded
            ))
        ));
    }
}
