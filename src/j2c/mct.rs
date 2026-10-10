//! The irreversible multi-component transformation, as specified in
//! Annex G.2 and G.3.

use super::ComponentData;
use super::codestream::{Header, WaveletTransform};
use super::tile::Tile;
use crate::error::{ColorError, Result, bail, err};
use crate::math::{self, Level, Simd, dispatch, f32x8};

/// Apply the inverse multi-component transform, as specified in G.2 and G.3.
pub(crate) fn apply_inverse(
    components: &mut [ComponentData],
    tile: &Tile<'_>,
    header: &Header<'_>,
) -> Result<()> {
    if components.len() < 3 {
        return if header.strict {
            err!(ColorError::Mct)
        } else {
            Ok(())
        };
    }

    let (s, _) = components.split_at_mut(3);
    let [s0, s1, s2] = s else { unreachable!() };

    let component_infos = &tile.component_infos;
    let transform = component_infos[0].wavelet_transform();

    if transform != component_infos[1].wavelet_transform()
        || component_infos[1].wavelet_transform() != component_infos[2].wavelet_transform()
    {
        bail!(ColorError::Mct);
    }

    if component_infos[..3].iter().any(|info| {
        info.num_resolution_levels() != component_infos[0].num_resolution_levels()
            || info.size_info.horizontal_resolution != component_infos[0].size_info.horizontal_resolution
            || info.size_info.vertical_resolution != component_infos[0].size_info.vertical_resolution
            || info.size_info.precision != component_infos[0].size_info.precision
    }) {
        bail!(ColorError::Mct);
    }

    if s0.container.len() != s1.container.len() || s1.container.len() != s2.container.len() {
        bail!(ColorError::Mct);
    }

    // MCT is tile-local. Do not touch padding or another tile's samples.
    let rect = tile.output_rect(header);
    let width = header.size_data.image_width() as usize;
    let count = rect.width() as usize;
    for y in rect.y0..rect.y1 {
        let start = y as usize * width + rect.x0 as usize;
        apply_inner(
            transform,
            &mut s0.container[start..start + count],
            &mut s1.container[start..start + count],
            &mut s2.container[start..start + count],
        );
    }

    Ok(())
}

fn apply_inner(transform: WaveletTransform, s0: &mut [f32], s1: &mut [f32], s2: &mut [f32]) {
    dispatch!(Level::new(), simd => apply_inner_impl(simd, transform, s0, s1, s2));
}

#[inline(always)]
fn apply_inner_impl<S: Simd>(
    simd: S,
    transform: WaveletTransform,
    s0: &mut [f32],
    s1: &mut [f32],
    s2: &mut [f32],
) {
    match transform {
        // Irreversible MCT, specified in G.3.
        WaveletTransform::Irreversible97 => {
            for ((y0, y1), y2) in s0
                .chunks_exact_mut(8)
                .zip(s1.chunks_exact_mut(8))
                .zip(s2.chunks_exact_mut(8))
            {
                let y_0 = f32x8::from_slice(simd, y0);
                let y_1 = f32x8::from_slice(simd, y1);
                let y_2 = f32x8::from_slice(simd, y2);

                let i0 = y_2.mul_add(f32x8::splat(simd, 1.402), y_0);
                let i1 = y_2.mul_add(
                    f32x8::splat(simd, -0.71414),
                    y_1.mul_add(f32x8::splat(simd, -0.34413), y_0),
                );
                let i2 = y_1.mul_add(f32x8::splat(simd, 1.772), y_0);

                i0.store(y0);
                i1.store(y1);
                i2.store(y2);
            }
        }
        // Reversible MCT, specified in G.2.
        WaveletTransform::Reversible53 => {
            for ((y0, y1), y2) in s0
                .chunks_exact_mut(8)
                .zip(s1.chunks_exact_mut(8))
                .zip(s2.chunks_exact_mut(8))
            {
                let y_0 = f32x8::from_slice(simd, y0);
                let y_1 = f32x8::from_slice(simd, y1);
                let y_2 = f32x8::from_slice(simd, y2);

                let i1 = y_0 - ((y_2 + y_1) * 0.25).floor();
                let i0 = y_2 + i1;
                let i2 = y_1 + i1;

                i0.store(y0);
                i1.store(y1);
                i2.store(y2);
            }
        }
    }
    let remainder = s0.len() / 8 * 8;
    for i in remainder..s0.len() {
        let y0 = s0[i];
        let y1 = s1[i];
        let y2 = s2[i];
        let (i0, i1, i2) = match transform {
            WaveletTransform::Irreversible97 => (
                math::mul_add(y2, 1.402, y0),
                math::mul_add(y2, -0.71414, math::mul_add(y1, -0.34413, y0)),
                math::mul_add(y1, 1.772, y0),
            ),
            WaveletTransform::Reversible53 => {
                let green = y0 - math::floor_f32((y2 + y1) * 0.25);
                (y2 + green, green, y1 + green)
            }
        };
        s0[i] = i0;
        s1[i] = i1;
        s2[i] = i2;
    }
}

