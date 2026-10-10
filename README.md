# ps-hayro-jpeg2000

[![Crates.io](https://img.shields.io/crates/v/ps-hayro-jpeg2000.svg)](https://crates.io/crates/ps-hayro-jpeg2000)
[![Documentation](https://docs.rs/ps-hayro-jpeg2000/badge.svg)](https://docs.rs/ps-hayro-jpeg2000)

<!-- cargo-rdme start -->

A memory-safe, pure-Rust JPEG 2000 decoder.

This is the pathscale fork of hayro-jpeg2000 by Laurenz Stampfl. The package
is `ps-hayro-jpeg2000`; its library remains `hayro_jpeg2000`.

Changes in this fork:

- Silent skipping of unknown boxes, with typed errors and no logging macros.
- Tile-specific COD/COC settings for reconstruction and tile-local inverse MCT.
- Signed component metadata and DC level shifting only for unsigned components.
- Raw four-component streams preserve four colour channels without assumed alpha.
- Validated CDEF permutations that move planes, with premultiplied opacity reported.
- Interleaved u8 and u16 output, scaled per component, rounded and clamped.
- Finite pixel and reconstruction-buffer limits, checked before plane allocation.
- Shared ICC profile bytes and borrowed input for single-segment code blocks.

Packed unsigned output biases signed component ranges to unsigned. Planar
`ComponentData::samples()` retains signed samples and `is_signed()` reports them.
Only a JP2 CDEF marks opacity; `has_premultiplied_alpha()` reports its representation.
Raw colour interpretation remains the embedder's responsibility.

`DecodeSettings` defaults to at most 2^28 decoded pixels and 2 GiB for component
planes, coefficients, IDWT output and scratch, and palette expansion planes.
The byte limit excludes compressed input, packet metadata, code-block scratch,
and packed output allocated by the caller.

`hayro-jpeg2000` can decode both raw JPEG 2000 codestreams (`.j2c`) and images wrapped
inside the JP2 container format. The decoder supports the vast majority of features
defined in the JPEG2000 core coding system (ISO/IEC 15444-1) as well as some color
spaces from the extensions (ISO/IEC 15444-2). There are still some missing pieces
for some "obscure" features(like for example support for progression order
changes in tile-parts), but all features that actually commonly appear in real-life
images should be supported (if not, please open an issue!).

The decoder abstracts away most of the internal complexity of JPEG2000
and yields decoded component planes or interleaved unsigned 8-bit and 16-bit
samples with greyscale, RGB, CMYK, ICC-based or unknown colour metadata.

## Example
```rust
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

## Testing
Upstream reports coverage of 20.000+ images scraped from random PDFs and a
large part of the `OpenJPEG` test suite. Fork correctness is verified through
external pdf-toolkit fixtures whose generator computes the expected pixels.
These upstream coverage claims do not establish verification of the fork.

## Performance
A decent amount of effort has already been put into optimizing this crate
(both in terms of raw performance but also memory allocations). However, there
are some more important optimizations that have not been implemented yet, so
there is definitely still room for improvement (and I am planning on implementing
them eventually).

Overall, you should expect this crate to have worse performance than `OpenJPEG`,
but the difference gap should not be too large.

## Safety
The default `std` feature adds no dependencies. Use `default-features = false`
for the dependency-free `no_std + alloc` decoder. SIMD is optional through
`fearless_simd`; it is not enabled by default. Unsafe code is forbidden in
this crate via a crate-level attribute.

## License
Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option, as upstream.

The OpenJPEG attribution notes in the bitplane decoder and colour conversion
formulas are preserved. Bundled ICC profiles retain their upstream CC0 licence.

<!-- cargo-rdme end -->

