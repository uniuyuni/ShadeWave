# Shade Wave

[日本語版](README_JA.md)

![Shade Wave](docs/screenshot%201.png)

Shade Wave is a macOS-focused RAW/RGB photo editor built with Python and Kivy.
It combines a linear-light image pipeline, mask-based local editing, color-management-aware export, and optional AI helpers such as SAM3, Depth Pro, SCUNet, and inpainting backends.

The project is currently optimized for local desktop use on Apple Silicon macOS. It is not a small pip-installable library, and several AI/model features require large external downloads.

## Quick Start

### Requirements

| Item | Requirement |
| --- | --- |
| OS | macOS, tested primarily on Apple Silicon |
| Python | Python 3.11 through pixi |
| Package manager | [pixi](https://pixi.sh/) |
| Compiler tools | Xcode Command Line Tools / Apple Clang |
| Metadata tool | ExifTool CLI, e.g. `brew install exiftool` |
| Network | Required during setup for external repositories, ICC profiles, and model weights |
| Optional token | Hugging Face token for `facebook/sam3.1` checkpoint download |
| Optional API key | `RUNWARE_API_KEY` for Runware-backed inpainting / object eraser features |

Install pixi first, then run:

```bash
git clone https://github.com/uniuyuni/platypus.git
cd platypus
./setup.sh
pixi run python main.py
```

`setup.sh` prepares the pixi environment, clones external projects into `external/`, builds native extensions, downloads ICC profiles, and fetches model assets when needed.

ExifTool must be available on `PATH` before using metadata, rating, or export metadata features.

The application stores user-editable settings and presets in:

```text
~/Pictures/Shade Wave
```

## Scope

Shade Wave can:

- Open common RGB images, camera RAW files, and OpenEXR images.
- Edit images through a layered, non-destructive parameter pipeline.
- Apply global corrections, local masks, geometry tools, film/look effects, denoise, dehaze, sharpening, grain, and vignette.
- Export to common delivery formats with ICC profile handling and metadata options.
- Use optional AI-assisted features when the required external models are available.

Shade Wave does not currently aim to be:

- A cross-platform packaged application.
- A lightweight Python library API.
- A fully documented end-user product with stable public extension points.
- A signed/notarized macOS distribution.

## Usage

### Launch the GUI

```bash
pixi run python main.py
```

Open images from the file viewer, adjust effects from the editor panels, and export through the export dialog. Edits are stored as sidecar parameters next to the source image or in the configured user data flow used by the app.

### Build a macOS `.app`

```bash
pixi run build-macos-app
```

The generated app is written to:

```text
dist/Shade Wave.app
```

You can also choose temporary build/output directories:

```bash
pixi run python scripts/build_macos_app_pyinstaller.py \
  --distpath /tmp/platypus-app-dist \
  --workpath /tmp/platypus-app-build
```

The `.app` build bundles the current environment and large runtime assets, so the result can be several gigabytes. Code signing and notarization are not handled by this script.

### Run Tests

```bash
pixi run python -m unittest discover -s tests -p "test_*.py"
```

Some tests and runtime paths depend on native libraries, external repositories, or model assets prepared by `setup.sh`.

### Keyboard Shortcuts

| Shortcut | Action |
| --- | --- |
| `0` | Toggle preview zoom at the mouse position, or the preview center when the mouse is outside the image |
| `M` | Temporarily hide the mask overlay |
| `Space` hold | Temporary preview drag/fast-display mode; release to redraw the normal preview |
| `Cmd/Ctrl + S` | Save the current sidecar parameters |
| `Cmd/Ctrl + C` | Copy current effect settings |
| `Cmd/Ctrl + V` | Paste copied effect settings |
| `Cmd/Ctrl + F` | Toggle preview focus mode |
| `Cmd/Ctrl + Z` | Undo |
| `Cmd/Ctrl + Shift + Z` | Redo |
| `Delete` / `Backspace` | Delete the selected line in the distortion line-guide editor |

`0` is ignored while a text input has focus. Some shortcuts only do work when an image is loaded and the corresponding editor state is active.

## Input and Output

### Supported Input

| Type | Extensions |
| --- | --- |
| RGB / standard images | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.gif`, `.heic`, `.jxl` |
| RAW | `.cr2`, `.cr3`, `.nef`, `.arw`, `.dng`, `.orf`, `.raf`, `.rw2`, `.sr2`, `.pef`, `.raw`, `.3fr`, `.fff` |
| HDR / scene-linear | `.exr` |

RAW loading uses the `libraw_enhanced` external backend. OpenEXR uses the OpenEXR Python bindings and is kept on a separate path from regular RGB loading.

### Supported Export

| Format | Notes |
| --- | --- |
| JPEG | Quality setting, metadata/rating support |
| TIFF | Deflate compression |
| PNG | Quality option passed through the export path |
| JPEG XL | Requires the installed image stack to support JXL |
| HEIF | Requires the installed image stack to support HEIF |
| OpenEXR | Scene-linear export, chromaticities written when available |

Export can resize, sharpen, apply dithering/quantization, embed ICC profiles for non-EXR formats, and copy selected EXIF/GPS metadata.

Available ICC profiles are loaded from `icc/`, including sRGB, Display P3, Adobe RGB, ProPhoto RGB, ACES, Rec.709, and Rec.2020 profiles when setup has completed successfully.

## Features

- RAW development: RAW decode, auto exposure, color temperature/tint, lens modifier integration, chromatic aberration correction.
- Global tone and color: exposure, contrast, tone, levels, curves, HLS, hue/saturation/luminance curves, color separation, LUT input/look stages.
- Detail and restoration: AI noise reduction, SCUNet, radiance denoise, dehaze, clarity, texture, microcontrast, unsharp mask.
- Geometry: rotation, crop, perspective/mesh-like geometry correction, distortion correction, subpixel shift, lens blur, mosaic, scratch/frosted glass style effects.
- Local editing: mask layers, composite masks, free draw masks, quick-select style edge refinement, mask geometry, headless mask replay during export.
- AI-assisted masks: SAM3 box/text segmentation, Depth Pro depth masks, face masks through helper backends.
- Creative looks: film simulation, lens simulation, cross filter, glow, orton, grain, vignette, solid color overlays.
- Metadata and ratings: export metadata options and rating propagation.
- Packaging: PyInstaller-based macOS `.app` creation for the current pixi environment.

## Architecture

The codebase is organized around a preview/export pipeline and effect objects.

```text
main.py                         Kivy application entry point and UI orchestration
effects.py                      Effect definitions and parameter/widget binding
pipeline.py                     Preview and export pipeline execution
export.py                       File export, color conversion, ICC/metadata handling
cores/                          Image-processing kernels and reusable algorithms
cores/mask2/                    Mask generation, headless masks, AI mask runtime
helpers/                        Integrations for external AI/native backends
effect_backends/                Optional/native backend adapters
widgets/                        Kivy widgets and editor UI components
external/                       Cloned third-party projects prepared by setup.sh
scripts/                        Build, packaging, and environment helper scripts
tests/                          Regression tests for pipeline, UI flow, and helpers
```

The high-level flow is:

1. `main.py` loads an image into an `ImageSet`.
2. Effect parameters are stored in `primary_param` and synchronized to widgets by `effects.py`.
3. `pipeline.py` applies effect levels in order, with preview caching and mask compositing.
4. Mask2 tools can generate or edit masks in the UI; export uses a headless mask pipeline to replay them.
5. `export.py` reruns the full pipeline, converts color for the selected output profile, then writes the file and metadata.

## External Components

`setup.sh` clones or prepares several large external projects under `external/`:

| Component | Purpose |
| --- | --- |
| `libraw_enhanced` | RAW decoding and Metal shader resources |
| `SAM3` | AI segmentation masks |
| `ml-depth-pro` | Depth estimation for depth masks |
| `SCUNet` | PyTorch denoise model weights |
| `SCUNet_CoreML` | Core ML SCUNet model and runtime |
| `radiance_denoise` | Native denoise backend |
| `demosaicnet_torch` | AI demosaic helper |

Most of these projects have their own licenses and model terms. Check each upstream repository before redistribution.

## Configuration

Important default settings live in `config.py` and are copied/migrated into `~/Pictures/Shade Wave/config.json` at runtime.

| Key | Default | Meaning |
| --- | --- | --- |
| `preview_size` | `640` | Minimum preview texture side |
| `raw_auto_exposure` | `true` | Enable RAW auto exposure by default |
| `gpu_device` | `mps` | Preferred AI device on Apple Silicon |
| `display_color_gamut` | `sRGB` | Display transform target |
| `cat` | `cat16` | Chromatic adaptation transform |
| `base_resolution_scale` | `[4096, 4096]` | Base processing resolution limit |
| `mesh_rbf_function` | `mls` | Mesh warp interpolation mode |

## Known Issues and Constraints

- Setup is large. SAM3 and Depth Pro checkpoints alone can take several gigabytes.
- SAM3 checkpoint download may require a Hugging Face token and model access approval.
- Runware-backed inpainting/object eraser features require `RUNWARE_API_KEY`; without it, that path is unavailable and may currently fail noisily.
- PyTorch MPS may fall back to CPU for unsupported operators, which can make some AI paths slow.
- Core ML tooling can lag behind the newest PyTorch releases; the project keeps compatibility through setup constraints and targeted install commands.
- The macOS app build is unsigned and not notarized.
- The PyInstaller bundle is large because it includes Torch/CoreML/native dependencies and model resources.
- Windows and Linux are not first-class targets at the moment.

## Development Notes

- Keep `requirements.txt` as the pip dependency source used by `setup.sh`.
- Keep native or model-heavy third-party projects under `external/`; use `utils.external_paths` when adding import paths.
- Prefer focused tests for pipeline, mask replay, effect parameter binding, export behavior, and UI flow regressions.
- Do not commit downloaded checkpoints or generated build outputs.
- Refactoring guidance lives in `docs/refactoring-guidelines.md`.

Useful commands:

```bash
pixi install
./setup.sh
pixi run python main.py
pixi run python -m unittest discover -s tests -p "test_*.py"
pixi run build-macos-app
```

## Version and Compatibility

Current application version is defined in `define.py`.

```text
Shade Wave 2.22.42
```

The active pixi environment targets `osx-arm64` and Python `>=3.11.13,<3.12`.

## License and Credits

Shade Wave / PLATYPUS is licensed under the GNU General Public License version 3 or later. See `LICENSE`.

The repository includes and integrates third-party code and optional external projects with their own terms, including RawTherapee-derived demosaicing code in `libraw_enhanced`, LibRaw, Adobe DNG SDK-derived temperature helpers, Colour Science-compatible colour conversion code, SAM3, Depth Pro, SCUNet, and radiance_denoise. See `THIRD_PARTY_NOTICES.md` before redistributing source, binaries, model weights, or packaged `.app` builds.

## Links

- Repository: https://github.com/uniuyuni/platypus
- Refactoring notes: `docs/refactoring-guidelines.md`
- SAM3 model access: https://huggingface.co/facebook/sam3.1
