# macOS App Distribution Notices

This document lists the license and notice files that should be included when
distributing a packaged Shade Wave `.app` build.

Shade Wave / PLATYPUS is licensed under the GNU General Public License version
3 or later (GPL-3.0-or-later). Binary distribution of the app must provide the
corresponding source code and the license notices for the project and bundled
third-party components.

## Files to Include

Include these files next to the distributed `.app`, inside the disk image, zip
archive, or other app distribution package:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- `licenses/DNG_SDK_LICENSE.txt`
- `README.md`
- `README_JA.md`

If `external/libraw_enhanced` is distributed as source or as a separately
packaged component, also include:

- `external/libraw_enhanced/LICENSE`
- `external/libraw_enhanced/README.md`

## Source Code Offer

Because Shade Wave is distributed under GPL-3.0-or-later, binary app releases
must make the corresponding source code available under the same license terms.
For public releases, include either:

- a link to the exact public source repository and revision used for the build,
  or
- a written offer explaining how to obtain the corresponding source code.

The source code should include local patches, build scripts, license files, and
the source for GPL-covered components required to rebuild the distributed app.

## Third-Party Notices

`THIRD_PARTY_NOTICES.md` records notable bundled or ported code, including:

- RawTherapee-derived demosaicing code in `external/libraw_enhanced`
- LibRaw
- Adobe DNG SDK-derived temperature/tint conversion helpers
- Colour Science compatible colour conversion code
- optional external AI and image-processing projects

The notice file is not a complete dependency license report for every Python,
pixi, conda, Homebrew, or system dynamic library that may be copied into a
packaged `.app`. Before publishing a binary build, generate or collect license
notices for the exact bundled runtime environment.

## Model Weights and Optional AI Components

Do not assume model weights have the same license as Shade Wave. If a release
bundle includes AI model weights or external AI project files, include the
corresponding upstream license or model terms for each item.

Common optional projects to verify:

- SAM3
- Depth Pro
- SCUNet
- radiance_denoise
- demosaicnet_torch

## App Bundle Placement

For a `.dmg` or `.zip` distribution, the notice files may be placed beside the
`.app` bundle. If the app is distributed without surrounding files, place copies
inside the bundle, for example:

- `Shade Wave.app/Contents/Resources/LICENSE`
- `Shade Wave.app/Contents/Resources/THIRD_PARTY_NOTICES.md`
- `Shade Wave.app/Contents/Resources/licenses/DNG_SDK_LICENSE.txt`

## Release Checklist

- Confirm `define.py` contains the intended release version.
- Confirm the distributed source revision matches the built app.
- Include `LICENSE`, `THIRD_PARTY_NOTICES.md`, and DNG SDK license text.
- Include notices for all bundled runtime libraries and codecs.
- Include license terms for any bundled AI models or optional external projects.
- Keep Adobe and DNG trademark references factual; do not imply endorsement.
