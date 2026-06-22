"""Locate the bundled radiance_codec dylib before radiance_codec is imported."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))

    exe_dir = Path(sys.executable).resolve().parent
    roots.append(exe_dir)
    roots.extend(exe_dir.parents)
    return roots


def _set_library_env() -> None:
    for root in _candidate_roots():
        for rel in (
            "libradiance_codec.dylib",
            "Frameworks/libradiance_codec.dylib",
            "Resources/libradiance_codec.dylib",
            "Contents/Frameworks/libradiance_codec.dylib",
            "Contents/Resources/libradiance_codec.dylib",
        ):
            lib = root / rel
            if lib.is_file():
                os.environ["RADIANCE_CODEC_LIBRARY"] = str(lib)
                return


if not os.environ.get("RADIANCE_CODEC_LIBRARY"):
    _set_library_env()
