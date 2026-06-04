#!/usr/bin/env python3
"""Report which native libraries are used for EXR export support."""

from __future__ import annotations

import ctypes.util
import importlib.util
import os
import sys

import pyvips


def _has_vips_operation(name: str) -> bool:
    try:
        pyvips.Operation.new_from_name(name)
        return True
    except pyvips.Error:
        return False


def main() -> int:
    print("Python:", sys.executable)
    print("sys.prefix:", sys.prefix)
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", ""))
    print("pyvips:", getattr(pyvips, "__version__", "unknown"))
    print("libvips:", ".".join(str(pyvips.version(i)) for i in range(3)))
    print("libvips path:", ctypes.util.find_library("vips") or "(not found)")
    print("pyvips openexrload:", "yes" if _has_vips_operation("openexrload") else "no")
    print("pyvips exrsave:", "yes" if _has_vips_operation("exrsave") else "no")
    print("OpenEXR Python:", "yes" if importlib.util.find_spec("OpenEXR") else "no")

    if _has_vips_operation("exrsave"):
        print("NOTE: this libvips build exposes exrsave.")
    else:
        print("NOTE: native libvips exposes OpenEXR load only; export uses the OpenEXR Python package.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
