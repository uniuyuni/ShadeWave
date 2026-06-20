"""
PyInstaller runtime hook for macOS app environment defaults.

Keep the frozen app close to the stable VS Code launch environment for native
runtime noise and PyTorch/MPS behavior.  Debug-only paths, PYTHONPATH, and
project-local conda variables are intentionally omitted because the bundle
should resolve its own packaged resources.
"""
from __future__ import annotations

import os
import sys

if getattr(sys, "frozen", False) and sys.platform == "darwin":
    os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
    os.environ.setdefault("KMP_WARNINGS", "0")
    os.environ.setdefault("LIBOMP_VERBOSE", "0")
    os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "FALSE")

    # SAM3 can hit MPS-unsupported PyTorch fused operators such as
    # aten::_addmm_activation.out.  CPU fallback avoids terminating the app.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

    # Match the app's intended Quick Select v4 defaults from launch.json.
    os.environ.setdefault("QS_DRAW_V4", "1")
    os.environ.setdefault("QS_V4_EDGE_SNAP", "1")

    # Use the Core ML SAM3 backbone in packaged macOS builds.
    os.environ.setdefault("PLATYPUS_SAM3_COREML_BACKBONE", "1")
