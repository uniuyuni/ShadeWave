"""Python-facing Color Separation backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import (
    BackendStatus,
    backend_preference,
    import_error_detail,
    native_backend_enabled,
    optional_backend,
    strict_enabled,
)
from . import color_separation_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_color_separation_cpu")


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_COLOR_SEPARATION_BACKEND")


def native_enabled() -> bool:
    return native_backend_enabled(_cpu_backend, _backend_preference())


def _native_strict() -> bool:
    return strict_enabled("PLATYPUS_COLOR_SEPARATION_STRICT")


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("color_separation", "effect_backends._color_separation_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "color_separation",
            "effect_backends.color_separation_reference",
            False,
            "cpu backend available; PLATYPUS_COLOR_SEPARATION_BACKEND requested reference",
        )
    detail = import_error_detail(_CPU_IMPORT_ERROR)
    return BackendStatus("color_separation", "effect_backends.color_separation_reference", False, detail)


def apply_color_separation(
    img_float32,
    shadow_chroma_clean=0.0,
    shadow_threshold=0.2,
    color_separation=0.0,
    chroma_clarity=0.0,
    color_density=0.0,
    subtractive_saturation=0.0,
    opponent_contrast=0.0,
):
    shadow_chroma_clean = float(shadow_chroma_clean)
    color_separation = float(color_separation)
    chroma_clarity = float(chroma_clarity)
    color_density = float(color_density)
    subtractive_saturation = float(subtractive_saturation)
    opponent_contrast = float(opponent_contrast)
    if (
        shadow_chroma_clean == 0.0
        and color_separation == 0.0
        and chroma_clarity == 0.0
        and color_density == 0.0
        and subtractive_saturation == 0.0
        and opponent_contrast == 0.0
    ):
        return img_float32
    image32 = np.asarray(img_float32, dtype=np.float32)
    if (
        native_enabled()
        and image32.ndim == 3
        and image32.shape[-1] == 3
    ):
        try:
            return _cpu_backend.apply_color_separation(
                np.ascontiguousarray(image32),
                float(shadow_chroma_clean),
                float(shadow_threshold),
                float(color_separation),
                float(chroma_clarity),
                float(color_density),
                float(subtractive_saturation),
                float(opponent_contrast),
            )
        except Exception:
            if _native_strict():
                raise

    return color_separation_reference.apply_color_separation(
        image32,
        shadow_chroma_clean,
        shadow_threshold,
        color_separation,
        chroma_clarity,
        color_density,
        subtractive_saturation,
        opponent_contrast,
    )


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "apply_color_separation",
]
