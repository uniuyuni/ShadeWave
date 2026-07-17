"""Python-facing Color Separation backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import color_separation_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_color_separation_cpu")
_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_color_separation_metal")

_SELECTOR = BackendSelector(
    "color_separation",
    globals(),
    env="PLATYPUS_COLOR_SEPARATION_BACKEND",
    native_strict_env="PLATYPUS_COLOR_SEPARATION_STRICT",
    metal_name="effect_backends._color_separation_metal",
    cpu_name="effect_backends._color_separation_cpu",
    reference_name="effect_backends.color_separation_reference",
    metal_disabled_values={"reference", "python", "cpu", "off", "0", "false", "no"},
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def native_enabled() -> bool:
    return _SELECTOR.native_enabled()


def _metal_backend_enabled() -> bool:
    return _SELECTOR.metal_enabled()


def _metal_device_available() -> bool:
    return _SELECTOR.metal_device_available()


def _native_strict() -> bool:
    return _SELECTOR.native_strict()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


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
        image32.ndim == 3
        and image32.shape[-1] == 3
        and _metal_backend is not None
        and _metal_backend_enabled()
        and _metal_device_available()
    ):
        try:
            return _metal_backend.apply_color_separation(
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
