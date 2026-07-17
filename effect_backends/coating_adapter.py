"""Python-facing lens coating backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import coating_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_coating_metal")

_SELECTOR = BackendSelector(
    "coating",
    globals(),
    env="PLATYPUS_COATING_BACKEND",
    metal_strict_env="PLATYPUS_COATING_METAL_STRICT",
    metal_name="effect_backends._coating_metal",
    reference_name="effect_backends.coating_reference",
    metal_disabled_values={"reference", "python", "opencv", "off", "0", "false", "no"},
    metal_forced_values={"metal"},
)


def presets():
    return coating_reference.PRESETS


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def _metal_backend_enabled() -> bool:
    return _SELECTOR.metal_enabled()


def _metal_strict() -> bool:
    return _SELECTOR.metal_strict()


def _metal_device_available() -> bool:
    return _SELECTOR.metal_device_available()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


def apply_preset(
    image: np.ndarray,
    preset_name: str,
    light_source_intensity: float = 1.0,
    resolution_scale: float = 1.0,
) -> np.ndarray:
    preset_map = coating_reference.PRESETS
    if preset_name not in preset_map:
        raise ValueError(f"Unknown preset: {preset_name}")
    preset = preset_map[preset_name]

    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        try:
            image32 = np.ascontiguousarray(image, dtype=np.float32)
            matrix = np.ascontiguousarray(preset["color_matrix"], dtype=np.float32)
            return _metal_backend.apply_coating(
                image32,
                matrix,
                float(preset["flare_factor"]) * float(light_source_intensity),
                float(preset["contrast_factor"]),
                float(preset["saturation_factor"]),
                float(resolution_scale),
            )
        except Exception:
            if _metal_strict():
                raise

    return coating_reference.apply_preset(
        image,
        preset_name,
        light_source_intensity=light_source_intensity,
        resolution_scale=resolution_scale,
    )


__all__ = ["apply_preset", "backend_status", "native_available", "presets"]
