"""Python-facing lens coating backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendStatus, backend_preference, import_error_detail, optional_backend, strict_enabled
from . import coating_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_coating_metal")


def presets():
    return coating_reference.PRESETS


def native_available() -> bool:
    return _metal_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_COATING_BACKEND")


def _metal_backend_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "opencv", "off", "0", "false", "no"}:
        return False
    return value in {"", "auto", "metal"}


def _metal_strict() -> bool:
    return strict_enabled("PLATYPUS_COATING_METAL_STRICT")


def _metal_device_available() -> bool:
    if _metal_backend is None:
        return False
    try:
        return bool(_metal_backend.metal_available())
    except Exception:
        return False


def backend_status() -> BackendStatus:
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        return BackendStatus("coating", "effect_backends._coating_metal", True)
    if _backend_preference() == "metal":
        if _metal_backend is not None:
            detail = "Metal backend is built, but no Metal device is available"
        else:
            detail = import_error_detail(_METAL_IMPORT_ERROR)
        return BackendStatus("coating", "effect_backends.coating_reference", False, detail)
    return BackendStatus("coating", "effect_backends.coating_reference", False, import_error_detail(_METAL_IMPORT_ERROR))


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
