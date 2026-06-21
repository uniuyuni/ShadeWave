"""Python-facing CrossFilter backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import (
    BackendStatus,
    backend_preference,
    import_error_detail,
    optional_backend,
    strict_enabled,
)
from . import cross_filter_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_cross_filter_metal")
_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_cross_filter_cpu")


def native_available() -> bool:
    return _metal_backend is not None or _cpu_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_CROSS_FILTER_BACKEND")


def _cpu_backend_enabled() -> bool:
    value = _backend_preference()
    if value in {"metal", ""} and _metal_backend is not None and _metal_device_available():
        return False
    if value in {"reference", "python", "opencv", "off", "0", "false", "no"}:
        return False
    return True


def _metal_backend_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "opencv", "cpu", "native", "off", "0", "false", "no"}:
        return False
    return value in {"", "auto", "metal"}


def _metal_strict() -> bool:
    return strict_enabled("PLATYPUS_CROSS_FILTER_METAL_STRICT")


def _metal_device_available() -> bool:
    if _metal_backend is None:
        return False
    try:
        return bool(_metal_backend.metal_available())
    except Exception:
        return False


def backend_status() -> BackendStatus:
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        return BackendStatus("cross_filter", "effect_backends._cross_filter_metal", True)
    if _backend_preference() == "metal":
        if _metal_backend is not None:
            detail = "Metal backend is built, but no Metal device is available"
        else:
            detail = import_error_detail(_METAL_IMPORT_ERROR)
        return BackendStatus("cross_filter", "effect_backends.cross_filter_reference", False, detail)
    if _cpu_backend is not None and _cpu_backend_enabled():
        return BackendStatus("cross_filter", "effect_backends._cross_filter_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "cross_filter",
            "effect_backends.cross_filter_reference",
            False,
            "cpu backend available; PLATYPUS_CROSS_FILTER_BACKEND requested reference",
        )
    detail = import_error_detail(_CPU_IMPORT_ERROR)
    return BackendStatus("cross_filter", "effect_backends.cross_filter_reference", False, detail)


def apply_cross_filter(
    img_rgb: np.ndarray,
    num_points: int = 6,
    length: int = 100,
    angle_deg: float = 0,
    threshold: float = 1.0,
    intensity: float = 1.0,
    spectral_strength: float = 0.2,
    line_thickness: float = 1.0,
    min_distance: int = 10,
    randomness: float = 0.0,
    speed_factor: int = 4,
    debug_mode: bool = False,
) -> np.ndarray:
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        image32 = np.ascontiguousarray(img_rgb, dtype=np.float32)
        try:
            return _metal_backend.apply_cross_filter(
                image32,
                int(num_points),
                int(length),
                float(angle_deg),
                float(threshold),
                float(intensity),
                float(spectral_strength),
                float(line_thickness),
                int(min_distance),
                float(randomness),
                int(speed_factor),
                bool(debug_mode),
            )
        except Exception:
            if _metal_strict():
                raise

    if _cpu_backend is not None and _cpu_backend_enabled():
        image32 = np.ascontiguousarray(img_rgb, dtype=np.float32)
        return _cpu_backend.apply_cross_filter(
            image32,
            int(num_points),
            int(length),
            float(angle_deg),
            float(threshold),
            float(intensity),
            float(spectral_strength),
            float(line_thickness),
            int(min_distance),
            float(randomness),
            int(speed_factor),
            bool(debug_mode),
        )

    return cross_filter_reference.apply_cross_filter(
        img_rgb,
        num_points=num_points,
        length=length,
        angle_deg=angle_deg,
        threshold=threshold,
        intensity=intensity,
        spectral_strength=spectral_strength,
        line_thickness=line_thickness,
        min_distance=min_distance,
        randomness=randomness,
        speed_factor=speed_factor,
        debug_mode=debug_mode,
    )
