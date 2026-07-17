"""Python-facing CrossFilter backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import cross_filter_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_cross_filter_metal")
_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_cross_filter_cpu")

_SELECTOR = BackendSelector(
    "cross_filter",
    globals(),
    env="PLATYPUS_CROSS_FILTER_BACKEND",
    metal_strict_env="PLATYPUS_CROSS_FILTER_METAL_STRICT",
    metal_name="effect_backends._cross_filter_metal",
    cpu_name="effect_backends._cross_filter_cpu",
    reference_name="effect_backends.cross_filter_reference",
    metal_disabled_values={"reference", "python", "opencv", "cpu", "native", "off", "0", "false", "no"},
    cpu_disabled_values={"reference", "python", "opencv", "off", "0", "false", "no"},
    metal_forced_values={"metal"},
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def _cpu_backend_enabled() -> bool:
    # Kept local: unlike the shared cpu decision, the apply path also yields
    # to Metal when the preference is empty/"metal" and a device is present.
    value = _backend_preference()
    if value in {"metal", ""} and _metal_backend is not None and _metal_device_available():
        return False
    if value in {"reference", "python", "opencv", "off", "0", "false", "no"}:
        return False
    return True


def _metal_backend_enabled() -> bool:
    return _SELECTOR.metal_enabled()


def _metal_strict() -> bool:
    return _SELECTOR.metal_strict()


def _metal_device_available() -> bool:
    return _SELECTOR.metal_device_available()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


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
