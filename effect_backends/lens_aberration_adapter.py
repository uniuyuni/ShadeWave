"""Python-facing Lens Aberration backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import lens_aberration_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_lens_aberration_metal")

_SELECTOR = BackendSelector(
    "lens_aberration",
    globals(),
    env="PLATYPUS_LENS_ABERRATION_BACKEND",
    metal_strict_env="PLATYPUS_LENS_ABERRATION_METAL_STRICT",
    metal_name="effect_backends._lens_aberration_metal",
    reference_name="effect_backends.lens_aberration_reference",
    metal_disabled_values={"reference", "python", "opencv", "off", "0", "false", "no"},
    metal_forced_values={"metal"},
)


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


def apply_lateral_chromatic_aberration(
    image: np.ndarray,
    strength: float,
    resolution_scale: float,
    radial: bool = True,
) -> np.ndarray:
    """Apply lateral chromatic aberration via Metal when possible, else reference."""
    if radial and _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        try:
            image32 = np.ascontiguousarray(image, dtype=np.float32)
            return _metal_backend.apply_lateral_ca(image32, float(strength), float(resolution_scale))
        except Exception:
            if _metal_strict():
                raise
    return lens_aberration_reference.apply_lateral_chromatic_aberration(
        image,
        strength=float(strength),
        resolution_scale=float(resolution_scale),
        radial=bool(radial),
    )


def apply_longitudinal_chromatic_aberration(
    image: np.ndarray,
    depth_map: np.ndarray,
    strength: float,
    focus_depth: float,
    resolution_scale: float,
) -> np.ndarray:
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        try:
            image32 = np.ascontiguousarray(image, dtype=np.float32)
            depth32 = np.ascontiguousarray(depth_map, dtype=np.float32)
            return _metal_backend.apply_longitudinal_ca(
                image32,
                depth32,
                float(strength),
                float(focus_depth),
                float(resolution_scale),
            )
        except Exception:
            if _metal_strict():
                raise
    return lens_aberration_reference.apply_longitudinal_chromatic_aberration(
        image,
        depth_map,
        strength=float(strength),
        focus_depth=float(focus_depth),
        resolution_scale=float(resolution_scale),
    )


def apply_spherical_aberration(
    image: np.ndarray,
    depth_map: np.ndarray | None,
    strength: float,
    aperture: float,
    focus_depth: float,
    highlight_threshold: float,
    resolution_scale: float,
) -> np.ndarray:
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        try:
            image32 = np.ascontiguousarray(image, dtype=np.float32)
            if depth_map is None:
                depth32 = np.empty(image32.shape[:2], dtype=np.float32)
                has_depth = False
            else:
                depth32 = np.ascontiguousarray(depth_map, dtype=np.float32)
                has_depth = True
            return _metal_backend.apply_spherical_ca(
                image32,
                depth32,
                bool(has_depth),
                float(strength),
                float(aperture),
                float(focus_depth),
                float(highlight_threshold),
                float(resolution_scale),
            )
        except Exception:
            if _metal_strict():
                raise
    return lens_aberration_reference.apply_spherical_aberration(
        image,
        depth_map,
        strength=float(strength),
        aperture=float(aperture),
        focus_depth=float(focus_depth),
        highlight_threshold=float(highlight_threshold),
        resolution_scale=float(resolution_scale),
    )


__all__ = [
    "apply_lateral_chromatic_aberration",
    "apply_longitudinal_chromatic_aberration",
    "apply_spherical_aberration",
    "backend_status",
    "native_available",
]
