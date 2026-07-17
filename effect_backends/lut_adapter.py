"""Python-facing 3D LUT backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import lut_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_lut_metal")

_SELECTOR = BackendSelector(
    "lut",
    globals(),
    env="PLATYPUS_LUT_BACKEND",
    metal_strict_env="PLATYPUS_LUT_METAL_STRICT",
    metal_name="effect_backends._lut_metal",
    reference_name="effect_backends.lut_reference",
    metal_disabled_values={"reference", "python", "numpy", "off", "0", "false", "no"},
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


def apply_lut3d(image: np.ndarray, table: np.ndarray, domain: np.ndarray, size: int) -> np.ndarray:
    """Apply a 3D LUT (trilinear) to ``image``.

    image:  (..., 3) float32. table: (size, size, size, 3) float32.
    domain: (2, 3) float32. Returns a new float32 array shaped like ``image``.
    """
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        try:
            image32 = np.ascontiguousarray(image, dtype=np.float32)
            table32 = np.ascontiguousarray(table, dtype=np.float32)
            domain32 = np.ascontiguousarray(domain, dtype=np.float32)
            return _metal_backend.apply_lut3d(image32, table32, domain32, int(size))
        except Exception:
            if _metal_strict():
                raise

    return lut_reference.apply_lut3d(image, table, domain, size)
