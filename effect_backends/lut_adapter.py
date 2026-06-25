"""Python-facing 3D LUT backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import (
    BackendStatus,
    backend_preference,
    import_error_detail,
    optional_backend,
    strict_enabled,
)
from . import lut_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_lut_metal")


def native_available() -> bool:
    return _metal_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_LUT_BACKEND")


def _metal_backend_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "numpy", "off", "0", "false", "no"}:
        return False
    return value in {"", "auto", "metal"}


def _metal_strict() -> bool:
    return strict_enabled("PLATYPUS_LUT_METAL_STRICT")


def _metal_device_available() -> bool:
    if _metal_backend is None:
        return False
    try:
        return bool(_metal_backend.metal_available())
    except Exception:
        return False


def backend_status() -> BackendStatus:
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        return BackendStatus("lut", "effect_backends._lut_metal", True)
    if _backend_preference() == "metal":
        if _metal_backend is not None:
            detail = "Metal backend is built, but no Metal device is available"
        else:
            detail = import_error_detail(_METAL_IMPORT_ERROR)
        return BackendStatus("lut", "effect_backends.lut_reference", False, detail)
    return BackendStatus(
        "lut",
        "effect_backends.lut_reference",
        False,
        import_error_detail(_METAL_IMPORT_ERROR),
    )


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
