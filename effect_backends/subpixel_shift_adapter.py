"""Python-facing Subpixel Shift backend adapter."""

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
from . import subpixel_shift_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_subpixel_shift_cpu")


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_SUBPIXEL_SHIFT_BACKEND")


def native_enabled() -> bool:
    return native_backend_enabled(_cpu_backend, _backend_preference())


def _native_strict() -> bool:
    return strict_enabled("PLATYPUS_SUBPIXEL_SHIFT_STRICT")


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("subpixel_shift", "effect_backends._subpixel_shift_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "subpixel_shift",
            "effect_backends.subpixel_shift_reference",
            False,
            "cpu backend available; PLATYPUS_SUBPIXEL_SHIFT_BACKEND requested reference",
        )
    detail = import_error_detail(_CPU_IMPORT_ERROR)
    return BackendStatus("subpixel_shift", "effect_backends.subpixel_shift_reference", False, detail)


def subpixel_shift(img_array, shift_x=0.5, shift_y=0.5):
    image32 = np.asarray(img_array, dtype=np.float32)
    if native_enabled() and image32.ndim == 3 and image32.shape[-1] == 3:
        try:
            return _cpu_backend.subpixel_shift(
                np.ascontiguousarray(image32),
                float(shift_x),
                float(shift_y),
            )
        except Exception:
            if _native_strict():
                raise

    return subpixel_shift_reference.subpixel_shift(image32, shift_x, shift_y)


def create_enhanced_image(img_array):
    image32 = np.asarray(img_array, dtype=np.float32)
    if native_enabled() and image32.ndim == 3 and image32.shape[-1] == 3:
        try:
            return _cpu_backend.create_enhanced_image(np.ascontiguousarray(image32))
        except Exception:
            if _native_strict():
                raise

    return subpixel_shift_reference.create_enhanced_image(image32)


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "subpixel_shift",
    "create_enhanced_image",
]
