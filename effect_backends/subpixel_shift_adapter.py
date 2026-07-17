"""Python-facing Subpixel Shift backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import subpixel_shift_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_subpixel_shift_cpu")

_SELECTOR = BackendSelector(
    "subpixel_shift",
    globals(),
    env="PLATYPUS_SUBPIXEL_SHIFT_BACKEND",
    native_strict_env="PLATYPUS_SUBPIXEL_SHIFT_STRICT",
    cpu_name="effect_backends._subpixel_shift_cpu",
    reference_name="effect_backends.subpixel_shift_reference",
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def native_enabled() -> bool:
    return _SELECTOR.native_enabled()


def _native_strict() -> bool:
    return _SELECTOR.native_strict()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


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
