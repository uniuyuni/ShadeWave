"""Python-facing local contrast backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import (
    BackendStatus,
    backend_preference,
    import_error_detail,
    optional_backend,
    strict_enabled,
)
from . import local_contrast_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_local_contrast_metal")


def native_available() -> bool:
    return _metal_backend is not None and _metal_device_available()


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_LOCAL_CONTRAST_BACKEND")


def native_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "off", "0", "false", "no"}:
        return False
    return _metal_backend is not None and _metal_device_available()


def _metal_device_available() -> bool:
    if _metal_backend is None:
        return False
    try:
        return bool(_metal_backend.metal_available())
    except Exception:
        return False


def _metal_strict() -> bool:
    return strict_enabled("PLATYPUS_LOCAL_CONTRAST_METAL_STRICT")


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("local_contrast", "effect_backends._local_contrast_metal", True)
    if _backend_preference() in {"metal", "gpu"}:
        if _metal_backend is not None:
            detail = "Metal backend is built, but no Metal device is available"
        else:
            detail = import_error_detail(_METAL_IMPORT_ERROR)
        return BackendStatus("local_contrast", "effect_backends.local_contrast_reference", False, detail)
    if _metal_backend is not None:
        return BackendStatus(
            "local_contrast",
            "effect_backends.local_contrast_reference",
            False,
            "Metal backend available; PLATYPUS_LOCAL_CONTRAST_BACKEND requested reference",
        )
    return BackendStatus("local_contrast", "effect_backends.local_contrast_reference", False)


def _metal_compatible(image) -> bool:
    arr = np.asarray(image)
    return arr.dtype == np.float32 and arr.ndim == 3 and arr.shape[-1] == 3


def apply_clarity(rgb_image, clarity_amount):
    if native_enabled() and _metal_compatible(rgb_image):
        try:
            return _metal_backend.apply_clarity(np.ascontiguousarray(rgb_image, dtype=np.float32), float(clarity_amount))
        except Exception:
            if _metal_strict():
                raise
    return local_contrast_reference.apply_clarity(rgb_image, clarity_amount)


def apply_texture(rgb_image, texture_amount):
    if native_enabled() and _metal_compatible(rgb_image):
        try:
            return _metal_backend.apply_texture(np.ascontiguousarray(rgb_image, dtype=np.float32), float(texture_amount))
        except Exception:
            if _metal_strict():
                raise
    return local_contrast_reference.apply_texture(rgb_image, texture_amount)


def apply_microcontrast(image, strength):
    if native_enabled() and _metal_compatible(image):
        try:
            return _metal_backend.apply_microcontrast(np.ascontiguousarray(image, dtype=np.float32), float(strength))
        except Exception:
            if _metal_strict():
                raise
    return local_contrast_reference.apply_microcontrast(image, strength)


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "apply_clarity",
    "apply_texture",
    "apply_microcontrast",
]
