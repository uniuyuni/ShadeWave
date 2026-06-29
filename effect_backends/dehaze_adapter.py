"""Python-facing Dehaze backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import (
    BackendStatus,
    backend_preference,
    import_error_detail,
    optional_backend,
    strict_enabled,
)
from . import dehaze_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_dehaze_metal")


def native_available() -> bool:
    return _metal_backend is not None and _metal_device_available()


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_DEHAZE_BACKEND")


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
    return strict_enabled("PLATYPUS_DEHAZE_METAL_STRICT")


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("dehaze", "effect_backends._dehaze_metal", True)
    if _backend_preference() in {"metal", "gpu"}:
        if _metal_backend is not None:
            detail = "Metal backend is built, but no Metal device is available"
        else:
            detail = import_error_detail(_METAL_IMPORT_ERROR)
        return BackendStatus("dehaze", "effect_backends.dehaze_reference", False, detail)
    if _metal_backend is not None:
        return BackendStatus(
            "dehaze",
            "effect_backends.dehaze_reference",
            False,
            "Metal backend available; PLATYPUS_DEHAZE_BACKEND requested reference",
        )
    return BackendStatus("dehaze", "effect_backends.dehaze_reference", False)


def _metal_compatible(image) -> bool:
    arr = np.asarray(image)
    return arr.dtype == np.float32 and arr.ndim == 3 and arr.shape[-1] == 3


def dehaze_image(img, strength=0.5):
    # Fog addition is a trivial full-frame blend; keeping it on CPU is faster
    # than paying the Metal command/buffer overhead in the current NumPy pipeline.
    if float(strength) >= 0.0 and native_enabled() and _metal_compatible(img):
        try:
            return _metal_backend.dehaze_image(np.ascontiguousarray(img, dtype=np.float32), float(strength))
        except Exception:
            if _metal_strict():
                raise
    return dehaze_reference.dehaze_image(img, strength)


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "dehaze_image",
]
