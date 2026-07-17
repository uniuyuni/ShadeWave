"""Python-facing local contrast backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import local_contrast_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_local_contrast_metal")

_SELECTOR = BackendSelector(
    "local_contrast",
    globals(),
    env="PLATYPUS_LOCAL_CONTRAST_BACKEND",
    metal_strict_env="PLATYPUS_LOCAL_CONTRAST_METAL_STRICT",
    metal_name="effect_backends._local_contrast_metal",
    reference_name="effect_backends.local_contrast_reference",
    metal_enabled_values=None,
    metal_disabled_values={"reference", "python", "off", "0", "false", "no"},
    metal_forced_values={"metal", "gpu"},
    available_requires_device=True,
    reference_requested_detail=True,
    fallback_import_detail=False,
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def native_enabled() -> bool:
    return _SELECTOR.metal_ready()


def _metal_device_available() -> bool:
    return _SELECTOR.metal_device_available()


def _metal_strict() -> bool:
    return _SELECTOR.metal_strict()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


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
