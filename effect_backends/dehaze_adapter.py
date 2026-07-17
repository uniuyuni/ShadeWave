"""Python-facing Dehaze backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import dehaze_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_dehaze_metal")

_SELECTOR = BackendSelector(
    "dehaze",
    globals(),
    env="PLATYPUS_DEHAZE_BACKEND",
    metal_strict_env="PLATYPUS_DEHAZE_METAL_STRICT",
    metal_name="effect_backends._dehaze_metal",
    reference_name="effect_backends.dehaze_reference",
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
