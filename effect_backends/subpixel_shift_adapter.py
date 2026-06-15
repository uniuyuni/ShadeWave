"""Python-facing Subpixel Shift backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os

import numpy as np

from . import subpixel_shift_reference


@dataclass(frozen=True)
class BackendStatus:
    effect: str
    backend: str
    native: bool
    detail: str = ""


try:
    _cpu_backend = importlib.import_module(f"{__package__}._subpixel_shift_cpu")
    _CPU_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local build state.
    _cpu_backend = None
    _CPU_IMPORT_ERROR = exc


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return os.getenv("PLATYPUS_SUBPIXEL_SHIFT_BACKEND", "").strip().lower()


def native_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "off", "0", "false", "no"}:
        return False
    return _cpu_backend is not None


def _native_strict() -> bool:
    value = os.getenv("PLATYPUS_SUBPIXEL_SHIFT_STRICT", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
    detail = "" if _CPU_IMPORT_ERROR is None else str(_CPU_IMPORT_ERROR)
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
