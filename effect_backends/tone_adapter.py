"""Python-facing Tone backend adapter."""

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
from . import tone_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_tone_cpu")


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_TONE_BACKEND")


def native_enabled() -> bool:
    return native_backend_enabled(_cpu_backend, _backend_preference())


def _native_strict() -> bool:
    return strict_enabled("PLATYPUS_TONE_STRICT")


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("tone", "effect_backends._tone_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "tone",
            "effect_backends.tone_reference",
            False,
            "cpu backend available; PLATYPUS_TONE_BACKEND requested reference",
        )
    detail = import_error_detail(_CPU_IMPORT_ERROR)
    return BackendStatus("tone", "effect_backends.tone_reference", False, detail)


def adjust_tone(
    img,
    highlights=0,
    shadows=0,
    midtone=0,
    white_level=0,
    black_level=0,
    disp_scale=1.0,
    resolution_scale=1.0,
):
    image32 = np.asarray(img, dtype=np.float32)
    if native_enabled() and image32.ndim == 3 and image32.shape[-1] == 3:
        try:
            return _cpu_backend.adjust_tone(
                np.ascontiguousarray(image32),
                float(highlights),
                float(shadows),
                float(midtone),
                float(white_level),
                float(black_level),
                float(disp_scale),
                float(resolution_scale),
            )
        except Exception:
            if _native_strict():
                raise

    return tone_reference.adjust_tone(
        image32,
        highlights,
        shadows,
        midtone,
        white_level,
        black_level,
        disp_scale,
        resolution_scale,
    )


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "adjust_tone",
]
