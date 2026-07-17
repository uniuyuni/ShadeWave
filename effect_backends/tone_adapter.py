"""Python-facing Tone backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import tone_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_tone_cpu")

_SELECTOR = BackendSelector(
    "tone",
    globals(),
    env="PLATYPUS_TONE_BACKEND",
    native_strict_env="PLATYPUS_TONE_STRICT",
    cpu_name="effect_backends._tone_cpu",
    reference_name="effect_backends.tone_reference",
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
