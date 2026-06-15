"""Python-facing Tone backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os

import numpy as np

from . import tone_reference


@dataclass(frozen=True)
class BackendStatus:
    effect: str
    backend: str
    native: bool
    detail: str = ""


try:
    _cpu_backend = importlib.import_module(f"{__package__}._tone_cpu")
    _CPU_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local build state.
    _cpu_backend = None
    _CPU_IMPORT_ERROR = exc


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return os.getenv("PLATYPUS_TONE_BACKEND", "").strip().lower()


def native_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "off", "0", "false", "no"}:
        return False
    return _cpu_backend is not None


def _native_strict() -> bool:
    value = os.getenv("PLATYPUS_TONE_STRICT", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
    detail = "" if _CPU_IMPORT_ERROR is None else str(_CPU_IMPORT_ERROR)
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
