"""Python-facing Color Separation backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os

import numpy as np

from . import color_separation_reference


@dataclass(frozen=True)
class BackendStatus:
    effect: str
    backend: str
    native: bool
    detail: str = ""


try:
    _cpu_backend = importlib.import_module(f"{__package__}._color_separation_cpu")
    _CPU_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local build state.
    _cpu_backend = None
    _CPU_IMPORT_ERROR = exc


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return os.getenv("PLATYPUS_COLOR_SEPARATION_BACKEND", "").strip().lower()


def native_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "off", "0", "false", "no"}:
        return False
    return _cpu_backend is not None


def _native_strict() -> bool:
    value = os.getenv("PLATYPUS_COLOR_SEPARATION_STRICT", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("color_separation", "effect_backends._color_separation_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "color_separation",
            "effect_backends.color_separation_reference",
            False,
            "cpu backend available; PLATYPUS_COLOR_SEPARATION_BACKEND requested reference",
        )
    detail = "" if _CPU_IMPORT_ERROR is None else str(_CPU_IMPORT_ERROR)
    return BackendStatus("color_separation", "effect_backends.color_separation_reference", False, detail)


def apply_color_separation(
    img_float32,
    shadow_chroma_clean=0.0,
    shadow_threshold=0.2,
    color_separation=0.0,
    chroma_clarity=0.0,
    color_density=0.0,
    subtractive_saturation=0.0,
    opponent_contrast=0.0,
):
    shadow_chroma_clean = float(shadow_chroma_clean)
    color_separation = float(color_separation)
    chroma_clarity = float(chroma_clarity)
    color_density = float(color_density)
    subtractive_saturation = float(subtractive_saturation)
    opponent_contrast = float(opponent_contrast)
    if (
        shadow_chroma_clean == 0.0
        and color_separation == 0.0
        and chroma_clarity == 0.0
        and color_density == 0.0
        and subtractive_saturation == 0.0
        and opponent_contrast == 0.0
    ):
        return img_float32
    image32 = np.asarray(img_float32, dtype=np.float32)
    if (
        native_enabled()
        and image32.ndim == 3
        and image32.shape[-1] == 3
    ):
        try:
            return _cpu_backend.apply_color_separation(
                np.ascontiguousarray(image32),
                float(shadow_chroma_clean),
                float(shadow_threshold),
                float(color_separation),
                float(chroma_clarity),
                float(color_density),
                float(subtractive_saturation),
                float(opponent_contrast),
            )
        except Exception:
            if _native_strict():
                raise

    return color_separation_reference.apply_color_separation(
        image32,
        shadow_chroma_clean,
        shadow_threshold,
        color_separation,
        chroma_clarity,
        color_density,
        subtractive_saturation,
        opponent_contrast,
    )


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "apply_color_separation",
]
