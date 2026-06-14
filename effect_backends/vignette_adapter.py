"""Python-facing Vignette backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np

from . import vignette_reference


@dataclass(frozen=True)
class BackendStatus:
    effect: str
    backend: str
    native: bool
    detail: str = ""


try:
    _cpu_backend = importlib.import_module(f"{__package__}._vignette_cpu")
    _CPU_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local build state.
    _cpu_backend = None
    _CPU_IMPORT_ERROR = exc


def native_available() -> bool:
    return _cpu_backend is not None


def backend_status() -> BackendStatus:
    if _cpu_backend is not None:
        return BackendStatus("vignette", "effect_backends._vignette_cpu", True)
    detail = "" if _CPU_IMPORT_ERROR is None else str(_CPU_IMPORT_ERROR)
    return BackendStatus("vignette", "effect_backends.vignette_reference", False, detail)


def apply_vignette(
    image: np.ndarray,
    intensity: float,
    radius_percent: float,
    disp_info: Any,
    crop_rect: Any,
    offset: Any,
    gradient_softness: float = 4.0,
) -> np.ndarray:
    if _cpu_backend is not None:
        image32 = np.ascontiguousarray(image, dtype=np.float32)
        return _cpu_backend.apply_vignette(
            image32,
            float(intensity),
            float(radius_percent),
            disp_info,
            crop_rect,
            offset,
            float(gradient_softness),
        )

    return vignette_reference.apply_vignette(
        image,
        intensity,
        radius_percent,
        disp_info,
        crop_rect,
        offset,
        gradient_softness,
    )
