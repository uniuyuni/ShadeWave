"""Python-facing Vignette backend adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend_utils import BackendStatus, import_error_detail, optional_backend
from . import vignette_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_vignette_cpu")


def native_available() -> bool:
    return _cpu_backend is not None


def backend_status() -> BackendStatus:
    if _cpu_backend is not None:
        return BackendStatus("vignette", "effect_backends._vignette_cpu", True)
    detail = import_error_detail(_CPU_IMPORT_ERROR)
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


def create_vignette_mask(
    height: int,
    width: int,
    radius_percent: float,
    disp_info: Any,
    crop_rect: Any,
    offset: Any,
    gradient_softness: float = 4.0,
) -> np.ndarray:
    if _cpu_backend is not None and hasattr(_cpu_backend, "create_vignette_mask"):
        return _cpu_backend.create_vignette_mask(
            int(height),
            int(width),
            float(radius_percent),
            disp_info,
            crop_rect,
            offset,
            float(gradient_softness),
        )

    dx, dy, _, _, scale = disp_info
    x1, y1, x2, y2 = crop_rect
    offset_x, offset_y = offset
    center_x = (x1 + (x2 - x1) / 2.0 - dx) * scale + offset_x
    center_y = (y1 + (y2 - y1) / 2.0 - dy) * scale + offset_y
    mm = max(x2 - x1, y2 - y1) * scale
    max_radius = np.float32(np.sqrt(np.float32(mm * mm + mm * mm))) / np.float32(2.0)
    radius = max_radius * (np.float32(radius_percent) / np.float32(100.0))

    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x.astype(np.float32) - np.float32(center_x)) ** 2 + (y.astype(np.float32) - np.float32(center_y)) ** 2)
    if float(radius) == 0.0:
        val = np.ones((height, width), dtype=np.float32)
    else:
        val = np.clip(dist / radius, 0.0, 1.0).astype(np.float32, copy=False)
    mask = np.power(val, np.float32(max(0.1, float(gradient_softness)))).astype(np.float32, copy=False)
    return (mask * mask * (np.float32(3.0) - np.float32(2.0) * mask)).astype(np.float32, copy=False)


def apply_vignette_mask(image: np.ndarray, mask: np.ndarray, intensity: float) -> np.ndarray:
    if _cpu_backend is not None and hasattr(_cpu_backend, "apply_vignette_mask"):
        image32 = np.ascontiguousarray(image, dtype=np.float32)
        mask32 = np.ascontiguousarray(mask, dtype=np.float32)
        return _cpu_backend.apply_vignette_mask(image32, mask32, float(intensity))

    image32 = np.asarray(image, dtype=np.float32)
    mask32 = np.asarray(mask, dtype=np.float32)
    amount = np.float32(float(intensity) / 100.0)
    mask3 = mask32[..., np.newaxis] if image32.ndim == 3 else mask32
    if amount < 0.0:
        return (image32 * (np.float32(1.0) + amount * mask3)).astype(np.float32, copy=False)
    return (image32 + (np.float32(1.0) - image32) * (amount * mask3)).astype(np.float32, copy=False)
