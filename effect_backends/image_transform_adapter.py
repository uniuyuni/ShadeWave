"""Python-facing image transform backend adapter."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import image_transform_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_image_transform_metal")

_METAL_DEVICE_AVAILABLE_CACHE: bool | None = None


def _metal_device_available() -> bool:
    # Cached because this adapter sits on the interactive preview hot path.
    global _METAL_DEVICE_AVAILABLE_CACHE
    if _METAL_DEVICE_AVAILABLE_CACHE is not None:
        return _METAL_DEVICE_AVAILABLE_CACHE
    if _metal_backend is None:
        _METAL_DEVICE_AVAILABLE_CACHE = False
        return False
    try:
        _METAL_DEVICE_AVAILABLE_CACHE = bool(_metal_backend.metal_available())
    except Exception:
        _METAL_DEVICE_AVAILABLE_CACHE = False
    return _METAL_DEVICE_AVAILABLE_CACHE


def _clear_metal_device_available_cache() -> None:
    global _METAL_DEVICE_AVAILABLE_CACHE
    _METAL_DEVICE_AVAILABLE_CACHE = None


_SELECTOR = BackendSelector(
    "image_transform",
    globals(),
    env="PLATYPUS_IMAGE_TRANSFORM_BACKEND",
    metal_strict_env="PLATYPUS_IMAGE_TRANSFORM_METAL_STRICT",
    metal_name="effect_backends._image_transform_metal",
    reference_name="effect_backends.image_transform_reference",
    metal_enabled_values={"", "auto", "metal", "gpu"},
    metal_disabled_values={"reference", "python", "opencv", "off", "0", "false", "no"},
    metal_forced_values={"metal", "gpu"},
    available_requires_device=True,
    device_available=_metal_device_available,
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def _metal_backend_enabled() -> bool:
    return _SELECTOR.metal_enabled()


def _metal_strict() -> bool:
    return _SELECTOR.metal_strict()


def _metal_forced() -> bool:
    return _backend_preference() in {"metal", "gpu"}


def _area_mode() -> str:
    value = os.getenv("PLATYPUS_IMAGE_TRANSFORM_AREA_MODE", "exact").strip().lower()
    if value in {"reference", "opencv", "cpu"}:
        return "reference"
    if value in {"exact", "quality", "area"}:
        return "exact"
    return "linear"


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


def fit_crop_to_canvas(
    image: np.ndarray,
    source_rect: Sequence[int | float],
    canvas_width: int,
    canvas_height: int,
    draw_width: int,
    draw_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
    interpolation: str | int = "area",
) -> np.ndarray:
    use_metal = False
    metal_interpolation = interpolation
    if (
        _metal_backend is not None
        and _metal_backend_enabled()
        and _metal_device_available()
        and isinstance(interpolation, str)
        and interpolation in {"nearest", "linear", "area"}
    ):
        if interpolation == "area":
            area_mode = _area_mode()
            if area_mode == "reference":
                use_metal = False
            elif area_mode == "exact":
                use_metal = True
                metal_interpolation = "area"
            else:
                use_metal = True
                metal_interpolation = "linear"
        else:
            use_metal = True

    if use_metal:
        image32 = np.ascontiguousarray(image, dtype=np.float32)
        try:
            return _metal_backend.fit_crop_to_canvas(
                image32,
                source_rect,
                int(canvas_width),
                int(canvas_height),
                int(draw_width),
                int(draw_height),
                int(offset_x),
                int(offset_y),
                metal_interpolation,
            )
        except Exception:
            if _metal_strict():
                raise

    return image_transform_reference.fit_crop_to_canvas(
        image,
        source_rect,
        canvas_width,
        canvas_height,
        draw_width,
        draw_height,
        offset_x,
        offset_y,
        interpolation,
    )


def transform_to_canvas(*args, **kwargs):
    image = args[0] if args else kwargs.get("image")
    interpolation = kwargs.get("interpolation", args[5] if len(args) > 5 else "linear")
    border_mode = kwargs.get("border_mode", args[6] if len(args) > 6 else "reflect")

    if (
        _metal_backend is not None
        and _metal_backend_enabled()
        and _metal_device_available()
        and isinstance(image, np.ndarray)
        and image.dtype == np.float32
        and image.ndim == 3
        and image.shape[2] == 3
        and interpolation == "linear"
        and border_mode in {"reflect", "constant"}
    ):
        try:
            return _metal_backend.transform_to_canvas(*args, **kwargs)
        except Exception:
            if _metal_strict():
                raise

    return image_transform_reference.transform_to_canvas(*args, **kwargs)


def transform_crop_to_canvas(*args, **kwargs):
    image = args[0] if args else kwargs.get("image")
    interpolation = kwargs.get("interpolation", args[12] if len(args) > 12 else "linear")
    border_mode = kwargs.get("border_mode", args[13] if len(args) > 13 else "reflect")
    lens_scale = kwargs.get("lens_scale", args[15] if len(args) > 15 else 1.0)

    if (
        _metal_backend is not None
        and _metal_backend_enabled()
        and _metal_device_available()
        and isinstance(image, np.ndarray)
        and image.dtype == np.float32
        and image.ndim == 3
        and image.shape[2] == 3
        and interpolation in {"nearest", "linear", "area"}
        and border_mode in {"reflect", "constant"}
        and abs(float(lens_scale) - 1.0) <= 0.01
    ):
        try:
            metal_kwargs = dict(kwargs)
            return _metal_backend.transform_crop_to_canvas(*args, **metal_kwargs)
        except Exception:
            if _metal_strict():
                raise

    return image_transform_reference.transform_crop_to_canvas(*args, **kwargs)
