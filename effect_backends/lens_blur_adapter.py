"""Python-facing depth-of-field lens blur backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import (
    BackendStatus,
    backend_preference,
    import_error_detail,
    optional_backend,
    strict_enabled,
)
from . import lens_blur_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_lens_blur_metal")


def native_available() -> bool:
    return _metal_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_LENS_BLUR_BACKEND")


def _metal_backend_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "opencv", "cpu", "off", "0", "false", "no"}:
        return False
    return value in {"", "auto", "metal"}


def _metal_strict() -> bool:
    return strict_enabled("PLATYPUS_LENS_BLUR_METAL_STRICT")


def _metal_device_available() -> bool:
    if _metal_backend is None:
        return False
    try:
        return bool(_metal_backend.metal_available())
    except Exception:
        return False


def backend_status() -> BackendStatus:
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        return BackendStatus("lens_blur", "effect_backends._lens_blur_metal", True)
    if _backend_preference() == "metal":
        if _metal_backend is not None:
            detail = "Metal backend is built, but no Metal device is available"
        else:
            detail = import_error_detail(_METAL_IMPORT_ERROR)
        return BackendStatus("lens_blur", "effect_backends.lens_blur_reference", False, detail)
    return BackendStatus("lens_blur", "effect_backends.lens_blur_reference", False, import_error_detail(_METAL_IMPORT_ERROR))


def apply_lensblur(
    image: np.ndarray,
    depth_map: np.ndarray | None = None,
    focus_depth: float = 0.8,
    max_coc_radius: int = 25,
    num_levels: int = 25,
    chromatic_aberration: float = 0.04,
    spherical_aberration: float = 0.6,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if max_coc_radius <= 0:
        max_coc_radius = 1

    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        try:
            # CoC 半径マップ計算は軽量なので Python 側(reference と共有)で行い、
            # 重いレベルスタック生成と合成のみ Metal に渡す。
            coc, _ = lens_blur_reference.compute_coc_radius(
                image.shape, depth_map, focus_depth, max_coc_radius
            )
            return _metal_backend.apply_lensblur(
                np.ascontiguousarray(image, dtype=np.float32),
                np.ascontiguousarray(coc, dtype=np.float32),
                int(num_levels),
                float(max_coc_radius),
                float(chromatic_aberration),
                float(spherical_aberration),
            )
        except Exception:
            if _metal_strict():
                raise

    return lens_blur_reference.apply_lensblur(
        image,
        depth_map=depth_map,
        focus_depth=focus_depth,
        max_coc_radius=max_coc_radius,
        num_levels=num_levels,
        chromatic_aberration=chromatic_aberration,
        spherical_aberration=spherical_aberration,
    )


__all__ = [
    "apply_lensblur",
    "backend_status",
    "native_available",
]
