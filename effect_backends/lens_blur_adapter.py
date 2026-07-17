"""Python-facing depth-of-field lens blur backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import lens_blur_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_lens_blur_metal")

_SELECTOR = BackendSelector(
    "lens_blur",
    globals(),
    env="PLATYPUS_LENS_BLUR_BACKEND",
    metal_strict_env="PLATYPUS_LENS_BLUR_METAL_STRICT",
    metal_name="effect_backends._lens_blur_metal",
    reference_name="effect_backends.lens_blur_reference",
    metal_disabled_values={"reference", "python", "opencv", "cpu", "off", "0", "false", "no"},
    metal_forced_values={"metal"},
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def _metal_backend_enabled() -> bool:
    return _SELECTOR.metal_enabled()


def _metal_strict() -> bool:
    return _SELECTOR.metal_strict()


def _metal_device_available() -> bool:
    return _SELECTOR.metal_device_available()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


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
