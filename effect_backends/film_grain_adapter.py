"""Python-facing Film Grain backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import film_grain_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_film_grain_cpu")
_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_film_grain_metal")

_SELECTOR = BackendSelector(
    "film_grain",
    globals(),
    env="PLATYPUS_FILM_GRAIN_BACKEND",
    native_strict_env="PLATYPUS_FILM_GRAIN_STRICT",
    metal_name="effect_backends._film_grain_metal",
    cpu_name="effect_backends._film_grain_cpu",
    reference_name="effect_backends.film_grain_reference",
    metal_disabled_values={"reference", "python", "cpu", "off", "0", "false", "no"},
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _metal_backend_enabled() -> bool:
    return _SELECTOR.metal_enabled()


def _metal_device_available() -> bool:
    return _SELECTOR.metal_device_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def native_enabled() -> bool:
    return _SELECTOR.native_enabled()


def _native_strict() -> bool:
    return _SELECTOR.native_strict()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


def apply_film_grain(
    image,
    amount=0.0,
    grain_size=2.0,
    roughness=50.0,
    shadow=60.0,
    highlight=30.0,
    color=10.0,
    seed=0,
):
    amount = float(np.clip(amount, 0.0, 100.0))
    if amount <= 0.0:
        return image if getattr(image, "dtype", None) == np.float32 else np.asarray(image, dtype=np.float32)

    image32 = np.asarray(image, dtype=np.float32)
    if (
        image32.ndim == 3
        and image32.shape[-1] >= 3
        and _metal_backend is not None
        and _metal_backend_enabled()
        and _metal_device_available()
    ):
        try:
            return _metal_backend.apply_film_grain(
                np.ascontiguousarray(image32),
                amount,
                float(grain_size),
                float(roughness),
                float(shadow),
                float(highlight),
                float(color),
                int(seed),
            )
        except Exception:
            if _native_strict():
                raise

    if native_enabled() and image32.ndim == 3 and image32.shape[-1] >= 3:
        try:
            return _cpu_backend.apply_film_grain(
                np.ascontiguousarray(image32),
                amount,
                float(grain_size),
                float(roughness),
                float(shadow),
                float(highlight),
                float(color),
                int(seed),
            )
        except Exception:
            if _native_strict():
                raise

    return film_grain_reference.apply_film_grain(
        image32,
        amount,
        grain_size,
        roughness,
        shadow,
        highlight,
        color,
        seed,
    )


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "apply_film_grain",
]
