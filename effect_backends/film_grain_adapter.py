"""Python-facing Film Grain backend adapter."""

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
from . import film_grain_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_film_grain_cpu")


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_FILM_GRAIN_BACKEND")


def native_enabled() -> bool:
    return native_backend_enabled(_cpu_backend, _backend_preference())


def _native_strict() -> bool:
    return strict_enabled("PLATYPUS_FILM_GRAIN_STRICT")


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("film_grain", "effect_backends._film_grain_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "film_grain",
            "effect_backends.film_grain_reference",
            False,
            "cpu backend available; PLATYPUS_FILM_GRAIN_BACKEND requested reference",
        )
    detail = import_error_detail(_CPU_IMPORT_ERROR)
    return BackendStatus("film_grain", "effect_backends.film_grain_reference", False, detail)


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
