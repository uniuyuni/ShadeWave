"""Python-facing Film Process backend adapter."""

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
from . import film_process_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_film_process_cpu")


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_FILM_PROCESS_BACKEND")


def native_enabled() -> bool:
    return native_backend_enabled(_cpu_backend, _backend_preference())


def _native_strict() -> bool:
    return strict_enabled("PLATYPUS_FILM_PROCESS_STRICT")


def backend_status() -> BackendStatus:
    if native_enabled():
        return BackendStatus("film_process", "effect_backends._film_process_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "film_process",
            "effect_backends.film_process_reference",
            False,
            "cpu backend available; PLATYPUS_FILM_PROCESS_BACKEND requested reference",
        )
    detail = import_error_detail(_CPU_IMPORT_ERROR)
    return BackendStatus("film_process", "effect_backends.film_process_reference", False, detail)


def apply_film_process(
    image,
    mode="Off",
    latitude=55.0,
    contrast=50.0,
    color_bias=0.0,
    color_drift=0.0,
    dye_purity=75.0,
    layer_crosstalk=30.0,
    halation=0.0,
    aging=0.0,
):
    image32 = np.asarray(image, dtype=np.float32)

    mode_name = film_process_reference._mode_name(mode)
    if mode_name == "Off" or image32.ndim != 3 or image32.shape[-1] < 3:
        return image32

    # Native path: 3-channel only (extra channels fall back to the reference,
    # which preserves them). Halation is a spatial op, so it runs in Python and
    # the haloed RGB is shared with both backends to keep parity exact.
    if native_enabled() and image32.shape[-1] == 3:
        try:
            rgb = np.nan_to_num(image32, nan=0.0, posinf=4.0, neginf=0.0)
            rgb = np.maximum(rgb, 0.0)
            rgb = film_process_reference._apply_halation(rgb, halation)
            return _cpu_backend.apply_film_process(
                np.ascontiguousarray(rgb, dtype=np.float32),
                int(film_process_reference._mode_index(mode_name)),
                float(film_process_reference._clip01(latitude)),
                float(film_process_reference._clip01(contrast)),
                float(film_process_reference._signed01(color_bias)),
                float(film_process_reference._signed01(color_drift)),
                float(film_process_reference._clip01(dye_purity)),
                float(film_process_reference._clip01(layer_crosstalk)),
                float(film_process_reference._clip01(aging)),
            )
        except Exception:
            if _native_strict():
                raise

    return film_process_reference.apply_film_process(
        image32,
        mode_name,
        latitude,
        contrast,
        color_bias,
        color_drift,
        dye_purity,
        layer_crosstalk,
        halation,
        aging,
    )


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "apply_film_process",
]
