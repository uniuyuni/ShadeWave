"""Python-facing Film Process backend adapter."""

from __future__ import annotations

import numpy as np

from .backend_utils import BackendSelector, BackendStatus, optional_backend
from . import film_process_reference


_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_film_process_cpu")

_SELECTOR = BackendSelector(
    "film_process",
    globals(),
    env="PLATYPUS_FILM_PROCESS_BACKEND",
    native_strict_env="PLATYPUS_FILM_PROCESS_STRICT",
    cpu_name="effect_backends._film_process_cpu",
    reference_name="effect_backends.film_process_reference",
)


def native_available() -> bool:
    return _SELECTOR.native_available()


def _backend_preference() -> str:
    return _SELECTOR.preference()


def native_enabled() -> bool:
    return _SELECTOR.native_enabled()


def _native_strict() -> bool:
    return _SELECTOR.native_strict()


def backend_status() -> BackendStatus:
    return _SELECTOR.status()


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
