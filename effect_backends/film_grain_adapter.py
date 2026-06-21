"""Python-facing Film Grain backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os

import numpy as np

from . import film_grain_reference


@dataclass(frozen=True)
class BackendStatus:
    effect: str
    backend: str
    native: bool
    detail: str = ""


try:
    _cpu_backend = importlib.import_module(f"{__package__}._film_grain_cpu")
    _CPU_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local build state.
    _cpu_backend = None
    _CPU_IMPORT_ERROR = exc


def native_available() -> bool:
    return _cpu_backend is not None


def _backend_preference() -> str:
    return os.getenv("PLATYPUS_FILM_GRAIN_BACKEND", "").strip().lower()


def native_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "off", "0", "false", "no"}:
        return False
    return _cpu_backend is not None


def _native_strict() -> bool:
    value = os.getenv("PLATYPUS_FILM_GRAIN_STRICT", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
    detail = "" if _CPU_IMPORT_ERROR is None else str(_CPU_IMPORT_ERROR)
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
