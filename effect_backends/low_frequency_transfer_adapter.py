"""Python-facing low frequency transfer backend adapter."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .backend_utils import (
    BackendStatus,
    backend_preference,
    import_error_detail,
    native_backend_enabled,
    optional_backend,
    strict_enabled,
)
from . import low_frequency_transfer_reference


_metal_backend, _METAL_IMPORT_ERROR = optional_backend(__package__, "_low_frequency_transfer_metal")
_cpu_backend, _CPU_IMPORT_ERROR = optional_backend(__package__, "_low_frequency_transfer_cpu")


def native_available() -> bool:
    return (_metal_backend is not None and _metal_device_available()) or _cpu_backend is not None


def _backend_preference() -> str:
    return backend_preference("PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND")


def native_enabled() -> bool:
    return native_backend_enabled(
        _cpu_backend,
        _backend_preference(),
        disabled_values={"reference", "python", "opencv", "off", "0", "false", "no"},
    )


def _metal_backend_enabled() -> bool:
    value = _backend_preference()
    if value in {"reference", "python", "opencv", "cpu", "cpu_exact", "off", "0", "false", "no"}:
        return False
    return value in {"", "auto", "metal", "exact", "gpu"}


def _metal_strict() -> bool:
    return strict_enabled("PLATYPUS_LOW_FREQUENCY_TRANSFER_METAL_STRICT")


def _metal_device_available() -> bool:
    if _metal_backend is None:
        return False
    try:
        return bool(_metal_backend.metal_available())
    except Exception:
        return False


def _native_strict() -> bool:
    return strict_enabled("PLATYPUS_LOW_FREQUENCY_TRANSFER_STRICT")


def _luminance_transfer_strength(value) -> float:
    if value is None:
        value = os.getenv("PLATYPUS_LOW_FREQUENCY_TRANSFER_LUMA_STRENGTH", "1.0")
    try:
        strength = float(value)
    except (TypeError, ValueError):
        strength = 0.0
    return float(np.clip(strength, 0.0, 1.0))


def backend_status() -> BackendStatus:
    if _metal_backend is not None and _metal_backend_enabled() and _metal_device_available():
        return BackendStatus("low_frequency_transfer", "effect_backends._low_frequency_transfer_metal", True)
    if _backend_preference() in {"metal", "gpu"}:
        if _metal_backend is not None:
            detail = "Metal backend is built, but no Metal device is available"
        else:
            detail = import_error_detail(_METAL_IMPORT_ERROR)
        return BackendStatus("low_frequency_transfer", "effect_backends.low_frequency_transfer_reference", False, detail)
    if _cpu_backend is not None and native_enabled():
        return BackendStatus("low_frequency_transfer", "effect_backends._low_frequency_transfer_cpu", True)
    if _cpu_backend is not None:
        return BackendStatus(
            "low_frequency_transfer",
            "effect_backends.low_frequency_transfer_reference",
            False,
            "cpu backend available; PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND requested reference",
        )
    detail = import_error_detail(_CPU_IMPORT_ERROR)
    return BackendStatus("low_frequency_transfer", "effect_backends.low_frequency_transfer_reference", False, detail)


def _normalize_inputs(restored_img, reference_img):
    restored = np.asarray(restored_img, dtype=np.float32)
    reference = np.asarray(reference_img, dtype=np.float32)
    h, w = restored.shape[:2]
    if reference.shape[:2] != (h, w):
        reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored, reference


def _downsample_factor(height: int, width: int, sigma: float, downsample=None) -> int:
    if downsample is None:
        value = os.getenv("PLATYPUS_LOW_FREQUENCY_TRANSFER_DOWNSAMPLE", "off")
    else:
        value = downsample
    value = str(value).strip().lower()
    if value in {"", "off", "0", "false", "no", "exact", "full"}:
        return 1
    if value not in {"auto", "on", "true", "yes"}:
        try:
            return max(1, int(value))
        except ValueError:
            return 1

    if sigma < 16.0 or min(height, width) < 256:
        return 1
    factor = max(2, min(8, int(round(float(sigma) / 10.0))))
    while factor > 1 and (height // factor < 64 or width // factor < 64):
        factor //= 2
    return max(1, factor)


def _exact_backend_requested(preference: str) -> bool:
    return preference in {"exact", "cpu_exact", "full"}


def _apply_lowres_native(
    restored: np.ndarray,
    reference: np.ndarray,
    sigma: float,
    highlight_threshold,
    highlight_transition: float,
    highlight_detail_strength: float,
    luminance_transfer_strength: float,
    factor: int,
):
    h, w = restored.shape[:2]
    small_size = (max(1, w // factor), max(1, h // factor))
    restored_small = cv2.resize(restored, small_size, interpolation=cv2.INTER_AREA)
    reference_small = cv2.resize(reference, small_size, interpolation=cv2.INTER_AREA)
    sigma_small = max(float(sigma) / float(factor), 0.01)

    low_diff = np.asarray(reference_small - restored_small, dtype=np.float32)
    low_diff = cv2.GaussianBlur(low_diff, (0, 0), sigmaX=sigma_small, sigmaY=sigma_small)

    if highlight_threshold is not None:
        low_restored = cv2.GaussianBlur(restored_small, (0, 0), sigmaX=sigma_small, sigmaY=sigma_small)
    else:
        low_restored = low_diff

    return _cpu_backend.compose_lowres(
        np.ascontiguousarray(restored),
        np.ascontiguousarray(reference),
        np.ascontiguousarray(low_diff, dtype=np.float32),
        np.ascontiguousarray(low_restored, dtype=np.float32),
        highlight_threshold is not None,
        0.0 if highlight_threshold is None else float(highlight_threshold),
        float(highlight_transition),
        float(highlight_detail_strength),
        float(luminance_transfer_strength),
    )


def apply_low_frequency_transfer(
    restored_img,
    reference_img,
    sigma=30,
    highlight_threshold=None,
    highlight_transition=0.35,
    highlight_detail_strength=0.25,
    luminance_transfer_strength=None,
    downsample=None,
):
    restored, reference = _normalize_inputs(restored_img, reference_img)
    luma_strength = _luminance_transfer_strength(luminance_transfer_strength)
    preference = _backend_preference()
    factor = 1 if _exact_backend_requested(preference) else _downsample_factor(
        restored.shape[0],
        restored.shape[1],
        float(sigma),
        downsample,
    )

    if (
        factor > 1
        and _cpu_backend is not None
        and native_enabled()
        and restored.ndim in {2, 3}
        and restored.shape == reference.shape
    ):
        try:
            return _apply_lowres_native(
                restored,
                reference,
                float(sigma),
                highlight_threshold,
                float(highlight_transition),
                float(highlight_detail_strength),
                luma_strength,
                factor,
            )
        except Exception:
            if _native_strict():
                raise

    if (
        _metal_backend is not None
        and _metal_backend_enabled()
        and _metal_device_available()
        and restored.ndim in {2, 3}
        and restored.shape == reference.shape
    ):
        try:
            return _metal_backend.apply_low_frequency_transfer(
                np.ascontiguousarray(restored),
                np.ascontiguousarray(reference),
                float(sigma),
                highlight_threshold is not None,
                0.0 if highlight_threshold is None else float(highlight_threshold),
                float(highlight_transition),
                float(highlight_detail_strength),
                luma_strength,
            )
        except Exception:
            if _metal_strict():
                raise

    if native_enabled() and restored.ndim in {2, 3} and restored.shape == reference.shape:
        try:
            return _cpu_backend.apply_low_frequency_transfer(
                np.ascontiguousarray(restored),
                np.ascontiguousarray(reference),
                float(sigma),
                highlight_threshold is not None,
                0.0 if highlight_threshold is None else float(highlight_threshold),
                float(highlight_transition),
                float(highlight_detail_strength),
                luma_strength,
            )
        except Exception:
            if _native_strict():
                raise

    return low_frequency_transfer_reference.apply_low_frequency_transfer(
        restored,
        reference,
        sigma,
        highlight_threshold,
        highlight_transition,
        highlight_detail_strength,
        luma_strength,
    )


__all__ = [
    "BackendStatus",
    "backend_status",
    "native_available",
    "native_enabled",
    "apply_low_frequency_transfer",
]
