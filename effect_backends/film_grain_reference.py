"""Reference Film Grain implementation."""

from __future__ import annotations

import cv2
import numpy as np


def _grain_noise_layer(height: int, width: int, grain_size: float, rng) -> np.ndarray:
    grain_size = max(0.35, float(grain_size))
    if grain_size <= 0.75:
        noise = rng.standard_normal((height, width)).astype(np.float32)
    else:
        small_h = max(2, int(np.ceil(height / grain_size)))
        small_w = max(2, int(np.ceil(width / grain_size)))
        noise = rng.standard_normal((small_h, small_w)).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)

    noise -= float(np.mean(noise))
    std = float(np.std(noise))
    if std > 1e-6:
        noise /= std
    return noise.astype(np.float32, copy=False)


def _grain_seed(seed: int, height: int, width: int) -> int:
    seed = int(seed) & 0xFFFFFFFF
    if seed == 0:
        seed = 0x6D2B79F5
    seed ^= (height * 73856093) & 0xFFFFFFFF
    seed ^= (width * 19349663) & 0xFFFFFFFF
    return seed & 0xFFFFFFFF


def apply_film_grain(
    image: np.ndarray,
    amount: float = 0.0,
    grain_size: float = 2.0,
    roughness: float = 50.0,
    shadow: float = 60.0,
    highlight: float = 30.0,
    color: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Film grain V2.

    The grain is deterministic, zero-mean, mostly monochrome, and added without
    hard clipping so late-stage highlight detail is not crushed by the effect.
    """
    amount = float(np.clip(amount, 0.0, 100.0))
    if amount <= 0.0:
        return image if getattr(image, "dtype", None) == np.float32 else np.asarray(image, dtype=np.float32)

    src = np.asarray(image, dtype=np.float32)
    if src.ndim != 3 or src.shape[2] < 3:
        return src

    height, width = src.shape[:2]
    out = src.copy()
    rgb = out[..., :3]

    rough = float(np.clip(roughness, 0.0, 100.0)) / 100.0
    shadow_gain = 0.35 + float(np.clip(shadow, 0.0, 100.0)) / 100.0 * 1.35
    highlight_gain = 0.15 + float(np.clip(highlight, 0.0, 100.0)) / 100.0 * 1.10
    color_gain = float(np.clip(color, 0.0, 100.0)) / 100.0
    base_size = max(0.35, float(grain_size))

    rng = np.random.default_rng(_grain_seed(seed, height, width))
    fine = _grain_noise_layer(height, width, base_size * 0.55, rng)
    mid = _grain_noise_layer(height, width, base_size, rng)
    coarse = _grain_noise_layer(height, width, base_size * 2.35, rng)
    mono = (0.25 + 0.55 * rough) * fine + 0.70 * mid + (0.55 * (1.0 - rough)) * coarse
    mono -= float(np.mean(mono))
    mono_std = float(np.std(mono))
    if mono_std > 1e-6:
        mono /= mono_std

    safe_rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    luma = np.clip(
        safe_rgb[..., 0] * 0.2126 + safe_rgb[..., 1] * 0.7152 + safe_rgb[..., 2] * 0.0722,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    shadow_w = np.power(1.0 - luma, 1.55)
    highlight_w = np.power(luma, 1.75)
    mid_w = 1.0 - np.power(np.abs(luma * 2.0 - 1.0), 1.65)
    response = 0.50 * mid_w + 0.42 * shadow_gain * shadow_w + 0.32 * highlight_gain * highlight_w
    headroom = np.minimum(luma, 1.0 - luma)
    protect = 0.45 + 0.55 * np.clip(headroom * 5.0, 0.0, 1.0)
    amplitude = (amount / 100.0) * 0.045 * response * protect

    rgb += mono[..., np.newaxis] * amplitude[..., np.newaxis]

    if color_gain > 0.0:
        chroma_u = _grain_noise_layer(height, width, base_size * 1.35, rng)
        chroma_v = _grain_noise_layer(height, width, base_size * 1.75, rng)
        chroma = np.empty((height, width, 3), dtype=np.float32)
        chroma[..., 0] = chroma_u * 0.82 + chroma_v * 0.28
        chroma[..., 1] = chroma_u * -0.45 + chroma_v * 0.42
        chroma[..., 2] = chroma_u * -0.37 + chroma_v * -0.70
        rgb += chroma * (amplitude * color_gain * 0.42)[..., np.newaxis]

    return out


__all__ = ["apply_film_grain"]
