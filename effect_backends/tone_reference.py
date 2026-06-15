"""Reference implementation for the Tone effect.

This intentionally mirrors the historical cores.core.adjust_tone path and is
used as the numerical fallback / parity target for native backends.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numba import njit, prange

from threads import lock_numba


def gaussian_blur_cv(src, ksize=(3, 3), sigma=0.0):
    if ksize == (0, 0) and sigma == 0.0:
        return src
    return cv2.GaussianBlur(src, ksize, sigma)


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def get_luminance(img):
    h, w, _c = img.shape
    y = np.empty((h, w), dtype=np.float32)
    for i in prange(h):
        for j in range(w):
            y[i, j] = 0.2126 * img[i, j, 0] + 0.7152 * img[i, j, 1] + 0.0722 * img[i, j, 2]
    return y


@njit(fastmath=True, inline="always")
def _apply_midtones(val, midtone):
    if midtone == 0:
        return val
    if midtone > 0:
        c = midtone / 100.0 * 16.0
        return math.log(1.0 + val * c) / math.log(1.0 + c)
    c = -midtone / 100.0 * 16.0
    if abs(c) < 1e-6:
        return val
    log1pc = math.log(1.0 + c)
    normal_result = (math.exp(val * log1pc) - 1.0) / c
    derivative_at_1 = (1.0 + c) * log1pc / c
    if val <= 1.0:
        return normal_result
    return 1.0 + derivative_at_1 * (val - 1.0)


@njit(fastmath=True, inline="always")
def _apply_shadows(val, shadows):
    if shadows == 0:
        return val
    if shadows > 0:
        factor = shadows / 100.0 * 6.0
        influence = math.exp(-5.0 * val)
        return val * (1.0 + factor * influence)
    factor = shadows / 100.0
    influence = math.exp(-5.0 * val)
    raw_result = val * (1.0 + factor * influence)
    return max(raw_result, val * 0.1)


@njit(fastmath=True, inline="always")
def _apply_black(val, black_level):
    if black_level == 0:
        return val
    if black_level > 0:
        gamma = math.exp(-(black_level / 100.0) * 0.7)
    else:
        gamma = math.exp((-black_level / 100.0) * 0.7)
    return max(val, 0.0) ** gamma


@njit(fastmath=True, inline="always")
def _apply_highlight_pos(val, highlights):
    return val * (1.0 + highlights / 100.0 * 2.0)


@njit(fastmath=True, inline="always")
def _apply_highlight_neg(val, base, highlights):
    factor = -highlights / 100.0
    detail = val - base
    compressed_base = base / (1.0 + factor * max(base, 0.0))
    t = (base - 0.95) / 0.4
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    smooth_mask = t * t * (3.0 - 2.0 * t)
    adaptive_factor = 1.0 / (1.0 + 10.0 * abs(detail))
    effective_boost = 1.17 * adaptive_factor
    desired_boost = 1.0 + smooth_mask * factor * (effective_boost - 1.0)
    return compressed_base + detail * desired_boost


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_mid_shadow(y, midtone, shadows):
    h, w = y.shape
    res = np.empty_like(y)
    for i in prange(h):
        for j in range(w):
            val = _apply_midtones(y[i, j], midtone)
            res[i, j] = _apply_shadows(val, shadows)
    return res


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_high_pos_black(y, highlights, black_level):
    h, w = y.shape
    res = np.empty_like(y)
    for i in prange(h):
        for j in range(w):
            val = _apply_highlight_pos(y[i, j], highlights)
            res[i, j] = _apply_black(val, black_level)
    return res


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_high_neg_black(y, y_blur, highlights, black_level):
    h, w = y.shape
    res = np.empty_like(y)
    for i in prange(h):
        for j in range(w):
            val = _apply_highlight_neg(y[i, j], y_blur[i, j], highlights)
            res[i, j] = _apply_black(val, black_level)
    return res


@njit(fastmath=True, inline="always")
def _apply_white_pos(val, white_level, max_val):
    factor = white_level / 100.0 * 6.0
    base = val if max_val <= 1e-6 else val / max_val
    numer = math.log(1.0 + math.log(1.0 + base))
    denom = math.log(1.0 + math.log(1.0 + max(max_val, 2.0)))
    denominator = 1.0 if denom == 0 else 1.0 / denom
    return val * (1.0 + factor * (numer * denominator))


@njit(fastmath=True, inline="always")
def _apply_white_neg(val, base, white_level, max_val):
    factor = -white_level / 100.0
    detail = val - base
    safe_base = max(base, 0.0)
    denom = math.log(1.0 + math.log(1.0 + max(max_val, 2.0)))
    denominator = 1.0 if denom == 0 else 1.0 / denom
    target = math.log(1.0 + math.log(1.0 + safe_base)) * denominator
    compressed_base = min(safe_base, base * (1.0 - factor) + target * factor)
    t = (base - 0.95) / 0.4
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    smooth_mask = t * t * (3.0 - 2.0 * t)
    adaptive_factor = 1.0 / (1.0 + 10.0 * abs(detail))
    desired_boost = 1.0 + smooth_mask * factor * (1.17 * adaptive_factor - 1.0)
    if detail < 0:
        safe_boost = min(desired_boost, compressed_base / max(-detail, 1e-8))
    else:
        safe_boost = desired_boost
    return max(compressed_base + detail * safe_boost, 0.0)


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_white_pos_final(img, y_current, y_orig, white_level, max_val):
    h, w, c = img.shape
    res = np.empty_like(img)
    eps = 1e-6
    for i in prange(h):
        for j in range(w):
            val = _apply_white_pos(y_current[i, j], white_level, max_val)
            orig = y_orig[i, j]
            safe_orig = orig if orig >= eps else eps
            gain = val / safe_orig
            if orig < eps:
                gain = 1.0
            for k in range(c):
                res[i, j, k] = img[i, j, k] * gain
    return res


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_white_neg_final(img, y_current, y_blur, y_orig, white_level, max_val_blur):
    h, w, c = img.shape
    res = np.empty_like(img)
    eps = 1e-6
    for i in prange(h):
        for j in range(w):
            val = _apply_white_neg(y_current[i, j], y_blur[i, j], white_level, max_val_blur)
            orig = y_orig[i, j]
            safe_orig = orig if orig >= eps else eps
            gain = val / safe_orig
            if orig < eps:
                gain = 1.0
            for k in range(c):
                res[i, j, k] = img[i, j, k] * gain
    return res


def adjust_tone(
    img,
    highlights=0,
    shadows=0,
    midtone=0,
    white_level=0,
    black_level=0,
    disp_scale=1.0,
    resolution_scale=1.0,
):
    y_orig = get_luminance(img)
    current_y = _kernel_mid_shadow(y_orig, midtone, shadows)
    if highlights < 0:
        y_blur = gaussian_blur_cv(current_y, sigma=0.5 * resolution_scale)
        current_y = _kernel_high_neg_black(current_y, y_blur, highlights, black_level)
    else:
        current_y = _kernel_high_pos_black(current_y, highlights, black_level)

    if white_level < 0:
        y_blur = gaussian_blur_cv(current_y, sigma=0.5 * resolution_scale)
        return _kernel_white_neg_final(img, current_y, y_blur, y_orig, white_level, float(np.max(y_blur)))
    return _kernel_white_pos_final(img, current_y, y_orig, white_level, float(np.max(current_y)))
