import math

import numpy as np


_DEFAULT_ADJUSTMENT = {
    "switch_exposure_contrast": True,
    "exposure": 0.0,
    "contrast": 0,
    "switch_tone": True,
    "shadow": 0,
    "highlight": 0,
    "midtone": 0,
    "white": 0,
    "black": 0,
}


def _clamp(value, low, high):
    return min(high, max(low, value))


def _round_to_step(value, step):
    return round(value / step) * step


def _crop_image(image, crop_rect):
    if crop_rect is None:
        return image
    try:
        x1, y1, x2, y2 = [int(v) for v in crop_rect]
    except Exception:
        return image

    h, w = image.shape[:2]
    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return image
    return image[y1:y2, x1:x2]


def _sample_luminance(image, crop_rect=None, max_side=768):
    image = np.asarray(image)
    if image.ndim < 3 or image.shape[2] < 3:
        return None
    image = _crop_image(image, crop_rect)
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return None

    step = max(1, int(math.ceil(max(h, w) / float(max_side))))
    rgb = np.asarray(image[::step, ::step, :3], dtype=np.float32)
    finite = np.isfinite(rgb).all(axis=2)
    if not np.any(finite):
        return None

    rgb = rgb[finite]
    y = (
        rgb[:, 0] * np.float32(0.2126)
        + rgb[:, 1] * np.float32(0.7152)
        + rgb[:, 2] * np.float32(0.0722)
    )
    y = y[np.isfinite(y)]
    if y.size < 64:
        return None
    return np.maximum(y, 0.0).astype(np.float32, copy=False)


def compute_basic_auto_adjustment(image, crop_rect=None):
    """Estimate conservative basic correction parameters from a linear RGB image."""
    y = _sample_luminance(image, crop_rect=crop_rect)
    if y is None:
        return dict(_DEFAULT_ADJUSTMENT)

    p = np.percentile(y, [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 98, 99.5])
    p005, p01, _p05, p10, p25, p50, _p75, p90, _p95, p98, p995 = [float(v) for v in p]
    if p995 <= 1.0e-6:
        return dict(_DEFAULT_ADJUSTMENT)

    ev = math.log2(0.22 / max(p50, 1.0e-6))
    if p98 > 1.0e-6:
        highlight_guard = math.log2(1.18 / p98)
        ev = min(ev, highlight_guard + 0.25)
    ev = _round_to_step(_clamp(ev, -1.5, 1.5), 0.05)

    gain = 2.0 ** ev
    y_ev = y * np.float32(gain)
    p2 = np.percentile(y_ev, [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 98, 99.5])
    p005, p01, _p05, p10, p25, p50, _p75, p90, _p95, p98, p995 = [float(v) for v in p2]

    spread = p90 - p10
    if spread < 0.42:
        contrast = int(round(_clamp((0.42 - spread) * 65.0, 0, 24)))
    elif spread > 0.86:
        contrast = int(round(_clamp(-(spread - 0.86) * 24.0, -12, 0)))
    else:
        contrast = 0

    shadow_need = max(0.0, 0.11 - p10) * 135.0 + max(0.0, 0.20 - p25) * 35.0
    if p50 > 0.55:
        shadow_need *= 0.45
    shadow = int(round(_clamp(shadow_need, 0, 28)))

    highlight_need = max(0.0, p98 - 0.98) * 44.0 + max(0.0, p995 - 1.18) * 24.0
    highlight = -int(round(_clamp(highlight_need, 0, 34)))

    if p995 < 0.82 and p90 < 0.58:
        white = int(round(_clamp((0.82 - p995) * 24.0, 0, 14)))
    elif p995 > 1.25:
        white = -int(round(_clamp((p995 - 1.25) * 18.0, 0, 20)))
    else:
        white = 0

    black = -int(round(_clamp(max(0.0, p01 - 0.025) * 190.0, 0, 18)))
    if p005 < 0.002:
        black = min(0, black + 4)

    adjustment = dict(_DEFAULT_ADJUSTMENT)
    adjustment.update({
        "exposure": float(ev),
        "contrast": int(contrast),
        "shadow": int(shadow),
        "highlight": int(highlight),
        "white": int(white),
        "black": int(black),
    })
    return adjustment
