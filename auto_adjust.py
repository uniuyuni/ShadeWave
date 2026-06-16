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
    "switch_precence": True,
    "dehaze": 0,
    "clarity": 0,
    "texture": 0,
    "microcontrast": 0,
    "switch_saturation": True,
    "saturation": 0,
    "vibrance": 0,
    "switch_global": True,
    "shadow_chroma_clean": 0.0,
    "shadow_chroma_threshold": 0.2,
    "color_separation": 0.0,
    "chroma_clarity": 0.0,
    "color_density": 0.0,
    "subtractive_saturation": 0.0,
    "detail_tonemap": 0.0,
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
    sample = _sample_rgb(image, crop_rect=crop_rect, max_side=max_side)
    if sample is None:
        return None
    return sample["y"]


def _sample_rgb(image, crop_rect=None, max_side=768):
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
    finite_y = np.isfinite(y)
    y = y[finite_y]
    rgb = rgb[finite_y]
    if y.size < 64:
        return None
    y = np.maximum(y, 0.0).astype(np.float32, copy=False)
    rgb = np.maximum(rgb, 0.0).astype(np.float32, copy=False)
    return {"rgb": rgb, "y": y}


def _saturation_stats(rgb):
    if rgb is None or rgb.size == 0:
        return {
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "colorfulness": 0.0,
        }
    cmax = np.max(rgb, axis=1)
    cmin = np.min(rgb, axis=1)
    saturation = np.where(cmax > 1.0e-6, (cmax - cmin) / cmax, 0.0)
    saturation = saturation[np.isfinite(saturation)]
    if saturation.size == 0:
        return {
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "colorfulness": 0.0,
        }
    p50, p75, p90, p95 = [float(v) for v in np.percentile(saturation, [50, 75, 90, 95])]
    rg = rgb[:, 0] - rgb[:, 1]
    yb = (rgb[:, 0] + rgb[:, 1]) * 0.5 - rgb[:, 2]
    colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)))
    return {
        "median": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "colorfulness": colorfulness,
    }


def _round_int(value):
    return int(round(value))


def compute_basic_auto_adjustment(image, crop_rect=None):
    """Estimate balanced, presentation-ready correction parameters from a linear RGB image."""
    sample = _sample_rgb(image, crop_rect=crop_rect)
    if sample is None:
        return dict(_DEFAULT_ADJUSTMENT)
    rgb = sample["rgb"]
    y = sample["y"]

    p = np.percentile(y, [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 98, 99.5])
    p005, p01, p05, p10, p25, p50, p75, p90, p95, p98, p995 = [float(v) for v in p]
    if p995 <= 1.0e-6:
        return dict(_DEFAULT_ADJUSTMENT)
    orig_p50 = p50
    orig_p98 = p98

    highlight_load = _clamp((p98 - 0.92) / 0.55, 0.0, 1.0)
    shadow_load = _clamp((0.16 - p10) / 0.16, 0.0, 1.0)
    target_median = 0.245 - 0.025 * highlight_load + 0.015 * shadow_load
    ev = math.log2(target_median / max(p50, 1.0e-6))
    if p98 > 1.0e-6:
        highlight_guard = math.log2(1.10 / p98)
        ev = min(ev, highlight_guard + 0.18)
    if p01 < 0.006 and p50 < 0.18:
        ev += 0.12
    wide_tone_guard = _clamp((0.16 - orig_p50) / 0.16, 0.0, 1.0) * _clamp(
        (orig_p98 - 0.30) / 0.55, 0.0, 1.0
    )
    if wide_tone_guard > 0.0:
        ev = min(ev, max(0.45, 1.0 - 0.45 * wide_tone_guard))
    ev = _round_to_step(_clamp(ev, -1.65, 1.65), 0.05)

    gain = 2.0 ** ev
    y_ev = y * np.float32(gain)
    p2 = np.percentile(y_ev, [0.5, 1, 5, 10, 25, 50, 75, 90, 95, 98, 99.5])
    p005, p01, p05, p10, p25, p50, p75, p90, p95, p98, p995 = [float(v) for v in p2]

    spread = p90 - p10
    mid_spread = p75 - p25
    if spread < 0.45:
        contrast = _round_int(_clamp((0.45 - spread) * 78.0 + (0.20 - mid_spread) * 25.0, 0, 32))
    elif spread > 0.90:
        contrast = _round_int(_clamp(-(spread - 0.90) * 28.0, -16, 0))
    else:
        contrast = 0
    if contrast > 0 and wide_tone_guard > 0.0:
        contrast = min(contrast, _round_int(24 - 16 * wide_tone_guard))

    shadow_need = max(0.0, 0.12 - p10) * 150.0 + max(0.0, 0.22 - p25) * 42.0
    if p50 > 0.55:
        shadow_need *= 0.45
    shadow = _round_int(_clamp(shadow_need, 0, 34))
    if wide_tone_guard > 0.0:
        shadow = min(shadow, _round_int(28 - 20 * wide_tone_guard))

    highlight_need = max(0.0, p98 - 0.94) * 50.0 + max(0.0, p995 - 1.12) * 28.0
    highlight = -_round_int(_clamp(highlight_need, 0, 42))
    if highlight < 0 and wide_tone_guard > 0.0:
        highlight = -min(abs(highlight), _round_int(34 - 16 * wide_tone_guard))

    midtone = 0
    if 0.16 <= p50 <= 0.42 and spread < 0.66:
        midtone = _round_int(_clamp((0.30 - p50) * 18.0, -5, 7))

    if p995 < 0.88 and p90 < 0.63:
        white = _round_int(_clamp((0.88 - p995) * 30.0, 0, 18))
    elif p995 > 1.18:
        white = -_round_int(_clamp((p995 - 1.18) * 22.0, 0, 24))
    else:
        white = 0
    if white < 0 and wide_tone_guard > 0.0:
        white = -min(abs(white), _round_int(24 - 12 * wide_tone_guard))

    black = -_round_int(_clamp(max(0.0, p01 - 0.018) * 230.0 + max(0.0, p05 - 0.065) * 42.0, 0, 24))
    if p005 < 0.0025:
        black = min(0, black + 3)

    sat = _saturation_stats(rgb)
    low_sat = _clamp((0.34 - sat["median"]) / 0.34, 0.0, 1.0)
    high_sat = _clamp((sat["p95"] - 0.82) / 0.18, 0.0, 1.0)
    vibrant_scene = _clamp((sat["p90"] - 0.42) / 0.38, 0.0, 1.0)
    dark_penalty = _clamp((0.12 - p25) / 0.12, 0.0, 1.0)

    vibrance = _round_int(_clamp(16.0 * low_sat + 7.0 * (1.0 - high_sat) - 5.0 * dark_penalty, 0, 24))
    saturation = _round_int(_clamp(5.0 + 7.0 * low_sat + 3.0 * vibrant_scene - 10.0 * high_sat, 0, 14))

    haze_score = _clamp((0.50 - spread) / 0.34, 0.0, 1.0) * _clamp((p05 - 0.035) / 0.16, 0.0, 1.0)
    dehaze = _round_int(_clamp(haze_score * 18.0, 0, 16))
    if p98 > 1.05 or p50 < 0.10:
        dehaze = _round_int(dehaze * 0.55)

    clarity = _round_int(_clamp(5.0 + contrast * 0.22 + haze_score * 8.0 - high_sat * 3.0, 0, 16))
    texture = _round_int(_clamp(3.0 + (0.55 - mid_spread) * 8.0, 0, 10))
    microcontrast = _round_int(_clamp(4.0 + haze_score * 10.0 + (0.42 - spread) * 10.0, 0, 14))
    if wide_tone_guard > 0.0:
        presence_scale = 1.0 - 0.25 * wide_tone_guard
        clarity = _round_int(clarity * presence_scale)
        texture = _round_int(texture * presence_scale)
        microcontrast = _round_int(microcontrast * presence_scale)

    color_density = _round_int(_clamp(4.0 + vibrant_scene * 7.0 + low_sat * 4.0 - high_sat * 10.0, 0, 12))
    color_separation = _round_int(_clamp(3.0 + vibrant_scene * 7.0 + low_sat * 3.0 - high_sat * 7.0, 0, 12))
    chroma_clarity = _round_int(_clamp(2.0 + vibrant_scene * 5.0 + low_sat * 3.0 - dark_penalty * 4.0, 0, 10))
    subtractive_saturation = _round_int(_clamp(2.0 + vibrant_scene * 5.0 - high_sat * 6.0, 0, 8))
    shadow_chroma_clean = _round_to_step(_clamp(dark_penalty * (4.0 + high_sat * 6.0), 0.0, 8.0), 0.5)
    detail_tonemap = _round_int(_clamp(max(0.0, p995 - 0.92) * 8.0 + haze_score * 4.0, 0, 8))

    adjustment = dict(_DEFAULT_ADJUSTMENT)
    adjustment.update({
        "exposure": float(ev),
        "contrast": int(contrast),
        "shadow": int(shadow),
        "highlight": int(highlight),
        "midtone": int(midtone),
        "white": int(white),
        "black": int(black),
        "dehaze": int(dehaze),
        "clarity": int(clarity),
        "texture": int(texture),
        "microcontrast": int(microcontrast),
        "saturation": int(saturation),
        "vibrance": int(vibrance),
        "shadow_chroma_clean": float(shadow_chroma_clean),
        "color_separation": float(color_separation),
        "chroma_clarity": float(chroma_clarity),
        "color_density": float(color_density),
        "subtractive_saturation": float(subtractive_saturation),
        "detail_tonemap": float(detail_tonemap),
    })
    return adjustment
