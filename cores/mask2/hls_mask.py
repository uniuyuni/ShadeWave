"""Shared Mask2 hue/luminance/saturation selection helpers."""
from __future__ import annotations

import numpy as np

import cores.color as color
import cores.hlsrgb as hlsrgb


CHANNEL_INDEX = {
    "hue": 0,
    "lum": 1,
    "sat": 2,
}

DISTANCE_FULL = {
    "hue": 179.0,
    "lum": 255.0,
    "sat": 255.0,
}

RANGE_FULL = {
    "hue": 359.0,
    "lum": 255.0,
    "sat": 255.0,
}

_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def rgb_to_selection_hls(rgb):
    """Return Mask2 selection channels from linear working RGB.

    Hue and saturation keep the app's HLC-like chroma geometry, but the source
    RGB is first perceptually encoded so Mask2 samples are not compared in raw
    linear light. Luminance is computed from that encoded RGB without gain
    normalization, matching the user's brightness intuition more closely.
    """
    if rgb is None:
        return None

    rgb_linear = np.nan_to_num(np.asarray(rgb, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    rgb_linear = np.clip(rgb_linear, 0.0, None)
    encoded = color.prophoto_rgb_gamma_encode(rgb_linear).astype(np.float32, copy=False)

    hls = hlsrgb.rgb_to_hlc_gain(encoded)
    lum = np.tensordot(encoded[..., :3], _LUMA_WEIGHTS, axes=([-1], [0]))
    hls[..., 1] = np.clip(lum, 0.0, 1.0).astype(np.float32, copy=False)
    return hls


def apply_channel_mask(hls, mask, channel, center_xy, distance, range_min, range_max):
    if hls is None:
        return mask

    cimg = hls[..., CHANNEL_INDEX[channel]]
    selected = np.ones(cimg.shape, dtype=bool)
    applied = False

    distance = float(distance)
    distance_full = DISTANCE_FULL[channel]
    if distance != distance_full:
        applied = True
        inverted_distance = distance < 0.0
        distance = abs(distance)
        center = _sample_center(cimg, center_xy)
        if channel == "hue":
            distance_selected = _hue_distance_selected(cimg, center, distance)
        else:
            distance_selected = _linear_distance_selected(cimg, center, distance / 255.0)
        if inverted_distance:
            distance_selected = ~distance_selected
        selected &= distance_selected

    range_full = RANGE_FULL[channel]
    range_min = float(range_min)
    range_max = float(range_max)
    if range_min != 0.0 or range_max != range_full:
        applied = True
        if channel == "hue":
            selected &= _hue_range_selected(cimg, range_min, range_max)
        else:
            selected &= _linear_range_selected(cimg, range_min / range_full, range_max / range_full)

    if not applied:
        return mask

    return np.where(selected, mask, 0)


def _sample_center(channel_img, center_xy):
    h, w = channel_img.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    cx, cy = center_xy
    ix = int(np.clip(int(cx), 0, w - 1))
    iy = int(np.clip(int(cy), 0, h - 1))
    return float(channel_img[iy, ix])


def _hue_distance_selected(values, center, distance):
    distance = max(0.0, min(float(distance), 180.0))
    delta = np.abs(((values - center + 180.0) % 360.0) - 180.0)
    return delta <= distance


def _hue_range_selected(values, range_min, range_max):
    lo = float(range_min) % 360.0
    hi = float(range_max) % 360.0
    if lo <= hi:
        return (lo <= values) & (values <= hi)
    return (lo <= values) | (values <= hi)


def _linear_distance_selected(values, center, distance):
    distance = max(0.0, float(distance))
    lo = max(0.0, float(center) - distance)
    hi = min(1.0, float(center) + distance)
    return (lo <= values) & (values <= hi)


def _linear_range_selected(values, range_min, range_max):
    lo = max(0.0, min(1.0, float(range_min)))
    hi = max(0.0, min(1.0, float(range_max)))
    if hi < lo:
        lo, hi = hi, lo
    return (lo <= values) & (values <= hi)
