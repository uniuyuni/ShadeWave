"""Reference Color Separation implementation."""

from __future__ import annotations

import cv2
import numpy as np

import cores.hlsrgb as hlsrgb


def gaussian_blur_cv(src, ksize=(3, 3), sigma=0.0):
    if ksize == (0, 0) and sigma == 0.0:
        return src
    return cv2.GaussianBlur(src, ksize, sigma)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1.0e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _apply_subtractive_saturation(rgb, amount):
    amount = np.clip(float(amount), -1.0, 1.0)
    if amount == 0.0:
        return rgb

    src = np.asarray(rgb, dtype=np.float32)
    y = (0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]).astype(np.float32, copy=False)
    neutral = y[..., None]
    chroma_vec = src - neutral
    chroma = np.sqrt(np.sum(chroma_vec * chroma_vec, axis=-1))
    relative_chroma = chroma / (np.maximum(y, 0.0) + 1.0e-4)
    chroma_gate = smoothstep(0.025, 0.42, relative_chroma)
    midtone_gate = smoothstep(0.035, 0.24, y) * (1.0 - smoothstep(1.7, 4.0, y))

    if amount > 0.0:
        vivid_rolloff = 1.0 - 0.45 * smoothstep(0.95, 2.20, relative_chroma)
        sat_gain = 1.0 + amount * 0.55 * chroma_gate * midtone_gate * vivid_rolloff
        density = 1.0 - amount * 0.18 * chroma_gate * midtone_gate
    else:
        soften = -amount
        sat_gain = 1.0 - soften * 0.42 * chroma_gate * midtone_gate
        density = 1.0 + soften * 0.08 * chroma_gate * midtone_gate

    out = neutral + chroma_vec * sat_gain[..., None]
    return (out * density[..., None]).astype(np.float32, copy=False)


def apply_color_separation(
    img_float32,
    shadow_chroma_clean=0.0,
    shadow_threshold=0.2,
    color_separation=0.0,
    chroma_clarity=0.0,
    color_density=0.0,
    subtractive_saturation=0.0,
    opponent_contrast=0.0,
):
    """
    Clean low-luminance chroma and gently separate colors in linear RGB.

    The operation is intentionally conservative:
    - all-zero parameters are an exact identity,
    - vivid colors are protected from shadow chroma cleaning,
    - color separation is reduced on already vivid pixels,
    - newly introduced negative values are limited to the input lower bound.
    """
    shadow_chroma_clean = float(shadow_chroma_clean)
    shadow_threshold = float(shadow_threshold)
    color_separation = float(color_separation)
    chroma_clarity = float(chroma_clarity)
    color_density = float(color_density)
    subtractive_saturation = float(subtractive_saturation)
    opponent_contrast = float(opponent_contrast)
    if (shadow_chroma_clean == 0.0 and color_separation == 0.0
            and chroma_clarity == 0.0 and color_density == 0.0
            and subtractive_saturation == 0.0
            and opponent_contrast == 0.0):
        return img_float32

    src = np.asarray(img_float32, dtype=np.float32)
    ycbcr = hlsrgb.linear_rgb_to_ycbcr(src)
    y, cb, cr = cv2.split(ycbcr)

    chroma = np.sqrt(cb * cb + cr * cr)
    relative_chroma = chroma / (np.maximum(y, 0.0) + 1.0e-4)

    if shadow_chroma_clean > 0.0 and shadow_threshold > 0.0:
        threshold = max(shadow_threshold, 1.0e-4)
        shadow_mask = 1.0 - smoothstep(threshold * 0.35, threshold, y)
        vivid_protect = smoothstep(0.12, 0.45, relative_chroma)
        clean_amount = np.clip(shadow_chroma_clean, 0.0, 1.0) * 0.9
        clean_scale = 1.0 - clean_amount * shadow_mask * (1.0 - vivid_protect)
        cb = cb * clean_scale
        cr = cr * clean_scale

    if chroma_clarity != 0.0:
        chroma = np.sqrt(cb * cb + cr * cr)
        relative_chroma = chroma / (np.maximum(y, 0.0) + 1.0e-4)
        midtone_mask = smoothstep(0.035, 0.18, y)
        hdr_protect = 1.0 - smoothstep(1.6, 4.0, y)
        neutral_gate = smoothstep(0.015, 0.10, relative_chroma)
        vivid_limit = 1.0 - 0.45 * smoothstep(0.80, 1.80, relative_chroma)
        clarity_weight = midtone_mask * hdr_protect * neutral_gate * vivid_limit
        clarity_gain = np.clip(chroma_clarity, -1.0, 1.0)
        cb32 = cb.astype(np.float32, copy=False)
        cr32 = cr.astype(np.float32, copy=False)
        cb_local = gaussian_blur_cv(cb32, (0, 0), 1.2)
        cr_local = gaussian_blur_cv(cr32, (0, 0), 1.2)
        cb_base = gaussian_blur_cv(cb32, (0, 0), 7.0)
        cr_base = gaussian_blur_cv(cr32, (0, 0), 7.0)
        cb = cb + (cb_local - cb_base) * clarity_gain * 1.15 * clarity_weight
        cr = cr + (cr_local - cr_base) * clarity_gain * 1.15 * clarity_weight

    if color_separation > 0.0:
        chroma = np.sqrt(cb * cb + cr * cr)
        relative_chroma = chroma / (np.maximum(y, 0.0) + 1.0e-4)
        midtone_mask = smoothstep(0.04, 0.22, y)
        hdr_protect = 1.0 - smoothstep(1.6, 4.0, y)
        vivid_limit = 1.0 - 0.65 * smoothstep(0.30, 0.90, relative_chroma)
        sep_gain = 1.0 + np.clip(color_separation, 0.0, 1.0) * 0.35 * midtone_mask * hdr_protect * vivid_limit
        cb = cb * sep_gain
        cr = cr * sep_gain

    if color_density != 0.0:
        chroma = np.sqrt(cb * cb + cr * cr)
        relative_chroma = chroma / (np.maximum(y, 0.0) + 1.0e-4)
        midtone_mask = smoothstep(0.06, 0.24, y) * (1.0 - smoothstep(1.4, 3.2, y))
        neutral_gate = smoothstep(0.025, 0.18, relative_chroma)
        density_value = np.clip(color_density, -1.0, 1.0)
        if density_value > 0.0:
            vivid_rolloff = 1.0 - 0.85 * smoothstep(0.45, 1.05, relative_chroma)
            density_amount = density_value * midtone_mask * neutral_gate * vivid_rolloff
            target_chroma = chroma + 0.10 * np.tanh(chroma / 0.10)
            density_gain = 1.0 + density_amount * ((target_chroma / (chroma + 1.0e-6)) - 1.0)
        else:
            vivid_rolloff = 1.0 - 0.35 * smoothstep(0.70, 1.60, relative_chroma)
            density_amount = (-density_value) * midtone_mask * neutral_gate * vivid_rolloff
            density_gain = 1.0 - 0.40 * density_amount
        cb = cb * density_gain
        cr = cr * density_gain

    y = y.astype(np.float32, copy=False)
    cb = cb.astype(np.float32, copy=False)
    cr = cr.astype(np.float32, copy=False)
    out = hlsrgb.linear_ycbcr_to_rgb(cv2.merge((y, cb, cr))).astype(np.float32, copy=False)
    if subtractive_saturation != 0.0:
        out = _apply_subtractive_saturation(out, subtractive_saturation)
    if opponent_contrast > 0.0:
        r, g, b = cv2.split(out)
        y_opp = 0.2126 * r + 0.7152 * g + 0.0722 * b
        rg = r - g
        by = b - 0.5 * (r + g)
        opponent_strength = (np.abs(rg) + np.abs(by)) / (np.maximum(y_opp, 0.0) + 1.0e-4)
        midtone_mask = smoothstep(0.05, 0.24, y_opp)
        hdr_protect = 1.0 - smoothstep(1.6, 4.0, y_opp)
        vivid_rolloff = 1.0 - 0.70 * smoothstep(0.70, 1.80, opponent_strength)
        opponent_gain = 1.0 + np.clip(opponent_contrast, 0.0, 1.0) * 0.26 * midtone_mask * hdr_protect * vivid_rolloff
        rg = rg * opponent_gain
        by = by * opponent_gain
        g_new = y_opp - (0.2126 + 0.0722 * 0.5) * rg - 0.0722 * by
        r_new = g_new + rg
        b_new = g_new + 0.5 * rg + by
        out = cv2.merge((
            r_new.astype(np.float32, copy=False),
            g_new.astype(np.float32, copy=False),
            b_new.astype(np.float32, copy=False),
        ))
    lower_bound = np.minimum(src, 0.0)
    return np.maximum(out, lower_bound)
