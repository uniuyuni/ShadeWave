"""Reference implementation for low frequency transfer."""

from __future__ import annotations

import cv2
import numpy as np


def apply_low_frequency_transfer(
    restored_img,
    reference_img,
    sigma=30,
    highlight_threshold=None,
    highlight_transition=0.35,
    highlight_detail_strength=0.25,
    luminance_transfer_strength=1.0,
):
    h, w = restored_img.shape[:2]
    if reference_img.shape[:2] != (h, w):
        reference_img = cv2.resize(reference_img, (w, h), interpolation=cv2.INTER_LINEAR)

    restored_float = restored_img
    reference_float = reference_img

    low_freq_restored = cv2.GaussianBlur(restored_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    low_freq_reference = cv2.GaussianBlur(reference_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    low_freq_diff = low_freq_reference - low_freq_restored
    luma_strength = float(np.clip(luminance_transfer_strength, 0.0, 1.0))
    if luma_strength < 1.0:
        if low_freq_diff.ndim == 2:
            low_freq_diff = low_freq_diff * luma_strength
        else:
            lum_diff = (
                0.2126 * low_freq_diff[:, :, 0]
                + 0.7152 * low_freq_diff[:, :, 1]
                + 0.0722 * low_freq_diff[:, :, 2]
            )
            low_freq_diff = low_freq_diff - lum_diff[:, :, np.newaxis] * (1.0 - luma_strength)
    high_freq_restored = restored_float - low_freq_restored

    if highlight_threshold is not None:
        if reference_float.ndim == 2:
            luminance = reference_float
            mask = np.clip((luminance - highlight_threshold) / highlight_transition, 0.0, 1.0)
            mask = mask * mask * (3.0 - 2.0 * mask)
            detail_scale = 1.0 - mask * (1.0 - highlight_detail_strength)
        else:
            luminance = np.max(reference_float, axis=2)
            mask = np.clip((luminance - highlight_threshold) / highlight_transition, 0.0, 1.0)
            mask = mask * mask * (3.0 - 2.0 * mask)
            detail_scale = 1.0 - mask[..., np.newaxis] * (1.0 - highlight_detail_strength)
        high_freq_restored = high_freq_restored * detail_scale

    return restored_float + low_freq_diff - (restored_float - low_freq_restored) + high_freq_restored
