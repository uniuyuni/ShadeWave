"""Reference implementation for Subpixel Shift."""

from __future__ import annotations

import numpy as np


def subpixel_shift(img_array, shift_x=0.5, shift_y=0.5):
    """
    float32形式のRGB画像配列をサブピクセル単位でシフトする。
    """
    if not isinstance(img_array, np.ndarray) or img_array.dtype != np.float32:
        raise ValueError("Input must be a float32 numpy array")

    height, width = img_array.shape[:2]

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    X_shifted = X - shift_x
    Y_shifted = Y - shift_y

    x0 = np.floor(X_shifted).astype(int)
    y0 = np.floor(Y_shifted).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    wx1 = X_shifted - x0
    wx0 = 1 - wx1
    wy1 = Y_shifted - y0
    wy0 = 1 - wy1

    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    weights = (
        wy0[:, :, np.newaxis] * wx0[:, :, np.newaxis],
        wy0[:, :, np.newaxis] * wx1[:, :, np.newaxis],
        wy1[:, :, np.newaxis] * wx0[:, :, np.newaxis],
        wy1[:, :, np.newaxis] * wx1[:, :, np.newaxis],
    )

    samples = (
        img_array[y0, x0],
        img_array[y0, x1],
        img_array[y1, x0],
        img_array[y1, x1],
    )

    return sum(w * s for w, s in zip(weights, samples))


def create_enhanced_image(img_array):
    """
    4つの半ピクセルシフトした画像を合成して、より滑らかな画像を生成する。
    """
    shifts = [
        (-0.5, -0.5),
        (0.5, -0.5),
        (-0.5, 0.5),
        (0.5, 0.5),
    ]

    result = np.zeros_like(img_array)
    for shift_x, shift_y in shifts:
        shifted = subpixel_shift(img_array, shift_x, shift_y)
        result += shifted

    return result / len(shifts)
