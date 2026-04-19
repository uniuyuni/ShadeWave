"""
楕円グラデーションマスクのラスタ化（Kivy 非依存）。
widgets.mask_editor2.CircularGradientMask.draw_elliptical_gradient と同一アルゴリズム。
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def draw_elliptical_gradient(
    image_size,
    center,
    inner_axes,
    outer_axes,
    angle_rad,
    invert=False,
    smoothness=1,
):
    width, height = image_size

    if width <= 0 or height <= 0:
        return np.zeros((height, width), dtype=np.float32)

    angle_rad = -angle_rad

    rx_in, ry_in = inner_axes
    rx_out, ry_out = outer_axes

    rx_mid = (rx_in + rx_out) / 2.0
    ry_mid = (ry_in + ry_out) / 2.0

    sigma_x = abs(rx_out - rx_in) * 0.25 * smoothness
    sigma_y = abs(ry_out - ry_in) * 0.25 * smoothness

    if sigma_x < 0.1 and sigma_y < 0.1:
        sigma_x = 0.1
        sigma_y = 0.1

    target_sigma = 4.0
    min_sigma = min(sigma_x, sigma_y)
    dest_scale = target_sigma / min_sigma if min_sigma > target_sigma else 1.0
    dest_scale = min(dest_scale, 1.0)

    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )

    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)

    corners_centered = corners - center
    x_rot = corners_centered[:, 0] * cos_a - corners_centered[:, 1] * sin_a
    y_rot = corners_centered[:, 0] * sin_a + corners_centered[:, 1] * cos_a

    min_x = np.min(x_rot)
    max_x = np.max(x_rot)
    min_y = np.min(y_rot)
    max_y = np.max(y_rot)

    src_scale = dest_scale
    eff_sigma_x = sigma_x * src_scale
    eff_sigma_y = sigma_y * src_scale

    pad_x = int(math.ceil(3.0 * eff_sigma_x))
    pad_y = int(math.ceil(3.0 * eff_sigma_y))

    unrot_w = max_x - min_x
    unrot_h = max_y - min_y

    src_w = int(math.ceil(unrot_w * src_scale)) + 2 * pad_x
    src_h = int(math.ceil(unrot_h * src_scale)) + 2 * pad_y

    src_origin_x = min_x * src_scale - pad_x
    src_origin_y = min_y * src_scale - pad_y

    ell_cx = -src_origin_x
    ell_cy = -src_origin_y

    src_img = np.zeros((src_h, src_w), dtype=np.float32)

    if invert is False:
        bg_color = 1.0
        fg_color = 0.0
    else:
        bg_color = 0.0
        fg_color = 1.0

    src_img.fill(bg_color)

    cv2.ellipse(
        src_img,
        (int(ell_cx), int(ell_cy)),
        (int(rx_mid * src_scale), int(ry_mid * src_scale)),
        0,
        0,
        360,
        color=fg_color,
        thickness=-1,
    )

    src_img = cv2.GaussianBlur(src_img, (0, 0), sigmaX=eff_sigma_x, sigmaY=eff_sigma_y)

    dest_w = int(width * dest_scale)
    dest_h = int(height * dest_scale)

    if dest_w <= 0 or dest_h <= 0:
        return np.zeros((height, width), dtype=np.float32)

    cos_v = np.cos(angle_rad)
    sin_v = np.sin(angle_rad)

    a00 = cos_v
    a01 = -sin_v
    a10 = sin_v
    a11 = cos_v

    ox = src_origin_x
    oy = src_origin_y

    cx = center[0] * dest_scale
    cy = center[1] * dest_scale

    tx = ox * cos_v - oy * sin_v + cx
    ty = ox * sin_v + oy * cos_v + cy

    M = np.array([[a00, a01, tx], [a10, a11, ty]], dtype=np.float32)

    border_val = bg_color

    dst_small = cv2.warpAffine(
        src_img,
        M,
        (dest_w, dest_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_val),
    )

    if dest_scale < 1.0:
        dst_img = cv2.resize(dst_small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        dst_img = dst_small

    return dst_img
