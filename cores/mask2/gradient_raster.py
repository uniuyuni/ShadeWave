"""
線形グラデーションマスクのラスタ化（Kivy 非依存）。
widgets.mask_editor2.GradientMask.draw_gradient と同一アルゴリズム。
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def draw_linear_gradient(image_size, center, start_point, end_point, smoothness=1):
    width, height = image_size

    start_x, start_y = end_point
    end_x, end_y = start_point
    vec_start_end = np.array([end_x - start_x, end_y - start_y])
    length_start_end = np.linalg.norm(vec_start_end)

    if length_start_end == 0:
        return np.zeros((height, width), dtype=np.float32)

    sigma = (length_start_end * 0.25) * smoothness
    if sigma < 0.1:
        img = np.zeros((height, width), dtype=np.float32)
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        unit_vec = vec_start_end / length_start_end
        y_coords, x_coords = np.indices((height, width))
        projected = (x_coords - mid_x) * unit_vec[0] + (y_coords - mid_y) * unit_vec[1]
        img[projected >= 0] = 1.0
        return img

    target_sigma = 4.0
    scale = target_sigma / sigma if sigma > target_sigma else 1.0
    scale = min(scale, 1.0)

    small_w = int(math.ceil(width * scale))
    small_h = int(math.ceil(height * scale))

    if small_w <= 0 or small_h <= 0:
        return np.zeros((height, width), dtype=np.float32)

    img_small = np.zeros((small_h, small_w), dtype=np.float32)

    start_x_s = start_x * scale
    start_y_s = start_y * scale
    end_x_s = end_x * scale
    end_y_s = end_y * scale
    mid_x_s = (start_x_s + end_x_s) / 2
    mid_y_s = (start_y_s + end_y_s) / 2

    vec_s = np.array([end_x_s - start_x_s, end_y_s - start_y_s])
    len_s = np.linalg.norm(vec_s)
    if len_s == 0:
        return np.zeros((height, width), dtype=np.float32)
    unit_vec_s = vec_s / len_s

    y_coords_s, x_coords_s = np.indices((small_h, small_w))
    projected_s = (x_coords_s - mid_x_s) * unit_vec_s[0] + (y_coords_s - mid_y_s) * unit_vec_s[1]

    img_small[projected_s >= 0] = 1.0

    eff_sigma = sigma * scale
    img_small = cv2.GaussianBlur(img_small, (0, 0), sigmaX=eff_sigma, sigmaY=eff_sigma)

    if scale < 1.0:
        img = cv2.resize(img_small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        img = img_small

    return img
