"""mask Geometry の tcg-matrix-space 行列ヘルパ。

Composit の effects_param に格納された Mask Geometry params (rotation / flip /
translation / scale) から 3x3 ホモグラフィを構成し、`tcg_info['matrix']` に
左乗算で合成することで Composit 配下のマスクを一括変形する。
"""
from __future__ import annotations

import math
import numpy as np

import effects


def is_enabled(effects_param) -> bool:
    return bool(effects.Mask2Effect.get_param(effects_param, 'switch_mask_geometry'))


def get_hash_tuple(effects_param):
    if not is_enabled(effects_param):
        return (False,)
    return (
        True,
        float(effects.Mask2Effect.get_param(effects_param, 'mask_rotation')),
        int(effects.Mask2Effect.get_param(effects_param, 'mask_flip_mode')),
        float(effects.Mask2Effect.get_param(effects_param, 'mask_translation_x')),
        float(effects.Mask2Effect.get_param(effects_param, 'mask_translation_y')),
        float(effects.Mask2Effect.get_param(effects_param, 'mask_scale_x')),
        float(effects.Mask2Effect.get_param(effects_param, 'mask_scale_y')),
    )


def build_matrix_tcg(effects_param, original_img_size) -> np.ndarray:
    """tcg_info['matrix'] 空間 (画像中心原点、image-px 単位) における 3x3 ホモグラフィ。

    M = T @ R @ S  (点には scale → rotate → translate の順で適用される)
    theta = -radians(rot_deg) で画像 Geom の center_rotate の符号慣例 (rad=-rad) に揃える。
    translation は mask_translation_* (正規化値) × 画像短辺 で image-px に換算。
    flip は scale の符号反転として表現。
    """
    short_side = max(1, int(min(original_img_size)))
    rot_deg = float(effects.Mask2Effect.get_param(effects_param, 'mask_rotation'))
    flip = int(effects.Mask2Effect.get_param(effects_param, 'mask_flip_mode'))
    tx_norm = float(effects.Mask2Effect.get_param(effects_param, 'mask_translation_x'))
    ty_norm = float(effects.Mask2Effect.get_param(effects_param, 'mask_translation_y'))
    sx = float(effects.Mask2Effect.get_param(effects_param, 'mask_scale_x')) or 1e-3
    sy = float(effects.Mask2Effect.get_param(effects_param, 'mask_scale_y')) or 1e-3
    if flip & 1:
        sx = -sx
    if flip & 2:
        sy = -sy
    theta = -math.radians(rot_deg)
    c, s = math.cos(theta), math.sin(theta)
    T = np.array([
        [1, 0, tx_norm * short_side],
        [0, 1, ty_norm * short_side],
        [0, 0, 1],
    ], dtype=np.float64)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    return T @ R @ S
