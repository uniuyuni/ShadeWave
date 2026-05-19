"""Skin Smooth (Inverted High Pass) — Mask2 Draw Effects 用の肌補正フィルタ。

Photoshop の定番手法
    duplicate → High Pass(R) → Gaussian Blur(R/3) → Invert → Linear Light
は数学的には
    out = clip( 2 * GaussianBlur(img, sigma=R/3) - img , 0, 1 )
と等価。レイヤー・反転・ブレンド分岐を経ずに 1 回のガウシアンと線形演算で同じ結果が得られる。

半径 R は 4096 解像度基準で 16px を中心とし、効果に渡される
``efconfig.resolution_scale`` を介して各解像度（preview/full/export）へ自動追従する。
これは Photoshop の経験則（24px @ 6000px wide ≒ 寸法の 0.4%）と概ね一致する。

ユーザー UI は Amount（0-100）が主、Radius Bias（-100..+100）が補助。
Radius Bias は ``2 ** (bias/100)`` の指数倍率で半径を 0.5x〜2x の範囲で調整する。
"""

from __future__ import annotations

import cv2
import numpy as np


# 4096 px 基準の High Pass 半径。Photoshop 経験則の「寸法の約 0.4%」相当。
BASE_HIGH_PASS_RADIUS = 16.0
# Photoshop 標準の比率（Gaussian Blur radius = High Pass radius / 3）
GAUSSIAN_TO_HIGH_PASS_RATIO = 1.0 / 3.0


def compute_sigma(resolution_scale: float, radius_bias_minus100_to_100: float) -> float:
    """効果に渡される resolution_scale と Bias から GaussianBlur 用の sigma を求める。

    resolution_scale は cores.core.calc_resolution_scale の出力で、
    base_resolution_scale=[4096,4096] 比の幾何平均。
    """
    bias = max(-100.0, min(100.0, float(radius_bias_minus100_to_100))) / 100.0
    bias_mult = 2.0 ** bias  # -100=0.5x, 0=1.0x, +100=2.0x

    radius_px = BASE_HIGH_PASS_RADIUS * float(resolution_scale) * bias_mult
    sigma = radius_px * GAUSSIAN_TO_HIGH_PASS_RATIO
    # 1px 未満の半径は実質的に no-op になるが、cv2.GaussianBlur は sigma>0 を要求する。
    # 小さすぎる sigma は呼び出し側で early-return する想定。
    return float(sigma)


def inverted_high_pass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Linear Light + Inverted High Pass の合成結果を返す。

    img: float32 RGB（[0,1] 想定）。返り値も同形状の float32。
    sigma <= 0 の場合は入力をそのまま返す。
    """
    if sigma <= 0.0:
        return img
    blurred = cv2.GaussianBlur(
        img,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REPLICATE,
    )
    out = 2.0 * blurred - img
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def apply_skin_smooth(
    img: np.ndarray,
    *,
    amount_0_1: float,
    sigma: float,
) -> np.ndarray:
    """amount で原画像と IHP 結果を補間する薄いラッパー。"""
    a = float(amount_0_1)
    if a <= 0.0 or sigma <= 0.0:
        return img
    a = min(1.0, a)
    smoothed = inverted_high_pass(img, sigma)
    if a >= 1.0:
        return smoothed
    return (img * (1.0 - a) + smoothed * a).astype(np.float32, copy=False)
