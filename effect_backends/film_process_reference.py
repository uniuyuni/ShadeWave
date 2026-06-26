"""NumPy reference (and fallback) implementation of the Film Process model.

This is the numerical baseline that the native CPU backend must match within
tolerance. It is the verbatim port of the former ``cores/film_process.py`` with
an explicit-kwargs public API (no param dict) per the effect-backends design.
"""

from __future__ import annotations

import numpy as np

import cores.core as core


FILM_MODES = ("Off", "Negative", "Slide", "B&W")

# Shared with the adapter / native backend. Must stay in sync with the enum in
# film_process_capi.h.
_MODE_INDEX = {"Off": 0, "Negative": 1, "Slide": 2, "B&W": 3}


def _clip01(value):
    return float(np.clip(value, 0.0, 100.0)) / 100.0


def _signed01(value):
    return float(np.clip(value, -100.0, 100.0)) / 100.0


def _mode_name(value):
    if value in FILM_MODES:
        return value
    text = str(value or "Negative").strip().lower()
    if text in {"off", "none", "disabled", "disable"}:
        return "Off"
    if text in {"bw", "b&w", "black and white", "black-and-white", "monochrome"}:
        return "B&W"
    if text in {"slide", "positive", "reversal"}:
        return "Slide"
    return "Negative"


def _mode_index(value):
    return _MODE_INDEX[_mode_name(value)]


def _soft_density_response(layers, latitude, contrast, mode):
    latitude = np.clip(latitude, 0.0, 1.0)
    contrast = np.clip(contrast, 0.0, 1.0)

    gamma = 0.74 + contrast * 0.92
    if mode == "Slide":
        gamma += 0.22
    elif mode == "B&W":
        gamma += 0.08

    toe = 0.045 + latitude * 0.13
    shoulder = 0.72 + latitude * 1.12          # ハイライトのニー位置（latitude=ダイナミックレンジ）
    exposed = np.maximum(layers + toe, 0.0)
    dl = np.power(exposed, gamma)              # 線形濃度（上限なし）

    # フィルムのショルダー（Reinhard 風の丸め）は 0..1 に飽和する＝トーンマップ。これだと
    # HDR ハイライトが全部 ~1 へ潰れるので、ニー(shoulder)を超えた分を latitude 量だけ線形に
    # 足し戻し、ハイライトが 1.0 を超えて伸び続ける（=HDR を保持）ようにする。excess=0 の
    # 中間調/シャドウは旧式 density/(density+shoulder) と完全一致＝既存ルックを壊さない。
    comp = dl / (dl + shoulder)                # フィルムらしい明部ロールオフ（≤1）
    headroom = np.float32(0.02 + latitude * 0.13)   # 0.02(ほぼトーンマップ)…0.15(HDR を強く保持)
    excess = np.maximum(dl - shoulder, 0.0)    # ニーを超えたシーン由来のハイライト
    density = comp + headroom * excess

    # 黒（toe）が 0 になるよう正規化（toe は shoulder より十分小さく excess=0 なので comp のみ）。
    black = np.power(toe, gamma) / (np.power(toe, gamma) + shoulder)
    return np.maximum(density - black, 0.0)


def _apply_halation(rgb, amount):
    """空間処理（ハイライトのにじみ）。pointwise でないため native backend には載せず、
    adapter からも本関数を共有して native/reference に同一の haloed 入力を与える。"""
    amount = _clip01(amount)
    if amount <= 0.0:
        return rgb

    luma = core.cvtColorRGB2Gray(np.clip(rgb, 0.0, None))
    threshold = 0.72 - amount * 0.18
    highlights = np.maximum(luma - threshold, 0.0) / max(1.0 - threshold, 1e-6)
    if not np.any(highlights > 0.0):
        return rgb

    height, width = rgb.shape[:2]
    radius = max(1.0, min(height, width) * (0.004 + amount * 0.014))
    bloom = core.gaussian_blur_cv(highlights.astype(np.float32), (0, 0), radius)
    halo_color = np.array([1.0, 0.38, 0.16], dtype=np.float32)
    return rgb + bloom[..., np.newaxis] * halo_color * (0.10 + amount * 0.34)


def _apply_color_drift(rgb, amount):
    amount = _signed01(amount)
    if abs(amount) <= 1e-6:
        return rgb

    work = np.asarray(rgb, dtype=np.float32)
    safe = np.clip(work, 0.0, None).astype(np.float32, copy=False)
    luma = core.cvtColorRGB2Gray(safe)
    shadow_w = np.power(np.clip(1.0 - luma, 0.0, 1.0), 1.45)
    highlight_w = np.power(np.clip(luma, 0.0, 1.0), 1.65)
    mid_w = 1.0 - np.power(np.clip(np.abs(luma * 2.0 - 1.0), 0.0, 1.0), 1.8)

    direction = 1.0 if amount > 0.0 else -1.0
    strength = abs(amount)
    shadow_bias = np.array([-0.050, 0.026, 0.064], dtype=np.float32)
    highlight_bias = np.array([0.075, 0.018, -0.050], dtype=np.float32)
    mid_bias = np.array([0.020, -0.014, 0.010], dtype=np.float32)
    tonal_bias = (
        shadow_w[..., np.newaxis] * shadow_bias
        + highlight_w[..., np.newaxis] * highlight_bias
        + mid_w[..., np.newaxis] * mid_bias
    )

    rb_opponent = (safe[..., 0] - safe[..., 2])[..., np.newaxis]
    gm_opponent = (safe[..., 1] - np.mean(safe, axis=-1))[..., np.newaxis]
    channel_twist = np.concatenate(
        [
            -0.030 * gm_opponent,
            0.022 * rb_opponent,
            -0.026 * rb_opponent + 0.018 * gm_opponent,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    drifted = work + direction * strength * (tonal_bias + channel_twist)
    return np.maximum(drifted, 0.0).astype(np.float32, copy=False)


def apply_film_process(
    image,
    mode="Off",
    latitude=55.0,
    contrast=50.0,
    color_bias=0.0,
    color_drift=0.0,
    dye_purity=75.0,
    layer_crosstalk=30.0,
    halation=0.0,
    aging=0.0,
):
    """Apply a compact, structure-driven film process model to RGB float data."""
    src = np.asarray(image, dtype=np.float32)
    if src.ndim != 3 or src.shape[2] < 3:
        return src.astype(np.float32, copy=False)

    mode = _mode_name(mode)
    if mode == "Off":
        return src.astype(np.float32, copy=False)
    latitude = _clip01(latitude)
    contrast = _clip01(contrast)
    color_bias = _signed01(color_bias)
    dye_purity = _clip01(dye_purity)
    crosstalk = _clip01(layer_crosstalk)
    aging = _clip01(aging)

    out = src.copy()
    rgb = np.nan_to_num(out[..., :3], nan=0.0, posinf=4.0, neginf=0.0)
    rgb = np.maximum(rgb, 0.0)

    rgb = _apply_halation(rgb, halation)

    # フィルム的WB＝色温度(青⇔アンバー)＋ティント(緑⇔マゼンタ)を1スライダーに合成。
    # 緑を温度と逆向きに動かすことで純粋な色温度(緑中立)と差別化する。
    #   warm(+): R↑ G↓ B↓ = アンバー＋マゼンタ寄りの暖色
    #   cool(-): R↓ G↑ B↑ = 緑＋青＝ティール寄りの寒色
    warm_gain = np.array(
        [1.0 + color_bias * 0.18, 1.0 - color_bias * 0.08, 1.0 - color_bias * 0.18],
        dtype=np.float32,
    )
    age_gain = np.array([1.0 + aging * 0.10, 1.0 - aging * 0.08, 1.0 - aging * 0.24], dtype=np.float32)
    layers = rgb * np.maximum(warm_gain * age_gain, 0.05)

    # 各行の合計は 1.0（凸結合＝非負重みの加重平均）なので、出力は必ず入力 RGB の
    # レンジ内に収まりクリッピング/負値は発生しない。混色を強めるほど彩度が下がるが、
    # 対角成分（自己寄与）を各行で最大に保つことで完全なグレー潰れ＝破綻を避ける。
    mix_matrix = np.array(
        [
            [1.0 - 0.60 * crosstalk, 0.36 * crosstalk, 0.24 * crosstalk],
            [0.30 * crosstalk, 1.0 - 0.54 * crosstalk, 0.24 * crosstalk],
            [0.24 * crosstalk, 0.36 * crosstalk, 1.0 - 0.60 * crosstalk],
        ],
        dtype=np.float32,
    )
    layers = layers @ mix_matrix.T

    density = _soft_density_response(layers, latitude, contrast, mode)

    if mode == "Negative":
        positive = 1.0 - np.exp(-density * (1.55 + contrast * 1.10))
        positive = np.power(np.maximum(positive, 0.0), 0.90)   # プリントガンマ固定（latitude と分離）
        # 1-exp は ~1 に飽和し、せっかく density 側で保持した HDR ハイライトを再び潰す。
        # density が 1 を超えた分（=HDR ハイライト）を線形に足し戻して明部の伸びを残す。
        positive = positive + np.maximum(density - 1.0, 0.0) * np.float32(0.45)
    elif mode == "Slide":
        positive = np.power(np.maximum(density, 0.0), 0.72 + contrast * 0.48)
        positive = positive * (1.06 + contrast * 0.22)
    else:
        spectral_weights = np.array([0.28, 0.55, 0.17], dtype=np.float32)
        mono = np.sum(density * spectral_weights, axis=-1)
        mono = np.power(np.maximum(mono, 0.0), 0.78 + contrast * 0.42)
        positive = np.repeat(mono[..., np.newaxis], 3, axis=-1)

    if mode != "B&W":
        positive = np.asarray(positive, dtype=np.float32)
        positive = _apply_color_drift(positive, color_drift)
        luma = core.cvtColorRGB2Gray(np.clip(positive, 0.0, None).astype(np.float32, copy=False))
        luma3 = np.repeat(luma[..., np.newaxis], 3, axis=-1)
        purity = 0.42 + dye_purity * 0.88
        positive = luma3 + (positive - luma3) * purity

        impurity = (1.0 - dye_purity) * 0.22
        dye_leak = np.array(
            [[1.0 - impurity, impurity * 0.65, impurity * 0.35],
             [impurity * 0.35, 1.0 - impurity, impurity * 0.65],
             [impurity * 0.55, impurity * 0.45, 1.0 - impurity]],
            dtype=np.float32,
        )
        positive = positive @ dye_leak.T

    fog = aging * 0.095
    base_stain = np.array([1.0 + aging * 0.10, 1.0 + aging * 0.035, 1.0 - aging * 0.055], dtype=np.float32)
    positive = positive * (1.0 - fog) + fog
    positive *= base_stain

    # （旧）scan_gamma で latitude を最終ガンマ＝明るさにも流用していたが、latitude の役割を
    # 「ダイナミックレンジ」へ一本化するため撤去。明るさ/コントラストは contrast 側に集約。
    # HDR ハイライトを再圧縮しないよう最終ガンマは掛けず、下限のみ確保する。
    positive = np.maximum(positive, 0.0)

    out[..., :3] = positive.astype(np.float32, copy=False)
    return out.astype(np.float32, copy=False)
