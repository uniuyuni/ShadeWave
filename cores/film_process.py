import cv2
import numpy as np

import cores.core as core


FILM_MODES = ("Off", "Negative", "Slide", "B&W")


def _clip01(value):
    return float(np.clip(value, 0.0, 100.0)) / 100.0


def _signed01(value):
    return float(np.clip(value, -100.0, 100.0)) / 100.0


def _param(params, key, default):
    if params is None:
        return default
    return params.get(key, default)


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


def _soft_density_response(layers, latitude, contrast, mode):
    latitude = np.clip(latitude, 0.0, 1.0)
    contrast = np.clip(contrast, 0.0, 1.0)

    gamma = 0.74 + contrast * 0.92
    if mode == "Slide":
        gamma += 0.22
    elif mode == "B&W":
        gamma += 0.08

    toe = 0.045 + latitude * 0.13
    shoulder = 0.72 + latitude * 1.12
    exposed = np.maximum(layers + toe, 0.0)
    density = np.power(exposed, gamma)
    density = density / (density + shoulder)
    density -= np.power(toe, gamma) / (np.power(toe, gamma) + shoulder)
    return np.maximum(density, 0.0)


def _apply_halation(rgb, amount):
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


def apply_film_process(image, params=None):
    """Apply a compact, structure-driven film process model to RGB float data."""
    src = np.asarray(image, dtype=np.float32)
    if src.ndim != 3 or src.shape[2] < 3:
        return src.astype(np.float32, copy=False)

    mode = _mode_name(_param(params, "film_mode", "Off"))
    if mode == "Off":
        return src.astype(np.float32, copy=False)
    latitude = _clip01(_param(params, "film_latitude", 55.0))
    contrast = _clip01(_param(params, "film_contrast", 50.0))
    color_bias = _signed01(_param(params, "film_color_bias", 0.0))
    color_drift = _param(params, "film_color_drift", 0.0)
    dye_purity = _clip01(_param(params, "film_dye_purity", 75.0))
    crosstalk = _clip01(_param(params, "film_layer_crosstalk", 30.0))
    aging = _clip01(_param(params, "film_aging", 0.0))

    out = src.copy()
    rgb = np.nan_to_num(out[..., :3], nan=0.0, posinf=4.0, neginf=0.0)
    rgb = np.maximum(rgb, 0.0)

    rgb = _apply_halation(rgb, _param(params, "film_halation", 0.0))

    warm_gain = np.array(
        [1.0 + color_bias * 0.18, 1.0 + abs(color_bias) * 0.035, 1.0 - color_bias * 0.18],
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
        positive = np.power(np.maximum(positive, 0.0), 0.78 + latitude * 0.25)
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

    scan_gamma = 1.0 - (latitude - 0.5) * 0.22
    positive = np.power(np.maximum(positive, 0.0), scan_gamma)

    out[..., :3] = positive.astype(np.float32, copy=False)
    return out.astype(np.float32, copy=False)
