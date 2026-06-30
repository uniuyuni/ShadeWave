"""Reference lens coating simulation used as fallback and numeric baseline."""

from __future__ import annotations

import cv2
import numpy as np


def _presets():
    f32 = np.float32
    return {
        # 単層コーティング〜無コーティング（オールドレンズ風）
        "VINTAGE_NO_COAT": {
            "color_matrix": np.array([
                [1.05, 0.05, 0.05],  # 赤：少し強調
                [0.05, 0.95, 0.05],  # 緑：少し減衰（黄ばみ）
                [0.05, 0.05, 0.85],  # 青：大きく減衰（青抜け）
            ], dtype=f32),
            "flare_factor": 0.15,     # フレアが多い
            "contrast_factor": 0.85,  # コントラスト低め
            "saturation_factor": 0.9,
            "name": "Vintage No-Coat",
        },
        # 現代のマルチコーティング（ニュートラル）
        "MODERN_MULTI_COAT": {
            "color_matrix": np.array([
                [1.00, 0.00, 0.00],
                [0.00, 1.00, 0.00],
                [0.00, 0.00, 1.00],
            ], dtype=f32),
            "flare_factor": 0.02,     # フレア極少
            "contrast_factor": 1.05,  # コントラスト高め
            "saturation_factor": 1.0,
            "name": "Modern Multi-Coat",
        },
        # ライカ風（赤の発色が良く、微コントラストが高い）
        "LEICA_CLASSIC": {
            "color_matrix": np.array([
                [1.08, 0.02, 0.02],  # 赤：豊かに
                [0.02, 0.98, 0.02],  # 緑：自然
                [0.02, 0.02, 0.95],  # 青：少し抑えめ
            ], dtype=f32),
            "flare_factor": 0.05,     # 適度な耐性
            "contrast_factor": 1.10,  # マイクロコントラスト高
            "saturation_factor": 1.05,
            "name": "Leica Classic",
        },
        # ツァイス T* コーティング風（青みがかり、コントラスト鋭い）
        "ZEISS_TSTAR": {
            "color_matrix": np.array([
                [0.95, 0.02, 0.02],
                [0.02, 1.02, 0.02],
                [0.02, 0.02, 1.05],  # 青：強調
            ], dtype=f32),
            "flare_factor": 0.03,
            "contrast_factor": 1.15,
            "saturation_factor": 1.1,
            "name": "Zeiss T*",
        },
        # キヤノン風（暖色系、柔らかい）
        "CANON_L": {
            "color_matrix": np.array([
                [1.05, 0.03, 0.03],
                [0.03, 1.00, 0.03],
                [0.03, 0.03, 0.95],
            ], dtype=f32),
            "flare_factor": 0.04,
            "contrast_factor": 0.95,
            "saturation_factor": 1.05,
            "name": "Canon L",
        },
    }


PRESETS = _presets()


def apply_color_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """透過スペクトルによる色キャストを RGB 行列で適用する。"""
    m = np.asarray(matrix, dtype=np.float32)
    return np.matmul(image, m.T)


def apply_veiling_glare(image: np.ndarray, flare_factor: float, resolution_scale: float = 1.0) -> np.ndarray:
    """コーティング性能が悪い時のベーリングフレアで黒浮きとコントラスト低下を再現する。"""
    if flare_factor <= 0.0:
        return image

    # 画像の平均輝度を光源の強さの代わりに使い、ぼかしで画面全体への散乱を表現する。
    luminance = np.mean(image, axis=2, keepdims=True, dtype=np.float32)
    glow = cv2.GaussianBlur(luminance, (0, 0), sigmaX=max(1.0, 50.0 * float(resolution_scale)))
    if glow.ndim == 2:
        glow = glow[:, :, np.newaxis]
    glow = np.asarray(glow, dtype=np.float32)
    # フレアの色は白を基本に、コーティング残留色として少し暖色に寄せる。
    flare_color = np.array([1.0, 0.95, 0.9], dtype=np.float32).reshape(1, 1, 3)
    flare_intensity = np.float32(flare_factor * 0.2)
    return image + (glow * flare_color * flare_intensity)


def apply_micro_contrast(image: np.ndarray, contrast_factor: float, resolution_scale: float = 1.0) -> np.ndarray:
    """内部反射の抑制具合を、輝度のローカルコントラストとして表現する。"""
    if abs(contrast_factor - 1.0) < 0.01:
        return image

    luminance = np.mean(image, axis=2, keepdims=True, dtype=np.float32)
    blurred = cv2.GaussianBlur(luminance, (0, 0), sigmaX=max(1.0, 10.0 * float(resolution_scale)))
    if blurred.ndim == 2:
        blurred = blurred[:, :, np.newaxis]
    blurred = np.asarray(blurred, dtype=np.float32)
    # 詳細成分だけをスケールし、比率で RGB に戻すことで色相を保つ。
    detail = luminance - blurred
    enhanced_luminance = blurred + detail * np.float32(contrast_factor)
    ratio = enhanced_luminance / (luminance + 1e-6)
    return image * ratio


def apply_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 0.01:
        return image

    luminance = np.mean(image, axis=2, keepdims=True, dtype=np.float32)
    return luminance + (image - luminance) * np.float32(factor)


def apply_preset(
    image: np.ndarray,
    preset_name: str,
    light_source_intensity: float = 1.0,
    resolution_scale: float = 1.0,
) -> np.ndarray:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset = PRESETS[preset_name]
    result = np.asarray(image, dtype=np.float32).copy()
    result = apply_color_matrix(result, preset["color_matrix"])
    effective_flare = float(preset["flare_factor"] * light_source_intensity)
    result = apply_veiling_glare(result, effective_flare, resolution_scale=resolution_scale)
    result = apply_micro_contrast(result, preset["contrast_factor"], resolution_scale=resolution_scale)
    result = apply_saturation(result, preset["saturation_factor"])
    return result.astype(np.float32, copy=False)
