import cv2
import numpy as np
from numba import njit, prange

import cores.hlsrgb as hlsrgb
from threads import lock_numba


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _cvt_color_rgb_to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _estimate_depth_map(img, params=(0.121779, 0.959710, -0.780245), sigma=0.5):
    """
    色線形変換先行法（Color Attenuation Prior）を使用して深度マップを推定

    img: 線形 RGB（float32）。SDR でも HDR（>1）でも可。
    params: 線形モデルの係数 (β0, β1, β2) — 論文は OpenCV HSV の V,S 前提で学習。
    sigma: ガウシアンフィルタのシグマ値

    Zhu ら "Fast Single Image Haze Removal Using Color Attenuation Prior" の
    d ≈ β0 + β1*V + β2*S を、hlsrgb.rgb2hls の **L**（gain 正規化後の輝度）と **S**
    （彩度）に置換。OpenCV の HSV は float でも内部が SDR 寄りで V が HDR で不自然に
    なりやすいため。

    β は従来値のまま維持しているが、特徴の分布が変わるため効き具合が変わり得る。
    """
    img_f = np.ascontiguousarray(img, dtype=np.float32)
    # H[0,360), L[0,1], S[0,1], Gain（HDR 時 max(R,G,B)）
    hls = hlsrgb.rgb2hls(img_f)
    l_chan = hls[:, :, 1]
    s_chan = hls[:, :, 2]

    beta0, beta1, beta2 = params
    depth = beta0 + beta1 * l_chan + beta2 * s_chan

    # フィルタリングで深度マップを滑らかにする
    depth = cv2.GaussianBlur(depth, (0, 0), sigma)

    # 正規化（0-1の範囲に変換）。
    # 以前は画像ごとの min/max（コンテンツ依存）で正規化していたため、可視領域
    # （クロップ/ズーム）が変わると深度スケールも変わり、同一ピクセルでも全体表示と
    # 拡大表示で霞除去の効き方が変わってしまっていた（ズーム非整合）。
    # L, S は共に [0,1] なので、β 係数が理論的に取り得る範囲で正規化し、
    # コンテンツ非依存・ズーム非依存にする。
    depth_lo = beta0 + min(beta1, 0.0) + min(beta2, 0.0)
    depth_hi = beta0 + max(beta1, 0.0) + max(beta2, 0.0)
    depth = (depth - depth_lo) / (depth_hi - depth_lo + 1e-8)
    depth = np.clip(depth, 0.0, 1.0)

    return depth


def _estimate_atmospheric_light(img, depth_map, top_percent=0.001):
    """
    大気光を推定（最も深い点の上位N%を使用）

    img: 入力画像（RGB形式）
    depth_map: 深度マップ（値が大きいほど霧が濃い）
    top_percent: 使用する上位のピクセルの割合
    """
    # 画像サイズと上位N%のピクセル数を計算
    h, w = depth_map.shape
    size = h * w
    num_pixels = int(size * top_percent)

    # 深度マップに基づいてピクセルをソート
    indices = np.argsort(depth_map.flatten())[-num_pixels:]
    depth_pixels = np.zeros((size), dtype=bool)
    depth_pixels[indices] = True
    depth_pixels = depth_pixels.reshape(depth_map.shape)

    # 最も深いN%のピクセルから大気光を計算
    A = np.zeros(3, dtype=np.float32)
    for i in range(3):
        A[i] = np.mean(img[:, :, i][depth_pixels])

    return A


def _estimate_transmission(depth_map, strength=0.5, lower_bound=0.1):
    """
    深度マップから透過率を推定

    depth_map: 深度マップ（0-1の範囲）
    strength: 霞除去の強さ（0-1の範囲）
    lower_bound: 透過率の最小値
    """
    # 深度マップから透過率を計算 (t = e^(-β*d))
    beta = 1.0 * strength  # 散乱係数を強さパラメータに関連付け
    transmission = np.exp(-beta * depth_map)

    # 下限値を設定
    transmission = np.maximum(transmission, lower_bound)

    return transmission


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_dehaze_apply(img, A, transmission):
    h, w, c = img.shape
    res = np.empty_like(img)
    for i in prange(h):
        for j in range(w):
            t = transmission[i, j]
            t_clamped = max(t, 0.1)
            for k in range(c):
                res[i, j, k] = (img[i, j, k] - A[k]) / t_clamped + A[k]
    return res


@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_fog_apply_2d(img, transmission_map):
    h, w, c = img.shape
    res = np.empty_like(img)
    for i in prange(h):
        for j in range(w):
            t = transmission_map[i, j]
            inv_t = 1.0 - t
            for k in range(c):
                # img * t + 1.0 * (1-t)
                res[i, j, k] = img[i, j, k] * t + inv_t
    return res


def _protect_dehaze_shadows(img, dehazed, strength):
    img_f32 = np.asarray(img, dtype=np.float32)
    dehazed_f32 = np.asarray(dehazed, dtype=np.float32)
    if strength <= 0:
        return dehazed_f32

    y = _cvt_color_rgb_to_gray(np.maximum(img_f32, 0.0))
    shadow_end = np.float32(0.10 + 0.20 * np.clip(float(strength), 0.0, 1.0))
    amount = _smoothstep(np.float32(0.005), shadow_end, y).astype(np.float32, copy=False)
    return (img_f32 + (dehazed_f32 - img_f32) * amount[..., np.newaxis]).astype(np.float32, copy=False)


def dehaze_image(img, strength=0.5):
    """
    色線形変換先行法を使用した霞除去・霧追加 (Numba Optimized)

    img: float32 RGB（線形プロファイル想定）。SDR/HDR のいずれでも可。
    strength: 霞除去（正の値）または霧追加（負の値）の強さ、-1〜1 の範囲
    """
    img_f32 = np.ascontiguousarray(img, dtype=np.float32)

    if strength >= 0:
        # 霞除去モード
        # 深度マップの推定
        depth_map = _estimate_depth_map(img_f32)
        # 大気光の推定
        A = _estimate_atmospheric_light(img_f32, depth_map)

        effective_strength = strength
        # 透過率の推定
        transmission = _estimate_transmission(depth_map, effective_strength)

        # 霞補正された画像の計算（大気散乱モデル）
        result = _kernel_dehaze_apply(img_f32, A, transmission)
        result = _protect_dehaze_shadows(img_f32, result, strength)
        # (img - A)/t + A は A より暗い画素で負値を生む。線形 RGB の負値は後段の
        # ガンマ/色空間変換を壊すためクランプする（ハイライト保持のため上限は設けない）。
        result = np.maximum(result, np.float32(0.0))

    else:
        # ===== ヘイズ追加処理（霞を増やす）=====
        # Simple Atmospheric Scattering Modelを使用

        haze_strength = -strength  # 強度を正の値に変換

        # 画像サイズを取得
        h, w = img_f32.shape[:2]

        # 強度に応じて透過量を滑らかに調整
        min_trans = 0.4  # 最小透過量（最大霞）

        # 二次関数で滑らかな遷移を作成
        transmission_value = 1.0 - (1.0 - min_trans) * (haze_strength * haze_strength)

        # 均一な透過量で霞を生成
        transmission = np.ones((h, w), dtype=np.float32) * transmission_value

        # 散乱モデルによる霞の合成
        result = _kernel_fog_apply_2d(img_f32, transmission)

    return result
