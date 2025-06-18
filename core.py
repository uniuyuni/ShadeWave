
import sys
import io
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import jit
from functools import partial
import colour
import lensfunpy
from scipy.interpolate import splprep, splev
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter
import logging
import math
from numba import njit, prange
from PIL import ImageCms

import sigmoid
import dng_sdk
import utils

jax.config.update("jax_platform_name", "METAL")


def normalize_image(image_data):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_image = (image_data - min_val) / (max_val - min_val)
    return normalized_image

def calculate_ev_from_image(image_data):
    #if image_data.size == 0:
    #    raise ValueError("画像データが空です。")

    average_value = np.mean(image_data)

    # ここで基準を明確に設定
    # 例えば、EV0が0.5に相当する場合
    ev = np.log2(0.5 / average_value)  # 0.5を基準

    return ev, average_value

def calculate_correction_value(ev_histogram, ev_setting, maxvalue=4):
    correction_value = ev_setting - ev_histogram
    
    # 補正値を適切にクリッピング
    if correction_value > maxvalue:  # 過剰補正を避ける
        correction_value = maxvalue
    elif correction_value < -maxvalue:  # 過剰補正を避ける
        correction_value = -maxvalue

    return correction_value

# RGBからグレイスケールへの変換
@jit
def cvtColorRGB2Gray(rgb):
    # 変換元画像 RGB
    #gry = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gry = jnp.dot(rgb, jnp.array([0.2989, 0.5870, 0.1140]))

    return gry


def convert_RGB2TempTint(rgb):

    xyz = colour.RGB_to_XYZ(rgb, 'sRGB')

    xy = colour.XYZ_to_xy(xyz)

    dng = dng_sdk.dng_temperature.DngTemperature()
    dng.set_xy_coord(xy)

    return float(dng.fTemperature), float(dng.fTint), float(xyz[1])

def __invert_temp_tint(temp, tint, ref_temp):

    # 色温度の反転
    mired_temp = 1e6 / temp
    mired_ref = 1e6 / ref_temp
    inverted_temp = 1e6 / (mired_ref - (mired_temp - mired_ref) + sys.float_info.min)

    # ティントの反転
    inverted_tint = -tint

    return inverted_temp, inverted_tint

def invert_RGB2TempTint(rgb, ref_temp=5000.0):
    temp, tint, Y = convert_RGB2TempTint(rgb)

    invert_temp, invert_tint = __invert_temp_tint(temp, tint, ref_temp)

    return (invert_temp, invert_tint, Y)


def convert_TempTint2RGB(temp, tint, Y):

    dng = dng_sdk.dng_temperature.DngTemperature()
    dng.fTemperature = temp
    dng.fTint = tint

    xy = dng.get_xy_coord()

    xyz = colour.xy_to_XYZ(xy)
    xyz *= Y

    rgb = colour.XYZ_to_RGB(xyz, 'sRGB')

    return rgb.astype(np.float32)

def invert_TempTint2RGB(temp, tint, Y, reference_temp=5000.0):

    inverted_temp, inverted_tint = __invert_temp_tint(temp, tint, reference_temp)
    
    # DNG SDKの関数を使用して元のRGB値を取得
    r, g, b = convert_TempTint2RGB(inverted_temp, inverted_tint, Y)

    return [r, g, b]

def calc_resize_image(original_size, max_length):
    width, height = original_size

    if width > height:
        # 幅が長辺の場合
        scale_factor = max_length / width
    else:
        # 高さが長辺の場合
        scale_factor = max_length / height

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    return (new_width, new_height)

def rotation(img, angle, flip_mode=0):
    # 元の画像の高さと幅を取得
    height = img.shape[0]
    width = img.shape[1]
    
    # 回転の中心点を計算
    center = (int(width/2), int(height/2))
    
    # 回転行列を計算（スケール付き）
    trans = cv2.getRotationMatrix2D(center, angle, 1)

    # 回転後画像サイズ    
    size = max(width, height)
    
    # 変換行列に平行移動を追加
    trans[0, 2] += (size / 2) - center[0]
    trans[1, 2] += (size / 2) - center[1]

    # フリップモードの処理
    # bit0が1なら左右反転、bit1が1なら上下反転
    img_affine = img
    if flip_mode & 1:  # 左右反転
        img_affine = cv2.flip(img_affine, 1)
    
    if flip_mode & 2:  # 上下反転
        img_affine = cv2.flip(img_affine, 0)

    # 回転と中心補正を同時に行う
    img_affine = cv2.warpAffine(img_affine, trans, (size, size), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0))

    return img_affine

@partial(jit, static_argnums=(0, 1, 2))
def __create_gaussian_kernel(width: int, height: int, sigma: float):
    # kernelの中心と1個横のセルの値の比
    if sigma == 0.0:
        sigma = 0.3*(max(width, height)/2 - 1) + 0.8

    r = jnp.exp(-1 / (2 * sigma**2))

    # kernelの中心index
    mw = width // 2
    mh = height // 2

    # kernelの中心を原点とした座標が入った配列
    xs = jnp.arange(-mw, mw + 1)
    ys = jnp.arange(-mh, mh + 1)
    x, y = jnp.meshgrid(xs, ys)

    # rを絶対値の2乗分累乗すると、ベースとなる配列が求まる
    g = r**(x**2 + y**2)

    # 正規化
    return g / np.sum(g)

@partial(jit, static_argnums=(1, 2))
def gaussian_blur_jax(src, ksize=(3, 3), sigma=0.0):
    kw, kh = ksize
    kernel = __create_gaussian_kernel(kw, kh, sigma)

    if src.ndim >= 3:
        dest = []
        for i in range(src.shape[2]):
            dest.append(jscipy.signal.convolve2d(src[:, :, i], kernel, mode='same'))

        dest = jnp.stack(dest, -1)
    else:
        dest = jscipy.signal.convolve2d(src, kernel, mode='same')

    return dest

def gaussian_blur_cv(src, ksize=(3, 3), sigma=0.0):
    if ksize == (0, 0):
        return src
    return  cv2.GaussianBlur(src, ksize, sigma)

def gaussian_blur(src, ksize=(3, 3), sigma=0.0):
    return gaussian_blur_jax(src, ksize, sigma)

@partial(jit, static_argnums=1)
def lucy_richardson_gauss(srcf, iteration):

    # 出力用の画像を初期化
    destf = srcf.copy()

    for i in range(iteration):
        # ガウスぼかしを適用してぼけをシミュレーション
        bdest = gaussian_blur_jax(destf, ksize=(9, 9), sigma=0)

        # 元画像とぼけた画像の比を計算
        bdest = bdest + jnp.finfo(jnp.float32).eps
        ratio = jnp.divide(srcf, bdest)

        # 誤差の分配のために再びガウスぼかしを適用
        ratio_blur = gaussian_blur_jax(ratio, ksize=(9, 9), sigma=0)

        # 元の出力画像に誤差を乗算
        destf = jnp.multiply(destf, ratio_blur)
    
    return destf

def _lucy_richardson_gauss(srcf, iteration):
    array = lucy_richardson_gauss(srcf, iteration)
    array.block_until_ready()

    return np.array(array)

@partial(jit, static_argnums=(1,))
def tone_mapping(x, exposure=1.0):
    # Reinhard トーンマッピング
    return x / (x + exposure)

def tone_mapping_sync(x, exposure=1.0):
    array = tone_mapping(x, exposure)
    array.block_until_ready()

    return np.array(array)

def tone_mapping_np(x, exposure=1.0):
    # Reinhard トーンマッピング
    return x / (x + exposure)

def tone_mapping_cv(x, exposure=1.0):

    return cv2.divide(x, cv2.add(x, exposure))

def highlight_compress(image):
    return cv2.createTonemapReinhard(
        gamma=1.0, 
        intensity=0.0,
        light_adapt=0.5, 
        color_adapt=0.5
    ).process(image)

def correct_overexposed_areas(image_rgb: np.ndarray,
                            threshold_low=0.94,
                            threshold_high=1.0,
                            correction_color=(0.94, 0.94, 0.96),
                            blur_sigma=15.0) -> np.ndarray:
    """
    float32形式のRGB画像（0-1）の白飛び部分を自然に補正する関数
    
    Parameters:
    -----------
    image_rgb : np.ndarray
        入力画像（float32形式、RGB、値域0-1）
        shape: (height, width, 3)
    threshold_low : float
        補正を開始する明るさの閾値（デフォルト: 0.94）
    threshold_high : float
        完全な白とみなす閾値（デフォルト: 1.0）
    correction_color : tuple
        補正に使用する色（デフォルト: わずかに青みがかった白）
    blur_sigma : float
        ガウシアンブラーのシグマ値（デフォルト: 15.0）
        大きい値ほど広い範囲でブレンドされる
    """
    # 各チャンネルの輝度に基づいてマスクを作成
    # 各ピクセルの最小値を使用することで、より慎重に白領域を検出
    pixel_brightness = np.min(image_rgb, axis=2)
    
    # グラデーショナルなマスクを作成
    correction_mask = np.clip((pixel_brightness - threshold_low) / 
                            (threshold_high - threshold_low), 0, 1)
    
    # マスクを大きめのシグマ値でぼかす
    correction_mask = gaussian_filter(correction_mask, sigma=blur_sigma)
    
    # さらになめらかな補正のため、マスクを累乗して非線形に
    correction_mask = correction_mask ** 1.5  # 補正の急激な変化を抑える
    
    # 補正色のマップを作成
    correction = np.zeros_like(image_rgb, dtype=np.float32)
    for i in range(3):
        correction[:,:,i] = correction_color[i]
    
    # 補正を適用
    correction_mask = np.expand_dims(correction_mask, axis=2)
    
    # 元の画像と補正色をブレンド
    result = image_rgb * (1 - correction_mask) + correction * correction_mask
    
    return result

def apply_solid_color(image_rgb: np.ndarray, solid_color=(0.94, 0.94, 0.96)) -> np.ndarray:
    """
    float32形式のRGB画像（0-1）の白飛び部分を自然に補正する関数
    
    Parameters:
    -----------
    image_rgb : np.ndarray
        入力画像（float32形式、RGB、値域0-1）
        shape: (height, width, 3)
    solod_color : tuple
        補正に使用する色（デフォルト: わずかに青みがかった白）
    """
    # 補正色のマップを作成
    correction = np.zeros_like(image_rgb, dtype=np.float32)
    for i in range(3):
        correction[:,:,i] = solid_color[i]
        
    # 元の画像と補正色をブレンド
    result = correction
    
    return result

# ローパスフィルタ
def lowpass_filter(img, r):
    lpf = gaussian_blur(img, ksize=(r, r), sigma=0.0)

    return lpf

# ハイパスフィルタ
def highpass_filter(img, r):
    hpf = img - gaussian_blur(img, ksize=(r, r), sigma=0.0)+0.5

    return hpf    

# オーバーレイ合成
def blend_overlay(base, over):
    result = np.zeros(base.shape, dtype=np.float32)
    darker = base < 0.5
    base_inv = 1.0-base
    over_inv = 1.0-over
    result[darker] = base[darker] * over[darker] * 2
    #result[~darker] = (base[~darker]+over[~darker] - base[~darker]*over[~darker])*2-1
    result[~darker] = 1 - base_inv[~darker] * over_inv[~darker] * 2
    
    return result

# スクリーン合成
def blend_screen(base, over):
    result = 1 - (1.0-base)*(1-over)

    return result

def log_transform(x, base=np.e):
    """
    対数関数を適用する。x は0以上の値。
    出力は0-1の範囲に正規化されることを前提。
    """
    # x が0-1の範囲なら、出力も0-1に正規化
    # max_val = 1.0 (入力の最大値)
    return np.log(1 + x) / np.log(1 + 1.0)


# 露出補正
def adjust_exposure(rgb, ev):
    # img: 変換元画像
    # ev: 補正値 -4.0〜4.0

    return rgb * (2.0**ev)

# コントラスト補正
#@partial(jit, static_argnums=(1,2,))
def adjust_contrast(img, cf, c=0.5):
    # img: 変換元画像
    # cf: コントラストファクター -100.0〜100.0
    # c: 中心値 0〜1.0
    
    f = cf / 100.0 * 10.0  #-10.0〜10.0に変換

    if f == 0.0:
        adjust_img = img.copy()
    elif f >= 0.0:
        mm = max(1.0, np.max(img))
        adjust_img = sigmoid.scaled_sigmoid(img/mm, f, c/mm)*mm
    else:
        mm = max(1.0, np.max(img))
        adjust_img = sigmoid.scaled_inverse_sigmoid(img/mm, -f, c/mm)*mm
        
    return adjust_img
    """
    adjust = adjust_tone(img, cf, -cf)

    return adjust
    """

# レベル補正
@partial(jit, static_argnums=(1,2,3,))
def apply_level_adjustment(image, black_level, midtone_level, white_level):
    # image: 変換元イメージ
    # black_level: 黒レベル 0〜255
    # white_level: 白レベル 0〜255
    # midtone_level: 中間色レベル 0〜255

    # 16ビット画像の最大値
    max_val = 65535
    black_level *= 256
    white_level *= 256
    midtone_level *= 256

    # midtone_level を 1.0 を基準としてスケーリング (128が基準のため)
    midtone_factor = midtone_level / 32768.0

    # ルックアップテーブル (LUT) の作成 (0〜65535)
    lut = jnp.linspace(0, max_val, max_val + 1, dtype=np.float32)  # Liner space creation

    # Pre-calculate constants
    range_inv = 1.0 / (white_level - black_level)
    
    # LUT のスケーリングとクリッピング
    lut = jnp.clip((lut - black_level) * range_inv, 0, 1)  # Scale and clip
    lut = jnp.power(lut, midtone_factor) * max_val  # Apply midtone factor and scale
    lut = jnp.clip(lut, 0, max_val).astype(jnp.uint16)  # Final clip and type conversion
    
    # 画像全体にルックアップテーブルを適用
    adjusted_image = lut[jnp.clip(image*max_val, 0, max_val).astype(jnp.uint16)]
    adjusted_image = adjusted_image/max_val

    return adjusted_image

# 彩度補正と自然な彩度補正
@partial(jit, static_argnums=(1, 2))
def calc_saturation(s, sat, vib):

    # 彩度変更値と自然な彩度変更値を計算
    if sat >= 0:
        sat = 1.0 + sat/100.0
    else:
        sat = 1.0 + sat/100.0
    vib /= 50.0

    # 計算の破綻を防止（元データは壊さない）
    s = jnp.clip(s, 0.0, 1.0)

    # 自然な彩度調整
    if vib == 0.0:
        final_s = s

    elif vib > 0.0:
        # 通常の計算
        vib = vib**2
        final_s = jnp.log(1.0 + vib * s) / jnp.log(1.0 + vib)
    else:
        # 逆関数を使用
        vib = vib**2
        final_s = (jnp.exp(s * jnp.log(1.0 + vib)) - 1.0) / vib

    # 彩度を適用
    final_s = final_s * sat

    return final_s

def _calc_saturation(s, sat, vib):
    array = calc_saturation(s, sat, vib)
    array.block_until_ready()

    return np.array(array)


def calc_point_list_to_lut(point_list, max_value=1.0):
    """
    スプライン補間を使った基本的なLUT生成関数
    
    Parameters:
    -----------
    point_list : list of tuples
        (x, y)形式のコントロールポイントのリスト
    max_value : float
        LUTが対応する最大値（デフォルト1.0）
        
    Returns:
    --------
    ndarray
        65536エントリーのLUT
    """
    # ポイントをソート
    point_list = sorted((pl[0], pl[1]) for pl in point_list)
    
    # ポイントからx, y配列を取得
    x, y = zip(*point_list)
    x, y = np.array(x), np.array(y)
    
    # 3点以上ある場合はスプライン補間を使用
    if len(x) >= 3:
        # スプライン補間のパラメータ（次数は点の数-1か3の小さい方）
        k = min(3, len(x) - 1)
        # スプライン補間の計算
        tck, u = splprep([x, y], k=k, s=0)
        
        # [0, 1]の範囲で細かい点を生成
        fine_u = np.linspace(0, 1, 1000)
        fine_points = splev(fine_u, tck)
        
        # 生成された点を取得
        fine_x, fine_y = fine_points
        
        # この点を使って通常の線形補間でLUTを生成
        lut_size = 65536
        input_range = np.linspace(0, max_value, lut_size)
        
        # max_valueを超える部分は直線で外挿
        mask_in_range = input_range <= max(fine_x)
        lut = np.zeros(lut_size, dtype=np.float32)
        
        # 範囲内は補間
        lut[mask_in_range] = np.interp(
            input_range[mask_in_range], 
            fine_x, 
            fine_y
        )
        
        # 範囲外は直線外挿（最後の2点から傾きを計算）
        if np.any(~mask_in_range):
            # 最後の2点から傾きを計算
            last_idx = len(fine_x) - 1
            second_last_idx = last_idx - 1
            slope = (fine_y[last_idx] - fine_y[second_last_idx]) / (fine_x[last_idx] - fine_x[second_last_idx])
            
            # 直線外挿
            x_out = input_range[~mask_in_range]
            y_last = fine_y[last_idx]
            x_last = fine_x[last_idx]
            lut[~mask_in_range] = y_last + slope * (x_out - x_last)
    else:
        # 点が少ない場合は単純な線形補間
        lut_size = 65536
        input_range = np.linspace(0, max_value, lut_size)
        lut = np.interp(input_range, x, y).astype(np.float32)
    
    return lut

@partial(jit, static_argnums=(2,))
def apply_lut(img, lut, max_value=1.0):
    """
    画像にLUTを適用する関数
    max_value: LUTが対応する最大値（デフォルト1.0）
    """
    # スケーリングしてLUTのインデックスに変換
    scale_factor = 65535 / max_value
    lut_indices = jnp.clip(jnp.round(img * scale_factor), 0, 65535).astype(jnp.uint16)

    # LUTを適用
    result = jnp.take(lut, lut_indices)
    
    return result

def apply_mask(img1, msk, img2):
    return apply_mask_numba(img1, msk, img2)

# マスクイメージの適用
@jit
def apply_mask_jax(img1, msk, img2):
    _msk = msk[:, :, jnp.newaxis]
    img = img1 * (1.0 - _msk) + img2 * _msk

    return img

def apply_mask_np(img1, msk, img2):
    _msk = msk[:, :, np.newaxis]
    img = img1 * (1.0 - _msk) + img2 * _msk

    return img

def apply_mask_cv(img1, msk, img2):
    _msk = cv2.merge((msk, msk, msk))

    b = cv2.multiply(img1, cv2.subtract(1.0, _msk))
    f = cv2.multiply(img2, _msk)

    # Add the masked foreground and background.
    img = cv2.add(f, b)

    return img

@njit(parallel=True)
def apply_mask_numba(img1, msk, img2):
    """RGB（3チャンネル）専用の最適化版"""
    h, w = msk.shape
    result = np.empty((h, w, 3), dtype=np.float32)
    
    for i in prange(h):
        for j in prange(w):
            mask_val = msk[i, j]
            inv_mask = 1.0 - mask_val
            result[i, j, 0] = img1[i, j, 0] * inv_mask + img2[i, j, 0] * mask_val
            result[i, j, 1] = img1[i, j, 1] * inv_mask + img2[i, j, 1] * mask_val
            result[i, j, 2] = img1[i, j, 2] * inv_mask + img2[i, j, 2] * mask_val
    
    return result

@partial(jit, static_argnums=(1,2,5))
def apply_vignette(image, intensity, radius_percent, disp_info, crop_rect, offset, gradient_softness=4.0):
    """
    修正版 周辺光量落ち効果
    - 中心位置が正確にクロップ中心に一致
    - 効果の向きが正しく適用（負の値で周辺暗く、正の値で周辺明るく）
    - 滑らかなグラデーション
    - scaleを適切に考慮した効果適用
    - 元画像の座標系でのビネット中心を正確に反映
    
    Parameters:
        image: 入力画像 (float32, 0-1)
        intensity: 効果の強さ (-100 to 100)
        radius_percent: 効果の半径 (1-100%)
        disp_info: [x, y, w, h, scale] - 元画像における切り抜き情報
        gradient_softness: グラデーションの滑らかさ
    """

    intensity = intensity / 100.0
    radius_percent = radius_percent / 100.0
    gradient_softness = max(0.1, gradient_softness)
    
    h, w = image.shape[:2]
    
    if crop_rect is None:
        # クロップ情報がない場合は従来通り
        center_x, center_y = w/2, h/2

        mm = jax.lax.max(w, h)
        max_radius = jax.lax.sqrt(mm**2 + mm**2) / 2
    else:        
        dx, dy, _, _, scale = disp_info
        x1, y1, x2, y2 = crop_rect
        offset_x, offset_y = offset
            
        # クロップ画像内での元画像中心の位置
        center_x = (x1 + (x2 - x1) / 2 - dx) * scale + offset_x
        center_y = (y1 + (y2 - y1) / 2 - dy) * scale + offset_y
        
        mm = jax.lax.max((x2 - x1), (y2 - y1)) * scale.astype(jnp.float32)
        max_radius = jax.lax.sqrt(mm**2 + mm**2) / 2
    
    # 指定された半径パーセントに基づいて実際の半径を計算
    radius = max_radius * radius_percent
    
    # 距離マップ作成
    y_indices, x_indices = jnp.ogrid[:h, :w]
    dist = jnp.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    def smoothstep(x):
        return x * x * (3 - 2 * x)  # 3次多項式

    # マスク作成（0が中心、1が端）
    mask = jnp.clip(dist / radius, 0, 1)
    #mask = gaussian_blur(mask, (64, 64), 0)
    #mask = jnp.power(mask, gradient_softness)  # グラデーション調整
    mask = smoothstep(mask)
    
    # 効果適用（intensityの符号で方向を制御）
    vignette = jnp.where(intensity < 0, 1.0 + intensity * mask, 1.0 - intensity * mask)
    
    # カラー画像対応
    if len(image.shape) == 3:
        vignette = vignette[..., jnp.newaxis]
    
    # 効果適用
    result = jnp.clip(image * vignette, 0, 1) if intensity < 0 else jnp.clip(image + (1-image)*(1-vignette), 0, 1)
    return result.astype(jnp.float32)

# テクスチャサイズとクロップ情報から、新しい描画サイズと余白の大きさを得る
def crop_size_and_offset_from_texture(texture_width, texture_height, disp_info):

    # アスペクト比を計算
    crop_aspect = disp_info[2] / disp_info[3]
    texture_aspect = texture_width / texture_height

    if crop_aspect > texture_aspect:
        # 画像が横長の場合
        new_width = texture_width
        new_height = int(texture_width / crop_aspect)
    else:
        # 画像が縦長の場合
        new_width = int(texture_height * crop_aspect)
        new_height = texture_height

    # 中央に配置するためのオフセットを計算
    offset_x = (texture_width - new_width) // 2
    offset_y = (texture_height - new_height) // 2

    return (new_width, new_height, offset_x, offset_y)

def crop_image(image, disp_info, crop_rect, texture_width, texture_height, click_x, click_y, offset, is_zoomed):

    # 画像のサイズを取得
    image_height, image_width = image.shape[:2]

    new_width, new_height, offset_x, offset_y = crop_size_and_offset_from_texture(texture_width, texture_height, disp_info)

    # スケールを求める
    if disp_info[2] >= disp_info[3]:
        scale = texture_width/disp_info[2]
    else:
        scale = texture_height/disp_info[3]

    if not is_zoomed:
        # リサイズ
        dx, dy, dw, dh, _ = disp_info
        resized_img = cv2.resize(image[dy:dy+dh, dx:dx+dw], (new_width, new_height), interpolation=cv2.INTER_LINEAR_EXACT)

        # リサイズした画像を中央に配置
        result = np.pad(resized_img, ((offset_y, texture_height-(offset_y+new_height)), (offset_x, texture_width-(offset_x+new_width)), (0, 0)), constant_values=0)

        # 再設定
        disp_info = (dx, dy, dw, dh, scale)

    else:
        # クリック位置を元の画像の座標系に変換
        click_x = click_x - offset_x
        click_y = click_y - offset_y
        click_image_x = click_x / scale
        click_image_y = click_y / scale

        # 切り抜き範囲を計算
        crop_width = int(texture_width)
        crop_height = int(texture_height)

        if offset == (0, 0):
            # クリック位置を中心にする
            crop_x = disp_info[0] + click_image_x - crop_width // 2
            crop_y = disp_info[1] + click_image_y - crop_height // 2
        else:
            # スクロール
            crop_x = disp_info[0]
            crop_y = disp_info[1]

        # クロップ
        result, disp_info = crop_image_info(image, (crop_x, crop_y, crop_width, crop_height, 1.0), crop_rect, offset)
    
    return result, disp_info


def crop_image_info(image, disp_info, crop_rect, offset=(0, 0)):
    
    # 情報取得
    image_height, image_width = image.shape[:2]
    disp_x, disp_y, disp_width, disp_height, scale = disp_info

    # オフセット適用
    x = int(disp_x + offset[0])
    y = int(disp_y + offset[1])

    # 画像の範囲外にならないように調整
    #x = int(max(0, min(x, image_width - disp_width)))
    #y = int(max(0, min(y, image_height - disp_height)))
    x = int(max(crop_rect[0], min(x, crop_rect[2] - disp_width)))
    y = int(max(crop_rect[1], min(y, crop_rect[3] - disp_height)))

    # 画像を切り抜く
    #cropped_img = jax.lax.slice(image, (disp_y, disp_x, 0), (disp_y+disp_height, disp_x+disp_width, 3))
    cropped_img = image[y:y+disp_height, x:x+disp_width]

    return cropped_img, (x, y, disp_width, disp_height, scale)


def get_multiple_mask_bbox(mask):
    """
    マスク画像から複数の独立した領域それぞれのバウンディングボックスを計算する
    
    Args:
        mask マスク画像（2次元のnumpy配列）
        
    Returns:
        各領域の(x_min, y_min, x_max, y_max)座標のリスト
            空のマスクの場合は空リストを返す
    """
    # マスクが空かチェック
    if not np.any(mask > 0):
        return []
    
    # 連結成分のラベリングを実行
    labeled_array, num_features = label(mask > 0)
    
    bboxes = []
    # 各ラベルについてバウンディングボックスを計算
    for label_id in range(1, num_features + 1):
        # 現在のラベルのマスクを作成
        current_mask = labeled_array == label_id
        
        # 行と列それぞれについて、マスクが存在する座標を取得
        rows = np.any(current_mask, axis=1)
        cols = np.any(current_mask, axis=0)
        
        # 最小と最大の座標を取得
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        bboxes.append((int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)))
    
    return bboxes


# 上下または左右の余白を追加
def adjust_shape(img):
    imax = max(img.shape[1], img.shape[0])

    # イメージを正方形にする
    offset_y = (imax-img.shape[0])//2
    offset_x = (imax-img.shape[1])//2
    img = np.pad(img, ((offset_y, imax-(offset_y+img.shape[0])), (offset_x, imax-(offset_x+img.shape[1])), (0, 0)), constant_values=0)

    return img


@partial(jit, static_argnums=(1,2,3,4,5,))
def adjust_tone(img, highlights=0, shadows=0, midtone=0, white_level=0, black_level=0):
    """
    Lightroom風のシャドウ、ハイライト、白レベル、黒レベル調整を行う関数。
    
    Parameters:
        img (np.ndarray): 入力画像 (float32, RGB)
        shadows (float): シャドウの調整 (-100～100)
        highlights (float): ハイライトの調整 (-100～100)
        midtone (float): 中間調の調整 (-100～100)
        white_level (float): 白レベルの調整 (-100～100)
        black_level (float): 黒レベルの調整 (-100～100)
    
    Returns:
        np.ndarray: 調整後の画像（クリッピングなし）
        mask(tuple): 効果マスクリスト（Noneあり）
    """

    def _conditional_operation(x, img, pos_func, neg_func):
        func = pos_func if x > 0 else neg_func if x < 0 else lambda img, x: (img, None)
        
        return func(img, x)

    # 中間調の調整
    def enhance_midtones_positive(img, midtone):
        C = midtone / 100 * midtone_scale
        return jnp.log(1 + img * C) / jax.lax.log(1 + C), None

    def enhance_midtones_negative(img, midtone):
        C = -midtone / 100 * midtone_scale

        # 通常の範囲(0〜1)の計算
        normal_result = (jnp.exp(img * jax.lax.log(1 + C)) - 1) / C
        
        # 1.0の時の関数値と傾きを計算
        f_1 = ((1 + C) - 1) / C  # (np.exp(1.0 * math.log(1 + C)) - 1) / C を簡略化
        derivative_at_1 = (1 + C) * jax.lax.log(1 + C) / C
        
        # 1.0超の範囲 - 線形拡張する
        extended_result = f_1 + derivative_at_1 * (img - 1.0)
        
        # 条件マスクを使って結果を組み合わせる
        return jnp.where(img <= 1.0, normal_result, extended_result), None

    midtone_scale = 4
    img, _ = _conditional_operation(midtone, img, enhance_midtones_positive, enhance_midtones_negative)

    # シャドウ（暗部）の調整
    def enhance_shadow_positive(img, shadows):
        factor = shadows / 100
        influence = jnp.exp(-5 * img)
        mask = factor * influence
        return img * (1 + mask), mask
    
    def enhance_shadow_negative(img, shadows):
        factor = shadows / 100
        influence = jnp.exp(-5 * img)
        min_val = img * 0.1;  # 最小でも元の値の10%は維持
        mask = (1 + factor * influence)
        raw_result = img * mask
        return jnp.maximum(raw_result, min_val), None

    img, shadows_mask = _conditional_operation(shadows, img, enhance_shadow_positive, enhance_shadow_negative)
    
    # ハイライト（明部）の調整
    def enhance_highlight_positive(img, highlights):
        factor = highlights / 100 * highlight_scale
        max_val = jnp.max(img)
        base = img / max_val  # 0-1に正規化
        expansion = 1 + factor * (jnp.log1p(jnp.log1p(base)) / jax.lax.log1p(jax.lax.log1p(jax.lax.max(max_val, 2.0))))
        return img * expansion, None

    def enhance_highlight_negative(img, highlights):
        factor = -highlights / 100
        max_val = jnp.max(img)
        target = jnp.log1p(jnp.log1p(img)) / jax.lax.log1p(jax.lax.log1p(jax.lax.max(max_val, 2.0)))
        mask = target * factor
        return img * (1-factor) + mask, mask

    highlight_scale = 4
    img, highlight_mask = _conditional_operation(highlights, img, enhance_highlight_positive, enhance_highlight_negative)

    """
    # 黒レベル（全体の暗い部分の引き下げ）
    def enhance_black_positive(img, black_level):
        factor = black_level / 100
        influence = jnp.exp(black_level_const * img)
        return img * (1 + factor * influence)
    
    def enhance_black_negative(img, black_level):
        factor = black_level / 100
        influence = jnp.exp(black_level_const * img)
        min_val = img * 0.05;  # 最小でも元の値の5%は維持
        raw_result = img * (1 + factor * influence)
        return jnp.maximum(raw_result, min_val)

    black_level_const = -1
    img = _conditional_operation(black_level, img, enhance_black_positive, enhance_black_negative)

    # 白レベル（全体の明るい部分の引き上げ）
    def enhance_white_positive(img, white_level):
        factor = white_level / 100 * white_level_scale
        max_val = jnp.max(img)
        base = img / max_val  # 0-1に正規化
        expansion = 1 + factor * (jnp.log1p(jnp.log1p(jnp.log1p(base))) / jax.lax.log1p(jax.lax.log1p(jax.lax.log1p(jax.lax.max(max_val, 2.0)))))
        return img * expansion
    
    def enhance_white_negative(img, white_level):
        factor = -white_level / 100
        max_val = jnp.max(img)
        target = jnp.log1p(jnp.log1p(jnp.log1p(img))) / jax.lax.log1p(jax.lax.log1p(jax.lax.log1p(jax.lax.max(max_val, 2.0))))
        return img * (1-factor) + target * factor

    white_level_scale = 4
    img = _conditional_operation(white_level, img, enhance_white_positive, enhance_white_negative)
    """

    return img, (shadows_mask, highlight_mask)


def get_exif_image_size(exif_data):
    top, left = exif_data.get("RawImageCropTopLeft", "0 0").split()
    top, left = int(top), int(left)

    width, height = exif_data.get("RawImageCroppedSize", "0x0").split('x')
    width, height = int(width), int(height)
    if width == 0 and height == 0:
        width, height = exif_data.get("ImageSize", "0x0").split('x')
        width, height = int(width), int(height)
        if width == 0 and height == 0:
            raise AttributeError("Not Find image size data")
        
    return (top, left, width, height)

def set_exif_image_size(exif_data, top, left, width, height):
    setflag = False
    
    if exif_data.get("RawImageCropTopLeft", None) is not None:
        exif_data["RawImageCropTopLeft"] = str(top) + " " + str(left)

    if exif_data.get("RawImageCroppedSize", None) is not None:
        exif_data["RawImageCroppedSize"] = str(width) + "x" + str(height)
        setflag = True

    if setflag == False:
        exif_data["ImageSize"] = str(width) + "x" + str(height)
    
def get_exif_image_size_with_orientation(exif_data):
        # クロップとexifデータの回転
        top, left, width, height = get_exif_image_size(exif_data)
        if "Orientation" in exif_data:
            rad, flip = utils.split_orientation(utils.str_to_orientation(exif_data.get("Orientation", "")))
            if rad < 0.0:
                top, left = left, top
                width, height = height, width

        return (top, left, width, height)

def _estimate_depth_map(img, params=(0.121779, 0.959710, -0.780245), sigma=0.5):
    """
    色線形変換先行法（Color Attenuation Prior）を使用して深度マップを推定
    
    img: 入力画像（0-1の範囲のfloat32、BGR形式）
    params: 線形モデルの係数 (β0, β1, β2)
    sigma: ガウシアンフィルタのシグマ値
    
    Zhu らの論文 "Fast Single Image Haze Removal Using Color Attenuation Prior" に基づく
    """
    # RGB画像をHSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # 彩度と明度の差分を計算
    diff = s - v
    
    # 線形モデルを使用して深度を推定: d = β0 + β1 * v + β2 * s
    beta0, beta1, beta2 = params
    depth = beta0 + beta1 * v + beta2 * s
    
    # フィルタリングで深度マップを滑らかにする
    depth = cv2.GaussianBlur(depth, (0, 0), sigma)
    
    # 正規化（0-1の範囲に変換）
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)
    
    return depth

def _estimate_atmospheric_light(img, depth_map, top_percent=0.001):
    """
    大気光を推定（最も深い点の上位N%を使用）
    
    img: 入力画像（BGR形式）
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
        A[i] = np.mean(img[:,:,i][depth_pixels])
    
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


def dehaze_image(img, strength=0.5):
    """
    色線形変換先行法を使用した霞除去・霧追加
    
    img: 入力画像（0-1の範囲のfloat32、BGR形式）
    strength: 霞除去（正の値）または霧追加（負の値）の強さ、-1から1の範囲
    """
    
    # 深度マップの推定
    depth_map = _estimate_depth_map(img)
    
    # 大気光の推定
    A = _estimate_atmospheric_light(img, depth_map)
    
    # 強さに基づいて調整
    if strength >= 0:
        # 霞除去モード
        effective_strength = strength
        # 透過率の推定
        transmission = _estimate_transmission(depth_map, effective_strength)

        # 霞補正された画像の計算（大気散乱モデル）
        result = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            result[:,:,i] = (img[:,:,i] - A[i]) / np.maximum(transmission, 0.1) + A[i]
    
    else:
        # ===== ヘイズ追加処理（霞を増やす）=====
        # Simple Atmospheric Scattering Modelを使用
        
        haze_strength = -strength  # 強度を正の値に変換
        
        # 大気光の色（純粋な白）
        atmospheric_light = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # RGB形式で白
        
        # 画像サイズを取得
        h, w = img.shape[:2]
        
        # 強度に応じて透過量を滑らかに調整
        # 強度0では透過量1.0（霞なし）
        # 強度-1では最小透過量（最大霞）
        # 二次関数的なイージングを適用して滑らかな遷移を実現
        min_trans = 0.4  # 最小透過量（最大霞）
        
        # 二次関数で滑らかな遷移を作成
        # haze_strength=0→transmission=1.0（元画像）
        # haze_strength=1→transmission=min_trans（最大霞）
        transmission_value = 1.0 - (1.0 - min_trans) * (haze_strength * haze_strength)
        
        # 均一な透過量で霞を生成
        transmission = np.ones((h, w), dtype=np.float32) * transmission_value
        
        # 透過量マップを3チャンネルに拡張
        transmission = np.stack([transmission] * 3, axis=2)
        
        # 散乱モデルによる霞の合成
        result = img * transmission + atmospheric_light * (1 - transmission)

    return result


def bicubic_kernel_vectorized(x, a=-0.5):
    """
    バイキュービック補間カーネル関数（ベクトル化版）
    
    Parameters:
    -----------
    x : numpy.ndarray
        距離値
    a : float
        バイキュービックパラメータ (デフォルト: -0.5)
        
    Returns:
    --------
    numpy.ndarray
        カーネルの重み
    """
    x = np.abs(x)
    result = np.zeros_like(x, dtype=np.float32)
    
    # |x| <= 1 の場合
    mask1 = x <= 1
    result[mask1] = ((a + 2) * x[mask1]**3 - (a + 3) * x[mask1]**2 + 1)
    
    # 1 < |x| < 2 の場合
    mask2 = np.logical_and(x > 1, x < 2)
    result[mask2] = (a * x[mask2]**3 - 5 * a * x[mask2]**2 + 8 * a * x[mask2] - 4 * a)
    
    return result

def resize_bicubic_vectorized(image, target_height, target_width):
    """
    ベクトル化されたバイキュービック補間による画像リサイズ
    
    Parameters:
    -----------
    image : numpy.ndarray
        入力画像、float32型のNumPy配列 (height, width) または (height, width, channels)
    target_height : int
        目標の高さ
    target_width : int
        目標の幅
        
    Returns:
    --------
    numpy.ndarray
        リサイズされた画像、float32型のNumPy配列
    """
    # float32型に変換
    image = image.astype(np.float32)
    
    # 画像の寸法を取得
    original_height, original_width = image.shape[:2]
    
    # グレースケール画像の処理
    is_grayscale = len(image.shape) == 2
    if is_grayscale:
        # 処理を統一するためチャンネル次元を追加
        image = image[:, :, np.newaxis]
    
    channels = image.shape[2]
    
    # スケーリング係数の計算
    if target_height > 1:
        scale_y = float(original_height - 1) / (target_height - 1)
    else:
        scale_y = 0.0
        
    if target_width > 1:
        scale_x = float(original_width - 1) / (target_width - 1)
    else:
        scale_x = 0.0
    
    # 出力ピクセル座標のグリッドを作成
    y_coords = np.arange(target_height, dtype=np.float32)
    x_coords = np.arange(target_width, dtype=np.float32)
    
    # 元画像の対応座標を計算
    y_coords = y_coords * scale_y
    x_coords = x_coords * scale_x
    
    # 整数部分と小数部分に分解
    y_floor = np.floor(y_coords).astype(np.int32)
    x_floor = np.floor(x_coords).astype(np.int32)
    
    y_frac = y_coords - y_floor
    x_frac = x_coords - x_floor
    
    # 範囲外アクセスを防ぐためのインデックス調整
    y_indices = np.zeros((target_height, 4), dtype=np.int32)
    x_indices = np.zeros((target_width, 4), dtype=np.int32)
    
    for i in range(-1, 3):
        y_idx = np.clip(y_floor + i, 0, original_height - 1)
        x_idx = np.clip(x_floor + i, 0, original_width - 1)
        y_indices[:, i+1] = y_idx
        x_indices[:, i+1] = x_idx
    
    # バイキュービックカーネルの重みを計算
    y_weights = np.zeros((target_height, 4), dtype=np.float32)
    x_weights = np.zeros((target_width, 4), dtype=np.float32)
    
    for i in range(-1, 3):
        y_dist = y_frac[:, np.newaxis] - i
        x_dist = x_frac[:, np.newaxis] - i
        y_weights[:, i+1] = bicubic_kernel_vectorized(y_dist).reshape(-1)
        x_weights[:, i+1] = bicubic_kernel_vectorized(x_dist).reshape(-1)
    
    # 出力画像の初期化
    output = np.zeros((target_height, target_width, channels), dtype=np.float32)
    
    # バイキュービック補間を適用
    for c in range(channels):
        for i in range(4):
            for j in range(4):
                # 元画像から必要な位置のピクセルを取得
                pixel_values = image[y_indices[:, i][:, np.newaxis], x_indices[:, j][np.newaxis, :], c]
                
                # 重みの行列積を計算
                weights = y_weights[:, i][:, np.newaxis] * x_weights[:, j][np.newaxis, :]
                
                # 重み付きピクセル値を出力に加算
                output[:, :, c] += pixel_values * weights
    
    # 元がグレースケールの場合、チャンネル次元を削除
    if is_grayscale:
        output = output[:, :, 0]
    
    return output

def resize_bicubic_fully_vectorized(image, target_height, target_width):
    """
    完全ベクトル化されたバイキュービック補間による画像リサイズ
    大きな画像でのメモリ使用量に注意
    
    Parameters:
    -----------
    image : numpy.ndarray
        入力画像、float32型のNumPy配列 (height, width) または (height, width, channels)
    target_height : int
        目標の高さ
    target_width : int
        目標の幅
        
    Returns:
    --------
    numpy.ndarray
        リサイズされた画像、float32型のNumPy配列
    """
    # float32型に変換
    image = image.astype(np.float32)
    
    # 画像の寸法を取得
    original_height, original_width = image.shape[:2]
    
    # グレースケール画像の処理
    is_grayscale = len(image.shape) == 2
    if is_grayscale:
        # 処理を統一するためチャンネル次元を追加
        image = image[:, :, np.newaxis]
    
    channels = image.shape[2]
    
    # スケーリング係数の計算
    if target_height > 1:
        scale_y = float(original_height - 1) / (target_height - 1)
    else:
        scale_y = 0.0
        
    if target_width > 1:
        scale_x = float(original_width - 1) / (target_width - 1)
    else:
        scale_x = 0.0
    
    # 出力ピクセル座標のグリッドを作成
    y_grid, x_grid = np.meshgrid(np.arange(target_height, dtype=np.float32),
                                 np.arange(target_width, dtype=np.float32),
                                 indexing='ij')
    
    # 元画像の対応座標を計算
    y_original = y_grid * scale_y
    x_original = x_grid * scale_x
    
    # 整数部分と小数部分に分解
    y_floor = np.floor(y_original).astype(np.int32)
    x_floor = np.floor(x_original).astype(np.int32)
    
    y_frac = y_original - y_floor
    x_frac = x_original - x_floor
    
    # 出力画像の初期化
    output = np.zeros((target_height, target_width, channels), dtype=np.float32)
    
    # 各チャンネルに対して補間を計算
    for c in range(channels):
        channel_sum = np.zeros((target_height, target_width), dtype=np.float32)
        
        # 16個の近傍点に対して重みを計算し適用
        for i in range(-1, 3):
            # y方向のインデックスとカーネル重み
            y_idx = np.clip(y_floor + i, 0, original_height - 1)
            y_dist = np.abs(y_frac - i)
            y_weight = bicubic_kernel_vectorized(y_dist)
            
            for j in range(-1, 3):
                # x方向のインデックスとカーネル重み
                x_idx = np.clip(x_floor + j, 0, original_width - 1)
                x_dist = np.abs(x_frac - j)
                x_weight = bicubic_kernel_vectorized(x_dist)
                
                # 重みの計算（ブロードキャスト）
                weight = y_weight * x_weight
                
                # 対応するピクセル値を取得
                pixel_value = image[y_idx, x_idx, c]
                
                # 重み付きピクセル値を加算
                channel_sum += pixel_value * weight
        
        output[:, :, c] = channel_sum
    
    # 元がグレースケールの場合、チャンネル次元を削除
    if is_grayscale:
        output = output[:, :, 0]
    
    return output

#@partial(jit, static_argnums=(1,2,))
def smooth_step(x, edge0, edge1):
    """
    エルミート補間を用いた滑らかなステップ関数
    x が edge0 未満なら0、edge1 以上なら1、その間は滑らかな補間を行う
    """
    # クランプ
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    # エルミート補間
    return t * t * (3.0 - 2.0 * t)

#@partial(jit, static_argnums=(1,2,3))
def circular_smooth_step(hue, center, width, fade_width):
    """
    円環上の色相空間で滑らかな重みを計算する
    
    Args:
        hue: 入力色相 (0-360)
        center: 中心色相
        width: 完全に適用する幅 (半分)
        fade_width: フェードする幅
    
    Returns:
        0-1の重み
    """
    # 色相の円環性を考慮して距離を計算
    dist = np.abs((((hue - center) % 360) + 180) % 360 - 180)
    
    # 完全適用領域なら1.0
    full_region = dist <= width
    
    # フェード領域なら徐々に減衰
    fade_region = np.logical_and(dist > width, dist <= width + fade_width)
    
    # フェード領域では滑らかなステップ関数を適用
    fade_weight = smooth_step(dist, width + fade_width, width)
    
    # 条件に応じた重みを返す
    return np.where(full_region, 1.0, np.where(fade_region, fade_weight, 0.0))

#@partial(jit, static_argnums=(1,2,))
#@jit
def adjust_hls_with_weight(hls_img, weight, adjust):
    """
    重み付きでHLS値を調整する
    
    Args:
        hls_img: HLS形式の画像配列
        weight: 各ピクセルの調整の重み (0-1)
        adjust: 調整値の配列 [色相(-180〜180), 明度(-4〜4), 彩度(-1〜1)]
    """
    # 重みを適用した調整
    # 色相: -180〜180度の範囲で直接適用
    h_adj = weight[..., None] * adjust[0]
    
    # 明度: 露出係数として -4〜4 の範囲で適用 (2^adjust[1])
    l_factor = 2.0 ** (weight[..., None] * adjust[1])
    
    # 彩度: -1.0なら彩度*0、0なら変化なし、1.0なら彩度*2
    s_factor = 1.0 + weight[..., None] * adjust[2]
    
    # 各チャンネルに適用
    h = (hls_img[..., 0:1] + h_adj) % 360
    l = hls_img[..., 1:2] * l_factor
    s = hls_img[..., 2:3] * s_factor
    
    # 結果を結合
    return np.concatenate([h, l, s], axis=-1)

#@partial(jit, static_argnums=(1,2,))
def calculate_ls_weight(hls_img, l_range=(0.0, 1.0), s_range=(0.0, 1.0)):
    """
    輝度と彩度に基づく重みを計算
    
    Args:
        hls_img: HLS形式の画像配列
        l_range: 明度の有効範囲 (min, max)
        s_range: 彩度の有効範囲 (min, max)
    
    Returns:
        輝度と彩度に基づく0-1の重み
    """
    l = hls_img[..., 1]
    s = hls_img[..., 2]
    
    # 明度に基づく重み (フェードイン、フェードアウト)
    l_min, l_max = l_range
    l_fade_in = smooth_step(l, l_min, l_min + 0.1)
    l_fade_out = 1.0 - smooth_step(l, l_max - 0.1, l_max)
    l_weight = l_fade_in * l_fade_out
    
    # 彩度に基づく重み (フェードイン、フェードアウト)
    s_min, s_max = s_range
    s_fade_in = smooth_step(s, s_min, s_min + 0.1)
    s_fade_out = 1.0 - smooth_step(s, s_max - 0.1, s_max)
    s_weight = s_fade_in * s_fade_out
    
    # 明度と彩度の重みを組み合わせる
    return l_weight * s_weight

#@partial(jit, static_argnums=(1,))
def adjust_hls_colors(hls_img, color_settings):
    """
    複数の色相範囲を一度に調整する
    
    Args:
        hls_img: HLS形式の画像配列
        color_settings: 色設定の辞書のリスト。各辞書には以下のキーが必要：
            - name: 色の名前
            - center: 中心色相
            - width: 完全適用幅 (半径)
            - fade_width: フェード幅
            - adjust: [色相調整, 明度調整, 彩度調整]
            - l_range: (オプション) 明度の有効範囲 (min, max)
            - s_range: (オプション) 彩度の有効範囲 (min, max)
    
    Returns:
        調整されたHLS画像
    """
    result = hls_img.copy()
    
    for setting in color_settings:
        hue = result[..., 0]
        center = setting['center']
        width = setting['width']
        fade_width = setting['fade_width']
        adjust = setting['adjust']
        
        # 色相に基づく重みを計算
        hue_weight = circular_smooth_step(hue, center, width, fade_width)
        
        # 輝度と彩度の範囲を取得（指定がなければデフォルト）
        l_range = setting.get('l_range', (0.0, 1.0))
        s_range = setting.get('s_range', (0.0, 1.0))
        
        # 輝度と彩度に基づく重みを計算
        #ls_weight = calculate_ls_weight(result, l_range, s_range)
        
        # 最終的な重みを計算
        final_weight = hue_weight# * ls_weight

        # 重みをぼかす
        #final_weight = gaussian_blur(final_weight, (127, 127), 0)
        final_weight = cv2.GaussianBlur(final_weight, (127, 127), 0)
        
        # 重みを使って調整
        result = adjust_hls_with_weight(result, final_weight, adjust)
    
    return result

def adjust_hls_color_one(hls_img, color_name, h, l, s):
    # 色相の設定
    COLOR_SETTING = {
        'red': {
            'center': 22.65,  # 赤の中心値
            'width': 22.5,  # 完全適用幅 (±10度)
            'fade_width': 22.5/2,  # フェード幅 (10-22.5度でフェード)
            'l_range': (0.1, 0.9),  # 明度の有効範囲
            's_range': (0.2, 1.0),  # 彩度の有効範囲
            'adjust': [0.1, 0.05, 0.1],  # [色相, 明度, 彩度] の調整値
        },
        'orange': {
            'center': 33.75,
            'width': 11.25,
            'fade_width': 11.25/2,
            'l_range': (0.1, 0.9),  # 明度の有効範囲
            's_range': (0.1, 0.9),  # 彩度の有効範囲
            'adjust': [0.05, 0.1, 0.1],
        },
        'yellow': {
            'center': 60,
            'width': 15,
            'fade_width': 15/2,
            'l_range': (0.1, 0.9),  # 明度の有効範囲
            's_range': (0.2, 1.0),  # 彩度の有効範囲
            'adjust': [0, 0.1, 0.05],
        },
        'green': {
            'center': 112.5,
            'width': 37.5,
            'fade_width': 37.5/2,
            'l_range': (0.1, 0.9),  # 明度の有効範囲
            's_range': (0.2, 1.0),  # 彩度の有効範囲
            'adjust': [-0.05, 0, 0.1],
        },
        'cyan': {
            'center': 180,
            'width': 30,
            'fade_width': 30/2,
            'l_range': (0.1, 0.9),  # 明度の有効範囲
            's_range': (0.2, 1.0),  # 彩度の有効範囲
            'adjust': [0, -0.05, 0],
        },
        'blue': {
            'center': 240,
            'width': 30,
            'fade_width': 30/2,
            'l_range': (0.1, 0.9),  # 明度の有効範囲
            's_range': (0.2, 1.0),  # 彩度の有効範囲
            'adjust': [0.05, 0, 0.15],
        },
        'purple': {
            'center': 285,
            'width': 15,
            'fade_width': 15/2,
            'l_range': (0.1, 0.9),  # 明度の有効範囲
            's_range': (0.2, 1.0),  # 彩度の有効範囲
            'adjust': [0.1, 0.05, 0],
        },
        'magenta': {
            'center': 318.75,
            'width': 18.75,
            'fade_width': 18.75/2,
            'l_range': (0.1, 0.9),  # 明度の有効範囲
            's_range': (0.2, 1.0),  # 彩度の有効範囲
            'adjust': [0.05, 0.1, 0.05],
        },
        'sky': {
            'center': 210,
            'width': 30,
            'fade_width': 20,
            'l_range': (0.3, 0.9),
            's_range': (0.4, 1.0),
            'adjust': [5, 0.2, 0.1],  # [色相, 輝度, 彩度]
        },
        'skin': {
            'center': 30,
            'width': 20,
            'fade_width': 15,
            'l_range': (0.2, 0.8),
            's_range': (0.3, 0.7),
            'adjust': [-2, 0.1, -0.05],
        }
    }

    color_setting_one = [COLOR_SETTING[color_name]]
    color_setting_one[0]['adjust'] = [h, l, s]
    adjusted_hls = adjust_hls_colors(hls_img, color_setting_one)

    return adjusted_hls


@njit(parallel=True, fastmath=True)
def floyd_steinberg_dither_fast(image):
    """
    Numbaで高速化したFloyd-Steinbergディザリング
    - 並列処理とメモリ最適化を適用
    - 入力: [H, W, 3] float32 (0.0~1.0)
    - 出力: [H, W, 3] uint8
    """
    h, w, c = image.shape
    img = image.copy()
    
    for y in prange(h):
        for x in range(w):
            for ch in range(c):
                old_val = img[y, x, ch]
                new_val = np.round(old_val * 255) / 255
                quant_error = old_val - new_val
                img[y, x, ch] = new_val

                # 誤差分散
                if x < w-1:
                    img[y, x+1, ch] = min(1.0, max(0.0, img[y, x+1, ch] + quant_error * 0.4375))  # 7/16
                if y < h-1:
                    if x > 0:
                        img[y+1, x-1, ch] = min(1.0, max(0.0, img[y+1, x-1, ch] + quant_error * 0.1875))  # 3/16
                    img[y+1, x, ch] = min(1.0, max(0.0, img[y+1, x, ch] + quant_error * 0.3125))  # 5/16
                    if x < w-1:
                        img[y+1, x+1, ch] = min(1.0, max(0.0, img[y+1, x+1, ch] + quant_error * 0.0625))  # 1/16

    return (img * 255).astype(np.uint8)

@njit(nogil=True, parallel=True, fastmath=True)
def fast_median_filter(img, kernel_size=3, num_bins=256):
    """
    量子化とヒストグラムベースの高速メディアンフィルタ
    float32画像を高速処理可能
    
    Parameters:
        img (np.ndarray): 入力画像 (float32)
        kernel_size (int): カーネルサイズ (奇数)
        num_bins (int): 量子化ビン数 (速度/精度のトレードオフ)
    
    Returns:
        np.ndarray: フィルタリング後の画像 (float32)
    """
    h, w = img.shape
    pad = kernel_size // 2
    median_index = (kernel_size * kernel_size) // 2
    
    # 画像の最小値/最大値を計算
    min_val = np.min(img)
    max_val = np.max(img)
    scale = (num_bins - 1) / (max_val - min_val + 1e-7)
    
    # 量子化画像の作成
    quantized = ((img - min_val) * scale).astype(np.float32)
    
    # パディング追加 (reflectモード)
    padded = np.zeros((h + 2*pad, w + 2*pad), dtype=np.float32)
    padded[pad:-pad, pad:-pad] = quantized
    for i in range(pad):
        padded[i, pad:-pad] = quantized[pad-i-1]  # 上端
        padded[-(i+1), pad:-pad] = quantized[-(pad-i)]  # 下端
        padded[pad:-pad, i] = quantized[:, pad-i-1]  # 左端
        padded[pad:-pad, -(i+1)] = quantized[:, -(pad-i)]  # 右端
    
    # 出力画像初期化
    result = np.zeros((h, w), dtype=np.float32)
    
    # メイン処理 (並列化)
    for y in prange(h):
        hist = np.zeros(num_bins, dtype=np.uint16)
        # 初期ヒストグラム構築
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                val = padded[y + ky, kx]
                hist[int(val)] += 1
        
        # 行方向にスライディング
        for x in range(w):
            # 中央値計算
            cumsum = 0
            for b in range(num_bins):
                cumsum += hist[b]
                if cumsum > median_index:
                    result[y, x] = min_val + b / scale
                    break
            
            # ヒストグラム更新 (左カラム削除/右カラム追加)
            if x < w - 1:
                for ky in range(kernel_size):
                    # 左カラム削除
                    left_val = padded[y + ky, x]
                    hist[int(left_val)] -= 1
                    # 右カラム追加
                    right_val = padded[y + ky, x + kernel_size]
                    hist[int(right_val)] += 1
                    
    return result

ICC_PROFILE_TO_COLOR_SPACE = {
    'sRGB IEC61966-2.1': 'sRGB',
    'Adobe RGB (1998)': 'Adobe RGB (1998)',
    'ProPhoto RGB': 'ProPhoto RGB',
}

def get_icc_profile_name(pil_image):
    icc_data = pil_image.info.get("icc_profile")
    
    if not icc_data:
        return 'sRGB IEC61966-2.1'

    profile = ImageCms.getOpenProfile(io.BytesIO(icc_data))
    
    return profile.profile.profile_description
