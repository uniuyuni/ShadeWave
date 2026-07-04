
import sys
import io
import cv2
import math
import os
import re
import numpy as np

import logging
import numba
from numba.experimental import jitclass
from numba import njit, prange
from PIL import ImageCms
import json
from typing import Any, Dict
import base64

from effect_backends import colour_functions_adapter as colour_functions
import cores.sigmoid as sigmoid
import cores.dng_temperature as dng_temperature
import utils.utils as utils
import params
import config
from threads import lock_numba
import cores.hlsrgb as hlsrgb
from effect_backends import image_transform_adapter

def normalize_image(image_data):
    # 画像データを正規化
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_image = (image_data - min_val) / (max_val - min_val)
    return normalized_image

def calc_ev_from_image(image_data):
    # EV値を計算
    average_value = np.mean(image_data)

    # ここで基準を明確に設定
    # 例えば、EV0が0.5に相当する場合
    ev = np.log2(0.5 / average_value)  # 0.5を基準

    return float(ev), float(average_value)

#--------------------------------------------------

def cvtColorRGB2Gray(rgb):
    # RGBからグレイスケールへの変換
    gry = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    return gry

#--------------------------------------------------

def convert_RGB2TempTint(rgb):

    xyz = colour_functions.RGB_to_XYZ(rgb, 'ProPhoto RGB')

    xy = colour_functions.XYZ_to_xy(xyz)

    dng = dng_temperature.DngTemperature()
    dng.set_xy_coord(xy)

    return (float(dng.fTemperature), float(dng.fTint), float(xyz[1]))

def _invert_temp_tint(temp, tint, ref_temp):

    # 色温度の反転
    mired_temp = 1e6 / temp
    mired_ref = 1e6 / ref_temp
    inverted_temp = 1e6 / (mired_ref - (mired_temp - mired_ref) + sys.float_info.min)

    # ティントの反転
    inverted_tint = -tint

    return (inverted_temp, inverted_tint)

def invert_RGB2TempTint(rgb, ref_temp=5000.0):
    temp, tint, Y = convert_RGB2TempTint(rgb)

    invert_temp, invert_tint = _invert_temp_tint(temp, tint, ref_temp)

    return (invert_temp, invert_tint, Y)


def convert_TempTint2RGB(temp, tint, Y):

    dng = dng_temperature.DngTemperature()
    dng.fTemperature = temp
    dng.fTint = tint

    xy = dng.get_xy_coord()

    xyz = colour_functions.xy_to_XYZ(xy)
    xyz *= Y

    rgb = colour_functions.XYZ_to_RGB(xyz, 'ProPhoto RGB')

    return rgb.astype(np.float32)

def invert_TempTint2RGB(temp, tint, Y, reference_temp=5000.0):

    inverted_temp, inverted_tint = _invert_temp_tint(temp, tint, reference_temp)
    
    # DNG SDKの関数を使用して元のRGB値を取得
    r, g, b = convert_TempTint2RGB(inverted_temp, inverted_tint, Y)

    return [r, g, b]

#--------------------------------------------------

def rotation_canvas_matrix(image_shape, angle, flip_mode=0):
    height, width = image_shape[:2]

    # 回転の中心点を計算
    center = (int(width/2), int(height/2))

    # 回転行列を計算（スケール付き）
    trans = cv2.getRotationMatrix2D(center, angle, 1)

    # 回転後画像サイズ    
    size = max(width, height)
    
    # 変換行列に平行移動を追加
    trans[0, 2] += (size / 2) - center[0]
    trans[1, 2] += (size / 2) - center[1]

    # フリップ処理をアフィン変換行列に統合
    if flip_mode & 1:  # 左右反転
        # x -> width - 1 - x
        # 行列操作: 3列目に (width-1)*1列目 を足し、1列目の符号反転
        # M * [[-1, 0, w-1], [0, 1, 0], [0, 0, 1]]
        m00, m01, m02 = trans[0]
        m10, m11, m12 = trans[1]
        trans[0, 0] = -m00
        trans[0, 2] = m00*(width-1) + m02
        trans[1, 0] = -m10
        trans[1, 2] = m10*(width-1) + m12
        
    if flip_mode & 2:  # 上下反転
        # y -> height - 1 - y
        # M * [[1, 0, 0], [0, -1, h-1], [0, 0, 1]]
        m00, m01, m02 = trans[0]
        m10, m11, m12 = trans[1]
        trans[0, 1] = -m01
        trans[0, 2] = m01*(height-1) + m02
        trans[1, 1] = -m11
        trans[1, 2] = m11*(height-1) + m12

    return trans, size


def combined_rotation_canvas_matrix(image_shape, angle, flip_mode=0, matrix=None):
    trans, size = rotation_canvas_matrix(image_shape, angle, flip_mode)
    if matrix is None:
        return trans, size, "affine"

    trans3x3 = np.eye(3)
    trans3x3[:2, :] = trans

    T = np.array([
        [1, 0, size / 2],
        [0, 1, size / 2],
        [0, 0, 1]
    ])
    T_inv = np.linalg.inv(T)
    trans_centered = T_inv @ trans3x3 @ T
    combined = matrix @ trans_centered
    return T @ combined @ T_inv, size, "perspective"


def transform_points(matrix, points, transform_type="affine"):
    """変換行列で2D点群を変換する。

    points: shape (N, 2) の配列。
    transform_type: "affine"(2x3) または "perspective"(3x3)。
    戻り値: shape (N, 2) の変換後座標。
    """
    pts = np.asarray(points, dtype=np.float64)
    if transform_type == "perspective":
        homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
        out = homog @ np.asarray(matrix, dtype=np.float64).T
        w = out[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return out[:, :2] / w
    m = np.asarray(matrix, dtype=np.float64)
    return pts @ m[:, :2].T + m[:, 2]


def content_quad_norm(image_shape, transform_matrix, size, transform_type="affine"):
    """ジオメトリ変換後の有効画像コンテンツ四辺形を、size正方形キャンバスで
    正規化([0,1])した4頂点として返す。

    image_shape: 変換入力画像の shape (H, W, ...)。
    戻り値: shape (4, 2) の正規化頂点 (x, y)。
    """
    h, w = image_shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    quad = transform_points(transform_matrix, corners, transform_type)
    return quad / float(size)


def content_quad_mask(height, width, quad):
    """正規化([0,1])コンテンツ四辺形から (height, width) の float32 マスクを生成する。

    四辺形内部=1.0、外部=0.0。クロップ編集中の黒塗り/オーバーレイクリップで共用する。
    """
    pts = np.asarray(quad, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    mask = np.zeros((int(height), int(width)), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 1.0)
    return mask


def rotation(img, angle, flip_mode=0, matrix=None, inter_mode='bilinear', border_mode="reflect"):
    transform_matrix, size, transform_type = combined_rotation_canvas_matrix(img.shape, angle, flip_mode, matrix)

    if transform_type == "perspective":
        img_affine = image_transform_adapter.transform_to_canvas(
            img,
            transform_matrix,
            size,
            size,
            transform_type="perspective",
            interpolation="cubic" if inter_mode == "bicubic" else "linear",
            border_mode=border_mode,
        )

    else:
        img_affine = image_transform_adapter.transform_to_canvas(
            img,
            transform_matrix,
            size,
            size,
            transform_type="affine",
            interpolation="cubic" if inter_mode == "bicubic" else "linear",
            border_mode=border_mode,
        )

    return img_affine

def gaussian_blur_cv(src, ksize=(3, 3), sigma=0.0):
    if ksize == (0, 0) and sigma == 0.0:
        return src
    return  cv2.GaussianBlur(src, ksize, sigma)


def gaussian_blur(src, ksize=(3, 3), sigma=0.0):
    return gaussian_blur_cv(src, (int(ksize[0]) | 1, int(ksize[1]) | 1), sigma)

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def create_distortion_map(param_vec, width, height):
    """
    歪曲マップを作成する関数 (Numba版)
    """
    cx = width / 2.0
    cy = height / 2.0
    k1, k2, k3, p1, p2 = param_vec
    
    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty((height, width), dtype=np.float32)
    
    for i in prange(height):
        y_norm = i - cy
        y2 = y_norm * y_norm
        for j in range(width):
            x_norm = j - cx
            x2 = x_norm * x_norm
            r2 = x2 + y2
            r4 = r2 * r2
            r6 = r2 * r4
            
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            tangential_x = 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2.0 * x_norm**2)
            # Fix tangential y formula: p1 * (r2 + 2*y2) + 2*p2*x*y
            tangential_y = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * x_norm * y_norm
            
            map_x[i, j] = x_norm * radial + tangential_x + cx
            map_y[i, j] = y_norm * radial + tangential_y + cy
            
    return map_x, map_y

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def apply_lens_distortion(image, map_x, map_y, scale=1.0, interpolation='linear'):
    """
    レンズ歪曲収差を適用する関数 (Numba版)
    Note: scale and interpolation args kept for compatibility but ignored/fixed to linear.
    """
    h, w, c = image.shape
    res = np.empty_like(image)
    
    for i in prange(h):
        for j in range(w):
            x = map_x[i, j]
            y = map_y[i, j]
            
            # Bilinear interpolation
            x0 = int(x)
            y0 = int(y)
            x1 = x0 + 1
            y1 = y0 + 1
            
            dx = x - x0
            dy = y - y0
            
            # Check bounds
            if x0 >= 0 and x1 < w and y0 >= 0 and y1 < h:
                for k in range(c):
                    v00 = image[y0, x0, k]
                    v10 = image[y0, x1, k]
                    v01 = image[y1, x0, k]
                    v11 = image[y1, x1, k]
                    
                    v0 = v00 * (1.0-dx) + v10 * dx
                    v1 = v01 * (1.0-dx) + v11 * dx
                    # Interpolate y
                    val = v0 * (1.0-dy) + v1 * dy
                    res[i, j, k] = val
            else:
                 for k in range(c):
                     res[i, j, k] = 0.0
    return res

_detail_tonemap_warned = False


def _aces_highlight_compress(image):
    import cores.aces_tonemapping as aces_tonemapping
    
    return aces_tonemapping.aces_tonemapping(image, 0.7, config.get_config('gpu_device'))


def detail_preserving_tonemap(image, strength=1.0):
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return image

    src = np.asarray(image, dtype=np.float32)
    try:
        import libraw_enhanced as lre
        tonemap = getattr(lre, "detail_preserving_tonemap", None)
        if tonemap is None:
            raise AttributeError("libraw_enhanced.detail_preserving_tonemap is unavailable")
        mapped = tonemap(np.ascontiguousarray(src), use_gpu_acceleration=True)
    except Exception as exc:
        global _detail_tonemap_warned
        if not _detail_tonemap_warned:
            logging.warning("detail_preserving_tonemap unavailable; falling back to ACES tonemap: %s", exc)
            _detail_tonemap_warned = True
        mapped = _aces_highlight_compress(src)

    mapped = np.asarray(mapped, dtype=np.float32)
    if strength >= 1.0:
        return mapped
    return src + (mapped - src) * strength


def highlight_compress(image):
    return detail_preserving_tonemap(image, 1.0)

def apply_solid_color(image_rgb: np.ndarray, solid_color=(0.94, 0.94, 0.96), opacity=0.5) -> np.ndarray:
    """
    ルミノシティマスクと併用してfloat32形式のRGB画像（0-1）の白飛び部分を自然に補正する関数
    
    Parameters:
    -----------
    image_rgb : np.ndarray
        入力画像（float32形式、RGB、値域0-1）
        shape: (height, width, 3)
    solod_color : tuple
        補正に使用する色（デフォルト: わずかに青みがかった白）
    """
    opacity = float(opacity)
    if opacity <= 0.0:
        return image_rgb

    color = np.asarray(solid_color, dtype=np.float32)
    return image_rgb * np.float32(1.0 - opacity) + color * np.float32(opacity)

#--------------------------------------------------
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
    # Reinhard 領域 (f(x)=x/(1+x)) で screen を取って戻すと a+b+ab に帰着する。
    # 標準 screen 1-(1-a)(1-b) は HDR (>1) で破綻して負値や色ズレを生むのに対し、
    # この式は全域 C∞・単調・非負・常に >=max(a,b) で破綻しない。
    return base + over + base * over

#--------------------------------------------------
# 露出補正
def adjust_exposure(rgb, ev):
    return rgb * (2.0 ** ev)

#--------------------------------------------------

def adjust_contrast(img, cf, c=0.5):
    # コントラスト補正
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

def _median_sample(y, target_samples=250_000):
    """median計算用の間引きビュー。

    medianはソートが必要で画素数に対して非常に重い（40MP超で400ms超）。
    ピボットは厳密な中央値である必要はない（外れ値に強い代表値が取れれば十分）ため、
    格子状ストライドで target_samples 程度まで間引いてから計算する。
    小さい画像（しきい値以下）はストライド1＝従来通り厳密な中央値になる。
    """
    h, w = y.shape[:2]
    n = h * w
    if n <= target_samples:
        return y
    stride = max(1, int(np.sqrt(n / target_samples)))
    return y[::stride, ::stride]

def adjust_luminance_contrast(img, cf, c=None):
    """輝度ベースのコントラスト補正。

    RGB に直接補正をかけず、輝度だけをピボット中心に拡大/縮小する。
    ピボットは既定で画像の輝度中央値を使うため、暗め/明るめの画像でも
    露出補正ではなく画像内の明暗差を広げる挙動になりやすい。

    正の補正は RGB 比率を保つゲインで戻す。負の補正は YCbCr の Y だけを差し替え、
    色差は軽く抑えることで、暗部持ち上げ時の彩度増幅と白っぽい浮きを避ける。
    """
    if cf == 0:
        return img.copy()

    img_f32 = np.asarray(img, dtype=np.float32)
    y = cvtColorRGB2Gray(img_f32)
    pivot = float(np.clip(np.median(np.clip(_median_sample(y), 0.0, 1.0)), 0.25, 0.75)) if c is None else c
    effective_cf = float(cf)
    if effective_cf < 0.0:
        effective_cf *= 0.5
    factor = np.float32(max(0.0, 1.0 + effective_cf / 100.0))
    if effective_cf > 0.0:
        shadow_weight = (0.2 + 0.8 * smoothstep(0.0, pivot, y)).astype(np.float32, copy=False)
        local_factor = 1.0 + (factor - 1.0) * np.where(y < pivot, shadow_weight, 1.0)
        adjusted_y = pivot + (y - pivot) * local_factor
    else:
        adjusted_y = pivot + (y - pivot) * factor
    adjusted_y = np.maximum(adjusted_y, 0.0).astype(np.float32, copy=False)

    if cf < 0:
        lift = adjusted_y - y
        shadow_lift_weight = (smoothstep(0.0, 0.055, y) * (0.25 + 0.75 * smoothstep(0.055, 0.24, y))).astype(np.float32, copy=False)
        adjusted_y = np.where(lift > 0.0, y + lift * shadow_lift_weight, adjusted_y).astype(np.float32, copy=False)
        ycbcr = hlsrgb.linear_rgb_to_ycbcr(img_f32)
        _, cb, cr = cv2.split(ycbcr)
        chroma_scale = (1.0 - min(1.0, -effective_cf / 100.0) * 0.12 * shadow_lift_weight).astype(np.float32, copy=False)
        out_ycbcr = cv2.merge((
            adjusted_y.astype(np.float32, copy=False),
            (cb * chroma_scale).astype(np.float32, copy=False),
            (cr * chroma_scale).astype(np.float32, copy=False),
        ))
        result = hlsrgb.linear_ycbcr_to_rgb(out_ycbcr).astype(np.float32, copy=False)
        return np.maximum(result, np.minimum(img_f32, 0.0))

    eps = np.float32(1e-6)
    gain = np.divide(
        adjusted_y,
        y,
        out=np.ones_like(y, dtype=np.float32),
        where=np.abs(y) > eps,
    )
    return (img_f32 * gain[..., np.newaxis]).astype(np.float32, copy=False)

def apply_level_adjustment(image, black_level=0, midtone_level=128, white_level=255):
    """
    Photoshop風のレベル補正を適用する関数（float32ネイティブ演算）

    Args:
        image: 入力画像 (float32, 0.0–1.0の範囲。HDRは1.0超もあり)
        black_level: 黒レベル (0-255)
        midtone_level: 中間調レベル (0-255, 128が中性)
        white_level: 白レベル (0-255)

    Returns:
        調整された画像 (float32, 0.0–1.0の範囲)
    """

    # パラメータを直接 float32 の 0–1 正規化値に変換
    # （16bit 経由のスケールアップを廃止し精度を向上）
    inv255 = np.float32(1.0 / 255.0)
    black_f = np.float32(black_level) * inv255
    white_f = np.float32(white_level) * inv255

    # midtone を黒–白の範囲でクリップして正規化
    clipped_midtone = max(min(midtone_level, white_level), black_level)
    if white_level > black_level:
        midtone_normalized = (clipped_midtone - black_level) / (white_level - black_level)
    else:
        midtone_normalized = 0.5  # 範囲が無効な場合は中性値

    # 正規化された midtone をガンマ値に変換
    # 0.5 が中性（ガンマ 1.0）、0 に近いほど明るく、1 に近いほど暗く
    if midtone_normalized < 0.5:
        gamma = np.float32(0.1 + (midtone_normalized / 0.5) * 0.9)
    else:
        gamma = np.float32(1.0 + ((midtone_normalized - 0.5) / 0.5) * 8.99)

    # ---- float32 ネイティブで演算 ----------------------------------------
    # 1. 黒レベル以下を 0 にクリップ
    img_f32 = image.astype(np.float32)
    adjusted = np.maximum(img_f32 - black_f, np.float32(0.0))

    # 2. 入力範囲を 0–1 に正規化
    input_range = np.float32(max(white_f - black_f, np.float32(1.0 / 255.0)))  # 0 除算防止
    normalized = adjusted / input_range

    # 3. ガンマ補正（SDR 領域のみ）・HDR 領域は線形加算で保護
    if gamma != np.float32(1.0):
        sdr_part = np.clip(normalized, np.float32(0.0), np.float32(1.0))
        hdr_part = np.maximum(normalized - np.float32(1.0), np.float32(0.0))
        # HDR（>1.0）にガンマをかけると指数爆発するため、線形加算で階調を保持
        result = (np.power(sdr_part, gamma) + hdr_part).astype(np.float32)
    else:
        result = normalized.astype(np.float32)

    return result

#--------------------------------------------------
# 彩度補正と自然な彩度補正

def calc_saturation(hsl_s, sat, vib):

    # 彩度変更値と自然な彩度変更値を計算
    sat = 1.0 + sat / 100.0
    vib = (vib / 50.0)

    # 自然な彩度調整
    if vib == 0.0:
        final_s = hsl_s

    elif vib > 0.0:
        # 通常の計算
        vib = vib ** 2.0
        final_s = np.log(1.0 + vib * hsl_s, dtype=np.float32) / np.log(1.0 + vib, dtype=np.float32)
    else:
        # 逆関数を使用
        vib = vib ** 2.0
        final_s = (np.exp(hsl_s * np.log(1.0 + vib, dtype=np.float32)) - 1.0) / vib

    # 彩度を適用
    final_s = final_s * sat

    return final_s

#--------------------------------------------------

def calc_point_list_to_lut(point_list, max_value=1.0):
    from scipy.interpolate import PchipInterpolator
    """
    コントロールポイントから1D LUTを生成する関数
    
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
    lut_size = 65536
    input_range = np.linspace(0, max_value, lut_size, dtype=np.float32)

    points = np.asarray(point_list, dtype=np.float32)
    if points.size == 0:
        return input_range.copy()

    points = points.reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return input_range.copy()

    order = np.argsort(points[:, 0], kind="stable")
    points = points[order]

    # 同じX位置の点は最後に追加/移動された点を優先する。
    unique_x = np.unique(points[:, 0])
    if len(unique_x) != len(points):
        last_indices = []
        for x_value in unique_x:
            last_indices.append(np.flatnonzero(points[:, 0] == x_value)[-1])
        points = points[np.array(last_indices, dtype=np.int64)]

    x = points[:, 0]
    y = points[:, 1]

    if len(x) == 1:
        lut = np.full(lut_size, y[0], dtype=np.float32)
    elif len(x) >= 3:
        interpolator = PchipInterpolator(x, y, extrapolate=False)
        lut = interpolator(input_range).astype(np.float32)
        lut[input_range < x[0]] = y[0]
        lut[input_range > x[-1]] = y[-1]
    else:
        lut = np.interp(input_range, x, y, left=y[0], right=y[-1]).astype(np.float32)

    if x[0] <= input_range[0]:
        lut[0] = y[0]
    if x[-1] >= input_range[-1]:
        lut[-1] = y[-1]
    
    return lut

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _apply_lut_kernel(img, lut, scale_factor):
    """LUT参照(scale→round→clip→gather)を1回のループに融合したもの。

    元の実装(np.round→np.clip→astype→np.take)は同じ計算を4つの中間配列に
    分けて行っており、np.take によるランダムアクセスgatherが支配的コストだった
    (1024x1024で~5ms/8ms)。ここで融合しても計算式・丸め・クリップ挙動は完全に
    同一で、最終値もビット一致する(中間配列の確保をなくすだけ)。
    非連続ストライドの2D配列(hls[...,ch] のスライス等)もそのまま扱える。
    """
    rows, cols = img.shape
    out = np.empty((rows, cols), dtype=np.float32)
    lut_max = lut.shape[0] - 1
    for i in prange(rows):
        for j in range(cols):
            idx = np.int64(round(img[i, j] * scale_factor))
            if idx < 0:
                idx = 0
            elif idx > lut_max:
                idx = lut_max
            out[i, j] = lut[idx]
    return out


def apply_lut(img, lut, max_value=1.0, overrange="clip"):
    """
    画像にLUTを適用する関数
    max_value: LUTが対応する最大値（デフォルト1.0）
    overrange:
        "clip"     - 従来通りLUT範囲外を端に丸める
        "preserve" - max_valueを超える値はLUT終端の補正量だけを足して階調を保持する
        "scale"    - max_valueを超える値はLUT終端値を係数として階調を保持する
    """
    img = np.asarray(img, dtype=np.float32)

    # スケーリングしてLUTのインデックスに変換
    scale_factor = 65535 / max_value
    if img.ndim == 2:
        # 単一チャンネル(vs系カーブのH/L/S、Tonecurveのgray合成など)はnumba融合版を使う。
        # 3ch/(H,W,1)入力は下の従来経路(挙動を変えたくない箇所)にフォールバックする。
        result = _apply_lut_kernel(img, lut, np.float32(scale_factor))
    else:
        lut_indices = np.clip(np.round(img * scale_factor), 0, 65535).astype(np.uint16)
        result = np.take(lut, lut_indices)

    if overrange == "preserve":
        high_mask = img > max_value
        if np.any(high_mask):
            result = result.astype(np.float32, copy=True)
            result[high_mask] = img[high_mask] + (np.float32(lut[-1]) - np.float32(max_value))
    elif overrange == "scale":
        high_mask = img > max_value
        if np.any(high_mask):
            result = result.astype(np.float32, copy=True)
            gain = np.float32(lut[-1]) / np.float32(max_value)
            result[high_mask] = img[high_mask] * gain
    
    return result

#--------------------------------------------------
# マスクイメージの適用

#def apply_mask(img1, msk, img2):
#
#    _msk = msk[:, :, np.newaxis] if msk.ndim == 2 else msk
#    img = img1 * (1.0 - _msk) + img2 * _msk
#
#    return img

@lock_numba
@njit('f4[:,:,:](f4[:,:,:], f4[:,:], f4[:,:,:])', parallel=True, fastmath=True, cache=True)
def apply_mask(img1, msk, img2):

    """マスクが（3チャンネル）専用の最適化版"""
    if msk.ndim == 3:
        h, w, c = msk.shape
        result = np.empty_like(img1)

        for i in prange(h):
            for j in range(w):
                mask_val = msk[i, j, 0]
                inv_mask = 1.0 - mask_val
                result[i, j, 0] = img1[i, j, 0] * (1.0 - msk[i, j, 0]) + img2[i, j, 0] * msk[i, j, 0]
                result[i, j, 1] = img1[i, j, 1] * (1.0 - msk[i, j, 1]) + img2[i, j, 1] * msk[i, j, 1]
                result[i, j, 2] = img1[i, j, 2] * (1.0 - msk[i, j, 2]) + img2[i, j, 2] * msk[i, j, 2]

        return result

    """RGB（3チャンネル）専用の最適化版"""
    h, w = msk.shape
    result = np.empty_like(img1)
    
    for i in prange(h):
        for j in range(w):
            mask_val = msk[i, j]
            inv_mask = 1.0 - mask_val
            result[i, j, 0] = img1[i, j, 0] * inv_mask + img2[i, j, 0] * mask_val
            result[i, j, 1] = img1[i, j, 1] * inv_mask + img2[i, j, 1] * mask_val
            result[i, j, 2] = img1[i, j, 2] * inv_mask + img2[i, j, 2] * mask_val
    
    return result


def _mask2_param_percent(param, key):
    return float(param.get(key, 0)) / 100.0


def _ks_reflectance_to_ratio(reflectance):
    r = np.clip(reflectance, 1e-6, 1.0)
    return ((1.0 - r) * (1.0 - r)) / (2.0 * r)


def _ks_ratio_to_reflectance(ks_ratio):
    x = np.maximum(ks_ratio, 0.0)
    return 1.0 + x - np.sqrt(x * x + 2.0 * x)


def mix_pigment_white_black_ks_rgb(rgb, black_amount=0.0, white_amount=0.0):
    """RGB 反射率を K/S 比へ写して白・黒顔料を混ぜる軽量近似。"""
    result = np.asarray(rgb, dtype=np.float32)

    def mix_pigment(src, amount, pigment_k, pigment_s):
        amount = np.clip(np.asarray(amount, dtype=np.float32), 0.0, 1.0)
        if np.max(amount) <= 0.0:
            return src
        ks_src = _ks_reflectance_to_ratio(src)
        k_mix = (1.0 - amount) * ks_src + amount * pigment_k
        s_mix = (1.0 - amount) + amount * pigment_s
        return _ks_ratio_to_reflectance(k_mix / np.maximum(s_mix, 1e-6)).astype(np.float32)

    result = mix_pigment(result, black_amount, 8.0, 0.05)
    result = mix_pigment(result, white_amount, 0.0, 1.0)
    return result


def _resize_mask_draw_input(src, target_shape, interpolation):
    target_h, target_w = target_shape[:2]
    if src.shape[:2] == (target_h, target_w):
        return src
    resized = cv2.resize(src, (target_w, target_h), interpolation=interpolation)
    if src.ndim == 3 and resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    return resized


def _blend_mode_composite(backdrop, source, mode):
    """Photoshop 風レイヤーブレンドモード。backdrop/source は [0,1] にクリップ済みの前提。"""
    if mode == "Multiply":
        return backdrop * source
    if mode == "Screen":
        return backdrop + source - backdrop * source
    if mode == "Overlay":
        return np.where(backdrop <= 0.5, 2.0 * backdrop * source, 1.0 - 2.0 * (1.0 - backdrop) * (1.0 - source))
    if mode == "Hard Light":
        return np.where(source <= 0.5, 2.0 * backdrop * source, 1.0 - 2.0 * (1.0 - backdrop) * (1.0 - source))
    if mode == "Soft Light":
        d = np.where(backdrop <= 0.25, ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop, np.sqrt(np.maximum(backdrop, 0.0)))
        return np.where(
            source <= 0.5,
            backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop),
            backdrop + (2.0 * source - 1.0) * (d - backdrop),
        )
    if mode == "Darken":
        return np.minimum(backdrop, source)
    if mode == "Lighten":
        return np.maximum(backdrop, source)
    if mode == "Difference":
        return np.abs(backdrop - source)
    if mode == "Exclusion":
        return backdrop + source - 2.0 * backdrop * source
    if mode == "Linear Dodge (Add)":
        return backdrop + source
    if mode == "Linear Burn":
        return backdrop + source - 1.0
    return source  # Normal


def apply_mask_draw_effects(base, msk, layer_img, mask2_param, resolution_scale=1.0):
    """Mask2 の Photoshop 風 Draw Effects を適用してからマスク合成する。

    resolution_scale: 効果の半径を解像度に追従させるための係数。
    pipeline 側で efconfig.resolution_scale を渡す。
    """
    base = np.asarray(base, dtype=np.float32)
    layer_img = _resize_mask_draw_input(
        np.asarray(layer_img, dtype=np.float32),
        base.shape,
        cv2.INTER_LINEAR,
    )
    raw_mask = _resize_mask_draw_input(
        np.asarray(msk, dtype=np.float32),
        base.shape,
        cv2.INTER_LINEAR,
    )
    mask_alpha = np.clip(raw_mask, 0.0, 1.0)
    mask_alpha = mask_alpha[:, :, np.newaxis] if mask_alpha.ndim == 2 else mask_alpha
    mask_boost = np.maximum(raw_mask, 1.0)
    mask_boost = mask_boost[:, :, np.newaxis] if mask_boost.ndim == 2 else mask_boost

    effect_img = base + (layer_img - base) * mask_boost
    blend_mode = mask2_param.get("mask2_blend_mode", "Normal")
    if not mask2_param.get("switch_mask2_draw_effects", True):
        if blend_mode == "Normal":
            return base * (1.0 - mask_alpha) + effect_img * mask_alpha
        blended = _blend_mode_composite(np.clip(base, 0.0, 1.0), np.clip(effect_img, 0.0, 1.0), blend_mode)
        return base * (1.0 - mask_alpha) + blended * mask_alpha

    backdrop = np.clip(np.asarray(base, dtype=np.float32), 0.0, 1.0)
    source = np.clip(effect_img, 0.0, 1.0)
    eps = 1e-6

    # Skin Smooth (Inverted High Pass) — Dodge/Burn より先に肌のムラを均しておくと
    # コントラスト系の効果が安定する。
    skin_amount_raw = _mask2_param_percent(mask2_param, "mask2_skin_smooth_amount")
    if skin_amount_raw > 0.0:
        from cores import skin_smooth as _skin_smooth
        radius_bias = float(mask2_param.get("mask2_skin_smooth_radius_bias", 0))
        sigma = _skin_smooth.compute_sigma(resolution_scale, radius_bias)
        if sigma > 0.0:
            smoothed_full = _skin_smooth.inverted_high_pass(
                np.clip(effect_img, 0.0, 1.0), sigma
            )
            skin_amount = np.clip(skin_amount_raw * mask_boost, 0.0, 1.0)
            effect_img = effect_img * (1.0 - skin_amount) + smoothed_full * skin_amount
            source = np.clip(effect_img, 0.0, 1.0)

    dodge_amount = _mask2_param_percent(mask2_param, "mask2_color_dodge")
    if dodge_amount > 0.0:
        dodge_amount = np.clip(dodge_amount * mask_boost, 0.0, 1.0)
        dodge = np.clip(backdrop / np.maximum(1.0 - source, eps), 0.0, 1.0)
        effect_img = effect_img * (1.0 - dodge_amount) + dodge * dodge_amount
        source = np.clip(effect_img, 0.0, 1.0)

    burn_amount = _mask2_param_percent(mask2_param, "mask2_color_burn")
    if burn_amount > 0.0:
        burn_amount = np.clip(burn_amount * mask_boost, 0.0, 1.0)
        burn = np.clip(1.0 - ((1.0 - backdrop) / np.maximum(source, eps)), 0.0, 1.0)
        effect_img = effect_img * (1.0 - burn_amount) + burn * burn_amount
        source = np.clip(effect_img, 0.0, 1.0)

    black_amount = _mask2_param_percent(mask2_param, "mask2_mix_black") * mask_boost
    white_amount = _mask2_param_percent(mask2_param, "mask2_mix_white") * mask_boost
    if np.max(black_amount) > 0.0 or np.max(white_amount) > 0.0:
        effect_img = mix_pigment_white_black_ks_rgb(
            np.clip(effect_img, 0.0, 1.0),
            black_amount=black_amount,
            white_amount=white_amount,
        )

    if blend_mode == "Normal":
        return base * (1.0 - mask_alpha) + effect_img * mask_alpha
    blended = _blend_mode_composite(np.clip(base, 0.0, 1.0), np.clip(effect_img, 0.0, 1.0), blend_mode)
    return base * (1.0 - mask_alpha) + blended * mask_alpha

#--------------------------------------------------
# 周辺減光効果
#def apply_vignette(image, intensity, radius_percent, disp_info, crop_rect, offset, gradient_softness=4.0):
#    """
#    修正版 周辺光量落ち効果
#    - 中心位置が正確にクロップ中心に一致
#    - 効果の向きが正しく適用（負の値で周辺暗く、正の値で周辺明るく）
#    - 滑らかなグラデーション
#    - scaleを適切に考慮した効果適用
#    - 元画像の座標系でのビネット中心を正確に反映
#    
#    Parameters:
#        image: 入力画像 (float32, 0-1)
#        intensity: 効果の強さ (-100 to 100)
#        radius_percent: 効果の半径 (1-100%)
#        disp_info: [x, y, w, h, scale] - 元画像における切り抜き情報
#        gradient_softness: グラデーションの滑らかさ
#    """
#
#    intensity = intensity / 100.0
#    radius_percent = radius_percent / 100.0
#    gradient_softness = max(0.1, gradient_softness)
#    
#    h, w = image.shape[:2]
#    
#    if crop_rect is None:
#        # クロップ情報がない場合は従来通り
#        center_x, center_y = w/2, h/2
#
#        mm = jax.lax.max(w, h)
#        max_radius = jax.lax.sqrt(mm**2 + mm**2) / 2
#    else:        
#        dx, dy, _, _, scale = disp_info
#        x1, y1, x2, y2 = crop_rect
#        offset_x, offset_y = offset
#            
#        # クロップ画像内での元画像中心の位置
#        center_x = (x1 + (x2 - x1) / 2 - dx) * scale + offset_x
#        center_y = (y1 + (y2 - y1) / 2 - dy) * scale + offset_y
#        
#        mm = jax.lax.max((x2 - x1), (y2 - y1)) * scale.astype(np.float32)
#        max_radius = jax.lax.sqrt(mm**2 + mm**2) / 2
#    
#    # 指定された半径パーセントに基づいて実際の半径を計算
#    radius = max_radius * radius_percent
#    
#    # 距離マップ作成
#    y_indices, x_indices = np.ogrid[:h, :w]
#    dist = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
#    
#    def smoothstep(x):
#        return x * x * (3 - 2 * x)  # 3次多項式
#
#    # マスク作成（0が中心、1が端）
#    mask = np.clip(dist / radius, 0, 1)
#    #mask = gaussian_blur(mask, (64, 64), 0)
#    mask = np.power(mask, gradient_softness)  # グラデーション調整
#    mask = smoothstep(mask)
#    
#    # 効果適用（intensityの符号で方向を制御）
#    vignette = np.where(intensity < 0, 1.0 + intensity * mask, 1.0 - intensity * mask)
#    
#    # カラー画像対応
#    if image.ndim == 3:
#        vignette = vignette[..., np.newaxis]
#    
#    # 効果適用
#    result = np.clip(image * vignette, 0, 1) if intensity < 0 else np.clip(image + (1-image)*(1-vignette), 0, 1)
#    return result.astype(np.float32)
#--------------------------------------------------
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

def crop_image_with_disp_info(image, disp_info):
    # スケーリング
    org_h, org_w = image.shape[:2]
    cx, cy, cw, ch = int(disp_info[0] * disp_info[4]), int(disp_info[1] * disp_info[4]), int(disp_info[2] * disp_info[4]), int(disp_info[3] * disp_info[4])

    # 切り抜き
    result = image[cy:cy+ch, cx:cx+cw]

    # 中央へ配置
    new_h, new_w = result.shape[:2]
    result = np.pad(result, ((cy, org_h-(new_h+cy)), (cx, org_w-(new_w+cx))), mode="constant")

    return result

def crop_image(image, disp_info, crop_rect, texture_width, texture_height, click_x, click_y, is_zoomed, center_pos=None, zoom_ratio=1.0):

    # 画像のサイズを取得
    image_height, image_width = image.shape[:2]
    crop_rect = _clamp_crop_rect_to_image(crop_rect, image_width, image_height)
    disp_info = _clamp_disp_info_to_image(disp_info, image_width, image_height)

    new_width, new_height, offset_x, offset_y = crop_size_and_offset_from_texture(texture_width, texture_height, disp_info)
    debug_zoom_sync = os.getenv("PLATYPUS_DEBUG_MASK_ZOOM_SYNC", "0").strip().lower() in {"1", "true", "yes", "on"}
    if debug_zoom_sync:
        logging.warning(
            "[MASK_ZOOM_SYNC] crop_image enter image=%s texture=%sx%s is_zoomed=%s zoom_ratio=%.3f disp=%s crop_rect=%s click=(%.2f,%.2f) center_pos=%s draw_size=%sx%s offset=(%s,%s)",
            getattr(image, "shape", None), texture_width, texture_height, is_zoomed,
            zoom_ratio, disp_info, crop_rect, click_x, click_y, center_pos,
            new_width, new_height, offset_x, offset_y,
        )

    # スケールを求める
    if disp_info[2] >= disp_info[3]:
        scale = texture_width/disp_info[2]
    else:
        scale = texture_height/disp_info[3]

    if not is_zoomed:
        dx, dy, dw, dh, _ = disp_info
        result = image_transform_adapter.fit_crop_to_canvas(
            image,
            (dx, dy, dw, dh),
            texture_width,
            texture_height,
            new_width,
            new_height,
            offset_x,
            offset_y,
            "area",
        )

        # 再設定
        disp_info = (dx, dy, dw, dh, scale)

    else:
        crop_source_info, zoom_debug = zoom_crop_source_info(
            disp_info,
            crop_rect,
            texture_width,
            texture_height,
            click_x,
            click_y,
            center_pos,
            zoom_ratio,
            base_scale=scale,
            base_offset=(offset_x, offset_y),
        )

        # クロップ
        if debug_zoom_sync:
            logging.warning(
                "[MASK_ZOOM_SYNC] crop_image zoom_calc base_scale=%.6f click_image=(%.2f,%.2f) crop_xywh=(%.2f,%.2f,%s,%s)",
                scale,
                zoom_debug["click_image_x"],
                zoom_debug["click_image_y"],
                zoom_debug["crop_x"],
                zoom_debug["crop_y"],
                zoom_debug["crop_width"],
                zoom_debug["crop_height"],
            )
        result, disp_info = crop_image_info(image, crop_source_info, crop_rect)
        target_width = max(1, int(texture_width))
        target_height = max(1, int(texture_height))
        if result.shape[1] != target_width or result.shape[0] != target_height:
            interpolation = "nearest" if zoom_ratio >= 1.0 else "area"
            result = image_transform_adapter.fit_crop_to_canvas(
                result,
                (0, 0, result.shape[1], result.shape[0]),
                target_width,
                target_height,
                target_width,
                target_height,
                0,
                0,
                interpolation,
            )
        actual_scale = target_width / max(1, disp_info[2])
        disp_info = (disp_info[0], disp_info[1], disp_info[2], disp_info[3], actual_scale)
    if debug_zoom_sync:
        logging.warning(
            "[MASK_ZOOM_SYNC] crop_image leave result=%s out_disp=%s",
            getattr(result, "shape", None), disp_info,
        )
    
    return result, disp_info


def zoom_crop_source_info(disp_info, crop_rect, texture_width, texture_height, click_x, click_y, center_pos=None, zoom_ratio=1.0, base_scale=None, base_offset=None):
    if base_offset is None:
        _, _, offset_x, offset_y = crop_size_and_offset_from_texture(texture_width, texture_height, disp_info)
    else:
        offset_x, offset_y = base_offset

    if base_scale is None:
        if disp_info[2] >= disp_info[3]:
            base_scale = texture_width / disp_info[2]
        else:
            base_scale = texture_height / disp_info[3]

    zoom_ratio = min(4.0, max(0.5, float(zoom_ratio)))
    click_x = click_x - offset_x
    click_y = click_y - offset_y
    click_image_x = click_x / base_scale
    click_image_y = click_y / base_scale

    crop_width = max(1, int(round(texture_width / zoom_ratio)))
    crop_height = max(1, int(round(texture_height / zoom_ratio)))

    if center_pos is not None:
        crop_x = center_pos[0] - crop_width / 2.0
        crop_y = center_pos[1] - crop_height / 2.0
    else:
        if abs(base_scale - zoom_ratio) < 0.01:
            crop_x = disp_info[0] + disp_info[2] / 2.0 - crop_width / 2.0
            crop_y = disp_info[1] + disp_info[3] / 2.0 - crop_height / 2.0
        else:
            crop_x = disp_info[0] + click_image_x - crop_width / 2.0
            crop_y = disp_info[1] + click_image_y - crop_height / 2.0

    return (
        (crop_x, crop_y, crop_width, crop_height, zoom_ratio),
        {
            "click_image_x": click_image_x,
            "click_image_y": click_image_y,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_width": crop_width,
            "crop_height": crop_height,
        },
    )


def transform_crop_image(
    image,
    transform_matrix,
    transform_width,
    transform_height,
    disp_info,
    texture_width,
    texture_height,
    border_mode="reflect",
    transform_type="affine",
    lens_strength=0.0,
    lens_scale=1.0,
    mesh_map_x=None,
    mesh_map_y=None,
    interpolation="area",
):
    disp_info = _clamp_disp_info_to_image(disp_info, int(transform_width), int(transform_height))
    new_width, new_height, offset_x, offset_y = crop_size_and_offset_from_texture(texture_width, texture_height, disp_info)

    if disp_info[2] >= disp_info[3]:
        scale = texture_width / disp_info[2]
    else:
        scale = texture_height / disp_info[3]

    dx, dy, dw, dh, _ = disp_info
    result = image_transform_adapter.transform_crop_to_canvas(
        image,
        transform_matrix,
        (dx, dy, dw, dh),
        int(transform_width),
        int(transform_height),
        int(texture_width),
        int(texture_height),
        int(new_width),
        int(new_height),
        int(offset_x),
        int(offset_y),
        transform_type=transform_type,
        interpolation=interpolation,
        border_mode=border_mode,
        lens_strength=lens_strength,
        lens_scale=lens_scale,
        mesh_map_x=mesh_map_x,
        mesh_map_y=mesh_map_y,
    )
    return result, (dx, dy, dw, dh, scale)


def transform_zoom_crop_image(
    image,
    transform_matrix,
    transform_width,
    transform_height,
    disp_info,
    crop_rect,
    texture_width,
    texture_height,
    click_x,
    click_y,
    center_pos=None,
    zoom_ratio=1.0,
    border_mode="reflect",
    transform_type="affine",
    lens_strength=0.0,
    lens_scale=1.0,
    mesh_map_x=None,
    mesh_map_y=None,
    interpolation=None,
):
    crop_rect = _clamp_crop_rect_to_image(crop_rect, int(transform_width), int(transform_height))
    disp_info = _clamp_disp_info_to_image(disp_info, int(transform_width), int(transform_height))
    _, _, offset_x, offset_y = crop_size_and_offset_from_texture(texture_width, texture_height, disp_info)

    if disp_info[2] >= disp_info[3]:
        scale = texture_width / disp_info[2]
    else:
        scale = texture_height / disp_info[3]

    crop_source_info, _ = zoom_crop_source_info(
        disp_info,
        crop_rect,
        texture_width,
        texture_height,
        click_x,
        click_y,
        center_pos,
        zoom_ratio,
        base_scale=scale,
        base_offset=(offset_x, offset_y),
    )
    disp_info = _clamp_disp_info_to_crop_rect(crop_source_info, crop_rect)
    dx, dy, dw, dh, _ = disp_info
    target_width = max(1, int(texture_width))
    target_height = max(1, int(texture_height))
    if interpolation is None:
        interpolation = "nearest" if float(zoom_ratio) >= 1.0 else "area"
    result = image_transform_adapter.transform_crop_to_canvas(
        image,
        transform_matrix,
        (dx, dy, dw, dh),
        int(transform_width),
        int(transform_height),
        target_width,
        target_height,
        target_width,
        target_height,
        0,
        0,
        transform_type=transform_type,
        interpolation=interpolation,
        border_mode=border_mode,
        lens_strength=lens_strength,
        lens_scale=lens_scale,
        mesh_map_x=mesh_map_x,
        mesh_map_y=mesh_map_y,
    )
    actual_scale = target_width / max(1, dw)
    return result, (dx, dy, dw, dh, actual_scale)


def crop_image_info(image, disp_info, crop_rect):
    
    # 情報取得
    image_height, image_width = image.shape[:2]
    crop_rect = _clamp_crop_rect_to_image(crop_rect, image_width, image_height)
    disp_info = _clamp_disp_info_to_crop_rect(disp_info, crop_rect)
    disp_x, disp_y, disp_width, disp_height, scale = disp_info

    # オフセット適用は削除（呼び出し側で計算済み）
    x = int(disp_x)
    y = int(disp_y)

    # 画像の範囲外にならないように調整
    x = int(max(crop_rect[0], min(x, crop_rect[2] - disp_width)))
    y = int(max(crop_rect[1], min(y, crop_rect[3] - disp_height)))

    # 画像を切り抜く
    cropped_img = image[y:y+disp_height, x:x+disp_width]

    return cropped_img, (x, y, disp_width, disp_height, scale)


def _clamp_crop_rect_to_image(crop_rect, image_width, image_height):
    if crop_rect is None:
        return (0, 0, max(1, image_width), max(1, image_height))

    x1, y1, x2, y2 = crop_rect
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return (x1, y1, x2, y2)


def _clamp_disp_info_to_crop_rect(disp_info, crop_rect):
    x1, y1, x2, y2 = crop_rect
    crop_width = max(1, x2 - x1)
    crop_height = max(1, y2 - y1)

    if disp_info is None:
        return (x1, y1, crop_width, crop_height, 1.0)

    disp_x, disp_y, disp_width, disp_height, scale = disp_info
    disp_width = int(max(1, min(round(disp_width), crop_width)))
    disp_height = int(max(1, min(round(disp_height), crop_height)))
    max_x = x2 - disp_width
    max_y = y2 - disp_height
    disp_x = int(max(x1, min(round(disp_x), max_x)))
    disp_y = int(max(y1, min(round(disp_y), max_y)))
    return (disp_x, disp_y, disp_width, disp_height, scale)


def _clamp_disp_info_to_image(disp_info, image_width, image_height):
    return _clamp_disp_info_to_crop_rect(
        disp_info,
        (0, 0, max(1, image_width), max(1, image_height)),
    )

#--------------------------------------------------
def get_multiple_mask_bbox(mask):
    """
    マスク画像から複数の独立した領域それぞれのバウンディングボックスを計算する
    
    Args:
        mask マスク画像（2次元のnumpy配列）
        
    Returns:
        各領域の(x, y, w, h)のリスト
            空のマスクの場合は空リストを返す
    """
    from scipy.ndimage import label
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
        
        bboxes.append((int(x_min), int(y_min), int(x_max-x_min+1), int(y_max-y_min+1)))
    
    return bboxes

#--------------------------------------------------
# 上下または左右の余白を追加
def adjust_shape_to_square(img, mode="constant"):
    imax = max(img.shape[1], img.shape[0])

    # イメージを正方形にする
    offset_y = (imax-img.shape[0])//2
    offset_x = (imax-img.shape[1])//2
    img = np.pad(img, ((offset_y, imax-(offset_y+img.shape[0])), (offset_x, imax-(offset_x+img.shape[1])), (0, 0)), mode=mode)

    return img

#--------------------------------------------------
@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def get_luminance(img):
    h, w, c = img.shape
    y = np.empty((h, w), dtype=np.float32)
    # Rec.709: 0.2126, 0.7152, 0.0722
    for i in prange(h):
        for j in range(w):
            y[i, j] = 0.2126 * img[i, j, 0] + 0.7152 * img[i, j, 1] + 0.0722 * img[i, j, 2]
    return y

@njit(fastmath=True, inline='always')
def _apply_midtones(val, midtone):
    if midtone == 0: return val
    if midtone > 0:
        midtone_scale = 16.0
        C = midtone / 100.0 * midtone_scale
        return math.log(1.0 + val * C) / math.log(1.0 + C)
    else:
        midtone_scale = 16.0
        C = -midtone / 100.0 * midtone_scale
        if abs(C) < 1e-6: return val
        log1pC = math.log(1.0 + C)
        normal_result = (math.exp(val * log1pC) - 1.0) / C
        f_1 = ((1.0 + C) - 1.0) / C
        derivative_at_1 = (1.0 + C) * log1pC / C
        if val <= 1.0:
            return normal_result
        else:
            return f_1 + derivative_at_1 * (val - 1.0)

@njit(fastmath=True, inline='always')
def _apply_shadows(val, shadows):
    if shadows == 0: return val
    if shadows > 0:
        shadow_scale = 6.0
        factor = shadows / 100.0 * shadow_scale
        influence = math.exp(-5.0 * val)
        mask = factor * influence
        return val * (1.0 + mask)
    else:
        factor = shadows / 100.0
        influence = math.exp(-5.0 * val)
        min_val = val * 0.1
        mask = (1.0 + factor * influence)
        raw_result = val * mask
        return max(raw_result, min_val)

@njit(fastmath=True, inline='always')
def _apply_black(val, black_level):
    if black_level == 0: return val
    if black_level > 0:
        value = black_level / 100.0
        gamma = math.exp(-value * 0.7)
        return max(val, 0.0) ** gamma
    else:
        value = -black_level / 100.0
        gamma = math.exp(value * 0.7)
        return max(val, 0.0) ** gamma

@njit(fastmath=True, inline='always')
def _apply_highlight_pos(val, highlights):
    strength = highlights / 100.0 * 2.0
    return val * (1.0 + strength)

@njit(fastmath=True, inline='always')
def _apply_highlight_neg(val, base, highlights):
    factor = -highlights / 100.0
    detail = val - base
    strength = factor * 1.0
    compressed_base = base / (1.0 + strength * max(base, 0.0))
    
    threshold = 0.95
    transition_width = 0.4
    t = (base - threshold) / transition_width
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    smooth_mask = t * t * (3.0 - 2.0 * t)
    
    suppression_alpha = 10.0
    adaptive_factor = 1.0 / (1.0 + suppression_alpha * abs(detail))
    # |detail| が小さいハイライト内の微小コントラストを広げる（大きめにすると縁が乗りやすい）
    detail_boost = 1.17
    effective_boost = detail_boost * adaptive_factor
    
    desired_boost = 1.0 + smooth_mask * factor * (effective_boost - 1.0)
    compressed_val = compressed_base + detail * desired_boost
    return compressed_val

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_mid_shadow(y, midtone, shadows):
    h, w = y.shape
    res = np.empty_like(y)
    for i in prange(h):
        for j in range(w):
            val = y[i, j]
            val = _apply_midtones(val, midtone)
            val = _apply_shadows(val, shadows)
            res[i, j] = val
    return res

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_high_pos_black(y, highlights, black_level):
    h, w = y.shape
    res = np.empty_like(y)
    for i in prange(h):
        for j in range(w):
            val = y[i, j]
            val = _apply_highlight_pos(val, highlights)
            val = _apply_black(val, black_level)
            res[i, j] = val
    return res

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_high_neg_black(y, y_blur, highlights, black_level):
    h, w = y.shape
    res = np.empty_like(y)
    for i in prange(h):
        for j in range(w):
            val = y[i, j]
            base = y_blur[i, j]
            val = _apply_highlight_neg(val, base, highlights)
            val = _apply_black(val, black_level)
            res[i, j] = val
    return res

@njit(fastmath=True, inline='always')
def _apply_white_pos(val, white_level, max_val):
    scale = 6.0
    factor = white_level / 100.0 * scale
    if max_val <= 1e-6:
        base = val
    else:
        base = val / max_val
    
    numer_inner = math.log(1.0 + base)
    numer = math.log(1.0 + numer_inner)
    
    denom_inner = math.log(1.0 + max(max_val, 2.0))
    denom = math.log(1.0 + denom_inner)
    
    if denom == 0: denominator = 1.0
    else: denominator = 1.0 / denom
    
    expansion = 1.0 + factor * (numer * denominator)
    return val * expansion

@njit(fastmath=True, inline='always')
def _apply_white_neg(val, base, white_level, max_val):
    factor = -white_level / 100.0
    detail = val - base
    safe_base = max(base, 0.0)
    
    denom_inner = math.log(1.0 + max(max_val, 2.0))
    denom = math.log(1.0 + denom_inner)
    if denom == 0: denominator = 1.0
    else: denominator = 1.0 / denom
    
    target_inner = math.log(1.0 + safe_base)
    target = math.log(1.0 + target_inner) * denominator
    
    compressed_base = min(safe_base, base * (1.0 - factor) + target * factor)
    
    threshold = 0.95
    transition_width = 0.4
    t = (base - threshold) / transition_width
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    smooth_mask = t * t * (3.0 - 2.0 * t)
    
    suppression_alpha = 10.0
    adaptive_factor = 1.0 / (1.0 + suppression_alpha * abs(detail))
    # _apply_highlight_neg の detail_boost(1.085) に対し (boost-1) を約2倍 (~0.085 -> ~0.17)
    detail_boost = 1.17
    effective_boost = detail_boost * adaptive_factor
    
    desired_boost = 1.0 + smooth_mask * factor * (effective_boost - 1.0)
    
    if detail < 0:
        safe_boost = min(desired_boost, compressed_base / max(-detail, 1e-8))
    else:
        safe_boost = desired_boost
        
    compressed_val = compressed_base + detail * safe_boost
    return max(compressed_val, 0.0)

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_white_pos_final(img, y_current, y_orig, white_level, max_val):
    h, w, c = img.shape
    res = np.empty_like(img)
    eps = 1e-6
    for i in prange(h):
        for j in range(w):
            val = y_current[i, j]
            val = _apply_white_pos(val, white_level, max_val)
            
            orig = y_orig[i, j]
            safe_orig = orig if orig >= eps else eps
            gain = val / safe_orig
            if orig < eps: gain = 1.0
            
            for k in range(c):
                res[i, j, k] = img[i, j, k] * gain
    return res

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_white_neg_final(img, y_current, y_blur, y_orig, white_level, max_val_blur):
    h, w, c = img.shape
    res = np.empty_like(img)
    eps = 1e-6
    for i in prange(h):
        for j in range(w):
            val = y_current[i, j]
            base = y_blur[i, j]
            val = _apply_white_neg(val, base, white_level, max_val_blur)
            
            orig = y_orig[i, j]
            safe_orig = orig if orig >= eps else eps
            gain = val / safe_orig
            if orig < eps: gain = 1.0
            
            for k in range(c):
                res[i, j, k] = img[i, j, k] * gain
    return res

# 画像のサイズを取得する関数
def get_exif_image_size(exif_data):
    top, left = exif_data.get("RawImageCropTopLeft", "0 0").split()
    top, left = int(top), int(left)

    _size_tag = ["RawImageCroppedSize", "FullImageSize", "RawImageSize", "ImageSize"]
    for tag in _size_tag:
        if exif_data.get(tag, None) is not None:
            width, height = exif_data.get(tag, "0x0").split('x')
            width, height = int(width), int(height)
            if width != 0 and height != 0:
                return (top, left, width, height)

    raise AttributeError("Not Find image size data")
        
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
            o = utils.normalize_exif_orientation(exif_data.get("Orientation"))
            rad, flip = utils.split_orientation(o)
            if rad < 0.0:
                top, left = left, top
                width, height = height, width

        return (top, left, width, height)


# ガウスカーネル生成関数
@lock_numba
@njit(parallel=True, fastmath=True, cache=True, boundscheck=False, error_model="numpy")
def _gaussian_kernel(size, sigma):
    if size % 2 == 0:
        size += 1  # 奇数に保証
    kernel = np.zeros(size, dtype=np.float32)
    center = size // 2
    sum_val = 0.0
    
    for i in prange(size):
        x = i - center
        kernel[i] = np.exp(-x*x / (2*sigma*sigma))
        sum_val += kernel[i]
    
    return kernel / sum_val

# 手動クリッピング関数
@njit(parallel=True, fastmath=True, cache=True, boundscheck=False, error_model="numpy")
def _manual_clip(x, min_val, max_val):
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    return x

@njit(parallel=True, fastmath=True, cache=True, boundscheck=False, error_model="numpy")
def _smooth_step(x, edge0, edge1):
    """手動クリッピングを使用した滑らかなステップ関数"""
    t = (x - edge0) / (edge1 - edge0)
    t = _manual_clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@njit(parallel=True, fastmath=True, cache=True, boundscheck=False, error_model="numpy")
def _circular_smooth_step(hue, center, width, fade_width):
    """円環滑らかステップ関数"""
    # 円環距離計算
    diff = hue - center
    dist = np.abs(((diff + 180) % 360) - 180)
    
    if dist <= width:
        return 1.0
    elif dist <= width + fade_width:
        # 逆方向の補間: distが大きいほど値が小さい
        return 1.0 - _smooth_step(dist, width, width + fade_width)
    else:
        return 0.0

# ベクトル化された円環ステップ関数
@lock_numba
@njit(parallel=True, fastmath=True, cache=True, boundscheck=False, error_model="numpy")
def _vectorized_circular_smooth_step(hue_map, center, width, fade_width):
    h, w = hue_map.shape
    result = np.empty((h, w), dtype=np.float32)
    
    for i in prange(h):
        for j in prange(w):
            result[i, j] = _circular_smooth_step(hue_map[i, j], center, width, fade_width)
    
    return result

@lock_numba
@njit("f4[:,:,:](f4[:,:,:],f4[:,:],f4[:])", parallel=True, fastmath=True)
def _adjust_hls_with_weight(hls_img, weight, adjust):
    h, w, c = hls_img.shape
    output = np.empty_like(hls_img)
    
    h_adj = adjust[0]
    l_factor = 2.0 ** (adjust[1] * 2)
    s_factor = 1.0 + adjust[2]
    
    for i in prange(h):
        for j in range(w):
            w_val = weight[i, j]
            
            # 色相調整
            new_h = (hls_img[i, j, 0] + w_val * h_adj) % 360
            
            # 明度調整
            new_l = hls_img[i, j, 1] * (l_factor ** w_val)
            
            # 彩度調整 (Vibrance logic)
            s_adj = adjust[2]
            if s_adj > 0.0:
                # Vibrance Boost
                w_adj = s_adj * w_val
                new_s = hls_img[i, j, 2] + hls_img[i, j, 2] * (1.0 - hls_img[i, j, 2]) * w_adj * 2.0
            else:
                # Desaturation (Linear Interpolation)
                # Power function (0.0 ** w) breaks at s_factor=0 (-100%), causing artifacts.
                # Linear: S * (1 + adjust * w)
                new_s = hls_img[i, j, 2] * (1.0 + adjust[2] * w_val)

            # クリッピング
            #new_l = _manual_clip(new_l, 0.0, 1.0)
            #new_s = _manual_clip(new_s, 0.0, 1.0)
            
            output[i, j, 0] = new_h
            output[i, j, 1] = new_l
            output[i, j, 2] = new_s
            
            # チャンネル数が4以上の場合（Gainマップ等）、残りのチャンネルをコピー
            if c > 3:
                for k in range(3, c):
                     output[i, j, k] = hls_img[i, j, k]
    
    return output

@lock_numba
@njit("f4[:,:,:](f4[:,:,:],f4[:,:,:])", parallel=True, fastmath=True)
def _apply_hls_adjust_map(hls_img, total_adjust):
    h, w, c = hls_img.shape
    output = np.empty_like(hls_img)
    
    for i in prange(h):
        for j in range(w):
            # 累積調整値の取得
            adj_h = total_adjust[i, j, 0]
            adj_l = total_adjust[i, j, 1]
            adj_s = total_adjust[i, j, 2]
            
            # --- 色相調整 ---
            new_h = (hls_img[i, j, 0] + adj_h) % 360.0

            # --- 明度調整 (指数関数) ---
            # 実際の明るさは gain にあるため、輝度調整は gain を乗算する。L(正規化輝度)を乗算すると
            # RGB に一定値が足されて彩度が漏れ、かつ階調にも対応しないため。L は色の性質として保持。
            l_factor = 2.0 ** (adj_l * 2.0)

            # --- 彩度調整 ---
            if adj_s > 0.0:
                 # Vibrance Boost
                 new_s = hls_img[i, j, 2] + hls_img[i, j, 2] * (1.0 - hls_img[i, j, 2]) * adj_s * 2.0
            else:
                 # Linear Desaturation
                 new_s = hls_img[i, j, 2] * (1.0 + adj_s)

            # 負の値のクリッピング (過剰な重複によるマイナス防止)
            if new_s < 0.0: new_s = 0.0

            output[i, j, 0] = new_h
            output[i, j, 2] = new_s

            if c > 3:
                # gain(=実輝度)に明度調整を適用、L は保持
                output[i, j, 1] = hls_img[i, j, 1]
                output[i, j, 3] = hls_img[i, j, 3] * l_factor
                for k in range(4, c):
                     output[i, j, k] = hls_img[i, j, k]
            else:
                # gain が無い(3ch)場合のフォールバック: 従来通り L を操作
                output[i, j, 1] = hls_img[i, j, 1] * l_factor
                     
    return output

@lock_numba
@njit("f4[:,:](f4[:,:,:],f4,f4[:],f4[:],f4[:],f4[:])", parallel=True, fastmath=False)
def _calculate_elliptical_weight(hls_img, center_h, width_h, fade_h, l_range, s_range):
    h, w, _ = hls_img.shape
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    l_min, l_max = l_range
    s_min, s_max = s_range
    fade_ls = 0.15 # Fixed fade width for L/S
    
    for i in prange(h):
        for j in prange(w):
            hue = hls_img[i, j, 0]
            # 階調選択は実輝度(L×gain)で行う。L単体は正規化輝度で明暗を持たないため。
            if hls_img.shape[2] > 3:
                l = hls_img[i, j, 1] * hls_img[i, j, 3]
            else:
                l = hls_img[i, j, 1]
            s = hls_img[i, j, 2]
            
            # 1. Hue Excess Distance (Asymmetric)
            signed_diff = hue - center_h
            # Wrap around 180 degrees
            if signed_diff > 180.0:
                signed_diff -= 360.0
            elif signed_diff < -180.0:
                signed_diff += 360.0
            
            # Determine Side (Left=0, Right=1)
            side_idx = 0 if signed_diff < 0 else 1
            abs_diff = abs(signed_diff)
            
            w_h = width_h[side_idx]
            f_h = fade_h[side_idx]
            
            excess_h = 0.0
            if abs_diff > w_h:
                if f_h > 1e-5:
                    excess_h = (abs_diff - w_h) / f_h
                else:
                    excess_h = 100.0 # Sharp cutoff
            
            # 2. L Excess Distance
            excess_l = 0.0
            if l < l_min:
                excess_l = (l_min - l) / fade_ls
            elif l > l_max:
                excess_l = (l - l_max) / fade_ls
                
            # 3. S Excess Distance
            excess_s = 0.0
            if s < s_min:
                # STRICT FADE for Lower Bound (to exclude Gray/Noise)
                # fade_ls (0.15) is too loose for s_min (0.02).
                # Use a much sharper fade (e.g., 0.005 or s_min/2)
                strict_fade = 0.005
                excess_s = (s_min - s) / strict_fade
            elif s > s_max:
                excess_s = (s - s_max) / fade_ls
                
            # 4. Elliptical Combination (Euclidean Norm of Excess)
            dist_sq = excess_h*excess_h + excess_l*excess_l + excess_s*excess_s
            dist = np.sqrt(dist_sq)
            
            # 5. Smooth Falloff
            # Inside plateau (dist=0) -> 1.0
            # At fade limit (dist=1) -> 0.0
            # Using smooth_step for S-curve falloff
            weight = 1.0 - _smooth_step(dist, 0.0, 1.0)
            
            weight_map[i, j] = weight
    
    return weight_map

# 色設定クラス
color_setting_spec = [
    ('center', numba.float32),
    ('width', numba.float32[:]),      # [Left, Right]
    ('fade_width', numba.float32[:]), # [Left, Right]
    ('adjust', numba.float32[:]),
    ('l_range', numba.float32[:]),
    ('s_range', numba.float32[:]),
    ('kernel_size', numba.int32),
]

@jitclass(color_setting_spec)
class ColorSetting:
    def __init__(self):
        self.center = 0.0
        self.width = np.zeros(2, dtype=np.float32)
        self.fade_width = np.zeros(2, dtype=np.float32)
        self.adjust = np.zeros(3, dtype=np.float32)
        self.l_range = np.zeros(2, dtype=np.float32)
        self.s_range = np.zeros(2, dtype=np.float32)
        self.kernel_size = 3

# メイン処理関数
def adjust_hls_colors(hls_img, color_settings, resolution_scale=1.0):

    # Numba設定に変換
    numba_settings = []
    for s in color_settings:
        cs = ColorSetting()
        cs.center = np.float32(s['center'])
        
        # Handle Width (Scalar or List)
        w_val = s['width']
        if np.isscalar(w_val):
            cs.width = np.array([w_val, w_val], dtype=np.float32)
        else:
            cs.width = np.array(w_val, dtype=np.float32)
            
        # Handle Fade Width (Scalar or List)
        f_val = s['fade_width']
        if np.isscalar(f_val):
            cs.fade_width = np.array([f_val, f_val], dtype=np.float32)
        else:
            cs.fade_width = np.array(f_val, dtype=np.float32)

        adjust = np.array(s['adjust'], dtype=np.float32)
        if adjust[0] >= 180.0:
            adjust[0] -= 360.0
        elif adjust[0] < -180.0:
            adjust[0] += 360.0
        cs.adjust = adjust
        cs.l_range = np.array(s['l_range'], dtype=np.float32)
        cs.s_range = np.array(s['s_range'], dtype=np.float32)
        cs.kernel_size = np.float32(s['kernel_size'])
        numba_settings.append(cs)
    
    # カーネルサイズ計算
    kernel_size = max(3, int(cs.kernel_size * resolution_scale))
    if kernel_size % 2 == 0: 
        kernel_size += 1
    sigma = max(1.0, kernel_size / 2.0)
    kernel = _gaussian_kernel(kernel_size, sigma)
    
    # 選択マスク計算用の画像
    mask_source = hls_img
    
    # 累積調整マップの初期化
    h, w = hls_img.shape[:2]
    total_adjust = np.zeros((h, w, 3), dtype=np.float32)

    for setting in numba_settings:
        # Elliptical Weighting (H, L, S combined Isotropically)
        final_weight = _calculate_elliptical_weight(
            mask_source, 
            setting.center, 
            setting.width, 
            setting.fade_width, 
            setting.l_range, 
            setting.s_range
        )

        # ガウシアンブラー適用
        if kernel_size > 1:
            final_weight = gaussian_blur_cv(final_weight, (kernel_size, kernel_size), 0)
        
        # 調整値を累積加算（Broadcasting: (H,W) -> (H,W,1) * (3,) -> (H,W,3)）
        # setting.adjust: [H, L, S]
        
        # NumbaでのBroadcastingがうまくいかない場合があるため、明示的にループまたはreshape
        # _accumulate_adjust(total_adjust, final_weight, setting.adjust) のような関数でも良いが、
        # ここはSimple Broadcastingを期待。もしエラーなら修正。
        # Numba supports broadcasting.
        
        # reshape weight to (H,W,1)
        w_expanded = final_weight.reshape(h, w, 1)
        total_adjust += w_expanded * setting.adjust
    
    # 累積した調整値を一括適用
    current_hls = _apply_hls_adjust_map(hls_img, total_adjust)
    
    return current_hls

HLS_COLOR_SETTING = {
    'red': {
        'center': 105.11,
        'width': [15.0, 9.5],
        'fade_width': [30.0, 18.9],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [0.1, 0.05, 0.1],
        'kernel_size': 64,
    },
    'orange': {
        'center': 142.99,
        'width': [9.5, 8.1],
        'fade_width': [18.9, 16.3],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [0.05, 0.1, 0.1],
        'kernel_size': 64,
    },
    'yellow': {
        'center': 175.55,
        'width': [8.1, 12.4],
        'fade_width': [16.3, 24.7],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [0, 0.1, 0.05],
        'kernel_size': 64,
    },
    'green': {
        'center': 225.0,
        'width': [12.4, 15.0],
        'fade_width': [24.7, 30.0],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [-0.05, 0, 0.1],
        'kernel_size': 64,
    },
    'cyan': {
        'center': 285.11,
        'width': [15.0, 17.6],
        'fade_width': [30.0, 35.2],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [0, -0.05, 0],
        'kernel_size': 64,
    },
    'blue': {
        'center': 355.55,
        'width': [17.6, 6.5],
        'fade_width': [35.2, 12.9],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [0.05, 0, 0.15],
        'kernel_size': 64,
    },
    'purple': {
        'center': 21.37,
        'width': [6.5, 5.9],
        'fade_width': [12.9, 11.8],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [0.1, 0.05, 0],
        'kernel_size': 64,
    },
    'magenta': {
        'center': 45.0,
        'width': [5.9, 15.0],
        'fade_width': [11.8, 30.0],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [0.05, 0.1, 0.05],
        'kernel_size': 64,
    },
    'sky': {
        'center': 320.0,
        'width': [20.0, 20.0],
        'fade_width': [30.0, 30.0],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [5, 0.2, 0.1],  # [色相, 輝度, 彩度]
        'kernel_size': 64,
    },
    'skin': {
        'center': 135.0,
        'width': [15.0, 16.0],
        'fade_width': [20.0, 16.3],
        'l_range': (0.01, 1.0),
        's_range': (0.02, 1.0),
        'adjust': [-2, 0.1, -0.05],
        'kernel_size': 32,
    },
    'enhance_red': {
        'center': 105.11,  # 赤の中心値
        'width': [15.0, 9.5],  # 完全適用幅 (±10度)
        'fade_width': [30.0, 18.9],  # フェード幅 (10-22.5度でフェード)
        'l_range': (0.1, 0.9),  # 明度の有効範囲
        's_range': (0.2, 1.0),  # 彩度の有効範囲
        'adjust': [0.1, 0.05, 0.1],  # [色相, 明度, 彩度] の調整値
        'kernel_size': 128,
    },
}

def adjust_hls_color_one(hls_img, color_name, h, l, s, resolution_scale=1.0):
    # 色相の設定
    color_setting_one = [HLS_COLOR_SETTING[color_name]]
    color_setting_one[0]['adjust'] = [h, l, s]
    adjusted_hls = adjust_hls_colors(hls_img, color_setting_one, resolution_scale)

    return np.array(adjusted_hls)


@lock_numba
@njit('u1[:,:,:](f4[:,:,:])', parallel=True, fastmath=True, cache=True)
def jjn_dither_uint8(img_float):
    """
    float32画像(0.0-1.0)をJJN法でディザリングしてuint8に変換
    """
    h, w, channels = img_float.shape
    
    # 出力バッファの初期化
    output = np.zeros((h, w, channels), dtype=np.uint8)
    
    # JJN法の拡散カーネル (周囲12ピクセルに誤差を拡散)
    kernel = [
        (0, 1, 7), (0, 2, 5),
        (1, -2, 3), (1, -1, 5), (1, 0, 7), (1, 1, 5), (1, 2, 3),
        (2, -2, 1), (2, -1, 3), (2, 0, 5), (2, 1, 3), (2, 2, 1)
    ]
    divisor = 48  # 係数の合計値
    
    # 各チャンネルを個別に処理
    for c in prange(channels):
        # 現在のチャンネル用の誤差バッファ
        error = np.zeros((h+4, w+4), dtype=np.float32)
        
        # ラスタ走査（左上→右下）
        for y in range(h):
            for x in range(w):
                # 現在のピクセル値 + 累積誤差 (xを+2シフト)
                current_val = img_float[y, x, c] + error[y, x+2]
                
                # 量子化（四捨五入）
                quantized_val = min(max(round(current_val * 255.0), 0.0), 255.0)
                
                # 出力値設定
                output[y, x, c] = int(quantized_val)
                
                # 量子化誤差の計算
                quant_error = current_val - (quantized_val / 255.0)
                
                # 誤差を周囲ピクセルに拡散
                for dy, dx, weight in kernel:
                    ey = y + dy
                    ex = x + dx
                    # xは+2オフセット
                    error[ey, ex+2] += quant_error * (weight / divisor)
        
    return output

@lock_numba
@njit('u2[:,:,:](f4[:,:,:])', parallel=True, fastmath=True, cache=True)
def jjn_dither_uint16(img_float):
    """
    float32画像(0.0-1.0)をJJN法でディザリングしてuint16に変換
    """
    h, w, channels = img_float.shape
    
    # 出力バッファの初期化
    output = np.zeros((h, w, channels), dtype=np.uint16)
    
    # JJN法の拡散カーネル
    kernel = [
        (0, 1, 7), (0, 2, 5),
        (1, -2, 3), (1, -1, 5), (1, 0, 7), (1, 1, 5), (1, 2, 3),
        (2, -2, 1), (2, -1, 3), (2, 0, 5), (2, 1, 3), (2, 2, 1)
    ]
    divisor = 48  # 係数の合計値
    
    # 各チャンネルを個別に処理（チャンネル間は並列化可能）
    for c in prange(channels):
        # 現在のチャンネル用の誤差バッファ
        # x方向は左右に余裕、yは下に余裕
        error = np.zeros((h+4, w+4), dtype=np.float32)
        
        # ラスタ走査（左上→右下）
        for y in range(h):
            for x in range(w):
                # 現在のピクセル値 + 累積誤差 (xを+2シフトしてアクセス)
                current_val = img_float[y, x, c] + error[y, x+2]
                
                # 量子化（四捨五入）
                quantized_val = min(max(round(current_val * 65535.0), 0.0), 65535.0)
                
                # 出力値設定
                output[y, x, c] = int(quantized_val)
                
                # 量子化誤差の計算
                quant_error = current_val - (quantized_val / 65535.0)
                
                # 誤差を周囲ピクセルに拡散
                for dy, dx, weight in kernel:
                    ey = y + dy
                    ex = x + dx
                    # xは+2オフセットして書き込み
                    error[ey, ex+2] += quant_error * (weight / divisor)
        
    return output


ICC_PROFILE_TO_COLOR_SPACE = {
    'sRGB': 'sRGB', # 何故かこれを返すデータがある
    'sRGB IEC61966-2.1': 'sRGB',
    'Adobe RGB (1998)': 'Adobe RGB (1998)',
    'ProPhoto RGB': 'ProPhoto RGB',
    'ACES2065-1': 'ACES2065-1',
    'ACEScg': 'ACEScg',
    'Display P3': 'Display P3',
    'ITU-R BT.2020': 'Rec.2020',
    'Rec.2020': 'Rec.2020',
    'ITU-R BT.709 Reference Display': 'Rec.709',
    'ITU-R BT.709': 'Rec.709',
}

def get_icc_profile_name(pil_image):
    icc_data = pil_image.info.get("icc_profile")
    
    if not icc_data:
        return 'sRGB IEC61966-2.1'

    profile = ImageCms.getOpenProfile(io.BytesIO(icc_data))
    
    return profile.profile.profile_description

def apply_zero_wrap(img, param, crop_editing=False):
    """
    Zero-wrapフィルタを適用する関数
    """
    # クロップ編集中は回転した正方形パディングのため、黒塗り範囲が矩形にならない。
    # GeometryEffect が param に格納した正規化コンテンツ四辺形からマスクを生成する。
    quad = param.get('_zero_wrap_content_quad')
    if crop_editing:
        if quad is not None:
            out_h, out_w = int(img.shape[0]), int(img.shape[1])
            mask = content_quad_mask(out_h, out_w, quad)
            content = int(np.count_nonzero(mask))
            zero_count = out_w * out_h - content
            img = img * mask[..., np.newaxis]
            return (img, zero_count)

    disp_info = params.get_disp_info(param)

    # 通常表示（crop_editing=False）はクロップ枠の外側を矩形で黒塗りするだけにする。
    # 枠内の回転コンテンツ外（reflect ミラー画素）はエクスポートでもミラーのまま残るため、
    # ここで quad マスクを掛けるとプレビューだけ斜めに削れてエクスポートと不一致になる
    # （後がけの Light Rays 等も切られる）。quad マスクは Ge タブ（crop_editing）専用。
    width = int((disp_info[2]) * disp_info[4])
    height = int((disp_info[3]) * disp_info[4])
    width, height = min(width, img.shape[1]), min(height, img.shape[0]) # 安全策
    wrap = np.ones((height, width), dtype=np.float32)
    # パイプライン出力はプレビュー用テクスチャ解像度と1ピクセル未満の差でずれることがある。
    # マスクは常に img と同じ形に合わせる（config の preview_* は使わない）。
    out_w, out_h = int(img.shape[1]), int(img.shape[0])
    offset_x, offset_y = (out_w - wrap.shape[1]) // 2, (out_h - wrap.shape[0]) // 2
    zero_count = out_w * out_h - wrap.shape[1] * wrap.shape[0]
    wrap = np.pad(
        wrap,
        (
            (offset_y, out_h - wrap.shape[0] - offset_y),
            (offset_x, out_w - wrap.shape[1] - offset_x),
        ),
        "constant",
        constant_values=0.0,
    )

    # クロップ中は処理しないがクロップしている範囲のzero_countだけ返す
    if not crop_editing:
        img = img * wrap[..., np.newaxis]

    return (img, zero_count)

def apply_out_of_range_exposure(img, overexposure, underexposure):

    if overexposure == True or underexposure == True:
        img = img.copy()

        if underexposure == True:
            mask = (img[..., 0] <= 0.0) & (img[..., 1] <= 0.0) & (img[..., 2] <= 0.0)
            img[mask] = [0.0, 0.0, 1.0]

        if overexposure == True:
            mask = (img[..., 0] >= 1.0) & (img[..., 1] >= 1.0) & (img[..., 2] >= 1.0)
            img[mask] = [0.0, 0.0, 0.0]

    return img

def calc_resolution_scale(current_resolution, scale=1.0):
        
    # 解像度比を計算（幅と高さの幾何平均を使用）
    ratio = np.sqrt(
        (current_resolution[0] / config.get_config('base_resolution_scale')[0]) *
        (current_resolution[1] / config.get_config('base_resolution_scale')[1])
    )

    return scale * ratio

def type_convert(img, target_type):
    """
    if isinstance(img, target_type):
        return img
    
    if target_type == np.ndarray:
        if isinstance(img, np.ndarray):
            return np.array(img)
        elif isinstance(img, cv2.UMat):
            return img.get()
    
    elif target_type == np.ndarray:
        if isinstance(img, np.ndarray):
            return np.array(img)
        elif isinstance(img, cv2.UMat):
            return np.array(img.get())
    
    elif target_type == cv2.UMat:
        if isinstance(img, np.ndarray):
            return cv2.UMat(img)
        elif isinstance(img, np.ndarray):
            return cv2.UMat(np.array(img))
    """
    return img

def calc_ev_from_exif(exif_data):
    Av = exif_data.get('ApertureValue', 1.0)
    ssv = exif_data.get('ShutterSpeedValue', "1/100")
    if type(ssv) == str:
        if '/' in ssv:
            _, Tv = ssv.split('/')
            Tv = float(_) / float(Tv)
        else:
            Tv = float(ssv)
    else:
        Tv = ssv

    return calc_ev_from_settings(Av, Tv, exif_data.get('ISO', 100))
    
def calc_ev_from_settings(Av: float, Tv: float, Sv: float) -> float:
    """
    カメラ設定値から直接露出値(Ev)を計算
    
    Args:
        f_number (float): F値 (例: 2.8)
        shutter_speed (float): シャッター速度[秒] (例: 1/100秒の場合は0.01)
        iso (float): ISO感度 (例: 100)
    
    Returns:
        float: 露出値(Ev)
    """
    # 各成分の計算
    Ev = math.log2((Av ** 2) / Tv)
    Sv = math.log2(Sv / 100.0)

    return Ev + Sv

def convert_to_float32(img):
    """
    画像のデータ型をfloat32に変換する関数

    Args:
        img (numpy.ndarray): 変換する画像データ

    Returns:
        numpy.ndarray: float32の画像データ
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255
    elif img.dtype == np.uint16 or img.dtype == '>u2' or img.dtype == '<u2':
        img = img.astype(np.float32)/65535
    elif img.dtype == np.uint32 or img.dtype == '>u4' or img.dtype == '<u4':
        img = img.astype(np.float32)/4294967295
    elif img.dtype == np.uint64:
        img = img.astype(np.float32)/18446744073709551615
    elif img.dtype == np.int8:
        img = img.astype(np.float32)/127
    elif img.dtype == np.int16:
        img = img.astype(np.float32)/32767
    elif img.dtype == np.int32:
        img = img.astype(np.float32)/2147483647
    elif img.dtype == np.int64:
        img = img.astype(np.float32)/9223372036854775807
    elif img.dtype == np.float16:
        img = img.astype(np.float32)
    elif img.dtype == np.float32:
        pass
    elif img.dtype == np.float64:
        img = img.astype(np.float32)
    else:
        raise ValueError(f"サポートされていないデータ型: {img.dtype}")

    return img

def get_initial_crop_rect(input_width, input_height):
    maxsize = max(input_width, input_height)
    padw = (maxsize - input_width) / 2 
    padh = (maxsize - input_height) / 2
    return (int(math.floor(padw)), int(math.floor(padh)), int(math.ceil(padw)+input_width), int(math.ceil(padh)+input_height))

def get_initial_disp_info(input_width, input_height, scale):
    # パディング付与
    x1, y1, crop_width, crop_height = 0, 0, input_width, input_height
    maxsize = max(input_width, input_height)
    padw = (maxsize - input_width) // 2 
    padh = (maxsize - input_height) // 2
    crop_x = int(x1 + padw)
    crop_y = int(y1 + padh)
    return (crop_x, crop_y, crop_width, crop_height, scale)

def convert_rect_to_info(crop_rect, scale):      
    x1, y1, x2, y2 = crop_rect
    w = x2 - x1
    h = y2 - y1        
    return (x1, y1, w, h, scale)

class CompactNumpyEncoder(json.JSONEncoder):
    """NumPyデータを最小容量で保存するカスタムエンコーダ"""
    
    def default(self, obj: Any) -> Any:
        # NumPy配列の処理
        if isinstance(obj, np.ndarray):
            return self._compress_array(obj)
        
        # NumPyスカラーの処理
        if isinstance(obj, np.generic):
            return obj.item()
        
        # bytesの処理
        if isinstance(obj, bytes):
            return {
                '__bytes__': True,
                'data': base64.b64encode(obj).decode('ascii')
            }
                    
        return super().default(obj)
    
    def _compress_array(self, array: np.ndarray) -> Dict[str, Any]:
        """配列を圧縮してBase64エンコード"""
        # データをバイト列に変換
        data_bytes = array.tobytes()
        
        # zlibで圧縮 (レベル9で最大圧縮)
        compressed = data_bytes #zlib.compress(data_bytes, level=9)
        
        # Base64エンコード
        encoded = base64.b64encode(compressed).decode('ascii')
        
        return {
            '__numpy_array__': True,
            'dtype': str(array.dtype),
            'shape': array.shape,
            'data': encoded
        }

def compact_numpy_decoder(obj: Dict) -> Any:
    """圧縮されたNumPyデータを復元"""
    if '__numpy_array__' in obj:
        # Base64デコード
        decoded = base64.b64decode(obj['data'])
        
        # zlib解凍
        decompressed = decoded #zlib.decompress(decoded)
        
        # NumPy配列に変換
        array = np.frombuffer(decompressed, dtype=np.dtype(obj['dtype']))
        return array.reshape(obj['shape'])

    if '__bytes__' in obj:
        return base64.b64decode(obj['data'])
    
    return obj

def auto_contrast_tonemap(image):
    """
    トーンカーブベースの自動コントラスト補正
    入力条件:
        - image: (H, W, 3) shapeのnumpy配列 (RGB, float32)
    
    処理内容:
        1. RGBから輝度(Y)を計算
        2. 輝度のヒストグラムからトーンカーブを生成
        3. トーンカーブをRGB各チャンネルに適用
    """
    # 入力画像のコピーを作成
    corrected = np.empty_like(image)
    
    # RGBから輝度Yを計算 (BT.709基準)
    luminance = cvtColorRGB2Gray(image)
    normaliced_lum = luminance / np.max(luminance)
        
    def s_curve(x, strength=0.5):
        """S字カーブ関数"""
        return x + strength * x * (1 - x) * (2 * x - 1)

    # トーンカーブを生成（ガンマ補正含む）
    tone_curve = np.zeros(65536, dtype=np.float32)
    for i in range(65536):
        value = i / 65535.0
        value = s_curve(value, strength=0.7)  # S字カーブ適用
        tone_curve[i] = value
    
    # 輝度値からトーンカーブを適用するためのインデックスマップを作成
    indices = np.clip((normaliced_lum * 65535).astype(np.uint32), 0, 65535)
    mapped_lum = tone_curve[indices]
    
    # 元の輝度とマッピング後の比率を計算
    ratio = np.divide(mapped_lum, luminance, 
                     where=np.abs(luminance) > 1e-6,
                     out=np.ones_like(luminance))
    
    # RGBチャンネルに比率を適用
    corrected = image * ratio[..., np.newaxis]
    
    return corrected

#-------------------------------------------------

_lensfun_db_instance = None

def _get_lensfun_db():
    global _lensfun_db_instance
    import lensfunpy
    if _lensfun_db_instance is None:
        _lensfun_db_instance = lensfunpy.Database()
    return _lensfun_db_instance


def _lensfun_number(value, default=None, *, prefer_f_number: bool = False):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _lensfun_number(item, None, prefer_f_number=prefer_f_number)
            if parsed is not None:
                return parsed
        return default
    if isinstance(value, (int, float, np.number)):
        result = float(value)
        return result if math.isfinite(result) else default

    text = str(value).strip()
    if not text:
        return default
    lower = text.lower()
    if lower in {"unknown", "undef", "undefined", "inf", "infinity", "nan", "close", "distant"}:
        return default
    text = text.replace(",", ".")

    if prefer_f_number:
        f_match = re.search(r"\bf\s*/\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if f_match:
            return float(f_match.group(1))
        if text.lower().startswith("f/"):
            text = text[2:].strip()

    fraction = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)\s*", text)
    if fraction:
        denominator = float(fraction.group(2))
        if abs(denominator) > 1e-12:
            return float(fraction.group(1)) / denominator
        return default

    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return default
    return float(match.group(0))


def _lensfun_aperture(exif_data):
    for key in ("FNumber", "Aperture"):
        parsed = _lensfun_number(exif_data.get(key), None, prefer_f_number=True)
        if parsed is not None and parsed > 0:
            return parsed

    aperture_value = exif_data.get("ApertureValue")
    parsed = _lensfun_number(aperture_value, None, prefer_f_number=True)
    if parsed is None or parsed <= 0:
        return None
    if isinstance(aperture_value, str) and re.search(r"\bf\s*/", aperture_value, flags=re.IGNORECASE):
        return parsed
    return float(2 ** (parsed / 2.0))


def setup_lensfun(img_size, exif_data):

    make =  exif_data.get('Make', None)
    model = exif_data.get('Model', None)
    lensmake = exif_data.get('LensMake', None)
    lensmodel = exif_data.get('LensModel', None)
    focal_length = _lensfun_number(exif_data.get('FocalLength', None), None)
    aperture = _lensfun_aperture(exif_data)
    distance = _lensfun_number(exif_data.get('SubjectDistance', exif_data.get('SubjectDistanceRange', None)), 100)

    logging.info(f"{make}, {model}")
    logging.info(f"{lensmake}, {lensmodel}")
    logging.info(f"{focal_length}, {aperture}, {distance}")

    if focal_length is None or aperture is None:
        logging.info("focal_length or aperture is None")
        return

    if distance is None:
        distance = 100

    import lensfunpy
    db = _get_lensfun_db()
    cams = db.find_cameras(make, model, loose_search=True)
    if len(cams) > 0:
        lens = db.find_lenses(cams[0], lensmake, lensmodel, loose_search=False)

    mod = None
    if len(cams) > 0:
        lens = db.find_lenses(cams[0], lensmake, lensmodel, loose_search=False)

        if len(lens) > 0:
            width, height = img_size
            mod = lensfunpy.Modifier(lens[0], cams[0].crop_factor, width, height)
            mod.initialize(focal_length, aperture, distance, pixel_format=np.float32)
            return mod

    return mod

def modify_lensfun(mod, img, is_cm=True, is_sd=True, is_gd=True):

    if mod is None:
        logging.warning("Lensfun is not initialized")
        return (img, False, False, False)

    modimg = img.copy()
    
    if is_cm == True:
        did_apply = mod.apply_color_modification(modimg)
        if did_apply == False:
            logging.warning("Apply Color Modification is Failed")
            is_cm = False

    combined_distortion_applied = False
    if is_sd == True and is_gd == True and hasattr(mod, "apply_subpixel_geometry_distortion"):
        undist_coords = mod.apply_subpixel_geometry_distortion()
        if undist_coords is None:
            logging.warning("Apply Subpixel Geometry Distortion is Failed")
        else:
            modimg[..., 0] = cv2.remap(modimg[..., 0], undist_coords[..., 0, :], None, cv2.INTER_CUBIC)
            modimg[..., 1] = cv2.remap(modimg[..., 1], undist_coords[..., 1, :], None, cv2.INTER_CUBIC)
            modimg[..., 2] = cv2.remap(modimg[..., 2], undist_coords[..., 2, :], None, cv2.INTER_CUBIC)
            combined_distortion_applied = True

    if is_sd == True and not combined_distortion_applied:
        undist_coords = mod.apply_subpixel_distortion()
        if undist_coords is None:
            logging.warning("Apply Subpixel Distortion is Failed")
            is_sd = False
        else:
            modimg[..., 0] = cv2.remap(modimg[..., 0], undist_coords[..., 0, :], None, cv2.INTER_CUBIC)
            modimg[..., 1] = cv2.remap(modimg[..., 1], undist_coords[..., 1, :], None, cv2.INTER_CUBIC)
            modimg[..., 2] = cv2.remap(modimg[..., 2], undist_coords[..., 2, :], None, cv2.INTER_CUBIC)

    if is_gd == True and not combined_distortion_applied:
        undist_coords = mod.apply_geometry_distortion()
        if undist_coords is None:
            logging.warning("Apply Geometry Distortion is Failed")
            is_gd = False
        else:
            modimg = cv2.remap(modimg, undist_coords, None, cv2.INTER_CUBIC)

    return (modimg, is_cm, is_sd, is_gd)


def get_lensfun_capability(mod, img):
    """
    ユーザー指定を混ぜない、lensfun の純粋な対応可否 (cm, sd, gd) を返す。
    失敗時はすべて False。
    """
    if mod is None or img is None:
        return (False, False, False)
    is_cm = False
    is_sd = False
    is_gd = False
    try:
        tmp = img.copy()
        is_cm = bool(mod.apply_color_modification(tmp))
    except Exception:
        is_cm = False
    try:
        is_sd = mod.apply_subpixel_distortion() is not None
    except Exception:
        is_sd = False
    try:
        is_gd = mod.apply_geometry_distortion() is not None
    except Exception:
        is_gd = False
    return (is_cm, is_sd, is_gd)

#-------------------------------------------------

#-------------------------------------------------

@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def _kernel_unsharp_mask_apply(img, blurred, amount):
    h, w, c = img.shape
    res = np.empty_like(img)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                val = img[i, j, k]
                blur_val = blurred[i, j, k]
                res[i, j, k] = val + amount * (val - blur_val)
    return res

def unsharp_mask(rgb_image, amount=1.0, sigma=1.0):
    """
    RGB画像にアンシャープマスク処理を適用 (Numba Optimized)
    
    引数:
        rgb_image (numpy.ndarray): RGB形式の入力画像 (float32, 0.0-1.0)
        amount (float): マスクの適用強度（デフォルト: 1.0）
        sigma (float): ガウシアンブラーの標準偏差（デフォルト: 1.0）
        
    戻り値:
        numpy.ndarray: シャープニングされたRGB画像 (float32)
    """
    # ガウシアンフィルタでぼかした画像を生成
    blurred = gaussian_blur_cv(rgb_image, (0, 0), sigma)
    
    # Numbaカーネルで適用
    sharpened = _kernel_unsharp_mask_apply(rgb_image, blurred, amount)
    
    return sharpened


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def boost_detail_from_tone_change(
    rgb_before,              # 補正前 (float, >=0)
    rgb_after,               # 補正後 (float, >=0)
    detail_strength=0.8,     # 全体強度
    sigma=1.8,               # 半径
    hi_start=0.8,            # 明部判定開始(輝度)
    hi_end=2.0,              # 明部判定最大(輝度)
    max_comp_stops=2.0,      # 何stop下げで強度1.0扱い
    gamma=2.0,               # 小さい補正を抑えるカーブ
    eps=1e-6
):
    a = np.asarray(rgb_before, dtype=np.float32)
    b = np.asarray(rgb_after, dtype=np.float32)

    Ya = np.maximum(0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2], eps)
    Yb = np.maximum(0.2126*b[...,0] + 0.7152*b[...,1] + 0.0722*b[...,2], eps)

    # 下がった分のstop量
    comp_stops = np.maximum(0.0, np.log2(Ya, dtype=np.float32) - np.log2(Yb, dtype=np.float32))
    comp_w = np.clip(comp_stops / max_comp_stops, 0.0, 1.0)

    # 明部マスクは補正前の輝度から作る方が安定
    L = np.log2(Ya, dtype=np.float32)
    hi_w = smoothstep(np.log2(hi_start, dtype=np.float32), np.log2(hi_end, dtype=np.float32), L)

    # log輝度ハイパス
    logY = np.log(Yb, dtype=np.float32)
    base = cv2.GaussianBlur(logY, (0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = logY - base

    gain = detail_strength * (comp_w ** gamma) * hi_w
    logY2 = logY + gain * detail
    Y2 = np.exp(logY2, dtype=np.float32)

    out = b * (Y2 / Yb)[..., None]
    return np.maximum(out, 0.0)
