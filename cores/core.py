
import sys
import io
import cv2
import math
import numpy as np

import logging
import numba
from numba.experimental import jitclass
from numba import njit, prange
from PIL import ImageCms
import json
from typing import Any, Dict
import base64

import cores.colour_functions as colour_functions
import cores.sigmoid as sigmoid
import dng_sdk.dng_temperature
import utils.utils as utils
import params
import config
from threads import lock_numba

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

    dng = dng_sdk.dng_temperature.DngTemperature()
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

    dng = dng_sdk.dng_temperature.DngTemperature()
    dng.fTemperature = temp
    dng.fTint = tint

    xy = dng.get_xy_coord()

    xyz = colour_functions.xy_to_XYZ(xy)
    xyz *= Y

    rgb = colour_functions.XYZ_to_RGB(xyz, 'ProPhoto RGB')

    return rgb.astype(np.float32)

def invert_TempTint2RGB(temp, tint, Y, reference_temp=5000.0):

    inverted_temp, inverted_tint = __invert_temp_tint(temp, tint, reference_temp)
    
    # DNG SDKの関数を使用して元のRGB値を取得
    r, g, b = convert_TempTint2RGB(inverted_temp, inverted_tint, Y)

    return [r, g, b]

#--------------------------------------------------

def rotation(img, angle, flip_mode=0, matrix=None, inter_mode='bilinear', border_mode="reflect"):
    # 元の画像の高さと幅を取得
    height, width = img.shape[:2]
    
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

    if matrix is not None:
        # transを3x3行列に拡張
        trans3x3 = np.eye(3)
        trans3x3[:2, :] = trans
        
        # transを中心原点座標系に変換
        # matrixは既にparams.add_matrixで中心原点座標系に変換済み
        T = np.array([
            [1, 0, size / 2],
            [0, 1, size / 2],
            [0, 0, 1]
        ])
        T_inv = np.linalg.inv(T)
        trans_centered = T_inv @ trans3x3 @ T
        
        # 中心原点座標系で合成: 先にtrans（回転・フリップ）を適用し、その後にmatrix（ジオメトリ補正）を適用
        combined = matrix @ trans_centered
        
        # 左上原点座標系に戻す
        final_matrix = T @ combined @ T_inv
        
        # こっちはパースペクティブ
        img_affine = cv2.warpPerspective(img, final_matrix, (size, size),
                                    flags=cv2.INTER_CUBIC if inter_mode == 'bicubic' else cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT if border_mode == "reflect" else cv2.BORDER_CONSTANT)

    else:
        # 回転と中心補正を同時に行う
        img_affine = cv2.warpAffine(img, trans, (size, size),
                                    flags=cv2.INTER_CUBIC if inter_mode == 'bicubic' else cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_REFLECT if border_mode == "reflect" else cv2.BORDER_CONSTANT)

    return img_affine

def gaussian_blur_cv(src, ksize=(3, 3), sigma=0.0):
    if ksize == (0, 0) and sigma == 0.0:
        return src
    return  cv2.GaussianBlur(src, ksize, sigma)


def gaussian_blur(src, ksize=(3, 3), sigma=0.0):
    return gaussian_blur_cv(src, (int(ksize[0]) | 1, int(ksize[1]) | 1), sigma)

@lock_numba
@njit('f4[:,:,:](f4[:,:,:],f4[:,:,:])', parallel=True, fastmath=True, cache=True)
def _lucy_ratio_step(srcf, bdest):
    eps = np.finfo(np.float32).eps
    h, w, c = srcf.shape
    ratio = np.empty_like(srcf)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                ratio[i, j, k] = srcf[i, j, k] / (bdest[i, j, k] + eps)
    return ratio

@lock_numba
@njit('f4[:,:,:](f4[:,:,:],f4[:,:,:])', parallel=True, fastmath=True, cache=True)
def _lucy_update_step(destf, ratio_blur):
    h, w, c = destf.shape
    res = np.empty_like(destf)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                val = destf[i, j, k] * ratio_blur[i, j, k]
                res[i, j, k] = val
    return res

def lucy_richardson_gauss(srcf, iteration):
    # 出力用の画像を初期化
    destf = srcf

    for i in range(iteration):
        # ガウスぼかしを適用してぼけをシミュレーション (OpenCV)
        bdest = gaussian_blur(destf, ksize=(9, 9), sigma=0)

        # 元画像とぼけた画像の比を計算 (Numba)
        ratio = _lucy_ratio_step(srcf, bdest)

        # 誤差の分配のために再びガウスぼかしを適用 (OpenCV)
        ratio_blur = gaussian_blur(ratio, ksize=(9, 9), sigma=0)

        # 元の出力画像に誤差を乗算 (Numba)
        destf = _lucy_update_step(destf, ratio_blur)
    
    return destf

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

def highlight_compress(image):
    import cores.aces_tonemapping as aces_tonemapping
    
    return aces_tonemapping.aces_tonemapping(image, 0.7, config.get_config('gpu_device'))

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
    # 補正色のマップを作成
    correction = np.zeros_like(image_rgb, dtype=np.float32)
    for i in range(3):
        correction[:,:,i] = solid_color[i]
        
    # 元の画像と補正色をブレンド
    return cv2.addWeighted(image_rgb, 1.0 - opacity, correction, opacity, 0.0)
    #return image_rgb * (1-opacity) + correction * opacity

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
    result = 1 - (1.0-base)*(1-over)

    return result

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

def apply_level_adjustment(image, black_level, midtone_level, white_level):
    """
    Photoshop風のレベル補正を適用する関数
    
    Args:
        image: 入力画像 (0.0-1.0の範囲)
        black_level: 黒レベル (0-255)
        midtone_level: 中間調レベル (0-255, 128が中性)
        white_level: 白レベル (0-255)
    
    Returns:
        調整された画像 (0.0-1.0の範囲)
    """
    
    # 16ビット画像の最大値
    max_val = 65535
    
    # 入力レベルを16ビット範囲に変換
    black_16bit = black_level * 256
    white_16bit = white_level * 256
    
    # midtone_levelを黒レベルと白レベルの範囲でクリップして再マッピング
    # Photoshopでは、midtone_levelは黒レベルと白レベルの間の相対位置を表す
    clipped_midtone = max(min(midtone_level, white_level), black_level)

    # 黒レベルと白レベルの範囲で正規化（0-1）
    if white_level > black_level:
        midtone_normalized = (clipped_midtone - black_level) / (white_level - black_level)
    else:
        midtone_normalized = 0.5  # 範囲が無効な場合は中性値
    
    # 正規化された値（0-1）をガンマ値に変換
    # 0.5が中性（ガンマ1.0）、0に近いほど明るく、1に近いほど暗く
    if midtone_normalized < 0.5:
        # 0-0.5の範囲を0.1-1.0のガンマ値にマッピング（明るく）
        gamma = 0.1 + (midtone_normalized / 0.5) * 0.9
    else:
        # 0.5-1.0の範囲を1.0-9.99のガンマ値にマッピング（暗く）
        gamma = 1.0 + ((midtone_normalized - 0.5) / 0.5) * 8.99
    
    # 入力画像を16ビット範囲に変換
    image_16bit = image * max_val
    
    # レベル調整の計算（Photoshop準拠）
    # 1. 黒レベル以下を0にクリップ
    adjusted = np.maximum(image_16bit - black_16bit, 0)
    
    # 2. 入力範囲を0-1に正規化
    input_range = white_16bit - black_16bit
    input_range = np.maximum(input_range, 1.0)  # 0除算を防ぐ
    normalized = adjusted / input_range
    
    # 3. 1.0以上をクリップ
    #normalized = np.minimum(normalized, 1.0)
    
    # 4. ガンマ補正を適用（正規化された0-1範囲に対して）
    result = np.power(normalized, gamma).astype(np.float32)
    
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
    from scipy.interpolate import splprep, splev
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

def apply_lut(img, lut, max_value=1.0):
    """
    画像にLUTを適用する関数
    max_value: LUTが対応する最大値（デフォルト1.0）
    """
    # スケーリングしてLUTのインデックスに変換
    scale_factor = 65535 / max_value
    lut_indices = np.clip(np.round(img * scale_factor), 0, 65535).astype(np.uint16)

    # LUTを適用
    result = np.take(lut, lut_indices)
    
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
@lock_numba
@njit(parallel=True, fastmath=True, cache=True)
def apply_vignette(image, intensity, radius_percent, disp_info, crop_rect, offset, gradient_softness=4.0):
    intensity = intensity / 100.0
    radius_percent = radius_percent / 100.0
    gradient_softness = max(0.1, gradient_softness)
    
    h, w = image.shape[:2]
    
    dx, dy, _, _, scale = disp_info
    
    x1, y1, x2, y2 = crop_rect
    offset_x, offset_y = offset
        
    center_x = (x1 + (x2 - x1) / 2 - dx) * scale + offset_x
    center_y = (y1 + (y2 - y1) / 2 - dy) * scale + offset_y
    
    mm = max((x2 - x1), (y2 - y1)) * scale
    max_radius = math.sqrt(mm**2 + mm**2) / 2
    
    radius = max_radius * radius_percent
    
    res = np.empty_like(image)
    
    # 3 channels assumed
    c = 3
    if image.ndim == 2:
        c = 1
    
    for y in prange(h):
        for x in prange(w):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            val = dist / radius
            if val > 1.0: val = 1.0
            elif val < 0.0: val = 0.0
            
            # pow and smoothstep
            mask = val ** gradient_softness
            mask = mask * mask * (3 - 2 * mask)
            
            if intensity < 0:
                vig = 1.0 + intensity * mask
                if c == 3:
                    for k in range(3):
                        v = image[y, x, k] * vig
                        # v is float32
                        res[y, x, k] = v
                else:
                    v = image[y, x] * vig
                    res[y, x] = v
            else:
                vig = 1.0 - intensity * mask
                if c == 3:
                     for k in range(3):
                        v = image[y, x, k]
                        # image + (1-image)*(1-vignette) -> v + (1-v)*(1-vig)
                        v_out = v + (1.0 - v) * (1.0 - vig)
                        res[y, x, k] = v_out
                else:
                    v = image[y, x]
                    v_out = v + (1.0 - v) * (1.0 - vig)
                    res[y, x] = v_out
    return res

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

def crop_image(image, disp_info, crop_rect, texture_width, texture_height, click_x, click_y, is_zoomed, center_pos=None):

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
        resized_img = cv2.resize(image[dy:dy+dh, dx:dx+dw], (new_width, new_height), interpolation=cv2.INTER_AREA)

        # リサイズした画像を中央に配置
        result = np.pad(resized_img, ((offset_y, texture_height-(offset_y+new_height)), (offset_x, texture_width-(offset_x+new_width)), (0, 0)), mode="constant")

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

        if center_pos is not None:
             # 中心座標指定
            crop_x = center_pos[0] - crop_width // 2
            crop_y = center_pos[1] - crop_height // 2

        else:
            # 既にズーム済み（scale == 1.0）なら位置を維持
            if abs(scale - 1.0) < 0.01:
                crop_x = disp_info[0]
                crop_y = disp_info[1]
            else:
                # クリック位置を中心にする
                crop_x = disp_info[0] + click_image_x - crop_width // 2
                crop_y = disp_info[1] + click_image_y - crop_height // 2

        # クロップ
        result, disp_info = crop_image_info(image, (crop_x, crop_y, crop_width, crop_height, 1.0), crop_rect)
    
    return result, disp_info


def crop_image_info(image, disp_info, crop_rect):
    
    # 情報取得
    image_height, image_width = image.shape[:2]
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
    
    threshold = 0.6
    transition_width = 0.4
    t = (base - threshold) / transition_width
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    smooth_mask = t * t * (3.0 - 2.0 * t)
    
    suppression_alpha = 10.0
    adaptive_factor = 1.0 / (1.0 + suppression_alpha * abs(detail))
    detail_boost = 1.02
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
    
    threshold = 0.8
    transition_width = 0.4
    t = (base - threshold) / transition_width
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    smooth_mask = t * t * (3.0 - 2.0 * t)
    
    suppression_alpha = 10.0
    adaptive_factor = 1.0 / (1.0 + suppression_alpha * abs(detail))
    detail_boost = 1.02
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

def adjust_tone(img, highlights=0, shadows=0, midtone=0, white_level=0, black_level=0, disp_scale=1.0, resolution_scale=1.0):
    """
    Lightroom風のシャドウ、ハイライト、白レベル、黒レベル調整を行う関数。
    (Numba実装版 - JAX/Numpy依存なし, 高速化)
    """
    # Step 1: Luminance
    y_orig = get_luminance(img)
    
    # Step 2: Mid -> Shadow
    current_y = _kernel_mid_shadow(y_orig, midtone, shadows)
    
    # Step 3: Highlight -> Black
    if highlights < 0:
        sigma = 20.0 * resolution_scale
        y_blur = gaussian_blur_cv(current_y, sigma=sigma)
        current_y = _kernel_high_neg_black(current_y, y_blur, highlights, black_level)
    else:
        current_y = _kernel_high_pos_black(current_y, highlights, black_level)
        
    # Step 4: White -> Final
    if white_level < 0:
        sigma = 20.0 * resolution_scale
        y_blur = gaussian_blur_cv(current_y, sigma=sigma)
        max_val_blur = np.max(y_blur)
        res = _kernel_white_neg_final(img, current_y, y_blur, y_orig, white_level, float(max_val_blur))
    else:
        max_val = np.max(current_y)
        res = _kernel_white_pos_final(img, current_y, y_orig, white_level, float(max_val))
        
    return res


# 画像のサイズを取得する関数
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
    
    img: 入力画像（0-1の範囲のfloat32、RGB形式）
    params: 線形モデルの係数 (β0, β1, β2)
    sigma: ガウシアンフィルタのシグマ値
    
    Zhu らの論文 "Fast Single Image Haze Removal Using Color Attenuation Prior" に基づく
    """
    # RGB画像をHSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # 線形モデルを使用して深度を推定: d = β0 + β1 * v + β2 * s
    beta0, beta1, beta2 = params
    depth = beta0 + beta1 * v + beta2 * s
    
    # フィルタリングで深度マップを滑らかにする
    depth = cv2.GaussianBlur(depth, (0, 0), sigma)
    
    # 正規化（0-1の範囲に変換）
    mmin = np.min(depth)
    mmax = np.max(depth)
    depth = (depth - mmin) / (mmax - mmin + 1e-8)
    
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

def dehaze_image(img, strength=0.5):
    """
    色線形変換先行法を使用した霞除去・霧追加 (Numba Optimized)
    
    img: 入力画像（0-1の範囲のfloat32、RGB形式）
    strength: 霞除去（正の値）または霧追加（負の値）の強さ、-1から1の範囲
    """
    
    if strength >= 0:
        # 霞除去モード
        # 深度マップの推定
        depth_map = _estimate_depth_map(img)
        # 大気光の推定
        A = _estimate_atmospheric_light(img, depth_map)
        
        effective_strength = strength
        # 透過率の推定
        transmission = _estimate_transmission(depth_map, effective_strength)

        # 霞補正された画像の計算（大気散乱モデル）
        result = _kernel_dehaze_apply(img, A, transmission)
    
    else:
        # ===== ヘイズ追加処理（霞を増やす）=====
        # Simple Atmospheric Scattering Modelを使用
        
        haze_strength = -strength  # 強度を正の値に変換
        
        # 画像サイズを取得
        h, w = img.shape[:2]
        
        # 強度に応じて透過量を滑らかに調整
        min_trans = 0.4  # 最小透過量（最大霞）
        
        # 二次関数で滑らかな遷移を作成
        transmission_value = 1.0 - (1.0 - min_trans) * (haze_strength * haze_strength)
        
        # 均一な透過量で霞を生成
        transmission = np.ones((h, w), dtype=np.float32) * transmission_value
        
        # 散乱モデルによる霞の合成
        result = _kernel_fog_apply_2d(img, transmission)

    return result

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
            # L_new = L * (2.0 ** (2.0 * TotalAdjL))
            l_factor = 2.0 ** (adj_l * 2.0)
            new_l = hls_img[i, j, 1] * l_factor
            
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
            output[i, j, 1] = new_l
            output[i, j, 2] = new_s
            
            # Extra Channels
            if c > 3:
                for k in range(3, c):
                     output[i, j, k] = hls_img[i, j, k]
                     
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

        cs.adjust = np.array(s['adjust'], dtype=np.float32)
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
        'width': [15.0, 8.0],
        'fade_width': [20.0, 10.0],
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
}

def get_icc_profile_name(pil_image):
    icc_data = pil_image.info.get("icc_profile")
    
    if not icc_data:
        return 'sRGB IEC61966-2.1'

    profile = ImageCms.getOpenProfile(io.BytesIO(icc_data))
    
    return profile.profile.profile_description

def apply_zero_wrap(img, param):
    """
    Zero-wrapフィルタを適用する関数
    """        
    disp_info = params.get_disp_info(param)
    width = int((disp_info[2]) * disp_info[4])
    height = int((disp_info[3]) * disp_info[4])
    width, height = min(width, img.shape[1]), min(height, img.shape[0]) # 安全策
    wrap = np.ones((height, width), dtype=np.float32)
    preview_width = config.get_config('preview_width')
    preview_height = config.get_config('preview_height')
    offset_x, offset_y = (preview_width - wrap.shape[1]) // 2, (preview_height - wrap.shape[0]) // 2
    wrap = np.pad(wrap, ((offset_y, preview_height-wrap.shape[0]-offset_y), (offset_x, preview_width-wrap.shape[1]-offset_x)), 'constant', constant_values=0.0)

    # クロップ中は処理しないがクロップしている範囲のzero_countだけ返す
    if param.get('crop_enable', False) == False:
        img = img * wrap[..., np.newaxis]

    return (img, wrap.size - np.count_nonzero(wrap))

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

def apply_film_grain(
    image: np.ndarray,
    intensity: float = 0.5,
    grain_size: float = 2.0,
    blue_sensitivity: float = 1.2,
    shadow_boost: float = 1.0,
    color_noise_ratio: float = 0.1
) -> np.ndarray:
    """
    改良版フィルム粒状感適用関数
    - shadow_boostの効果範囲を明確化
    - color_noise_ratioの効果を強調
    - 出力をfloat32で保証
    """
    # 入力検証
    H, W, _ = image.shape
    
    # 1. 明度マスク生成（効果を明確化）
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = lab[:,:,0] / 100.0
    
    # shadow_boostの効果を強化（0.5-2.0の範囲で効果が明確に）
    shadow_mask = np.exp(-shadow_boost * 10.0 * L)  # 係数を調整
    
    # 2. モノクログレイン生成
    mono_grain = np.random.randn(H, W).astype(np.float32)
    
    # 粒子サイズ再現
    kernel_size = max(3, int(grain_size * 3)) | 1
    mono_grain = gaussian_blur_cv(
        mono_grain, 
        (kernel_size, kernel_size), 
        grain_size
    )
    
    # 3. 色ノイズ成分（効果を明確化）
    color_noise = np.random.randn(H, W, 3).astype(np.float32) * 0.2  # ベース強度増加
    
    # 色ノイズにブラーを適用（低周波数化）
    color_noise = gaussian_blur_cv(
        color_noise, 
        (kernel_size, kernel_size), 
        grain_size*1.5
    )
    
    # 4. ノイズ合成（モノクロ + カラー）
    channel_weights = np.array([blue_sensitivity, 1.0, 0.8], dtype=np.float32)
    combined_grain = (
        mono_grain[..., np.newaxis] * channel_weights * (1 - color_noise_ratio) + 
        color_noise * color_noise_ratio
    )
    
    # 5. 適応的強度調整
    adaptive_intensity = intensity * 0.2 * shadow_mask[..., np.newaxis]  # 係数調整
    grain = combined_grain * adaptive_intensity
    
    # 6. ノイズ付加とクリッピング（float32保持）
    noisy_image = image + grain
    return noisy_image

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

from skimage.restoration import denoise_bilateral

def chromatic_aberration_correction(
    image: np.ndarray, 
    intensity: float = 1.0,
    edge_threshold: float = 0.05,
    channel_align_iter: int = 3
) -> np.ndarray:
    """
    製品レベルの倍率色収差補正
    
    パラメータ:
        image: 入力画像 (H×W×3, float32 [0,1]形式)
        intensity: 補正強度 (0.0 ~ 2.0)
        edge_threshold: エッジ検出閾値 (0.01 ~ 0.1)
        channel_align_iter: チャンネルアライメント反復回数
        
    戻り値:
        補正済み画像 (入力と同じ形式)
    """
    # 入力検証
    assert image.dtype == np.float32, "Input must be float32"
    assert image.ndim == 3 and image.shape[2] == 3, "Input must be RGB"
    assert np.max(image) <= 1.0 and np.min(image) >= 0.0, "Range must be [0,1]"
    
    # 元画像をコピー
    corrected = image.copy()
    h, w, _ = image.shape
    
    # グレースケール変換 (輝度ベース)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # エッジマスク生成 (Canny + 拡張)
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150) / 255
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)
    
    # チャンネルアライメント (緑チャンネル基準)
    aligned_rgb = np.zeros_like(image)
    aligned_rgb[:, :, 1] = image[:, :, 1]  # 緑チャンネルは基準
    
    # マルチスケール位相相関によるアライメント
    for channel in [0, 2]:  # Red and Blue channels
        chan_img = image[:, :, channel]
        ref_img = image[:, :, 1]  # Green channel reference
        
        total_dx, total_dy = 0.0, 0.0
        
        # マルチスケールアライメント
        for scale in range(channel_align_iter, 0, -1):
            scale_factor = 1 / (2 ** (scale - 1))
            scaled_ref = cv2.resize(ref_img, (0,0), fx=scale_factor, fy=scale_factor)
            scaled_chan = cv2.resize(chan_img, (0,0), fx=scale_factor, fy=scale_factor)
            
            # 位相相関によるシフト検出
            shift, _ = cv2.phaseCorrelate(
                scaled_ref.astype(np.float32), 
                scaled_chan.astype(np.float32)
            )
            
            total_dx += shift[0] * scale_factor
            total_dy += shift[1] * scale_factor
        
        # 平均シフト量計算
        dx = total_dx / channel_align_iter
        dy = total_dy / channel_align_iter
        
        # シフト適用
        M = np.float32([[1, 0, dx * intensity], [0, 1, dy * intensity]])
        aligned_rgb[:, :, channel] = cv2.warpAffine(
            chan_img, M, (w, h), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP
        )
    
    # 色収差マスク生成 (色差ベース)
    r_diff = np.abs(aligned_rgb[:, :, 0] - image[:, :, 1])
    b_diff = np.abs(aligned_rgb[:, :, 2] - image[:, :, 1])
    chroma_mask = np.clip((r_diff + b_diff) * 5.0, 0, 1)
    chroma_mask = np.where(edge_mask > 0, chroma_mask, 0)
    
    # 適応型バイラテラルフィルタリング
    for c in range(3):
        # エッジ領域のみを選択的に補正
        corrected_channel = corrected[:, :, c]
        aligned_channel = aligned_rgb[:, :, c]
        
        # バイラテラルフィルタパラメータ調整
        sigma_color = 0.05 + (0.1 * intensity)
        sigma_spatial = max(1.0, 3.0 * intensity)
        
        # マスク領域のみフィルタ適用
        filtered = denoise_bilateral(
            aligned_channel,
            win_size=5,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            channel_axis=None
        )
        
        # マスクに基づいてブレンド
        corrected[:, :, c] = np.where(
            chroma_mask > edge_threshold,
            filtered,
            corrected_channel
        )
    
    return corrected

#-------------------------------------------------

_lensfun_db_instance = None
def _get_lensfun_db():
    global _lensfun_db_instance
    import lensfunpy
    if _lensfun_db_instance is None:
        _lensfun_db_instance = lensfunpy.Database()
    return _lensfun_db_instance

def setup_lensfun(img_size, exif_data):
    global __lensfun_mod

    make =  exif_data.get('Make', None)
    model = exif_data.get('Model', None)
    lensmake = exif_data.get('LensMake', None)
    lensmodel = exif_data.get('LensModel', None)
    focal_length = exif_data.get('FocalLength', None)
    aperture = exif_data.get('ApertureValue',  exif_data.get('Aperture', None))
    distance = exif_data.get('SubjectDistanceRange', 100)

    logging.info(f"{make}, {model}")
    logging.info(f"{lensmake}, {lensmodel}")
    logging.info(f"{focal_length}, {aperture}, {distance}")

    if focal_length is None or aperture is None:
        logging.info("focal_length or aperture is None")
        return

    if distance == 'Unknown' or distance == 'Close':
        distance = 100

    import lensfunpy
    db = _get_lensfun_db()
    cams = db.find_cameras(make, model, loose_search=True)
    if len(cams) > 0:
        lens = db.find_lenses(cams[0], lensmake, lensmodel, loose_search=False)

        if len(lens) > 0:
            width, height = img_size
            __lensfun_mod = lensfunpy.Modifier(lens[0], cams[0].crop_factor, width, height)
            __lensfun_mod.initialize(float(focal_length[0:-3]), aperture, distance, pixel_format=np.float32)
            return

    __lensfun_mod = None

def clean_lensfun():
    global __lensfun_mod
    __lensfun_mod = None

__lensfun_mod = None

def modify_lensfun(img, is_cm=True, is_sd=True, is_gd=True):

    if __lensfun_mod is None:
        logging.warning("Lensfun is not initialized")
        return (img, False, False, False)

    mod = __lensfun_mod
    modimg = img
    if is_cm == True:
        modimg = img.copy()
        did_apply = mod.apply_color_modification(modimg)
        if did_apply == False:
            logging.warning("Apply Color Modification is Failed")
            is_cm = False

    if is_sd == True:
        undist_coords = mod.apply_subpixel_distortion()
        if undist_coords is None:
            logging.warning("Apply Subpixel Distortion is Failed")
            is_sd = False
        else:
            modimg[..., 0] = cv2.remap(modimg[..., 0], undist_coords[..., 0, :], None, cv2.INTER_LANCZOS4)
            modimg[..., 1] = cv2.remap(modimg[..., 1], undist_coords[..., 1, :], None, cv2.INTER_LANCZOS4)
            modimg[..., 2] = cv2.remap(modimg[..., 2], undist_coords[..., 2, :], None, cv2.INTER_LANCZOS4)

    if is_gd == True:
        undist_coords = mod.apply_geometry_distortion()
        if undist_coords is None:
            logging.warning("Apply Geometry Distortion is Failed")
            is_gd = False
        else:
            modimg = cv2.remap(modimg, undist_coords, None, cv2.INTER_LANCZOS4)

    return (modimg, is_cm, is_sd, is_gd)

#-------------------------------------------------

def light_denoise(img, its, col):

    # YCrCb色空間に変換 (HDR対応・リニア変換)
    # Yは輝度、Cr, Cbは色差
    # float32の場合、Y, Cr, Cb ともに概ね 0.0-1.0 (HDRならそれ以上) の範囲で扱われる
    # (Cr, Cbは 0.5 が中心)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # 輝度チャンネル(Y)のノイズ除去 (Guided Filter)
    if its > 0:
        # 半径
        radius = max(2, int(its * 0.01))
        # イプシロンの計算
        # Yは0-1スケール (HDR対応)
        # eps = (閾値)^2
        # its=100 で 0.2 (20%) 程度の変動を平滑化
        eps = ((its * 0.002) ** 2)
        
        # 分散安定化変換: sqrt(Y) をとることで、ショットノイズ(値に比例して分散が増える)を均一化する
        # これによりハイライト部でもノイズ除去が効くようになる
        sq_y = np.sqrt(np.maximum(y, 0))
        sq_y = cv2.ximgproc.guidedFilter(guide=sq_y, src=sq_y, radius=radius, eps=eps)
        y = sq_y ** 2

    # 色度チャンネル(Cr, Cb)のノイズ除去 (Joint Guided Filter)
    if col > 0:
        # 色度ノイズは強めにかけるため半径を大きく
        radius = max(4, int(col * 0.3))
        # 色差も0-1スケール
        eps = ((col * 0.005) ** 2)
        
        # 輝度(Y)をガイドにして色差(Cr, Cb)をフィルタリング
        cr = cv2.ximgproc.guidedFilter(guide=y, src=cr, radius=radius, eps=eps)
        cb = cv2.ximgproc.guidedFilter(guide=y, src=cb, radius=radius, eps=eps)
 
    # チャンネルを結合
    filtered_ycrcb = cv2.merge([y, cr, cb])
    return cv2.cvtColor(filtered_ycrcb, cv2.COLOR_YCrCb2RGB)

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

