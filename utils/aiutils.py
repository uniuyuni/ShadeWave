
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Cuda用
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()  # MPSバックエンド用

# log1p による HDR 的レンジ圧縮（SCUNet / DemosaicNet 等で k=8 を共有）
LOG1P_TONEMAP_K_DEFAULT = 8.0


def log1p_tonemap_forward(x, k=LOG1P_TONEMAP_K_DEFAULT, clip_nonnegative=True):
    """
    入力を log1p 正規化空間へ写す。clip_nonnegative=True のとき 0 未満をクリップ（SCUNet 前処理用）。
    """
    y = x
    if clip_nonnegative:
        y = np.clip(y, 0.0, None)
    return (np.log1p(k * y) / np.log1p(k)).astype(np.float32)


def log1p_tonemap_inverse(result, k=LOG1P_TONEMAP_K_DEFAULT):
    """log1p_tonemap_forward の逆変換。"""
    return (np.expm1(np.log1p(k) * result) / k).astype(np.float32)


def log1p_tonemap_forward_hdr(
    x,
    k=LOG1P_TONEMAP_K_DEFAULT,
    clip_nonnegative=True,
    white_point=None,
    white_percentile=99.9,
):
    """HDR入力をSCUNet向けの概ね[0,1] log空間へ写す。戻り値は(output, white_point)。"""
    y = np.asarray(x, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    if clip_nonnegative:
        y = np.clip(y, 0.0, None)
    if white_point is None:
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            white_point = 1.0
        else:
            white_point = float(np.percentile(finite, white_percentile))
    white_point = max(float(white_point), 1.0)
    denom = np.log1p(k * white_point)
    result = np.log1p(k * y) / max(float(denom), 1e-6)
    return np.clip(result, 0.0, 1.0).astype(np.float32), white_point


def log1p_tonemap_inverse_hdr(result, white_point, k=LOG1P_TONEMAP_K_DEFAULT):
    """log1p_tonemap_forward_hdr の逆変換。"""
    white_point = max(float(white_point), 1.0)
    return (np.expm1(np.log1p(k * white_point) * result) / k).astype(np.float32)


def calculate_expanded_crop(img_width, img_height, x, y, w, h, width, height):
    """
    関数のパラメータ説明:
    img_width, img_height: 画像の幅と高さ
    x, y, w, h: 切り抜きたい元の範囲（x, yは左上座標、w, hは幅と高さ）
    width, height: 拡張したい目標サイズ

    特徴:
    中心点基準の拡張: 元の範囲の中心を基準に対称的に拡張
    8の倍数サイズ保証: 拡張後のサイズは常に8の倍数
    画像範囲制約: 拡張範囲が画像を超える場合は反対側に拡張せず、画像内で可能な最大サイズを使用
    最小サイズ保証: 拡張後のサイズが元の範囲より小さい場合は元の範囲を返す
    座標調整: 最終的な座標が画像範囲内に収まるよう自動調整
    """

    # 拡張サイズが元の範囲より小さい場合、元の範囲を返す
    if width < w or height < h:
        return (x, y, w, h)
    
    # 中心座標を計算
    cx = x + w // 2
    cy = y + h // 2
    
    # 中心から各方向への最大余白を計算
    left = cx
    right = img_width - cx - 1
    top = cy
    bottom = img_height - cy - 1
    
    # 対称的に拡張可能な最大サイズを計算
    max_possible_width = 2 * min(left, right)
    max_possible_height = 2 * min(top, bottom)
    
    # 拡張サイズを8の倍数に切り上げ
    target_width = ((width + 7) // 8) * 8
    target_height = ((height + 7) // 8) * 8
    
    # 実際の拡張サイズを決定（画像範囲内で可能なサイズ）
    actual_width = min(target_width, max_possible_width)
    actual_height = min(target_height, max_possible_height)
    
    # 8の倍数に切り捨てて調整
    adj_width = (actual_width // 8) * 8
    adj_height = (actual_height // 8) * 8
    
    # 調整後のサイズが元の範囲より小さい場合は元の範囲を返す
    if adj_width < w or adj_height < h:
        return (x, y, w, h)
    
    # 拡張範囲の左上座標を計算
    new_x = cx - adj_width // 2
    new_y = cy - adj_height // 2
    
    # 座標が画像範囲内に収まっていることを確認
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    if new_x + adj_width > img_width:
        new_x = img_width - adj_width
    if new_y + adj_height > img_height:
        new_y = img_height - adj_height
    
    return (new_x, new_y, adj_width, adj_height)

def adjust_to_multiple(image, size=8, mode='constant'):
    """
    画像を指定した倍数（デフォルト8）のサイズにパディングする関数
    画像の下端・右端にパディングが追加される
    """

    # 画像の高さと幅を取得
    h, w = image.shape[:2]
    
    # sizeの倍数に切り上げた新しいサイズを計算
    new_h = (h + size-1) // size * size
    new_w = (w + size-1) // size * size
    
    # パディング量を計算
    pad_h = new_h - h
    pad_w = new_w - w
    
    # パディング幅を設定（次元ごとに指定）
    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2)
    
    # 画像の下側と右側をエッジ値でパディング
    padded_image = np.pad(image, pad_width=pad_width, mode=mode)
    
    return padded_image, (h, w)


def adjust_to_multiple_square(image, size=8, mode='constant'):
    """
    正方形かつ指定した倍数（デフォルト8）のサイズにパディングする関数
    画像の縦横を比較し、大きい方に合わせて正方形にし、さらにsizeの倍数に切り上げてパディングする
    """
    h, w = image.shape[:2]

    # sizeの倍数に切り上げた新しいサイズを計算
    new_w = (w + size-1) // size * size
    new_h = (h + size-1) // size * size
    
    # パディング量を計算
    if new_w < new_h:
        pad_w = new_h - w
        pad_h = new_h - h
    else:
        pad_w = new_w - w
        pad_h = new_w - h

    # パディング幅を設定（次元ごとに指定）
    pad_width = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2)
    
    # 画像の下側と右側をエッジ値でパディング
    padded = np.pad(image, pad_width=pad_width, mode=mode)

    return padded, (h, w)


def restore_original_size(padded_image, original_size):
    # 元のサイズを取得
    h_orig, w_orig = original_size
    
    # パディングされた部分を切り取って元のサイズに復元
    return padded_image[:h_orig, :w_orig, ...]

def downscaler(image, width, height):
    org_h, org_w = image.shape[:2]

    if height == 0:
        if width > org_w: width = org_w
        height = int(width * org_h / org_w)

    resize_img = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)

    return resize_img

def upscaler(image, width, height):
    import helpers.realesrgan_helper as realesrgan_helper

    outscale = width / image.shape[1]

    regan = realesrgan_helper.init_realesrgan()
    result = realesrgan_helper.inference_realesrgan(regan, image, outscale=outscale)

    result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LANCZOS4)

    return result


def print_model_structure(model, indent=0):
    """
    モデルの構造を詳細に表示
    """
    for name, module in model.named_children():
        logging.debug("%s├─ %s: %s", "  " * indent, name, module.__class__.__name__)
        
        if isinstance(module, nn.InstanceNorm2d):
            logging.debug("%s   └─ InstanceNorm2d detected!", "  " * indent)
        
        if len(list(module.children())) > 0:
            print_model_structure(module, indent + 1)
