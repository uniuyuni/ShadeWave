"""
座標変換ユーティリティ

3つの座標系を扱う:
- TCG座標系: 画像中心原点、X右、Y下、画像サイズで正規化（保存・API用）
- 画像座標系: 左上原点、X右、Y下（内部処理用）
- Kivy座標系: 左下原点、X右、Y上（GUI用）
"""

import numpy as np


def tcg_to_image_coords(tcg_point, image_shape):
    """
    TCG座標を画像座標に変換
    
    TCG座標系: 画像中心が原点、X右が正、Y下が正、画像サイズで正規化
    例: 画像が640x480の場合
        - 中心: (0, 0) → (319.5, 239.5)
        - 左上隅: (-0.5, -0.5) → (0, 0)
        - 右上隅: (0.5, -0.5) → (639, 0)
        - 右下隅: (0.5, 0.5) → (639, 479)
        - 左下隅: (-0.5, 0.5) → (0, 479)
    
    Args:
        tcg_point: tuple (x, y)、TCG座標系（正規化済み）
        image_shape: tuple (H, W)
    
    Returns:
        tuple (x, y)、画像座標系（ピクセル単位）
    """
    x_tcg, y_tcg = tcg_point
    height, width = image_shape
    
    # エッジケース: 1x1画像
    if width == 1 and height == 1:
        return (0.0, 0.0)
    
    # 正規化座標をピクセル座標に変換
    # TCG範囲[-0.5, 0.5]を画像範囲[0, width-1], [0, height-1]にマップ
    
    x_img = (x_tcg + 0.5) * (width - 1)
    y_img = (y_tcg + 0.5) * (height - 1)
    
    return (x_img, y_img)


def image_to_tcg_coords(image_point, image_shape):
    """
    画像座標をTCG座標に変換
    
    Args:
        image_point: tuple (x, y)、画像座標系（ピクセル単位）
        image_shape: tuple (H, W)
    
    Returns:
        tuple (x, y)、TCG座標系（正規化済み）
    """
    x_img, y_img = image_point
    height, width = image_shape
    
    # エッジケース: 1x1画像
    if width == 1 and height == 1:
        return (0.0, 0.0)
    
    # ピクセル座標を正規化座標に変換
    # 画像範囲[0, width-1], [0, height-1]をTCG範囲[-0.5, 0.5]にマップ
    
    x_tcg = x_img / (width - 1) - 0.5
    y_tcg = y_img / (height - 1) - 0.5
    
    return (x_tcg, y_tcg)


def kivy_to_tcg_coords(kivy_point, widget_size, image_shape):
    """
    Kivy座標をTCG座標に変換
    
    Args:
        kivy_point: tuple (x, y)、Kivy座標系
        widget_size: tuple (widget_width, widget_height)
        image_shape: tuple (H, W)
    
    Returns:
        tuple (x, y)、TCG座標系（正規化済み）
    """
    x_kivy, y_kivy = kivy_point
    widget_width, widget_height = widget_size
    height, width = image_shape
    
    # Kivy座標を正規化 [0, widget_size] → [0, 1]
    x_norm = x_kivy / widget_width
    y_norm = y_kivy / widget_height
    
    # Y軸反転 (Kivy上→画像下)
    y_norm_flipped = 1.0 - y_norm
    
    # 正規化座標[0, 1]をTCG座標[-0.5, 0.5]に変換
    x_tcg = x_norm - 0.5
    y_tcg = y_norm_flipped - 0.5
    
    return (x_tcg, y_tcg)


def tcg_to_kivy_coords(tcg_point, widget_size, image_shape):
    """
    TCG座標をKivy座標に変換
    
    Args:
        tcg_point: tuple (x, y)、TCG座標系（正規化済み）
        widget_size: tuple (widget_width, widget_height)
        image_shape: tuple (H, W)
    
    Returns:
        tuple (x, y)、Kivy座標系
    """
    x_tcg, y_tcg = tcg_point
    widget_width, widget_height = widget_size
    height, width = image_shape
    
    # TCG座標[-0.5, 0.5]を正規化座標[0, 1]に変換
    x_norm = x_tcg + 0.5
    y_norm = y_tcg + 0.5
    
    # Y軸反転 (画像下→Kivy上)
    y_norm_flipped = 1.0 - y_norm
    
    # 正規化座標をKivy座標に変換
    x_kivy = x_norm * widget_width
    y_kivy = y_norm_flipped * widget_height
    
    return (x_kivy, y_kivy)


def tcg_points_to_image(tcg_points, image_shape):
    """
    複数のTCG座標を画像座標に一括変換
    
    Args:
        tcg_points: list of tuples [(x, y), ...]
        image_shape: tuple (H, W)
    
    Returns:
        list of tuples [(x, y), ...]、画像座標系
    """
    return [tcg_to_image_coords(pt, image_shape) for pt in tcg_points]


def image_points_to_tcg(image_points, image_shape):
    """
    複数の画像座標をTCG座標に一括変換
    
    Args:
        image_points: list of tuples [(x, y), ...]
        image_shape: tuple (H, W)
    
    Returns:
        list of tuples [(x, y), ...]、TCG座標系
    """
    return [image_to_tcg_coords(pt, image_shape) for pt in image_points]


def tcg_lines_to_image(tcg_lines, image_shape):
    """
    複数の線をTCG座標から画像座標に変換
    
    Args:
        tcg_lines: list of tuples [((x1,y1), (x2,y2)), ...]
        image_shape: tuple (H, W)
    
    Returns:
        list of tuples [((x1,y1), (x2,y2)), ...]、画像座標系
    """
    result = []
    for (pt1, pt2) in tcg_lines:
        img_pt1 = tcg_to_image_coords(pt1, image_shape)
        img_pt2 = tcg_to_image_coords(pt2, image_shape)
        result.append((img_pt1, img_pt2))
    return result


def image_lines_to_tcg(image_lines, image_shape):
    """
    複数の線を画像座標からTCG座標に変換
    
    Args:
        image_lines: list of tuples [((x1,y1), (x2,y2)), ...]
        image_shape: tuple (H, W)
    
    Returns:
        list of tuples [((x1,y1), (x2,y2)), ...]、TCG座標系
    """
    result = []
    for (pt1, pt2) in image_lines:
        tcg_pt1 = image_to_tcg_coords(pt1, image_shape)
        tcg_pt2 = image_to_tcg_coords(pt2, image_shape)
        result.append((tcg_pt1, tcg_pt2))
    return result
