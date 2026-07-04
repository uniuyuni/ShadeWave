"""
4点補正API

4点を指定して透視変換による補正を行う
"""

import numpy as np
import cv2
import numba
from typing import Tuple, Optional


def correct_four_points(
    image: np.ndarray,
    src_points: list,
    dst_points: list,
    interpolation: str = 'bicubic',
    homography: np.ndarray = None
) -> np.ndarray:
    """
    4点を指定して透視変換
    
    Args:
        image: numpy.ndarray、dtype=float32、shape=(H, W, 3)
        src_points: list of 4 tuples、画像座標系の4点 [(x, y), ...]
            左上、右上、右下、左下の順
        interpolation: str、'bilinear' | 'bicubic'
    
    Returns:
        補正後画像
    """
    if image.dtype != np.float32:
        raise TypeError(f"image.dtype must be float32, got {image.dtype}")

    # 目的画像のサイズ
    height, width = image.shape[:2]

    if homography is not None:
        # 呼び出し側で計算・調整済み (減衰クランプ等) の順変換(src->dst)ホモグラフィを使う。
        # remap は出力->入力(dst->src)写像が必要なので逆行列にする。
        H = np.linalg.inv(np.asarray(homography, dtype=np.float32))
    else:
        if len(src_points) != 4:
            raise ValueError(f"src_points must have 4 points, got {len(src_points)}")

        if len(dst_points) != 4:
            raise ValueError(f"dst_points must have 4 points, got {len(dst_points)}")

        # 目標点
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # マップ生成 (dst->src)
        H = calculate_four_point_homography(src_points, dst_points)
    
    # メッシュグリッド生成 (shape: (H, W))
    # indexing='xy' (デフォルト) で X(幅方向), Y(高さ方向) のグリッドを作成
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    
    ones = np.ones(X.shape, dtype=np.float32)
    coords = np.stack([X, Y, ones], axis=-1)  # ホモグラファス座標 (H, W, 3)
    
    # 行列演算 (H, W, 3) @ (3, 3).T -> (H, W, 3)
    coords_h = coords @ H.T
    
    map1 = coords_h[:,:,0] / coords_h[:,:,2]  # 正規化 x/w
    map2 = coords_h[:,:,1] / coords_h[:,:,2]  # 正規化 y/w
    map1 = map1.astype(np.float32)
    map2 = map2.astype(np.float32)

    # 補間方法の選択
    interp_flags = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    if interpolation not in interp_flags:
        raise ValueError(f"未対応の補間方法: {interpolation}")
    
    # リマップ
    corrected = cv2.remap(
        image,
        map1,
        map2,
        interpolation=interp_flags[interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # 順変換行列を返す (Src -> Dst)
    return corrected, np.linalg.inv(H)

def calculate_four_point_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    4点からホモグラフィ行列（逆変換: dst -> src）を計算する
    
    Args:
        src_points: 画像座標系の4点 [(x, y), ...] (変換前)
        dst_points: 画像座標系の4点 [(x, y), ...] (変換後)
        
    Returns:
        H: 3x3 行列 (dst -> src)
    """
    # 逆変換行列（出力座標 -> 元座標）を求めるため (dst, src) の順
    # これにより、出力画像の各ピクセルが入力画像のどこに対応するかを求める
    if not isinstance(dst_points, np.ndarray):
        dst_points = np.array(dst_points, dtype=np.float32)
    if not isinstance(src_points, np.ndarray):
        src_points = np.array(src_points, dtype=np.float32)
        
    return cv2.getPerspectiveTransform(dst_points, src_points)

def detect_rectangle(
    image: np.ndarray,
    min_area_ratio: float = 0.1,
    max_area_ratio: float = 0.9,
    aspect_ratio_range: Tuple[float, float] = (0.3, 3.0)
) -> Optional[list]:
    """
    画像から矩形を検出（簡易版）
    
    Args:
        image: numpy.ndarray、dtype=float32、shape=(H, W, 3)
        min_area_ratio: float、最小面積比率
        max_area_ratio: float、最大面積比率
        aspect_ratio_range: tuple、アスペクト比の範囲 (min, max)
    
    Returns:
        list of 4 tuples (TCG座標) | None
    """
    if image.dtype != np.float32:
        raise TypeError(f"image.dtype must be float32, got {image.dtype}")
    
    height, width = image.shape[:2]
    
    # グレースケール変換
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # エッジ検出
    edges = cv2.Canny(gray, 50, 150)
    
    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 最大の輪郭を探す
    image_area = width * height
    best_contour = None
    best_area = 0
    
    for contour in contours:
        # 面積チェック
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        
        # 近似（4点に）
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) != 4:
            continue
        
        # アスペクト比チェック
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        
        aspect = max(w, h) / min(w, h)
        if aspect < aspect_ratio_range[0] or aspect > aspect_ratio_range[1]:
            continue
        
        if area > best_area:
            best_area = area
            best_contour = approx
    
    if best_contour is None:
        return None
    
    # 画像座標からTCG座標に変換
    points_tcg = []
    for point in best_contour.reshape(-1, 2):
        x_img, y_img = point
        x_tcg = x_img / width - 0.5
        y_tcg = y_img / height - 0.5  # Fixed inversion
        points_tcg.append((float(x_tcg), float(y_tcg)))
    
    # 左上、右上、右下、左下の順に並べ替え
    # 重心を計算
    center_x = sum(p[0] for p in points_tcg) / 4
    center_y = sum(p[1] for p in points_tcg) / 4
    
    # 角度でソート
    def angle_from_center(p):
        return np.arctan2(p[1] - center_y, p[0] - center_x)
    
    points_sorted = sorted(points_tcg, key=angle_from_center)
    
    # 左上から開始するように回転
    # 最も左上の点（x + yが最小）を見つける
    start_idx = min(range(4), key=lambda i: points_sorted[i][0] + points_sorted[i][1])
    points_ordered = points_sorted[start_idx:] + points_sorted[:start_idx]
    
    return points_ordered
