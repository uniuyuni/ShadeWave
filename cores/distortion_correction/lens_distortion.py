"""
レンズ歪み補正API

樽型・糸巻き型歪みの補正を行う
"""

import numpy as np
import cv2
import logging

# numbaのインポート（オプション）
try:
    from numba import jit
    _numba_available = True
except ImportError:
    _numba_available = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def _validate_image(image: np.ndarray):
    """入力画像の検証"""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be numpy.ndarray, got {type(image)}")
    if image.dtype != np.float32:
        raise TypeError(f"image.dtype must be float32, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")


def correct_lens_distortion(
    image: np.ndarray,
    strength: float,
    interpolation: str = 'bicubic',
    grid_size: int = 1,
    scale: float = 1.0
) -> np.ndarray:
    """
    レンズ歪み補正
    
    Args:
        image: numpy.ndarray、dtype=float32、shape=(H, W, 3)
        strength: float、-100.0〜+100.0
            負: 樽型歪み補正
            正: 糸巻き型歪み補正
        interpolation: str、'bilinear' | 'bicubic'
        grid_size: int、マップ生成のグリッドサイズ（1=全ピクセル、推奨: 2-4）
        scale: float、補正後のスケール倍率（1.0=等倍、>1.0=拡大、<1.0=縮小）
            樽型補正で黒余白を除去: 1.1〜1.3程度
            糸巻き型補正で端を保持: 0.8〜0.95程度
    
    Returns:
        補正後画像
    """
    _validate_image(image)
    
    if not -100.0 <= strength <= 100.0:
        raise ValueError(f"strength must be in range [-100.0, 100.0], got {strength}")
    if interpolation not in ['bilinear', 'bicubic']:
        raise ValueError(f"interpolation must be 'bilinear' or 'bicubic', got {interpolation}")
    if grid_size < 1:
        grid_size = 1
        logging.warning(f"grid_size must be >= 1, got {grid_size}")
    if scale <= 0:
        scale = 0.01
        logging.warning(f"scale must be > 0, got {scale}")
    
    # 補正を適用
    corrected = _apply_lens_distortion_correction(image, strength, interpolation, grid_size)
    
    # スケーリング
    if abs(scale - 1.0) > 0.01:
        corrected = _apply_scale(corrected, scale)
    
    return corrected


def _apply_lens_distortion_correction(image: np.ndarray, strength: float, interpolation: str, grid_size: int) -> np.ndarray:
    """レンズ歪み補正を適用"""
    height, width = image.shape[:2]
    k1 = strength / 200.0
    center_x, center_y = width / 2.0, height / 2.0
    max_radius = np.sqrt(center_x**2 + center_y**2)
    
    # マップ生成
    if _numba_available:
        map_x_grid, map_y_grid = _generate_distortion_map_numba(height, width, center_x, center_y, max_radius, k1, grid_size)
    else:
        map_x_grid, map_y_grid = _generate_distortion_map_python(height, width, center_x, center_y, max_radius, k1, grid_size)
    
    # リサイズ
    if grid_size > 1:
        map_x = cv2.resize(map_x_grid, (width, height), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y_grid, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        map_x, map_y = map_x_grid, map_y_grid
        
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
    corrected = cv2.remap(image, map_x, map_y, interp_flags[interpolation], borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return corrected


def _apply_scale(image: np.ndarray, scale: float) -> np.ndarray:
    """画像をスケーリング"""
    height, width = image.shape[:2]
    
    if scale > 1.0:
        # 拡大してクロップ
        new_width, new_height = int(width * scale), int(height * scale)
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        start_x, start_y = (new_width - width) // 2, (new_height - height) // 2
        result = scaled[start_y:start_y+height, start_x:start_x+width]
    elif scale < 1.0:
        # 縮小してパディング
        new_width, new_height = int(width * scale), int(height * scale)
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        result = np.zeros_like(image)
        start_x, start_y = (width - new_width) // 2, (height - new_height) // 2
        result[start_y:start_y+new_height, start_x:start_x+new_width] = scaled
    else:
        result = image
    
    return result


if _numba_available:
    @jit(nopython=True, cache=True)
    def _generate_distortion_map_numba(height, width, center_x, center_y, max_radius, k1, grid_size):
        """歪み補正マップを生成（Numba JIT版）"""
        grid_h, grid_w = height // grid_size, width // grid_size
        map_x = np.zeros((grid_h, grid_w), dtype=np.float32)
        map_y = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        for i in range(grid_h):
            for j in range(grid_w):
                y, x = i * grid_size, j * grid_size
                dx, dy = (x - center_x) / max_radius, (y - center_y) / max_radius
                r2 = dx * dx + dy * dy
                distortion = 1.0 + k1 * r2
                map_x[i, j] = center_x + dx * distortion * max_radius
                map_y[i, j] = center_y + dy * distortion * max_radius
        
        return map_x, map_y
else:
    _generate_distortion_map_numba = None


def _generate_distortion_map_python(height, width, center_x, center_y, max_radius, k1, grid_size):
    """歪み補正マップを生成（Python版）"""
    grid_h, grid_w = height // grid_size, width // grid_size
    map_x = np.zeros((grid_h, grid_w), dtype=np.float32)
    map_y = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    for i in range(grid_h):
        for j in range(grid_w):
            y, x = i * grid_size, j * grid_size
            dx, dy = (x - center_x) / max_radius, (y - center_y) / max_radius
            r2 = dx * dx + dy * dy
            distortion = 1.0 + k1 * r2
            map_x[i, j] = center_x + dx * distortion * max_radius
            map_y[i, j] = center_y + dy * distortion * max_radius
    
    return map_x, map_y


if not _numba_available:
    _generate_distortion_map_numba = _generate_distortion_map_python


def detect_lens_distortion(image: np.ndarray) -> float:
    """画像から歪みを自動検出（簡易版）"""
    _validate_image(image)
    
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is None or len(lines) < 5:
        raise ValueError("Not enough lines detected")
    
    height, width = image.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0
    
    # 端に近い直線を選択
    edge_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        min_dist = min(min(x1, x2), width - max(x1, x2), min(y1, y2), height - max(y1, y2))
        if min_dist < width * 0.2:
            edge_lines.append(line[0])
    
    if len(edge_lines) < 3:
        raise ValueError("Not enough edge lines")
    
    # 歪み推定
    curvature_sum = 0.0
    for line in edge_lines[:10]:
        x1, y1, x2, y2 = line
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = np.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            curvature_sum += slope * dist
    
    estimated_strength = np.clip(curvature_sum / len(edge_lines) * 0.1, -100.0, 100.0)
    return float(estimated_strength)
