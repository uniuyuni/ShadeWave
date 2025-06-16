import numpy as np
import cv2

def expand_mask(mask, pixels=1.0):
    """
    マスク画像の範囲を拡張する（膨張処理）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像
    pixels : float
        拡張するピクセル数
        
    Returns:
    --------
    numpy.ndarray : 拡張されたマスク画像
    """
    if pixels <= 0:
        return mask
    
    kernel_size = int(2 * pixels + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # float32に変換（既にfloat32の場合はコピーを避ける）
    if mask.dtype != np.float32:
        mask_float32 = mask.astype(np.float32)
    else:
        mask_float32 = mask
    
    return cv2.dilate(mask_float32, kernel, iterations=1)

def shrink_mask(mask, pixels=1.0):
    """
    マスク画像の範囲を縮小する（収縮処理）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像
    pixels : float
        縮小するピクセル数
        
    Returns:
    --------
    numpy.ndarray : 縮小されたマスク画像
    """
    if pixels <= 0:
        return mask
    
    kernel_size = int(2 * pixels + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # float32に変換（既にfloat32の場合はコピーを避ける）
    if mask.dtype != np.float32:
        mask_float32 = mask.astype(np.float32)
    else:
        mask_float32 = mask
    
    return cv2.erode(mask_float32, kernel, iterations=1)

def adjust_mask_range(mask, pixels):
    """
    マスク画像の範囲を調整する（正の値で拡張、負の値で縮小）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像
    pixels : float
        調整するピクセル数（正：拡張、負：縮小）
        
    Returns:
    --------
    numpy.ndarray : 調整されたマスク画像
    """
    if pixels > 0:
        return expand_mask(mask, pixels)
    elif pixels < 0:
        return shrink_mask(mask, abs(pixels))
    else:
        return mask

def _calculate_adaptive_pixels(area, base_pixels, min_pixels=1.0, max_pixels=None):
    """
    領域の面積に基づいて適応的なピクセル数を計算
    
    Parameters:
    -----------
    area : int
        領域の面積（ピクセル数）
    base_pixels : float
        基準となるピクセル数
    min_pixels : float
        最小ピクセル数
    max_pixels : float
        最大ピクセル数（Noneの場合は制限なし）
        
    Returns:
    --------
    float : 適応的に調整されたピクセル数
    """
    # 面積の平方根に比例してピクセル数を調整（円形を想定）
    # 基準面積を1000ピクセルとして正規化
    base_area = 1000.0
    area_ratio = np.sqrt(area / base_area)
    adaptive_pixels = max(min_pixels, base_pixels * area_ratio)
    
    if max_pixels is not None:
        adaptive_pixels = min(adaptive_pixels, max_pixels)
    
    return adaptive_pixels

def _find_components_and_holes(mask):
    """
    マスクから連結成分と穴を検出し、それぞれの情報を返す
    
    Returns:
    --------
    tuple : (foreground_components, hole_components)
        - foreground_components: [(mask, area), ...] 前景成分のリスト
        - hole_components: [(mask, area), ...] 穴成分のリスト
    """
    # バイナリマスクに変換（高速化のため閾値処理を最適化）
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # 外側の輪郭を検出（前景成分）
    contours_ext, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 全ての輪郭を階層付きで検出（穴を含む）
    contours_all, hierarchy = cv2.findContours(mask_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    foreground_components = []
    hole_components = []
    
    # マスクサイズを取得（再利用）
    mask_shape = mask.shape
    
    # 前景成分の処理
    for contour in contours_ext:
        area = cv2.contourArea(contour)
        if area > 0:
            component_mask = np.zeros(mask_shape, dtype=np.uint8)
            cv2.fillPoly(component_mask, [contour], 1)  # drawContoursより高速
            foreground_components.append((component_mask.astype(np.float32), area))
    
    # 穴成分の処理
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] >= 0:  # 親を持つ = 穴
                area = cv2.contourArea(contours_all[i])
                if area > 0:
                    hole_mask = np.zeros(mask_shape, dtype=np.uint8)
                    cv2.fillPoly(hole_mask, [contours_all[i]], 1)  # drawContoursより高速
                    hole_components.append((hole_mask.astype(np.float32), area))
    
    return foreground_components, hole_components

def expand_holes_only(mask, pixels=1.0, adaptive=True):
    """
    閉空間（穴）のみを拡張する（穴を大きくする）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像（1が前景、0が背景・穴）
    pixels : float
        基準となる拡張ピクセル数
    adaptive : bool
        穴のサイズに応じて適応的に調整するかどうか
        
    Returns:
    --------
    numpy.ndarray : 穴が拡張されたマスク画像
    """
    _, hole_components = _find_components_and_holes(mask)
    if not hole_components:
        return mask
    
    result = mask
    
    for hole_mask, area in hole_components:
        if adaptive:
            adj_pixels = _calculate_adaptive_pixels(area, pixels)
        else:
            adj_pixels = pixels
            
        if adj_pixels > 0:
            kernel_size = int(2 * adj_pixels + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            expanded_hole = cv2.dilate(hole_mask, kernel, iterations=1)
            result = result * (1.0 - expanded_hole)
    
    return result

def shrink_holes_only(mask, pixels=1.0, adaptive=True):
    """
    閉空間（穴）のみを縮小する（穴を小さくする）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像（1が前景、0が背景・穴）
    pixels : float
        基準となる縮小ピクセル数
    adaptive : bool
        穴のサイズに応じて適応的に調整するかどうか
        
    Returns:
    --------
    numpy.ndarray : 穴が縮小されたマスク画像
    """
    _, hole_components = _find_components_and_holes(mask)
    if not hole_components:
        return mask
    
    result = mask
    
    for hole_mask, area in hole_components:
        if adaptive:
            adj_pixels = _calculate_adaptive_pixels(area, pixels)
        else:
            adj_pixels = pixels
            
        if adj_pixels > 0:
            kernel_size = int(2 * adj_pixels + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            shrinked_hole = cv2.erode(hole_mask, kernel, iterations=1)
            # 穴を縮小した分だけ前景に変換
            result = np.maximum(result, hole_mask - shrinked_hole)
    
    return result

def adjust_holes_only(mask, pixels, adaptive=True):
    """
    閉空間（穴）のみを調整する（正の値で穴を大きく、負の値で穴を小さく）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像
    pixels : float
        調整するピクセル数（正：穴を拡張、負：穴を縮小）
    adaptive : bool
        穴のサイズに応じて適応的に調整するかどうか
        
    Returns:
    --------
    numpy.ndarray : 穴が調整されたマスク画像
    """
    if pixels > 0:
        return expand_holes_only(mask, pixels, adaptive)
    elif pixels < 0:
        return shrink_holes_only(mask, abs(pixels), adaptive)
    else:
        return mask

def expand_foreground_only(mask, pixels=1.0, adaptive=True):
    """
    前景（マスク領域）のみを拡張する（穴のサイズは変えない）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像
    pixels : float
        基準となる拡張ピクセル数
    adaptive : bool
        前景成分のサイズに応じて適応的に調整するかどうか
        
    Returns:
    --------
    numpy.ndarray : 前景のみが拡張されたマスク画像
    """
    foreground_components, hole_components = _find_components_and_holes(mask)
    
    if not foreground_components:
        return mask
    
    result = np.zeros_like(mask)
    
    # 各前景成分を個別に処理
    for fg_mask, area in foreground_components:
        if adaptive:
            adj_pixels = _calculate_adaptive_pixels(area, pixels)
        else:
            adj_pixels = pixels
            
        if adj_pixels > 0:
            # この成分のみを拡張
            expanded_component = expand_mask(fg_mask, adj_pixels)
            result = np.maximum(result, expanded_component)
    
    # 元の穴を復元
    for hole_mask, _ in hole_components:
        result = result * (1.0 - hole_mask)
    
    return result

def shrink_foreground_only(mask, pixels=1.0, adaptive=True):
    """
    前景（マスク領域）のみを縮小する（穴のサイズは変えない）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像
    pixels : float
        基準となる縮小ピクセル数
    adaptive : bool
        前景成分のサイズに応じて適応的に調整するかどうか
        
    Returns:
    --------
    numpy.ndarray : 前景のみが縮小されたマスク画像
    """
    foreground_components, hole_components = _find_components_and_holes(mask)
    
    if not foreground_components:
        return mask
    
    result = np.zeros_like(mask)
    
    # 各前景成分を個別に処理
    for fg_mask, area in foreground_components:
        if adaptive:
            adj_pixels = _calculate_adaptive_pixels(area, pixels)
        else:
            adj_pixels = pixels
            
        if adj_pixels > 0:
            # この成分のみを縮小
            shrinked_component = shrink_mask(fg_mask, adj_pixels)
            result = np.maximum(result, shrinked_component)
    
    # 元の穴を復元
    for hole_mask, _ in hole_components:
        result = result * (1.0 - hole_mask)
    
    return result

def adjust_foreground_only(mask, pixels, adaptive=True):
    """
    前景（マスク領域）のみを調整する（正の値で拡張、負の値で縮小、穴のサイズは変えない）
    
    Parameters:
    -----------
    mask : numpy.ndarray
        0-1.0の値を持つマスク画像
    pixels : float
        調整するピクセル数（正：前景拡張、負：前景縮小）
    adaptive : bool
        前景成分のサイズに応じて適応的に調整するかどうか
        
    Returns:
    --------
    numpy.ndarray : 前景のみが調整されたマスク画像
    """
    if pixels > 0:
        return expand_foreground_only(mask, pixels, adaptive)
    elif pixels < 0:
        return shrink_foreground_only(mask, abs(pixels), adaptive)
    else:
        return mask

def create_donut_mask(shape=(100, 100), outer_radius=30, inner_radius=15):
    """
    ドーナッツ型のテスト用マスクを作成
    """
    mask = np.zeros(shape, dtype=np.float32)
    center = (shape[0]//2, shape[1]//2)
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    # 外側の円
    outer_circle = ((x - center[1])**2 + (y - center[0])**2) <= outer_radius**2
    # 内側の円（穴）
    inner_circle = ((x - center[1])**2 + (y - center[0])**2) <= inner_radius**2
    
    # ドーナッツ形状
    mask[outer_circle] = 1.0
    mask[inner_circle] = 0.0
    
    return mask