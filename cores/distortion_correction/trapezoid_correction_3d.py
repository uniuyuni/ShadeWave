
import numpy as np
import cv2

def correct_trapezoid_3d(
    image: np.ndarray,
    horizontal: float = 0,
    vertical: float = 0,
    rotation: float = 0,
    focal_length: float = 0,
    offset_x: float = 0,  # Unused for now in 3D rotation center logic, or could be used to shift center
    offset_y: float = 0,  # Unused for now in 3D rotation center logic
    interpolation: str = 'bicubic',
    homography: np.ndarray = None
) -> np.ndarray:
    """
    3D回転ベースの台形補正
    """
    if image.dtype != np.float32:
        raise TypeError(f"image.dtype must be float32, got {image.dtype}")

    height, width = image.shape[:2]

    if homography is not None:
        # 呼び出し側で計算・調整済み (減衰クランプ等) の順変換ホモグラフィをそのまま使う。
        H = np.asarray(homography, dtype=np.float64)
    else:
        H = calculate_trapezoid_homography(width, height, horizontal, vertical, rotation, focal_length, offset_x, offset_y)
    
    # 補間方法の選択
    interp_flags = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    if interpolation not in interp_flags:
        raise ValueError(f"未対応の補間方法: {interpolation}")

    corrected = cv2.warpPerspective(image, H, (width, height), flags=interp_flags[interpolation], borderValue=(0,0,0))

    return corrected, H

def calculate_trapezoid_homography(
    width: int,
    height: int,
    horizontal: float = 0,
    vertical: float = 0,
    rotation: float = 0,
    focal_length: float = 0,
    offset_x: float = 0,
    offset_y: float = 0
) -> np.ndarray:
    """
    台形補正用のホモグラフィ行列を計算する
    """
    # 画像中心
    cx, cy = width / 2.0, height / 2.0
    
    # オフセット適用（回転中心をずらす場合）
    rotation_center_x = cx + offset_x * width
    rotation_center_y = cy + offset_y * height

    # 焦点距離 (Focal Length)
    if focal_length <= 0:
        f = max(width, height)
    else:
        f = focal_length

    # 回転角度 (Degree -> Radian)
    rad_x = np.radians(vertical)
    rad_y = np.radians(horizontal)
    rad_z = np.radians(rotation)

    # 3D回転行列の構築
    # X軸周り
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x), 0],
        [0, np.sin(rad_x), np.cos(rad_x), 0],
        [0, 0, 0, 1]
    ])
    
    # Y軸周り
    Ry = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y), 0],
        [0, 1, 0, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y), 0],
        [0, 0, 0, 1]
    ])
    
    # Z軸周り
    Rz = np.array([
        [np.cos(rad_z), -np.sin(rad_z), 0, 0],
        [np.sin(rad_z), np.cos(rad_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # 合成回転行列 R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    # コーナーポイントの変換
    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    dst_corners = []
    
    for pt in corners:
        # 1. 中心を原点へ
        x = pt[0] - rotation_center_x
        y = pt[1] - rotation_center_y
        z = 0
        
        # 2. 回転
        vec = np.array([x, y, z, 1])
        rotated_vec = R @ vec
        
        rx, ry, rz = rotated_vec[0], rotated_vec[1], rotated_vec[2]
        
        # 3. 透視投影
        cam_z = rz + f
        
        if cam_z <= 1e-3:
            cam_z = 1e-3
            
        proj_x = f * rx / cam_z
        proj_y = f * ry / cam_z
        
        # 4. 座標系を戻す
        dst_x = proj_x + rotation_center_x
        dst_y = proj_y + rotation_center_y
        
        dst_corners.append([dst_x, dst_y])
        
    dst_corners = np.array(dst_corners, dtype=np.float32)
    
    # ホモグラフィ行列を計算
    # src(corners) -> dst(dst_corners)
    H = cv2.getPerspectiveTransform(corners, dst_corners)
    
    return H
