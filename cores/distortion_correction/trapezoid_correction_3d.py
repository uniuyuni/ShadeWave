
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
    interpolation: str = 'bicubic'
) -> np.ndarray:
    """
    3D回転ベースの台形補正
    
    vertical: X軸周りの回転（上下の遠近感）
    horizontal: Y軸周りの回転（左右の遠近感）
    rotation: Z軸周りの回転
    focal_length: 焦点距離（0の場合は画像の長辺を使用）
    """
    if image.dtype != np.float32:
        raise TypeError(f"image.dtype must be float32, got {image.dtype}")
    
    height, width = image.shape[:2]
    
    # 画像中心
    cx, cy = width / 2.0, height / 2.0
    
    # オフセット適用（回転中心をずらす場合）
    rotation_center_x = cx + offset_x * width
    rotation_center_y = cy + offset_y * height

    # 焦点距離 (Focal Length)
    # 値が大きいほど望遠（パースが弱い）、小さいほど広角（パースが強い）
    # 0の場合は画像サイズ程度（対角線や長辺）に設定
    if focal_length <= 0:
        f = max(width, height)
    else:
        # スライダー等の入力を想定し、長辺に対する倍率などで扱うか、ピクセルそのままか
        # ここでは pixel 単位の焦点距離として扱う
        f = focal_length

    
    # 回転角度 (Degree -> Radian)
    # vertical は X軸回転 (Tilt)
    # horizontal は Y軸回転 (Pan)
    # rotation は Z軸回転 (Roll)
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

    # 変換行列 T の構築
    
    # 1. 画像中心を原点 (0, 0, 0) に移動
    T1 = np.array([
        [1, 0, 0, -rotation_center_x],
        [0, 1, 0, -rotation_center_y],
        [0, 0, 1, 0], # Z=0
        [0, 0, 0, 1]
    ])
    
    # 2. カメラ位置まで Z 方向に移動 (平行移動)
    # これにより、画像がカメラの前方 f の位置に来るようにする
    # しかし、一般的な透視投影変換マトリックスを作るなら、
    # 単純に回転させてから透視投影する方が制御しやすい。
    
    # ここでは OpenCV の warpPerspective に渡す 3x3 行列を作るアプローチをとる
    # Reference: https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
    
    # 世界座標系変換行列(3x4)ではなく、ホモグラフィ行列(3x3)を直接作るアプローチ
    # カメラの内部パラメータ行列 K
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    
    # 回転行列 (3x3 part of R)
    R3 = R[:3, :3]
    
    # カメラの移動 (Translation)
    # 回転中心を画像中心にしたいので、並進は基本ゼロだが
    # 奥行き方向の操作が必要。
    # 通常の透視変換では、Z=f の平面にある画像を回転させるイメージ
    
    # 変換フロー:
    # 2D (u,v) -> Camera (x,y,z) where z=f -> Rotate -> Project back to 2D
    
    # 厳密には、warpPerspective の H は src -> dst への変換
    # src point p = (u, v, 1)
    # dst point p' = H @ p
    
    # シンプルなアプローチ:
    # 4隅の点を3D回転させて、その投影点との間のホモグラフィを計算する方法が確実。
    
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
        
        # 3. 透視投影
        # カメラは原点(0,0,0)にあり、スクリーンは Z=f にあると仮定、あるいは
        # 物体が Z=0 から動いて、カメラが Z=-f にあると考える
        # ここでは後者: カメラ位置は (0, 0, -f) と定義し、視線は +Z 方向
        
        rx, ry, rz = rotated_vec[0], rotated_vec[1], rotated_vec[2]
        
        # 投影: 視点(0,0,-f) から 点(rx,ry,rz) へのレイが Z=0 平面と交わる点...ではなく
        # 通常のピンホールモデル: x' = f * (x / z)
        # ここでは、物体中心が(0,0,f)にあるとして、カメラが原点にあるとするセットアップが楽
        
        # 再考:
        # 物体座標 (x, y, 0) relative to center
        # 回転後 (rx, ry, rz)
        # カメラから見た座標にするために Z軸方向に f だけ奥にずらす -> (rx, ry, rz + f)
        cam_z = rz + f
        
        # 平面への投影 (focal length f)
        # x_proj = f * rx / cam_z
        # y_proj = f * ry / cam_z
        
        # もし cam_z が 0 以下ならクリッピングが必要だが、
        # 通常の補正範囲なら大丈夫と仮定
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
