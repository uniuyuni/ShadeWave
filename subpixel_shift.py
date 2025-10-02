
import numpy as np
import cv2
import os

def subpixel_shift(img_array, shift_x=0.5, shift_y=0.5):
    """
    float32形式のRGB画像配列をサブピクセル単位でシフトする
    
    Parameters:
    img_array (ndarray): 入力画像配列 (HxWx3, float32, 値域[0,1])
    shift_x (float): X軸方向のシフト量（ピクセル単位、デフォルト0.5）
    shift_y (float): Y軸方向のシフト量（ピクセル単位、デフォルト0.5）
    
    Returns:
    ndarray: シフト後の画像配列 (HxWx3, float32, 値域[0,1])
    """
    # 入力配列の形状を確認
    if not isinstance(img_array, np.ndarray) or img_array.dtype != np.float32:
        raise ValueError("Input must be a float32 numpy array")
    
    height, width = img_array.shape[:2]
    
    # シフト後の座標を計算するためのグリッドを作成
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # シフト後の座標を計算
    X_shifted = X - shift_x
    Y_shifted = Y - shift_y
    
    # バイリニア補間のための重みを計算
    x0 = np.floor(X_shifted).astype(int)
    y0 = np.floor(Y_shifted).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    wx1 = X_shifted - x0
    wx0 = 1 - wx1
    wy1 = Y_shifted - y0
    wy0 = 1 - wy1
    
    # 境界をクリップ
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    
    # バイリニア補間を実行（全チャンネルを一度に処理）
    weights = (
        wy0[:,:,np.newaxis] * wx0[:,:,np.newaxis],  # 左上
        wy0[:,:,np.newaxis] * wx1[:,:,np.newaxis],  # 右上
        wy1[:,:,np.newaxis] * wx0[:,:,np.newaxis],  # 左下
        wy1[:,:,np.newaxis] * wx1[:,:,np.newaxis]   # 右下
    )
    
    samples = (
        img_array[y0, x0],  # 左上
        img_array[y0, x1],  # 右上
        img_array[y1, x0],  # 左下
        img_array[y1, x1]   # 右下
    )
    
    result = sum(w * s for w, s in zip(weights, samples))
    
    return result

def create_enhanced_image(img_array):
    """
    4つの半ピクセルシフトした画像を合成して、より滑らかな画像を生成する
    
    Parameters:
    img_array (ndarray): 入力画像配列 (HxWx3, float32, 値域[0,1])
    
    Returns:
    ndarray: 合成後の画像配列 (HxWx3, float32, 値域[0,1])
    """
    # 4つの異なる位置にシフト
    shifts = [
        (-0.5, -0.5),
        (0.5, -0.5),
        (-0.5, 0.5),
        (0.5, 0.5)
    ]
    
    # 結果を格納する配列を初期化
    result = np.zeros_like(img_array)
    
    # 各シフトバージョンを合成
    for shift_x, shift_y in shifts:
        shifted = subpixel_shift(img_array, shift_x, shift_y)
        result += shifted
    
    # 平均化
    return result / len(shifts)

if __name__ == '__main__':
    img = cv2.imread(os.getcwd() + "/picture/DSCF0002.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255

    img = create_enhanced_image(img)

    img = (cv2.cvtColor(img, cv2.COLOR_RGB2BGR)*255).astype(np.uint8)
    cv2.imshow('subpixel shift', img)
    cv2.waitKey(0)
