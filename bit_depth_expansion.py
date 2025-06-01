
import numpy as np
from scipy import signal
import cv2

def expand_14bit_to_16bit(image_14bit):
    """
    14ビットの画像を16ビットに拡張し、下位2ビットを周囲のピクセル情報から補完します
    
    Args:
        image_14bit (numpy.ndarray): 14ビットの入力画像
    
    Returns:
        numpy.ndarray: 16ビットに拡張された画像
    """
    # 入力が2次元配列でない場合は2次元に変換
    if len(image_14bit.shape) > 2:
        raise ValueError("入力画像は2次元のグレースケール画像である必要があります")
    
    # 14ビットの値を16ビットの上位14ビットにシフト
    image_16bit = image_14bit.astype(np.uint16) << 2
    image_14bit = image_14bit.astype(np.float32)
    
    # バイリニア補間用のカーネル
    kernel = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]], dtype=np.float32) / 16.0
    
    # 下位2ビットの補完
    lower_bits = signal.convolve2d(image_14bit, kernel, mode='same', boundary='symm')
    lower_bits = lower_bits.astype(np.uint16) & 3
    
    # 16ビット画像に下位2ビットを追加
    result = image_16bit | lower_bits
    
    return result

def edge_aware_expansion(image_14bit):
    """
    エッジを考慮した14ビットから16ビットへの拡張
    
    Args:
        image_14bit (numpy.ndarray): 14ビットの入力画像
    
    Returns:
        numpy.ndarray: 16ビットに拡張された画像
    """
    # 入力が2次元配列でない場合は2次元に変換
    if len(image_14bit.shape) > 2:
        raise ValueError("入力画像は2次元のグレースケール画像である必要があります")
    
    # 基本的な16ビットへのシフト
    image_16bit = image_14bit.astype(np.uint16) << 2
    image_14bit = image_14bit.astype(np.float32) * 4
    
    # エッジ検出用のSobelフィルタ
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)
    
    # エッジの強度と方向を計算
    grad_x = signal.convolve2d(image_14bit, sobel_x, mode='same', boundary='symm')
    grad_y = signal.convolve2d(image_14bit, sobel_y, mode='same', boundary='symm')
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    
    # エッジの方向に基づいて補間
    edge_direction = np.arctan2(grad_y, grad_x)
    
    # エッジの強度に応じて補間の重みを調整
    weight = 1.0 / (1.0 + edge_strength)
    
    # バイリニア補間用のカーネル
    kernel = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]], dtype=np.float32) / 16.0
    
    # 下位2ビットの補完
    lower_bits = signal.convolve2d(image_14bit, kernel, mode='same', boundary='symm')
    lower_bits = (lower_bits * weight).astype(np.uint16) & 3
    
    # 16ビット画像に下位2ビットを追加
    result = image_16bit | lower_bits
    
    return result

def process_rgb_image(image_14bit_rgb):
    """
    RGB画像の各チャンネルに対してビット深度拡張を適用
    
    Args:
        image_14bit_rgb (numpy.ndarray): 3次元のRGB画像
    
    Returns:
        numpy.ndarray: 処理後のRGB画像
    """
    result = np.zeros_like(image_14bit_rgb, dtype=np.uint16)
    for i in range(3):  # R, G, Bの各チャンネル
        result[:, :, i] = edge_aware_expansion(image_14bit_rgb[:, :, i])
    return result



def linear_interpolation_8to16bit(img_8bit):
    
    # 16bit形式に変換
    img_16bit = img_8bit.astype(np.uint16) << 8  # 上位8bitに配置
    
    # ガウシアンフィルタで補間値生成
    kernel_size = (3, 3)  # カーネルサイズ
    sigma = 1.0  # 標準偏差
    interpolated = cv2.GaussianBlur(img_8bit, kernel_size, sigmaX=sigma)
    
    # 境界処理（鏡映境界）
    interpolated = cv2.copyMakeBorder(
        interpolated, 
        1, 1, 1, 1, 
        cv2.BORDER_REFLECT_101
    )[1:-1, 1:-1]  # 不要な境界をトリミング
    
    # 下位8bitに補間値を追加
    img_16bit |= interpolated.astype(np.uint16)
    
    return img_16bit
