import numpy as np
import cv2
import matplotlib.pyplot as plt

def scratch_effect(image, scratch_intensity=1.0, shift_parcent=1.0):
    """
    モザイク効果に特化した引っ掻きフィルター（高速化版）
    画像をより判別困難にする
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # 引っ掻き効果を段階的に適用
    num_passes = 3
    
    for pass_num in range(num_passes):
        scratch_size = int(5 * (pass_num + 1) * scratch_intensity)
        num_scratches = int(h * w * scratch_intensity / (100 * (pass_num + 1)))
        
        if num_scratches == 0:
            continue
            
        # 全ての座標を一度に生成（ベクトル化）
        x1_coords = np.random.randint(0, max(1, w - scratch_size), num_scratches)
        y1_coords = np.random.randint(0, max(1, h - scratch_size), num_scratches)
        x2_coords = np.minimum(x1_coords + scratch_size, w)
        y2_coords = np.minimum(y1_coords + scratch_size, h)
        
        # シフト量も一度に生成
        shift_x = np.random.randint(-scratch_size, scratch_size + 1, num_scratches)
        shift_y = np.random.randint(-scratch_size, scratch_size + 1, num_scratches)
        
        # ソース座標を計算
        src_x1_coords = np.clip(x1_coords + shift_x, 0, w - (x2_coords - x1_coords))
        src_y1_coords = np.clip(y1_coords + shift_y, 0, h - (y2_coords - y1_coords))
        src_x2_coords = src_x1_coords + (x2_coords - x1_coords)
        src_y2_coords = src_y1_coords + (y2_coords - y1_coords)
        
        # 有効な領域のマスクを作成
        valid_mask = (src_x2_coords <= w) & (src_y2_coords <= h)
        
        # 有効な座標のみを処理
        for i in np.where(valid_mask)[0]:
            y1, y2 = y1_coords[i], y2_coords[i]
            x1, x2 = x1_coords[i], x2_coords[i]
            src_y1, src_y2 = src_y1_coords[i], src_y2_coords[i]
            src_x1, src_x2 = src_x1_coords[i], src_x2_coords[i]
            
            # 領域をコピー
            result[y1:y2, x1:x2] = image[src_y1:src_y2, src_x1:src_x2]
    
    # ガウシアンブラーのカーネルサイズを調整（奇数にする必要がある）
    kernel_size = int(555 * shift_parcent)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    result = cv2.GaussianBlur(result, ksize=(kernel_size, 1), sigmaX=0)

    return result


def mosaic_effect(image, block_size=16):
    """
    モザイク効果を適用する関数
    [params]
    image: (H,W,3) float32形式のRGB画像（0.0-1.0）
    block_size: モザイクのブロックサイズ（ピクセル）
    """
    h, w = image.shape[:2]
    result = image.copy()

    block_size = int(block_size)
    
    # ブロック単位で平均色を計算
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y+block_size, h)
            x_end = min(x+block_size, w)
            block = image[y:y_end, x:x_end]
            avg_color = np.mean(block, axis=(0,1))
            result[y:y_end, x:x_end] = avg_color
    
    return result

    
def frosted_glass_effect(image, blur_radius=10, noise_scale=0.01):
    """
    フロストガラス効果を適用する関数
    [params]
    image: (H,W,3) float32形式のRGB画像（0.0-1.0）
    blur_radius: ぼかし強度
    noise_scale: ノイズの強度（0.0-0.1）
    """
    h, w = image.shape[:2]
    
    # ガウシアンブラーの最適化
    kernel_size = int(4 * blur_radius) | 1  # 奇数保証
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 
                             sigmaX=blur_radius, 
                             sigmaY=blur_radius,
                             borderType=cv2.BORDER_REPLICATE)
    
    # ノイズ生成（-1.0〜1.0の範囲）
    noise_x = (np.random.rand(h,w) * 2 - 1) * noise_scale * w
    noise_y = (np.random.rand(h,w) * 2 - 1) * noise_scale * h
    
    # 座標マップ生成
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map = (x_map + noise_x).astype(np.float32)
    y_map = (y_map + noise_y).astype(np.float32)
    
    # リマップ処理
    result = cv2.remap(blurred, x_map, y_map,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    
    return result



if __name__ == '__main__':
    # 入力画像の読み込み（0.0-1.0のfloat32に変換）
    input_img = cv2.imread("your_image.jpg").astype(np.float32) / 255.0
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # 各効果の適用
    scratch_img = scratch_effect(input_img, scratch_intensity=1.0, shift_parcent=1.5)
    mosaic_img = mosaic_effect(input_img, block_size=80)
    frosted_img = frosted_glass_effect(input_img, blur_radius=10, noise_scale=0.01)

    # 結果の保存
    scratch_img = cv2.cvtColor(scratch_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("scratch.jpg", (scratch_img*255).astype(np.uint8))
    mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("mosaic.jpg", (mosaic_img*255).astype(np.uint8))
    frosted_img = cv2.cvtColor(frosted_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("frosted.jpg", (frosted_img*255).astype(np.uint8))
