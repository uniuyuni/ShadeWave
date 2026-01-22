import cv2
import numpy as np
import random

# ==========================================
#  1. カーネル生成 (変更なし)
# ==========================================
def create_diffraction_kernel(length, decay_rate=8.0, symmetric=True):
    radius = length // 2
    if radius < 1: return np.ones((1, 1), dtype=np.float32)
    x = np.linspace(0, radius, radius)
    curve = np.exp(-decay_rate * (x / radius))
    curve[0] = 1.0
    if symmetric:
        kernel = np.concatenate((curve[::-1], curve[1:]))
    else:
        zeros = np.zeros(radius, dtype=np.float32)
        kernel = np.concatenate((zeros, curve))
    return kernel.reshape(1, -1).astype(np.float32)


# ==========================================
#  2. メインフィルター (輝度判定 + ピーク検出)
# ==========================================
def apply_cross_filter(img_rgb, num_points=6, length=100, angle_deg=0, 
                       threshold=1.0, intensity=1.0, spectral_strength=0.2, 
                       line_thickness=1.0, max_blob_size=0, min_blob_area=0,
                       randomness=0.0, speed_factor=4):
    """
    輝度(Luminance)に基づいて光源を検出し、
    各光源エリア内の「最大輝度点(ピーク)」から光条を発生させます。
    """
    h, w = img_rgb.shape[:2]
    
    # --- A. 輝度マップの計算 (RGB -> Luminance) ---
    # ITU-R BT.601 係数を使用 (R:0.299, G:0.587, B:0.114)
    # これにより「人間の目に明るく見える場所」を正しく判定できます。
    # img_rgb[:,:,0]がRと仮定しています。OpenCVで読み込んだ直後ならRGB変換済みを確認してください。
    
    # 高速化のためアインシュタインの縮約記法または単純な掛け算を使用
    # (H, W, 3) * (3,) -> (H, W)
    luminance_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    luminance_map = np.dot(img_rgb, luminance_weights)
    
    # --- B. 光源検出 ---
    # 輝度に基づいて閾値処理
    binary_mask = (luminance_map > threshold).astype(np.uint8) * 255
    
    # 連結成分解析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # --- C. インパルス生成 ---
    sh, sw = h // speed_factor, w // speed_factor
    if sh < 1 or sw < 1: sh, sw = h, w
    impulse_mini = np.zeros((sh, sw, 3), dtype=np.float32)
    
    for i in range(1, num_labels):
        x, y, bw, bh = stats[i, :4]
        area = stats[i, cv2.CC_STAT_AREA]

        # フィルタリング
        if max_blob_size > 0 and max(bw, bh) > max_blob_size: continue
        if area < min_blob_area: continue

        # --- ピーク検出 (輝度マップを使用) ---
        # 1. このブロブ(ラベルi)の範囲の輝度を取得
        roi_lum = luminance_map[y:y+bh, x:x+bw]
        roi_labels = labels[y:y+bh, x:x+bw]
        
        # 2. マスク作成
        mask_roi = (roi_labels == i).astype(np.uint8)
        
        # 3. マスク内で最も「輝度が高い」場所を探す
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi_lum, mask=mask_roi)
        
        # もし最大輝度が閾値以下なら無視 (念のため)
        if max_val <= threshold:
            continue
            
        # 4. 座標変換 (ROI座標 -> 全体座標)
        peak_x = x + max_loc[0]
        peak_y = y + max_loc[1]
        
        # ---------------------------

        # 色取得 (ピーク位置の元のRGB色を使う)
        color = img_rgb[peak_y, peak_x]

        if randomness > 0:
            gain = random.uniform(1.0 - randomness, 1.0 + randomness)
            color = color * gain

        # 縮小座標へ変換
        sx, sy = int(peak_x / speed_factor), int(peak_y / speed_factor)
        if 0 <= sx < sw and 0 <= sy < sh:
            boost = speed_factor * 1.5
            impulse_mini[sy, sx] = color * boost

    # --- D. アンチエイリアス ---
    if line_thickness > 1.0:
        sigma = (line_thickness - 1.0) * 0.5
        ksize_blur = int(sigma * 6) | 1
        if ksize_blur >= 3:
            impulse_mini = cv2.GaussianBlur(impulse_mini, (ksize_blur, ksize_blur), sigma)

    # --- E. 光条生成 ---
    mini_length = int(length / speed_factor)
    if mini_length < 1: mini_length = 1
    
    pad_len = int(mini_length * 1.5)
    impulse_padded = cv2.copyMakeBorder(impulse_mini, pad_len, pad_len, pad_len, pad_len, cv2.BORDER_CONSTANT, value=(0,0,0))
    ph, pw = impulse_padded.shape[:2]
    center_pad = (pw // 2, ph // 2)

    accumulated_streaks = np.zeros_like(impulse_padded)
    base_k_len = mini_length
    if base_k_len % 2 == 0: base_k_len += 1
    width_kernel = np.ones((1, 1), dtype=np.float32)
    spectral_scales = [1.0 + spectral_strength, 1.0, 1.0 - spectral_strength]
    
    if num_points % 2 == 0:
        num_passes = num_points // 2
        rot_step = 180.0 / num_passes if num_passes > 0 else 0
        use_symmetric = True 
    else:
        num_passes = num_points
        rot_step = 360.0 / num_passes
        use_symmetric = False
    if num_passes == 0: num_passes = 1 

    for i in range(num_passes):
        current_angle = angle_deg + (i * rot_step)
        M = cv2.getRotationMatrix2D(center_pad, current_angle, 1.0)
        rotated = cv2.warpAffine(impulse_padded, M, (pw, ph), flags=cv2.INTER_LINEAR)
        
        filtered_channels = []
        for ch in range(3): 
            ch_len = int(base_k_len * spectral_scales[ch])
            if ch_len % 2 == 0: ch_len += 1
            len_kernel = create_diffraction_kernel(ch_len, decay_rate=8.0, symmetric=use_symmetric)
            ch_img = rotated[:, :, ch]
            processed = cv2.sepFilter2D(ch_img, -1, len_kernel, width_kernel)
            filtered_channels.append(processed)
        merged = cv2.merge(filtered_channels)
        M_inv = cv2.getRotationMatrix2D(center_pad, -current_angle, 1.0)
        unrotated = cv2.warpAffine(merged, M_inv, (pw, ph), flags=cv2.INTER_LINEAR)
        accumulated_streaks += unrotated

    streaks_mini = accumulated_streaks[pad_len:pad_len+sh, pad_len:pad_len+sw]
    streaks_full = cv2.resize(streaks_mini, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return img_rgb + (streaks_full * intensity)


# ==========================================
#  テスト: 輝度 vs 最大値RGB の違い確認
# ==========================================
if __name__ == "__main__":
    
    def generate_luminance_test(h, w):
        img = np.zeros((h, w, 3), dtype=np.float32)
        
        # 1. 「青い」高輝度点 (RGB最大値=5.0 だが、輝度は低い)
        # 輝度 = 0.114 * 5.0 = 0.57
        # -> threshold=1.0 なら消えるべき
        cv2.circle(img, (200, 200), 10, (0.0, 0.0, 5.0), -1)
        
        # 2. 「白い」高輝度点 (RGB最大値=5.0 で、輝度も高い)
        # 輝度 = 5.0
        # -> threshold=1.0 なら残るべき
        cv2.circle(img, (400, 200), 10, (5.0, 5.0, 5.0), -1)
        
        return img

    H, W = 600, 400
    input_img = generate_luminance_test(H, W)

    print("Checking Luminance Thresholding...")
    
    # 閾値を 1.0 に設定
    # 以前のロジック(RGB Max)なら、両方とも最大値5.0なので両方光るはず。
    # 今回のロジック(Luminance)なら、青(輝度0.57)は消え、白(輝度5.0)だけ光るはず。
    res = apply_cross_filter(input_img, threshold=1.0, length=100, speed_factor=4)

    # 表示
    disp = (np.clip(res, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    
    cv2.putText(disp, "Blue(5.0)", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(disp, "White(5.0)", (350, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    
    cv2.imshow("Luminance Test", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()