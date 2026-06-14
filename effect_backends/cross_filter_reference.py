import cv2
import numpy as np
import random

# ==========================================
#  1. カーネル生成関数
# ==========================================
def create_diffraction_kernel(length, decay_rate=8.0, symmetric=True):
    """
    光条の減衰パターン（1次元カーネル）を生成します。
    """
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
#  2. メインフィルター関数 (ピーク検出・完全版)
# ==========================================
def apply_cross_filter(img_rgb, num_points=6, length=100, angle_deg=0, 
                       threshold=1.0, intensity=1.0, spectral_strength=0.2, 
                       line_thickness=1.0, min_distance=10,
                       randomness=0.0, speed_factor=4,
                       debug_mode=False):
    """
    物理ベースのクロスフィルター（光条）効果を画像に適用します。
    
    【アルゴリズムの特徴】
    従来の「塊(Blob)検出」を廃止し、「局所ピーク(Local Maxima)検出」を採用しました。
    これにより、閾値を下げて検出範囲が広がっても、中心の1点だけが検出されるため、
    「サイズオーバーで消える」という矛盾が起きず、ドーナツボケも自然に無視されます。

    Args:
        img_rgb (np.ndarray):
            入力画像データ (Height, Width, 3)。
            float32型 (HDRデータ対応) を推奨。

        num_points (int, optional):
            光条の本数。
            - 偶数 (4, 6, 8...): 対称型（十字など）。
            - 奇数 (1, 3, 5...): 非対称型（星型）。

        length (int, optional):
            光条の長さ（ピクセル単位）。
            speed_factorにより高速化されるため、長い値でも高速に動作します。

        angle_deg (float, optional):
            フィルターの回転角度（度数法）。

        threshold (float, optional):
            ★最重要パラメータ
            光源として認識する「ピークの最低輝度」。
            この値より明るい「頂点」だけが光ります。
            - HDR画像: 1.0 〜 5.0 推奨。
            - SDR画像: 0.9 〜 0.95 推奨。
            値を下げると暗い点も光り、上げると明るい点だけ光ります（直感的）。

        intensity (float, optional):
            光条の合成強度。
            デフォルト 1.0。強く光らせたい場合は 2.0 〜 5.0 に設定。

        spectral_strength (float, optional):
            分光（色収差）の強さ (0.0 〜 1.0)。
            値を上げると光条の端が虹色に分離します。

        line_thickness (float, optional):
            光条の太さ（鋭さ）。
            - 1.0: ブラーなし（最鋭）。
            - 1.1以上: 値を上げるほどソフトな光になります。

        min_distance (int, optional):
            ★重要パラメータ
            検出するピーク同士の最小距離（ピクセル）。
            近くに複数の明るい点がある場合、最も明るい1つだけを残して間引きます。
            - 値が小さい(1〜3): ノイズや細かいテクスチャもすべて光ります。
            - 値が大きい(10〜20): まとまった光源の中心だけが光ります。

        randomness (float, optional):
            各光条の明るさのランダムなばらつき (0.0 〜 1.0)。

        speed_factor (int, optional):
            高速化係数。
            - 1: 最高画質（低速）。
            - 4: 推奨。画像を1/4サイズで計算します。
              光条のようなボケ要素は縮小しても劣化が目立たず、計算が劇的に速くなります。

        debug_mode (bool, optional):
            Trueにすると、光条を描画する代わりに「検出されたピーク位置」に
            赤い点を打った画像を返します。パラメータ調整時に便利です。

    Returns:
        np.ndarray: 効果適用後の画像 (float32)。
    """
    h, w = img_rgb.shape[:2]
    
    # ---------------------------------------------------------
    # A. 輝度マップ計算 (RGB -> Luminance)
    # ---------------------------------------------------------
    # 人間の目の感度に合わせて重み付け (R:0.299, G:0.587, B:0.114)
    luminance_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    luminance_map = np.dot(img_rgb, luminance_weights)
    
    # ---------------------------------------------------------
    # B. ピーク検出 (Local Maxima Detection)
    # ---------------------------------------------------------
    # 「自分の周囲(min_distance)の中で、自分が一番明るいか？」を判定します。
    # これにより、ブロブを作らずにピンポイントで光源を特定できます。
    
    # 探索範囲のカーネルサイズ (奇数)
    ksize = int(min_distance * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    
    # 1. ダイレーション（膨張処理）: 各画素に「近傍エリア内の最大値」を代入
    dilated_map = cv2.dilate(luminance_map, kernel)
    
    # 2. ピーク判定: 「元の値」が「近傍最大値」と一致すれば、そこが山頂
    local_max_mask = (luminance_map == dilated_map)
    
    # 3. 閾値判定: 山頂であっても、thresholdより暗ければ無視
    threshold_mask = (luminance_map > threshold)
    
    # 最終的なピーク位置 (AND演算)
    peak_mask = local_max_mask & threshold_mask
    
    # 座標リスト取得
    peak_ys, peak_xs = np.where(peak_mask)
    
    # ---------------------------------------------------------
    # C. インパルス生成
    # ---------------------------------------------------------
    sh, sw = h // speed_factor, w // speed_factor
    if sh < 1 or sw < 1: sh, sw = h, w
    
    debug_img = img_rgb.copy() if debug_mode else None
    impulse_mini = np.zeros((sh, sw, 3), dtype=np.float32)
    
    for i in range(len(peak_ys)):
        py, px = peak_ys[i], peak_xs[i]
        
        # 色取得 (ピーク位置の色)
        color = img_rgb[py, px]
        
        # ランダム性
        if randomness > 0:
            gain = random.uniform(1.0 - randomness, 1.0 + randomness)
            color = color * gain

        # デバッグ表示
        if debug_mode:
            # 赤い点を打つ
            cv2.circle(debug_img, (px, py), 4, (0, 0, 10.0), -1)
        else:
            # 縮小座標へ変換してプロット
            sx, sy = int(px / speed_factor), int(py / speed_factor)
            if 0 <= sx < sw and 0 <= sy < sh:
                # 縮小分のエネルギー減衰を補正
                boost = speed_factor * 1.5
                impulse_mini[sy, sx] = color * boost

    if debug_mode: return debug_img

    # ---------------------------------------------------------
    # D. アンチエイリアス / 太さ制御
    # ---------------------------------------------------------
    if line_thickness > 1.0:
        sigma = (line_thickness - 1.0) * 0.5
        ksize_blur = int(sigma * 6) | 1
        if ksize_blur >= 3:
            impulse_mini = cv2.GaussianBlur(impulse_mini, (ksize_blur, ksize_blur), sigma)

    # ---------------------------------------------------------
    # E. 光条生成 (回折シミュレーション)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # F. 合成
    # ---------------------------------------------------------
    streaks_mini = accumulated_streaks[pad_len:pad_len+sh, pad_len:pad_len+sw]
    streaks_full = cv2.resize(streaks_mini, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return img_rgb + (streaks_full * intensity)


# ==========================================
#  テスト: 安定性の確認
# ==========================================
if __name__ == "__main__":
    
    def generate_peak_test(h, w):
        img = np.zeros((h, w, 3), dtype=np.float32)
        # 1. 巨大なハローを持つ強力な光源
        # 中心(200,200)は輝度5.0、周辺(R=50)まで輝度1.2が広がる
        # 以前のコードでは閾値を1.0にするとブロブが巨大化して消えていましたが、今回は光ります。
        cv2.circle(img, (200, 200), 50, (1.2, 1.2, 1.2), -1) 
        cv2.circle(img, (200, 200), 5,  (5.0, 5.0, 5.0), -1) 

        # 2. ドーナツボケ
        # 中心(400,200)は暗い(0.0) -> ピークではないので検出されません。
        cv2.circle(img, (400, 200), 40, (5.0, 5.0, 5.0), -1)
        cv2.circle(img, (400, 200), 20, (0.0, 0.0, 0.0), -1)
        
        return img

    H, W = 600, 400
    input_img = generate_peak_test(H, W)

    print("Testing Peak Detection Method...")
    
    # 閾値を 1.0 に下げても、ピーク検出なら範囲が広がらないため安定して光ります。
    res_low = apply_cross_filter(input_img, threshold=1.0, length=100)
    
    # 閾値を 2.0 に上げても、同様に光ります。
    res_high = apply_cross_filter(input_img, threshold=2.0, length=100)

    # 表示
    disp_low  = (np.clip(res_low,  0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    disp_high = (np.clip(res_high, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    
    cv2.putText(disp_low,  "Th=1.0 (Stable)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(disp_high, "Th=2.0 (Stable)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Peak Method Final", np.hstack((disp_low, disp_high)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
