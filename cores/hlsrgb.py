
import numpy as np
from numba import njit, prange
import time

@njit(parallel=True)
def rgb2hls(img):
    """
    Convert RGB image to HLS + Gain (HDR safe normalization).
    
    Returns 4 channels: (H, L, S, Gain)
    H: 0-360
    L: 0.0-1.0 (Normalized brightness)
    S: 0.0-1.0 (Saturation)
    Gain: >= 1.0 (HDR multiplier)
    
    Logic:
    Gain = max(1.0, max(R, G, B))
    RGB_norm = RGB / Gain
    L = max(RGB_norm) (Value)
    S = delta / max(RGB_norm)
    """
    rows, cols, _ = img.shape
    out = np.empty((rows, cols, 4), dtype=img.dtype)
    
    for i in prange(rows):
        for j in prange(cols):
            r = img[i, j, 0]
            g = img[i, j, 1]
            b = img[i, j, 2]
            
            # Clamp negatives for stability
            r_c = 0.0 if r < 0.0 else r
            g_c = 0.0 if g < 0.0 else g
            b_c = 0.0 if b < 0.0 else b
            
            max_val = max(r_c, g_c, b_c)
            
            # Calculate Gain
            gain = 1.0
            if max_val > 1.0:
                gain = max_val
            
            # Normalize RGB
            # If gain > 1, max_val becomes 1.0 after normalization
            # If gain = 1, max_val is <= 1.0
            
            # We don't need to actually divide r,g,b if we compute L, S smart.
            # L_norm = max_val / gain
            
            l = max_val / gain
            
            # S = (Max - Min) / Max
            # Since S is ratio, S(RGB) == S(RGB_norm).
            # So we can calculate S from original values.
            
            min_val = min(r_c, g_c, b_c)
            delta = max_val - min_val
            
            # Saturation Damping (epsilon)
            damp_epsilon = 0.005
            if gain > 1.0: 
                 # Adjust epsilon for HDR scale? 
                 # If value is 10, epsilon 0.003 is tiny.
                 # But we are calculating S based on raw values here.
                 # Actually, let's stick to raw calculation.
                 pass

            if max_val <= 1e-9:
                s = 0.0
                h = 0.0
            else:
                s = delta / (max_val + damp_epsilon)
                
                if delta < 1e-7:
                    h = 0.0
                elif max_val == r_c:
                    h = (g_c - b_c) / delta
                    if g_c < b_c:
                        h += 6.0
                elif max_val == g_c:
                    h = (b_c - r_c) / delta + 2.0
                else:
                    h = (r_c - g_c) / delta + 4.0
                h /= 6.0
            
            out[i, j, 0] = h * 360.0
            out[i, j, 1] = l
            out[i, j, 2] = s
            out[i, j, 3] = gain
            
    return out


# Rec.709の輝度係数（色差計算用）
KR = 0.2126
KG = 0.7152
KB = 0.0722

@njit(fastmath=True)
def rgb_to_hlc_gain(rgb):
    """
    RGB(HDR) -> HLC+Gain変換 (線形YCbCr風、Gain = max(R,G,B))
    
    Parameters:
    -----------
    rgb : ndarray, shape (H, W, 3), dtype=float32
        HDR RGB画像
    
    Returns:
    --------
    hlcg : ndarray, shape (H, W, 4), dtype=float32
        H: 色相 [0, 360)
        L: 輝度 [0, 1] - 線形加重平均
        C: 彩度 [0, 1] - 線形
        G: Gain - max(R,G,B)
    """
    H_img, W = rgb.shape[0], rgb.shape[1]
    hlcg = np.empty((H_img, W, 4), dtype=np.float32)
    
    for i in range(H_img): # なぜかクラッシュするからprangeが使えない
        for j in range(W):
            r, g, b = rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2]
            
            # Gain = max(R,G,B) に変更
            max_val = max(r, max(g, b))
            gain = max_val
            hlcg[i, j, 3] = gain
            
            if gain < 1e-10:
                hlcg[i, j, 0] = 0.0
                hlcg[i, j, 1] = 0.0
                hlcg[i, j, 2] = 0.0
                continue
            
            # 正規化
            r_norm = r / gain
            g_norm = g / gain
            b_norm = b / gain
            
            # 輝度（正規化空間）
            Y_norm = KR * r_norm + KG * g_norm + KB * b_norm
            L = Y_norm
            hlcg[i, j, 1] = L
            
            # 色差成分（線形）
            Cb = b_norm - Y_norm
            Cr = r_norm - Y_norm
            
            # 彩度（極座標）
            C = (Cb * Cb + Cr * Cr) ** 0.5
            
            # 理論上の最大Chroma
            # 正規化空間で max(R,G,B) = 1.0 の時
            # 最大の色差は約 sqrt((1-Y)^2 + (1-Y)^2) = sqrt(2) * (1-Y)
            # 最悪ケース: Y=0の時、C_max ≈ sqrt(2) ≈ 1.414
            C_max = 1.5  # 安全マージン
            C_norm = C / C_max
            C_norm = min(1.0, C_norm)
            hlcg[i, j, 2] = C_norm
            
            # 色相
            if C < 1e-8:
                H = 0.0
            else:
                H = np.arctan2(Cr, Cb) * 180.0 / np.pi
                if H < 0.0:
                    H += 360.0
            
            hlcg[i, j, 0] = H
    
    return hlcg


@njit(parallel=True, fastmath=True)
def hlc_gain_to_rgb(hlcg):
    """
    HLC+Gain -> RGB(HDR)変換
    
    Parameters:
    -----------
    hlcg : ndarray, shape (H, W, 4), dtype=float32
        H, L, C, G
    
    Returns:
    --------
    rgb : ndarray, shape (H, W, 3), dtype=float32
    """
    H_img, W = hlcg.shape[0], hlcg.shape[1]
    rgb = np.empty((H_img, W, 3), dtype=np.float32)
    
    for i in prange(H_img):
        for j in range(W):
            H = hlcg[i, j, 0]
            L = hlcg[i, j, 1]
            C_norm = hlcg[i, j, 2]
            gain = hlcg[i, j, 3]
            
            # Y (正規化空間)
            Y_norm = L
            
            # Chromaを復元
            C_max = 1.5
            C = C_norm * C_max
            
            # 極座標 → 色差成分
            H_rad = H * np.pi / 180.0
            Cb = C * np.cos(H_rad)
            Cr = C * np.sin(H_rad)
            
            # Y, Cb, Cr -> R, G, B
            g_norm = Y_norm - (KR / KG) * Cr - (KB / KG) * Cb
            r_norm = Cr + Y_norm
            b_norm = Cb + Y_norm
            
            # Gainを適用
            rgb[i, j, 0] = r_norm * gain
            rgb[i, j, 1] = g_norm * gain
            rgb[i, j, 2] = b_norm * gain
    
    return rgb


def test_basic_colors_detailed():
    """8色の詳細テスト"""
    
    colors = [
        ('赤 (Red)',         1.0, 0.0, 0.0),
        ('オレンジ (Orange)', 1.0, 0.5, 0.0),
        ('黄色 (Yellow)',    1.0, 1.0, 0.0),
        ('緑 (Green)',       0.0, 1.0, 0.0),
        ('シアン (Cyan)',    0.0, 1.0, 1.0),
        ('青 (Blue)',        0.0, 0.0, 1.0),
        ('紫 (Violet)',      0.5, 0.0, 1.0),
        ('マゼンタ (Magenta)', 1.0, 0.0, 1.0),
    ]
    
    print("="*80)
    print("Pure Colors Test with Gain = max(R,G,B)")
    print("="*80)
    print(f"{'Color':<20} {'RGB Input':<15} {'H':<10} {'L':<8} {'C':<8} {'Gain':<8} {'RGB Restored':<15} {'Error'}")
    print("-"*80)
    
    for name, r, g, b in colors:
        # 変換
        H, L, C, G = rgb_to_hlc_gain_single(r, g, b)
        
        # 逆変換
        r_out, g_out, b_out = hlc_gain_to_rgb_single(H, L, C, G)
        
        # エラー計算
        error = max(abs(r - r_out), abs(g - g_out), abs(b - b_out))
        
        rgb_in = f"({r:.1f},{g:.1f},{b:.1f})"
        rgb_out_str = f"({r_out:.3f},{g_out:.3f},{b_out:.3f})"
        
        print(f"{name:<20} {rgb_in:<15} {H:>7.2f}° {L:>7.3f} {C:>7.3f} {G:>7.3f} {rgb_out_str:<15} {error:.2e}")
    
    print("\n" + "="*80)
    print("Hue Ranges (based on midpoints)")
    print("="*80)
    
    hues = []
    for name, r, g, b in colors:
        H, _, _, _ = rgb_to_hlc_gain_single(r, g, b)
        hues.append((name, H))
    
    for i in range(len(hues)):
        name, curr_h = hues[i]
        prev_h = hues[i - 1][1]
        next_h = hues[(i + 1) % len(hues)][1]
        
        # 前の色との中間点
        if curr_h < prev_h:
            start = (prev_h + curr_h + 360) / 2
            if start >= 360:
                start -= 360
        else:
            start = (prev_h + curr_h) / 2
        
        # 次の色との中間点
        if next_h < curr_h:
            end = (curr_h + next_h + 360) / 2
            if end >= 360:
                end -= 360
        else:
            end = (curr_h + next_h) / 2
        
        if start > end:
            print(f"{name:<20} {start:>7.2f}° ~ 360° / 0° ~ {end:>7.2f}°")
        else:
            print(f"{name:<20} {start:>7.2f}° ~ {end:>7.2f}°")


def rgb_to_hlc_gain_single(r, g, b):
    """単一ピクセル変換"""
    max_val = max(r, max(g, b))
    gain = max_val
    
    if gain < 1e-10:
        return 0.0, 0.0, 0.0, 0.0
    
    r_norm = r / gain
    g_norm = g / gain
    b_norm = b / gain
    
    Y_norm = KR * r_norm + KG * g_norm + KB * b_norm
    L = Y_norm
    
    Cb = b_norm - Y_norm
    Cr = r_norm - Y_norm
    
    C = (Cb * Cb + Cr * Cr) ** 0.5
    C_norm = min(1.0, C / 1.5)
    
    if C < 1e-8:
        H = 0.0
    else:
        H = np.arctan2(Cr, Cb) * 180.0 / np.pi
        if H < 0.0:
            H += 360.0
    
    return H, L, C_norm, gain


def hlc_gain_to_rgb_single(H, L, C_norm, gain):
    """単一ピクセル逆変換"""
    Y_norm = L
    C = C_norm * 1.5
    
    H_rad = H * np.pi / 180.0
    Cb = C * np.cos(H_rad)
    Cr = C * np.sin(H_rad)
    
    g_norm = Y_norm - (KR / KG) * Cr - (KB / KG) * Cb
    r_norm = Cr + Y_norm
    b_norm = Cb + Y_norm
    
    return r_norm * gain, g_norm * gain, b_norm * gain


def test_saturation_linearity():
    """彩度の線形性テスト"""
    print("\n" + "="*80)
    print("Saturation Linearity Test")
    print("="*80)
    print("Testing: Red (1,0,0) with varying saturation")
    print("-"*80)
    print(f"{'Original C':<15} {'RGB Output':<25} {'Restored C':<15} {'Error'}")
    print("-"*80)
    
    r, g, b = 1.0, 0.0, 0.0
    H, L, C_orig, G = rgb_to_hlc_gain_single(r, g, b)
    
    for c_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]:
        C_test = min(1.0, C_orig * c_factor)
        
        r_out, g_out, b_out = hlc_gain_to_rgb_single(H, L, C_test, G)
        
        # 再変換して確認
        H_back, L_back, C_back, G_back = rgb_to_hlc_gain_single(r_out, g_out, b_out)
        
        error = abs(C_test - C_back)
        rgb_str = f"({r_out:.3f},{g_out:.3f},{b_out:.3f})"
        
        print(f"{C_test:<15.3f} {rgb_str:<25} {C_back:<15.3f} {error:.2e}")


def test_gradient_quality():
    """グラデーションの品質テスト"""
    print("\n" + "="*80)
    print("Gradient Quality Test")
    print("="*80)
    
    # 微妙なグラデーション
    h, w = 1024, 256
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        val = 0.5 + 0.1 * (i / h)
        gradient[i, :] = val
    
    print(f"Input: {h}x{w} gradient from 0.5 to 0.6")
    
    # 変換
    start = time.time()
    hlcg = rgb_to_hlc_gain(gradient)
    t1 = time.time() - start
    
    start = time.time()
    restored = hlc_gain_to_rgb(hlcg)
    t2 = time.time() - start
    
    # エラー
    error = np.abs(gradient - restored)
    
    print(f"Timing: Forward {t1*1000:.2f}ms, Backward {t2*1000:.2f}ms")
    print(f"Max error: {error.max():.2e}")
    print(f"Mean error: {error.mean():.2e}")
    
    # グラデーションの滑らかさ
    center_col = gradient[:, w//2, 0]
    restored_col = restored[:, w//2, 0]
    
    diff_orig = np.diff(center_col)
    diff_rest = np.diff(restored_col)
    
    print(f"\nGradient smoothness:")
    print(f"  Original diff std: {diff_orig.std():.2e}")
    print(f"  Restored diff std: {diff_rest.std():.2e}")
    print(f"  Ratio: {diff_rest.std() / diff_orig.std():.6f}")
    
    # HLCの連続性
    H_col = hlcg[:, w//2, 0]
    L_col = hlcg[:, w//2, 1]
    C_col = hlcg[:, w//2, 2]
    
    print(f"\nHLC continuity:")
    print(f"  H diff std: {np.diff(H_col).std():.2e}")
    print(f"  L diff std: {np.diff(L_col).std():.2e}")
    print(f"  C diff std: {np.diff(C_col).std():.2e}")


if __name__ == "__main__":
    # 基本色のテスト
    test_basic_colors_detailed()
    
    # 彩度の線形性テスト
    test_saturation_linearity()
    
    # グラデーション品質テスト
    test_gradient_quality()
    
    print("\n" + "="*80)
    print("✓ All tests completed")
    print("="*80)