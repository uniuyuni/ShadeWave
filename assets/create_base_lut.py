import numpy as np
import os

def generate_prophoto_atmospheric_lut(size=33, output_path="ProPhotoLinear_Atmospheric.cube"):
    """
    Linear ProPhoto RGB -> Linear ProPhoto RGB
    知覚ベースのトーンコントラスト / 明確な中間調彩度 / 自然な色相クロス / 編集起点安全設計
    """
    def smoothstep(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # グリッド構築
    step = 1.0 / (size - 1)
    vals = np.arange(size) * step
    R, G, B = np.meshgrid(vals, vals, vals, indexing='ij')
    r, g, b = R.copy(), G.copy(), B.copy()

    # 初期知覚輝度
    L = 0.2126*r + 0.7152*g + 0.0722*b

    # --------------------------------------------------------
    # 1. 知覚コントラストベース（Linear空間用 滑らかなS字）
    # L*(1-L)*(L-0.5) は 0, 0.5, 1 で 0 になり、中間調のみ滑らかに持ち上げ/圧縮
    # --------------------------------------------------------
    contrast_k = 0.28  # 知覚コントラスト強度。0.0=フラット, 0.40=強め
    L_tone = L + contrast_k * L * (1.0 - L) * (L - 0.5)
    # チャネル均等スケールでトーンを適用（色相ズレ防止）
    scale = np.where(L > 1e-5, L_tone / L, 1.0)
    r *= scale; g *= scale; b *= scale
    L = 0.2126*r + 0.7152*g + 0.0722*b  # 更新

    # --------------------------------------------------------
    # 2. 知覚彩度形状（Linear換算 +12~15% 中間調ピーク）
    # Gamma空間の「+5%」相当をLinearで実現するため、係数を1.8~2.2倍にスケーリング
    # --------------------------------------------------------
    S = 1.0
    S += 0.14 * np.exp(-((L - 0.28)**2) / 0.028)   # 中間調密度（知覚ピーク~1.14x）
    S -= 0.09 * (1.0 - smoothstep(0.0, 0.08, L))    # シャドウ抑制（~0.91x）
    S -= 0.06 * smoothstep(0.78, 1.0, L)             # ハイライト抑制（~0.94x）
    S = np.clip(S, 0.82, 1.16)                       # 安全範囲

    r = L + S * (r - L)
    g = L + S * (g - L)
    b = L + S * (b - L)

    # --------------------------------------------------------
    # 3. 自然色相クロス（フィルム染料層の分光特性を模倣）
    # R→暖赤／G→シアン緑／B→深青 への微シフト。行和=1.0で設計後、知覚明度補正
    # --------------------------------------------------------
    hue_k = 1.0  # 0.0=無シフト, 1.5=明確な雰囲気
    M = np.array([
        [0.940,  0.060*hue_k,  0.000],
        [0.025*hue_k, 0.950,  0.025*hue_k],
        [0.000,  0.035*hue_k,  0.965]
    ])
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.dot(rgb, M.T)
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]

    # 知覚明度補正（色相クロスによる露出ズレを吸収）
    L_new = 0.2126*r + 0.7152*g + 0.0722*b
    lum_fix = np.where(L_new > 1e-5, L / L_new, 1.0)
    r *= lum_fix; g *= lum_fix; b *= lum_fix
    L = 0.2126*r + 0.7152*g + 0.0722*b

    # --------------------------------------------------------
    # 4. 明確だが滑らかなスプリットトーン
    # シャドウ: クールグレー / ハイライト: 自然光の暖かみ
    # Linear空間で知覚可能なレベル（ΔE≈3.0~4.0）に設定。後段WBで完全相殺可能。
    # --------------------------------------------------------
    sh_mask = 1.0 - smoothstep(0.06, 0.18, L)
    hi_mask = smoothstep(0.80, 0.92, L)
    
    r -= 0.018 * sh_mask
    b += 0.020 * sh_mask
    r += 0.014 * hi_mask
    g += 0.007 * hi_mask

    # --------------------------------------------------------
    # 5. D1: 安全クランプ & 出力
    # --------------------------------------------------------
    r = np.clip(r, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    with open(output_path, "w") as f:
        f.write(f"TITLE \"ProPhoto Linear Atmospheric: Contrast+MidSat+Hue+Split\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        for bz in range(size):
            for gy in range(size):
                for rx in range(size):
                    f.write(f"{r[rx, gy, bz]:.6f} {g[rx, gy, bz]:.6f} {b[rx, gy, bz]:.6f}\n")
                    
    print(f"✅ 生成完了: {output_path} ({size}^3 nodes)")

# 実行
generate_prophoto_atmospheric_lut(size=33)