import numpy as np
import os

def generate_prophoto_mature_lut(size=33, output_path="ProPhotoLinear_Mature.cube"):
    """
    Linear ProPhoto RGB -> Linear ProPhoto RGB
    微色相シフト / 明度依存彩度 / 滑らかなハイライトロールオフ / 編集起点最適化
    """
    def smoothstep(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # グリッド構築
    step = 1.0 / (size - 1)
    vals = np.arange(size) * step
    R, G, B = np.meshgrid(vals, vals, vals, indexing='ij')
    r, g, b = R.copy(), G.copy(), B.copy()

    # 初期明度
    L = 0.2126*r + 0.7152*g + 0.0722*b

    # --------------------------------------------------------
    # 1. A1: トーン形状（ほぼリニア＋ハイライトソフトロールオフ）
    # --------------------------------------------------------
    tone_scale = 1.0 - 0.055 * (L ** 2.8)  # 0.65付近から滑らかに圧縮
    r *= tone_scale; g *= tone_scale; b *= tone_scale
    L = 0.2126*r + 0.7152*g + 0.0722*b  # 更新

    # --------------------------------------------------------
    # 2. 微色相シフト（ライカ/ハッセル風の“色の立体感”）
    # 行和=1.0で明度保存を設計。強度は hue_strength で一括調整可能。
    # --------------------------------------------------------
    hue_strength = 1.0  # 0.0=無色相, 1.5=強め
    M = np.array([
        [0.985,  0.015*hue_strength,  0.000],
        [0.000,  0.978,              0.022*hue_strength],
        [0.012*hue_strength, 0.000,  0.988]
    ])
    
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.dot(rgb, M.T)
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]

    # 知覚明度補正（シフトによる露出ズレを吸収）
    L_new = 0.2126*r + 0.7152*g + 0.0722*b
    lum_scale = np.where(L_new > 1e-5, L / L_new, 1.0)
    r *= lum_scale; g *= lum_scale; b *= lum_scale
    L = 0.2126*r + 0.7152*g + 0.0722*b  # 再計算

    # --------------------------------------------------------
    # 3. B3: 明度依存彩度（中間調密度↑ / 両端自然減衰）
    # --------------------------------------------------------
    S = 1.0
    S += 0.09 * np.exp(-((L - 0.44)**2) / 0.062)   # 中間調ピーク
    S -= 0.06 * (1.0 - np.clip(L / 0.14, 0, 1))    # シャドウ抑制
    S -= 0.035 * np.clip((L - 0.82) / 0.18, 0, 1)  # ハイライト抑制
    S = np.clip(S, 0.86, 1.11)

    r = L + S * (r - L)
    g = L + S * (g - L)
    b = L + S * (b - L)

    # --------------------------------------------------------
    # 4. C1基調 + 微C3 スプリットトーン
    # --------------------------------------------------------
    sh_mask = 1.0 - np.clip(L / 0.22, 0, 1)
    hi_mask = np.clip((L - 0.84) / 0.16, 0, 1)
    
    r -= 0.011 * sh_mask
    b += 0.011 * sh_mask
    r += 0.008 * hi_mask
    g += 0.004 * hi_mask

    # --------------------------------------------------------
    # 5. D1: 安全クランプ & 出力
    # --------------------------------------------------------
    r = np.clip(r, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    with open(output_path, "w") as f:
        f.write(f"TITLE \"ProPhoto Linear Mature: HueRot+MidSat+Split\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        for bz in range(size):
            for gy in range(size):
                for rx in range(size):
                    f.write(f"{r[rx, gy, bz]:.6f} {g[rx, gy, bz]:.6f} {b[rx, gy, bz]:.6f}\n")
                    
    print(f"✅ 生成完了: {output_path} ({size}^3 nodes)")

# 実行
generate_prophoto_mature_lut(size=33)