
import jax
from jax import jit
import jax.numpy as jnp

@jit
def rgb_to_hls(rgb):
    """
    RGBからHLSへの変換関数
    
    Args:
        rgb: shape (..., 3) の配列、RGB値 (範囲制限なし)
    
    Returns:
        shape (..., 3) の配列、HLS値 (H: 0-360, L・S: 範囲制限なし)
    """
    # RGBの値は任意の範囲を取りうるが、内部計算のために一度正規化する
    # ただし、負の値や1を超える値も維持したまま相対的な比率で計算
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    # 最大値と最小値を計算
    maxc = jnp.maximum(jnp.maximum(r, g), b)
    minc = jnp.minimum(jnp.minimum(r, g), b)
    
    # 輝度 (Lightness) の計算
    l = (maxc + minc) / 2.0
    
    # 彩度 (Saturation) の計算
    delta = maxc - minc
    
    # 彩度の計算（delta=0の場合の除算回避）
    # 数値的安定性のために小さな値を追加
    epsilon = 1e-10
    
    s = jnp.where(
        jnp.isclose(maxc, minc),
        0.0,
        jnp.where(
            l <= 0.5,
            delta / (maxc + minc + epsilon),
            delta / (2.0 - maxc - minc + epsilon)
        )
    )
    
    # 数値誤差により負の値が出ることがあるため、0以上に制限
    s = jnp.maximum(s, 0.0)
    
    # 色相 (Hue) の計算
    # 全てのチャンネルが同じ値の場合（グレースケール）は色相を0とする
    h = jnp.zeros_like(l)
    
    # R, G, Bのどれが最大値かに基づいて色相を計算
    # R が最大の場合
    mask_r = jnp.isclose(r, maxc)
    h = jnp.where(
        mask_r,
        60.0 * ((g - b) / (delta + epsilon)),  # 数値的安定性のためepsilonを追加
        h
    )
    
    # G が最大の場合
    mask_g = jnp.isclose(g, maxc)
    h = jnp.where(
        mask_g,
        60.0 * (2.0 + (b - r) / (delta + epsilon)),  # 数値的安定性のためepsilonを追加
        h
    )
    
    # B が最大の場合
    mask_b = jnp.isclose(b, maxc)
    h = jnp.where(
        mask_b,
        60.0 * (4.0 + (r - g) / (delta + epsilon)),  # 数値的安定性のためepsilonを追加
        h
    )
    
    # 色相を0-360の範囲に正規化
    h = jnp.where(h < 0, h + 360.0, h)
    h = jnp.mod(h, 360.0)
    
    # delta=0（グレースケール）の場合は色相を0に設定
    h = jnp.where(jnp.isclose(delta, 0.0), 0.0, h)
    
    # HLS形式の結果を返す
    return jnp.stack([h, l, s], axis=-1)

@jit
def hls_to_rgb(hls):
    """
    HLSからRGBへの変換関数
    
    Args:
        hls: shape (..., 3) の配列、HLS値 (H: 任意、L・S: 範囲制限なし)
    
    Returns:
        shape (..., 3) の配列、RGB値
    """
    h, l, s = hls[..., 0], hls[..., 1], hls[..., 2]
    
    # 色相が負の場合は正の値に変換して0-360の範囲に正規化
    h = jnp.where(h < 0, h + (1 + jnp.floor(-h / 360)) * 360, h)
    h = jnp.mod(h, 360.0)
    
    # HLS -> RGB の変換
    def hue_to_rgb(p, q, t):
        # tを0-1の範囲に正規化
        t = jnp.mod(t, 1.0)
        
        result = jnp.where(
            t < 1/6,
            p + (q - p) * 6.0 * t,
            jnp.where(
                t < 1/2,
                q,
                jnp.where(
                    t < 2/3,
                    p + (q - p) * (2/3 - t) * 6.0,
                    p
                )
            )
        )
        
        # 数値的な安定性のために結果をクリップ
        # 極端なHLS値の場合でも、最終的なRGB値が負になるのを防ぐ
        return result
    
    # 彩度が0の場合は、R=G=B=Lとなる（グレースケール）
    # それ以外の場合は、色相に基づいてRGBを計算
    q = jnp.where(
        l < 0.5,
        l * (1.0 + s),
        l + s - l * s
    )
    p = 2.0 * l - q
    
    r = hue_to_rgb(p, q, (h / 360.0) + 1/3)
    g = hue_to_rgb(p, q, h / 360.0)
    b = hue_to_rgb(p, q, (h / 360.0) - 1/3)
    
    rgb = jnp.stack([r, g, b], axis=-1)
    
    # 極端なHLS値から変換された場合に負のRGB値が生じることがあるため
    # 最終結果をクリッピングする選択肢もある（ただし、元の仕様では範囲制限なしとのこと）
    # rgb = jnp.clip(rgb, 0.0, None)  # 負の値のみをクリップする場合
    
    return rgb

# 変換のテスト
if __name__ == '__main__':
    # 通常範囲内のRGB値
    rgb_normal = jnp.array([0.5, 0.2, 0.8])
    
    # 範囲外のRGB値
    rgb_out_of_range = jnp.array([1.5, -0.3, 2.0])
    
    # 変換
    hls_normal = rgb_to_hls(rgb_normal)
    hls_out_of_range = rgb_to_hls(rgb_out_of_range)
    
    # 逆変換
    rgb_normal_restored = hls_to_rgb(hls_normal)
    rgb_out_of_range_restored = hls_to_rgb(hls_out_of_range)
    
    
    print('rgb_normal', rgb_normal)
    print('hls_normal', hls_normal)
    print('rgb_normal_restored', rgb_normal_restored)
    print('rgb_out_of_range', rgb_out_of_range)
    print('hls_out_of_range', hls_out_of_range)
    print('rgb_out_of_range_restored', rgb_out_of_range_restored)
    
