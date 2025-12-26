
import cv2
import numpy as np

import cores.core as core
import cores.local_contrast as local_contrast

def reconstruct_highlight_details(hdr_img, is_enhance_red=True):
    """
    ハイライトディテールを回復する統合処理
    """
    # 飽和ピクセル復元用のマスク作成（HDR状態で作る）                      
    mask = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)

    # マスクの最大値を取得
    max_val = np.max(mask)

    # 目標の上限値 M を計算
    M = 1.0 + (max_val - 1.0) / 2.0

    # 最大値が1.0の場合は何もしない
    if np.isclose(M, 1.0):
        return hdr_img

    # 線形変換を適用
    #mask = np.clip((mask - 1.0) / (M - 1.0), 0.0, 1.0)
    threshold = 0.7
    mask = np.where(
        mask <= threshold,  # しきい値以下の値は0.0に
        0.0,
        np.where(
            mask >= M,
            1.0,  # M以上の値は1.0に
            (mask - threshold) / (M - threshold)  # しきい値〜Mの間を0.0〜1.0に線形補間
        )
    )

    # 超ハイライト領域を広げてコントラストをつける
    contrast = np.where(hdr_img > 1.0, hdr_img ** 1.5, hdr_img)

    # 赤のカラーバランスが崩れているので補正、ついでにディティールをはっきりさせる
    rgb = contrast
    if is_enhance_red:
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
        hls = core.adjust_hls_color_one(hls, 'enhance_red', 0, 80/100, 0)
        rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
    #rgb = local_contrast.apply_microcontrast(rgb, 100)
    result = core.apply_mask(contrast, mask, rgb) # ハイライトにのみ適用

    return result

def reconstruct_highlight_details2(source, mask):
    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    threshold = float(np.max(mask)) * 3 / 4
    mask[mask < threshold] = 0.0
    target = local_contrast.apply_microcontrast(source, 200)
    mask = mask[..., np.newaxis]
    img_array = source * (1-mask) + target * mask
    return img_array
