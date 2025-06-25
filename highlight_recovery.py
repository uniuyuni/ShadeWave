
import cv2
import numpy as np

import core
import local_contrast

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
    mask = np.where(
        mask <= 1.0,
        0.0,  # 1.0以下の値は0.0に
        np.where(
            mask >= M,
            1.0,  # M以上の値は1.0に
            (mask - 1.0) / (M - 1.0)  # 1.0〜Mの間を0.0〜1.0に線形補間
        )
    )
    
    #mask = mask > (1.0 + (np.max(mask) - 1.0) / 2.0)
    #mask = cv2.GaussianBlur(mask.astype(np.float32), (127, 127), sigmaX=0)
    #cv2.imwrite("mask.jpg", (mask * 255).astype(np.uint8))

    # 超ハイライト領域を広げてコントラストをつける
    contrast = np.where(hdr_img > 1.0, hdr_img ** 1.5, hdr_img)

    # 適応的トーンマッピング
    tonemapped = cv2.createTonemapReinhard(
        gamma=1.0, 
        intensity=0.5,
        light_adapt=0.8, 
        color_adapt=0.2
    ).process(contrast)

    # 全体にマイクロコントラストをかけてはっきりさせる
    #micro_contrast = local_contrast.apply_microcontrast(tonemapped, 60)
    micro_contrast = tonemapped
    
    # 赤のカラーバランスが崩れているので補正、ついでにディティールをはっきりさせる
    rgb = micro_contrast
    if is_enhance_red:
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
        hls = core.adjust_hls_color_one(hls, 'red', 0, 18/100, 0)
        rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
    rgb = local_contrast.apply_microcontrast(rgb, 150)
    result = core.apply_mask(micro_contrast, mask, rgb) # ハイライトにのみ適用

    return result
