
import cv2
import numpy as np

import core
import local_contrast

def reconstruct_highlight_details(hdr_img):
    """
    ハイライトディテールを回復する統合処理
    """
    # 飽和ピクセル復元用のマスク作成（HDR状態で作る）                      
    mask = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)
    mask = mask > (1.0 + (np.max(mask) - 1.0) / 2.0)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (127, 127), sigmaX=0)
    #cv2.imwrite("mask.jpg", (mask * 255).astype(np.uint8))

    # 超ハイライト領域を広げてコントラストをつける
    contrast = np.where(hdr_img > 1.0, hdr_img ** 1.5, hdr_img)

    # 適応的トーンマッピング
    tonemapped = cv2.createTonemapReinhard(
        gamma=1.5, 
        intensity=0.5,
        light_adapt=0.8, 
        color_adapt=0.2
    ).process(contrast)

    # 全体にマイクロコントラストをかけてはっきりさせる
    #micro_contrast = local_contrast.apply_microcontrast(tonemapped, 60)
    micro_contrast = tonemapped
    
    # 赤のカラーバランスが崩れているので補正、ついでにディティールをはっきりさせる
    hls = cv2.cvtColor(micro_contrast, cv2.COLOR_RGB2HLS_FULL)
    hls = core.adjust_hls_color_one(hls, 'red', 0, 18/100, 0)
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
    rgb = local_contrast.apply_microcontrast(rgb, 150)
    result = core.apply_mask(micro_contrast, mask, rgb) # ハイライトにのみ適用

    return result
