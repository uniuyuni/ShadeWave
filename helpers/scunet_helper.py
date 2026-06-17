
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import time
import logging

from effect_backends import low_frequency_transfer_adapter

import splitimage
import waitinfo
import cores.hlsrgb as hlsrgb
import utils.aiutils as aiutils

from SCUNet.models.network_scunet import SCUNet

# Configuration
_TILE_SIZE = 32*14
_TILE_OVERLAP = 32*2

def setup_scunet(is_color=True, device='cpu', is_half=False):

    model_path = "checkpoints/SCUNet/scunet_color_real_psnr.pth" if is_color else "checkpoints/SCUNet/scunet_gray_50.pth"

    """モデルを初期化してロードする"""
    model = SCUNet(in_nc=1 if "gray" in model_path else 3, config=[4]*7, dim=64, input_resolution=_TILE_SIZE)
    #model = SCUNet(in_nc=3, config=[4]*7, dim=64, input_resolution=_TILE_SIZE)
    model.load_state_dict(torch.load(model_path))
    #model = nn.DataParallel(model).eval().to(device)
    model = model.eval().to(device)

    # ユーザー設定
    user_config = {"is_gray": "gray" in model_path, "is_half": is_half}
    model.user_config = user_config

    if is_half:
        model.half()

    #model = model.to(memory_format=torch.channels_last)

    logging.info("SCUNet Model loaded.")
    return model

def predict_scunet(model, np_image):
    """
    numpy画像（float32, [0,1]範囲）をデノイズ
    入力: (H,W,3)のnumpy配列
    出力: (H,W,3)のnumpy配列
    """
    device = next(model.parameters()).device

    # 前処理
    is_n_dim2 = False
    if np_image.ndim == 2:
        is_n_dim2 = True
        np_image = np_image[..., np.newaxis]
    tensor_img = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).to(device)

    # 推論
    #with torch.no_grad():
    with torch.inference_mode():
        if model.user_config["is_half"]:
            tensor_img = tensor_img.half()
        #tensor_img = tensor_img.to(memory_format=torch.channels_last)
        restored = model(tensor_img)

    # 後処理
    restored = restored.squeeze().float().detach().cpu().numpy()
    if restored.ndim == 3:
        restored = restored.transpose(1, 2, 0)
        if is_n_dim2:
            restored = restored[..., 0]
    elif restored.ndim == 2: # in out_nc=1
        if is_n_dim2 == False:
            restored = restored[..., np.newaxis]

    return restored

def predict_scunet_helper(model, np_image):
    """
    SCUNet は [0,1] 付近を想定。HDR（>1 等）はグローバル min-max で潰さず、
    負値を 0 にクリップしたうえで hi=max(max,1) でスケールし、出力で戻す。
    """
    org_image = np.ascontiguousarray(np.asarray(np_image, dtype=np.float32))
    org_image = np.nan_to_num(org_image, nan=0.0, posinf=1.0, neginf=0.0)

    logging.info("SCUNet Predicting...")
    """
    imin = float(org_image.min())
    imax = float(org_image.max())
    if org_image.size == 0 or imax < 1e-12:
        logging.warning("SCUNet: empty or flat image, skipping.")
        return org_image.copy()

    np_image = org_image.copy()
    scale_back = 1.0
    if imax > 1.0 + 1e-6 or imin < -1e-6:
        logging.warning(
            f"SCUNet Input range [{imin}, {imax}] — scale by hi=max(max,1) (not global min-max)."
        )
        np_image = np.clip(np_image, 0.0, None)
        hi = max(float(np_image.max()), 1.0)
        np_image = np_image / hi
        scale_back = hi
    """

    k = aiutils.LOG1P_TONEMAP_K_DEFAULT
    np_image, hdr_white = aiutils.log1p_tonemap_forward_hdr(np_image, k=k, clip_nonnegative=True)
    split_images, split_info = splitimage.split_image_with_overlap(np_image, _TILE_SIZE, _TILE_SIZE, _TILE_OVERLAP)

    t1 = time.time()
    denoised_images = []
    for i, image in enumerate(split_images):
        if i % 10 == 0:
            t0 = time.time()
        waitinfo.set_text("ai_noise_reduction", f"{i+1} / {len(split_images)}")
        denoised_images.append(predict_scunet(model, image))
        if i % 10 == 9:
            logging.info(f"SCUNet Predict {i+1} / {len(split_images)} in {time.time() - t0:.2f} seconds")

    result = splitimage.combine_image_with_overlap(denoised_images, split_info)
    """
    if scale_back != 1.0:
        result = np.asarray(result * scale_back, dtype=np.float32)
    """
    result = aiutils.log1p_tonemap_inverse_hdr(result, hdr_white, k=k)

    logging.info("SCUNet Finalizing...")
    waitinfo.set_text("ai_noise_reduction", "Finalizing...")
    result = low_frequency_transfer_adapter.apply_low_frequency_transfer(
        result,
        org_image,
        sigma=75,
        highlight_threshold=0.70 * hdr_white,
        highlight_transition=0.40 * hdr_white,
        highlight_detail_strength=0.20,
        luminance_transfer_strength=0.0,
    )

    logging.info(f"SCUNet Completed. {time.time() - t1:.2f} seconds")
    waitinfo.set_text("ai_noise_reduction", "")

    return np.asarray(result, dtype=np.float32)
