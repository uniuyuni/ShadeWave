
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import time
import logging

import splitimage
import waitinfo
import cores.hlsrgb as hlsrgb
import utils.aiutils as aiutils

from SCUNet.models.network_scunet import SCUNet

# Configuration
_TILE_SIZE = 32*14
_TILE_OVERLAP = 32*2

def setup_scunet(is_color=True, device='cpu', is_half=False):

    model_path = "checkpoints/SCUNet/scunet_color_real_gan.pth" if is_color else "checkpoints/SCUNet/scunet_gray_50.pth"

    """モデルを初期化してロードする"""
    model = SCUNet(in_nc=1 if "gray" in model_path else 3, config=[4]*7, dim=64, input_resolution=_TILE_SIZE)
    #model = SCUNet(in_nc=3, config=[4]*7, dim=64, input_resolution=_TILE_SIZE)
    model.load_state_dict(torch.load(model_path))
    model = nn.DataParallel(model).eval().to(device)

    # ユーザー設定
    user_config = {"is_gray": "gray" in model_path, "is_half": is_half}
    model.user_config = user_config

    if is_half:
        model.half()

    logging.info("SCUNet Model loaded.")
    return model

def predict_scunet(model, np_image):
    """
    numpy画像（float32, [0,1]範囲）をデノイズ
    入力: (H,W,3)のnumpy配列
    出力: (H,W,3)のnumpy配列
    """
    device = next(model.parameters()).device

    # 輝度だけに掛ける
    if model.user_config["is_gray"]:
        hlcg_image = hlsrgb.rgb_to_hlc_gain(np_image)
        h, l, c, g = cv2.split(hlcg_image)
        np_image = l

    # 前処理
    if np_image.ndim == 2:
        np_image = np_image[..., np.newaxis]
    tensor_img = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        if model.user_config["is_half"]:
            tensor_img = tensor_img.half()
        restored = model(tensor_img)

    # 後処理
    restored = restored.squeeze().float().detach().cpu().numpy()
    if restored.ndim == 3:
        restored = restored.transpose(1, 2, 0)

    # 復元
    if model.user_config["is_gray"]:
        l = restored
        hlcg_image = cv2.merge([h, l, c, g])
        restored = hlsrgb.hlc_gain_to_rgb(hlcg_image)

    return restored

def predict_scunet_helper(model, np_image):

    org_image = np_image.copy()

    logging.info("SCUNet Predicting...")
    imin = np_image.min()
    imax = np_image.max()
    if 0.0 < imin or 1.0 < imax:
        logging.warning(f"SCUNet Input image range is [{imin}, {imax}].")
        if imax != imin:
            np_image = (np_image - imin) / (imax - imin)
        else:
             np_image = np_image - imin

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
    if 0.0 < imin or 1.0 < imax:
        if imax != imin:
            result = (result * (imax - imin)) + imin
        else:
            result = result + imin

    logging.info("SCUNet Finalizing...")
    waitinfo.set_text("ai_noise_reduction", "Finalizing...")
    result = aiutils.apply_low_frequency_transfer(result, org_image, sigma=75)

    logging.info(f"SCUNet Completed. {time.time() - t1:.2f} seconds")
    waitinfo.set_text("ai_noise_reduction", "")

    return result
