
import dashscope
from dashscope import ImageSynthesis
from dashscope import MultiModalConversation
import base64
from pathlib import Path
import json
import os
import requests
import cv2
import numpy as np
from PIL import Image

import splitimage
import utils.aiutils as aiutils
import cores.core as core

# DashScope APIキーを設定
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'


def _dashscope_request_timeout():
    try:
        return float(os.environ.get("DASHSCOPE_REQUEST_TIMEOUT", "60"))
    except (TypeError, ValueError):
        return 60.0

def numpy_to_base64_png(image_array):
    """
    float RGB numpy配列を8bit PNGのBase64文字列に変換
    
    Args:
        image_array: float型のnumpy配列 (H, W, 3), 値の範囲は0.0-1.0
    
    Returns:
        Base64エンコードされたPNG文字列
    """
    image_array = np.clip(image_array, 0.0, 1.0)
    image_uint8 = (image_array * 255).round().astype(np.uint8)
    
    # RGBからBGRに変換（OpenCVはBGR形式）
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    
    # PNGとしてエンコード
    success, encoded_image = cv2.imencode('.png', image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    if not success:
        raise ValueError("Image encoding failed.")
    
    # Base64エンコード
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')


def mask_to_base64_png(mask):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_uint8 = (np.clip(mask, 0.0, 1.0) * 255).round().astype(np.uint8)
    success, encoded_image = cv2.imencode('.png', mask_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not success:
        raise ValueError("Mask encoding failed.")
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')


def _red_mask_from_image(image_array, threshold=0.9):
    image_array = np.asarray(image_array)
    return (
        (image_array[..., 0] >= threshold)
        & (image_array[..., 1] <= (1.0 - threshold))
        & (image_array[..., 2] <= (1.0 - threshold))
    )


def _soft_edit_mask(mask, dilate_px=4, blur_px=3):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if not np.any(mask > 0):
        return mask[..., np.newaxis]

    kernel_size = max(1, int(dilate_px))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    edit_mask = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1).astype(np.float32)

    blur_px = max(0, int(blur_px))
    if blur_px > 0:
        ksize = blur_px * 2 + 1
        edit_mask = cv2.GaussianBlur(edit_mask, (ksize, ksize), 0)
        edit_mask = np.clip(edit_mask, 0.0, 1.0)

    return edit_mask[..., np.newaxis]


def _ensure_result_size(result_array, target_shape):
    if result_array is None:
        return None
    if result_array.shape[:2] == target_shape[:2]:
        return result_array
    return cv2.resize(result_array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def _extract_image_urls(output):
    def get_value(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return obj[key]
        except Exception:
            pass
        try:
            return getattr(obj, key)
        except Exception:
            return default

    urls = []
    for result in get_value(output, "results", []) or []:
        url = get_value(result, "url")
        if url:
            urls.append(url)
    for choice in get_value(output, "choices", []) or []:
        message = get_value(choice, "message", {})
        content = get_value(message, "content", []) or []
        for item in content:
            if isinstance(item, dict):
                url = item.get("image")
            else:
                url = get_value(item, "image")
            if url:
                urls.append(url)
    return urls


def _make_mask_marker_image(image, mask, alpha=1.0):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    alpha_mask = np.clip(mask * float(alpha), 0.0, 1.0)
    marker = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return image * (1.0 - alpha_mask) + marker * alpha_mask

def download_image_to_numpy(url):
    """
    URLから画像をダウンロードしてfloat RGB numpy配列に変換
    
    Args:
        url: 画像のURL
    
    Returns:
        float型のnumpy配列 (H, W, 3), 値の範囲は0.0-1.0
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # バイトデータから画像をデコード
    image_array = np.frombuffer(response.content, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise ValueError("Image decoding failed.")
    
    # BGRからRGBに変換
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # uint8 (0-255) を float (0.0-1.0) に変換
    image_float = image_rgb.astype(np.float32) / 255.0
    
    return image_float

def predict_helper(image, mask, bbox, predict_func):
    """
    指定された画像領域に対してマスクを適用し、分割・予測・合成処理を行うヘルパー関数。
    Args:
        client: 予測モデルへのクライアントオブジェクト。
        image (np.ndarray): 入力画像（H x W x 3）。
        mask (np.ndarray): マスク画像（H x W）、対象領域は1、それ以外は0。
        bbox (tuple): 対象領域のバウンディングボックス (x, y, w, h)。
    Returns:
        np.ndarray: 処理後の画像（元画像に予測結果を合成したもの）。
    処理概要:
        1. bboxを元に切り抜き範囲を拡張し、8の倍数かつ最低1024ピクセルに調整。
        2. 指定領域を切り抜き、マスク部分を赤色で塗りつぶす。
        3. 画像をオーバーラップ付きで分割し、各ブロックに対して予測処理を実施。
        4. 分割した予測画像を再結合し、境界部分をブレンドして滑らかに合成。
        5. 元画像の該当領域に合成結果を上書きして返す。
    注意:
        - 境界部分の不連続を減らすため、ブレンド幅を指定して合成処理を行う。
        - 予測処理は分割ブロックごとに実施される。
    """

    # 目標サイズを元の2倍または1024に設定
    target_width = max(1024, (bbox[2] * 2 + 7) // 8 * 8)  # 8の倍数に切り上げ
    target_height = max(1024, (bbox[3] * 2 + 7) // 8 * 8)  # 8の倍数に切り上げ

    # 拡張された切り抜き範囲を計算
    x, y, w, h = aiutils.calculate_expanded_crop(
        img_width=image.shape[1],
        img_height=image.shape[0],
        x=bbox[0],
        y=bbox[1],
        w=bbox[2],
        h=bbox[3],
        width=target_width,
        height=target_height
    )

    # 切り抜きとマスク適用
    crop_image = image[y:y+h, x:x+w, :]
    crop_mask = mask[y:y+h, x:x+w, np.newaxis]
    mskimg = _make_mask_marker_image(crop_image, crop_mask)

    # 画像を分割して処理
    blocks, split_info = splitimage.split_image_with_overlap(mskimg, 1024, 1024, 192)  # オーバーラップを大きめに設定
    original_blocks, _ = splitimage.split_image_with_overlap(crop_image, 1024, 1024, 192)
    mask_blocks, _ = splitimage.split_image_with_overlap(
        np.repeat(crop_mask.astype(np.float32), 3, axis=2),
        1024,
        1024,
        192,
    )
    predict_blocks = []
    for i, block in enumerate(blocks):
        #Image.fromarray((block * 255).astype(np.uint8)).save(f"../test/X-T5 Room input {i+1}.jpg")
        block_mask = mask_blocks[i][..., 0]
        if np.any(block_mask > 0):
            print(f"Predicting block {i+1}/{len(blocks)}...")
            try:
                pre_image = predict_func(block, block_mask)
            except TypeError:
                pre_image = predict_func(block)
            pre_image = _ensure_result_size(pre_image, block.shape)
            if pre_image is None:
                predict_blocks.append(original_blocks[i])
                continue
            edit_mask = _soft_edit_mask(block_mask)
            predict_blocks.append(pre_image * edit_mask + block * (1 - edit_mask))
        else:
            predict_blocks.append(block)
        #Image.fromarray((block * 255).astype(np.uint8)).save(f"../test/X-T5 Room output {i+1}.jpg")

    # 分割した処理画像を結合
    combine = splitimage.combine_image_with_overlap(predict_blocks, split_info)
    #Image.fromarray((combine * 255).astype(np.uint8)).save("../test/X-T5 Room combine.jpg")

    # エッジ部分をなめらかに合成するためのブレンド処理
    blend_width = 192  # ブレンドする幅（ピクセル）

    # 合成領域のマスクを作成
    blend_mask = np.ones_like(combine[..., 0])
    for i in range(blend_width):
        alpha = (i + 1) / blend_width
        # 上端
        blend_mask[i, :] *= alpha
        # 下端
        blend_mask[-(i+1), :] *= alpha
        # 左端
        blend_mask[:, i] *= alpha
        # 右端
        blend_mask[:, -(i+1)] *= alpha
    blend_mask = blend_mask[..., np.newaxis]

    # 元画像と合成画像をブレンド
    image_crop = image[y:y+h, x:x+w, :]
    image[y:y+h, x:x+w, :] = combine * blend_mask + image_crop * (1 - blend_mask)

    return image

def predict_erace(image, mask=None):
    # 設定
    PROMPT="""
    Only edit the pure red masked area.
    Remove the red masked content and fill it naturally so it blends with the surrounding area.
    Do not change anything outside the red area.
    Remove all red pixels.
    """
    NEGATIVE_PROMPT="""
    red stain, red object, outline, seam, patch, color shift, blur, style change, changes outside the red area
    """
    return predict(image, PROMPT, NEGATIVE_PROMPT)


def predict_with_mask(image_array, mask, prompt, negative_prompt=""):
    try:
        print("Encoding base image and mask...")
        image_base64 = numpy_to_base64_png(image_array)
        mask_base64 = mask_to_base64_png(mask)
        base_image_url = f"data:image/png;base64,{image_base64}"
        mask_image_url = f"data:image/png;base64,{mask_base64}"

        print("Masked image editing in progress...")
        call_kwargs = {
            "api_key": dashscope.api_key,
            "model": "qwen-image-2.0",
            "prompt": prompt,
            "base_image_url": base_image_url,
            "mask_image_url": mask_image_url,
            "function": "description_edit_with_mask",
            "n": 1,
            "watermark": False,
            "request_timeout": _dashscope_request_timeout(),
        }
        if negative_prompt and negative_prompt.strip():
            call_kwargs["negative_prompt"] = negative_prompt

        response = ImageSynthesis.call(**call_kwargs)
        if response.status_code != 200:
            print(f"Masked edit error: {response.code} - {response.message}")
            return None

        for image_url in _extract_image_urls(response.output):
            print(f"URL: {image_url}")
            print("Downloading masked edit image...")
            result_array = download_image_to_numpy(image_url)
            print(f"Done! Output size: {result_array.shape}")
            return result_array

        print("Masked edit image data not found.")
        print("Response:", response.output)
        return None
    except Exception as e:
        print(f"Masked edit failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict(image_array, prompt, negative_prompt=""):
    """
    Qwen Image Editを使って画像を編集
    
    Args:
        image_array: float型のnumpy配列 (H, W, 3), 値の範囲は0.0-1.0
        prompt: 編集指示のプロンプト
        negative_prompt: ネガティブプロンプト（避けたい要素）
    
    Returns:
        編集後の画像 (float型numpy配列、H, W, 3)、エラー時はNone
    """
    try:
        # numpy配列をBase64 PNGに変換
        print("Encoding image...")
        image_base64 = numpy_to_base64_png(image_array)
        
        # APIリクエストを送信
        print("Image editing in progress...")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": f"data:image/png;base64,{image_base64}"
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ]
        
        call_kwargs = {
            "api_key": dashscope.api_key,
            "model": "qwen-image-2.0",
            "messages": messages,
            "result_format": "message",
            "stream": False,
            "n": 1,
            "watermark": False,
            "request_timeout": _dashscope_request_timeout(),
        }
        if negative_prompt and negative_prompt.strip():
            call_kwargs["negative_prompt"] = negative_prompt

        response = MultiModalConversation.call(**call_kwargs)
        
        # レスポンスを確認
        if response.status_code == 200:
            output = response.output
            
            if 'choices' in output and len(output['choices']) > 0:
                for image_url in _extract_image_urls(output):
                    print(f"URL: {image_url}")
                    print("Downloading image...")
                    result_array = download_image_to_numpy(image_url)
                    print(f"Done! Output size: {result_array.shape}")
                    return result_array
            
            print("Image data not found.")
            print("Response:", output)
            return None
        else:
            print(f"Error: {response.code} - {response.message}")
            return None
            
    except Exception as e:
        print(f"An error has occurred.: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 使用例
if __name__ == "__main__":
    
    image = Image.open("../test/X-T5 Room image.jpg")
    image = np.array(image).astype(np.float32) / 255.0

    mask = Image.open("../test/X-T5 Room mask.png")
    mask = np.array(mask).astype(np.float32) / 255.0

    bboxs = core.get_multiple_mask_bbox(mask)
    if len(bboxs) > 0:
        predict_image = predict_helper(image, mask, bboxs[0], predict_erace)
        Image.fromarray((predict_image * 255).astype(np.uint8)).save("../test/X-T5 Room complete.jpg")
