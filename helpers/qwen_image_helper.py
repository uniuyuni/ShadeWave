
import dashscope
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
import aiutil
import core

# DashScope APIキーを設定
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def numpy_to_base64_png(image_array):
    """
    float RGB numpy配列を16bit PNGのBase64文字列に変換
    
    Args:
        image_array: float型のnumpy配列 (H, W, 3), 値の範囲は0.0-1.0
    
    Returns:
        Base64エンコードされたPNG文字列
    """
    # float (0.0-1.0) を uint16 (0-65535) に変換
    image_uint16 = (image_array * 65535).astype(np.uint16)
    
    # RGBからBGRに変換（OpenCVはBGR形式）
    image_bgr = cv2.cvtColor(image_uint16, cv2.COLOR_RGB2BGR)
    
    # PNGとしてエンコード
    success, encoded_image = cv2.imencode('.png', image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    if not success:
        raise ValueError("Image encoding failed.")
    
    # Base64エンコード
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

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
    x, y, w, h = aiutil.calculate_expanded_crop(
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
    mskimg = crop_image * (1 - crop_mask) + crop_mask * np.array([1.0, 0.0, 0.0])  # マスク部分を赤で塗りつぶし

    # 画像を分割して処理
    blocks, split_info = splitimage.split_image_with_overlap(mskimg, 1024, 1024, 192)  # オーバーラップを大きめに設定
    predict_blocks = []
    for i, block in enumerate(blocks):
        Image.fromarray((block * 255).astype(np.uint8)).save(f"test/X-T5 Room input {i+1}.jpg")
        if np.any((block == [1.0, 0.0, 0.0]).all(axis=-1)):
            print(f"Predicting block {i+1}/{len(blocks)}...")
            pre_image = predict_func(block)
            block[..., :] = pre_image  # 予測結果でブロックを更新することで境界の不連続を減らす
            predict_blocks.append(pre_image)            
        else:
            predict_blocks.append(block)
        Image.fromarray((block * 255).astype(np.uint8)).save(f"test/X-T5 Room output {i+1}.jpg")

    # 分割した処理画像を結合
    combine = splitimage.combine_image_with_overlap(predict_blocks, split_info)
    Image.fromarray((combine * 255).astype(np.uint8)).save("test/X-T5 Room combine.jpg")

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

def predict_erace(image):
    # 設定
    PROMPT="""
    赤（色コード 255,0,0）の領域を周囲に馴染むように自然に削除
    """
    NEGATIVE_PROMPT="""
    赤の領域外の修正
    """

    return predict(image, PROMPT, NEGATIVE_PROMPT)

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
        
        response = MultiModalConversation.call(
            model='qwen-image-edit',
            messages=messages,
            temperature=0.0,
            negative_prompt=negative_prompt,
            watermark=False,
            prompt_extend=True,
        )
        
        # レスポンスを確認
        if response.status_code == 200:
            output = response.output
            
            # 編集された画像URLを取得
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    
                    # コンテンツから画像URLを抽出
                    for item in content:
                        if isinstance(item, dict) and 'image' in item:
                            image_url = item['image']
                            print(f"URL: {image_url}")
                            
                            # URLから画像をダウンロードしてnumpy配列に変換
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
    
    image = Image.open("test/X-T5 Room image.jpg")
    image = np.array(image).astype(np.float32) / 255.0

    mask = Image.open("test/X-T5 Room mask.png")
    mask = np.array(mask).astype(np.float32) / 255.0

    bboxs = core.get_multiple_mask_bbox(mask)
    if len(bboxs) > 0:
        predict_image = predict_helper(image, mask, bboxs[0], predict_erace)
        Image.fromarray((predict_image * 255).astype(np.uint8)).save("test/X-T5 Room complete.jpg")
