

import numpy as np
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

import splitimage
import aiutil
import core

def setup():
    """
    セットアップ関数。genai.Client のインスタンスを生成して返します。
    Returns:
        genai.Client: 新しい genai クライアントインスタンス。
    """

    return genai.Client()

def predict_helper(client, image, mask, bbox):
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
        if True:
#        if np.any((block == [1.0, 0.0, 0.0]).all(axis=-1)):
            pre_image = predict(client, block, i)
            block[..., :] = pre_image  # 予測結果でブロックを更新することで境界の不連続を減らす
            predict_blocks.append(pre_image)            
        else:
            predict_blocks.append(block)

    # 分割した処理画像を結合
    combine = splitimage.combine_image_with_overlap(predict_blocks, split_info)
    Image.fromarray((combine * 255).astype(np.uint8)).save("X-T5 Room combine.jpg")

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


def predict(client, fp32_image, num=0):
    """
    画像の赤いマスク領域のみをインペイントする関数。
    Args:
        client: 画像生成モデルへのAPIクライアント。
        fp32_image (np.ndarray): 0.0〜1.0の範囲のfloat32型画像配列。
        num (int, optional): 画像の識別番号や処理回数などに使用。デフォルトは0。
    Returns:
        np.ndarray: 赤いマスク領域がインペイントされたfloat32型画像配列。
    注意事項:
        - 赤いマスク領域のみを完全に除去し、背景に自然に馴染ませます。
        - 赤以外の領域は一切変更しません。
        - 画像の端から約192ピクセル以内にテクスチャ境界がある場合は、境界を自然にブレンドします。
        - 画像は大きな画像の一部であり、合成処理に影響するような位置変更や色調変更は行いません。
        - 処理後も赤いマスク領域が残っている場合は再帰的に処理します。
    """

    prompt="""
    ONLY inpaint the red masked areas. 
    CRITICAL COMMANDS:
    1. Remove ALL red masked pixels completely
    2. Fill red areas with perfectly matched background
    3. DO NOT TOUCH any non-red areas - not even slightly!
    4. No color changes, no brightness changes, no texture edits, no noise alteration outside red areas
    5. Keep every non-red pixel EXACTLY as original
    6. Zero modifications beyond red mask boundaries
    7. As the sole exception, if the texture boundary is located approximately 192 pixels from the top, bottom, left, or right edge of the image, blend the boundary to make it disappear. Avoid editing areas outside the boundary whenever possible.
    8. This image is a cropped section of a larger image. Other images exist above, below, to the left, and to the right, and will be composited later. Therefore, please refrain from performing any processing that could interfere with the compositing work, such as moving the object's position, resizing it, or altering the overall color tone.
    """
    print(f"Nano banana inpainting predict {num} {fp32_image.shape}.")

    pil_image = Image.fromarray((fp32_image * 255).astype(np.uint8))
#    my_file = client.files.upload(file="X-T5 Room.png")
    pil_image.save(f"X-T5 Room input {num}.jpg")

    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt, pil_image],
#        contents=[prompt, my_file],
        config=types.GenerateContentConfig(
#            system_instruction="You are an expert in image processing.",
            temperature=0.0,
        ),
    )

    if response.candidates is not None:
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                res_image = Image.open(BytesIO(part.inline_data.data))
                np_image = np.array(res_image)
                result = np_image.astype(np.float32) / 255.0
                if np.any(np.all(np_image == [255, 0, 0], axis=-1)):
                    result = predict(client, result, num)

                Image.fromarray((result * 255).astype(np.uint8)).save(f"X-T5 Room banana {num}.jpg")
                return result
        
    return fp32_image

if __name__ == "__main__":
    image = Image.open("X-T5 Room image.jpg")
    image = np.array(image).astype(np.float32) / 255.0

    mask = Image.open("X-T5 Room mask.png")
    mask = np.array(mask).astype(np.float32) / 255.0

    bboxs = core.get_multiple_mask_bbox(mask)
    if len(bboxs) > 0:
        client = setup()
        predict_image = predict_helper(client, image, mask, bboxs[0])
        Image.fromarray((predict_image * 255).astype(np.uint8)).save("X-T5 Room complete.jpg")
