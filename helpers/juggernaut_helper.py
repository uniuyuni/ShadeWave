
import numpy as np
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import os
import logging

import cores.splitimage as splitimage
import utils.aiutils as aiutils

# hf download RunDiffusion/Juggernaut-XI-v11 --local-dir ./checkpoints/Juggernaut-XI-Lightning

_MODEL_ID = "./checkpoints/Juggernaut-XI-Lightning"
_TILE_SIZE = 1024 # Lightning XLは1024x1024のタイルサイズで最適化されているため、分割時のタイルサイズは1024に設定
_OVERLAP_SIZE = 32*3
_EXPANSION_SCALE = 2/3

def _preprocess_numpy_inputs(img_np_f32, mask_np_f32):
    """
    入力の float32 NumPy 配列を Diffusers が受け付けられる PyTorch Tensor に変換する前処理
    
    img_np_f32:  形状 (H, W, 3)、値の範囲 0.0 ~ 1.0 の float32 ndarray
    mask_np_f32: 形状 (H, W) または (H, W, 1)、値の範囲 0.0 ~ 1.0 の float32 ndarray
    """
    # 1. 安全のために 0.0 ~ 1.0 にクリップ
    img_np = img_np_f32 # np.clip(img_np_f32, 0.0, 1.0)
    mask_np = mask_np_f32 # np.clip(mask_np_f32, 0.0, 1.0)
    
    # 2. PyTorch Tensor に変換 (この時点では float32 / CPU)
    img_tensor = torch.from_numpy(img_np)
    mask_tensor = torch.from_numpy(mask_np)
    
    # 3. 形状を配列の並びから (Channel, Height, Width) に変更
    # (H, W, 3) -> (3, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # マスクの形状を (1, H, W) に統一
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    elif mask_tensor.shape[-1] == 1:
        mask_tensor = mask_tensor.permute(2, 0, 1)

    # 4. バッチ次元を追加 (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0)
    
    # 5. 💡最重要: Diffusers が内部で要求する数値レンジへ変換
    # 画像データ: [0.0, 1.0] -> [-1.0, 1.0]
    img_tensor = img_tensor * 2.0 - 1.0
    
    # マスクデータ: 0.5 をしきい値として完全に 0.0 と 1.0 に二値化
    mask_tensor = torch.where(mask_tensor > 0.5, 1.0, 0.0)
    
    # 6. 8GB M1 Mac 用に float16 へキャスト（デバイスは CPU のまま維持）
    # ※ cpu_offload を有効にしているため、入力テンソルは CPU に置く必要があります
    img_tensor = img_tensor.to(dtype=torch.float16, device="cpu")
    mask_tensor = mask_tensor.to(dtype=torch.float16, device="cpu")
    
    return img_tensor, mask_tensor

def setup(device="mps"):    
    logging.info(f"Juggernaut XL ({_MODEL_ID}) をロード中...")
    
    # M1 Mac環境の安定性のために float32 を指定
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float32,
        use_safetensors=False,
        variant=None,
        local_files_only=True
    )

    # 💡 8GBメモリのための極限最適化
    # パイプラインを細分化して、必要なパーツのみを都度MPS(M1 GPU)に転送
    pipe.enable_model_cpu_offload(device=torch.device(device))
    pipe.enable_attention_slicing()
    #pipe.vae.to(dtype=torch.float32) # VAEだけは安定のためにfloat32化
    pipe.enable_vae_slicing()

    return pipe

def predict(pipe, image, mask):
    # Juggernaut XLのフォトリアリズムを引き出すためのプロンプトのコツ：
    # 独自のネガティブ埋め込み（Embedding）を使わない場合は、標準的なワードを明記します。
    #prompt = "scenic historic landscape, highly detailed photography, cinematic lighting, 8k resolution, crisp focus"
    #negative_prompt = "broken, anime, cartoon, graphic, blurry, low quality, distortion, artifacts, worst quality"
    prompt = "clean background, seamless texture, perfect blending, highly detailed scenery, cinematic lighting, crisp focus, photorealistic"
    negative_prompt = "extra objects, text, watermark, artifacts, blurry, low quality, worst quality, distortion, cartoon, anime"

    logging.info("Juggernaut XL でインペイント処理を実行中...")

    input_image_tensor, input_mask_tensor = _preprocess_numpy_inputs(image, mask)
    
    # 推論の実行
    # Juggernaut XLは通常 30〜40 ステップで本領を発揮しますが、
    # 8GB環境の速度とのトレードオフとして、まずは 25〜30 ステップあたりでテストするのがおすすめです。
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image_tensor,
        mask_image=input_mask_tensor,
        num_inference_steps=6,
        strength=1.0,            # 1.0に近いほど元のマスク内を完全に描き換えます
        guidance_scale=1.5,
        output_type="np"
    ).images[0]

    return image

def predict_helper(pipe, image, mask, bbox):
    """
    指定された画像領域に対してマスクを適用し、分割・予測・合成処理を行うヘルパー関数。
    Args:
        pipe: 予測モデルへのパイプラインオブジェクト。
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
    target_width = max(_TILE_SIZE, (int(bbox[2] * _EXPANSION_SCALE) + 7) // 8 * 8)  # 8の倍数に切り上げ
    target_height = max(_TILE_SIZE, (int(bbox[3] * _EXPANSION_SCALE) + 7) // 8 * 8)  # 8の倍数に切り上げ

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

    # 画像を分割して処理
    image_blocks, split_info = splitimage.split_image_with_overlap(crop_image, _TILE_SIZE, _TILE_SIZE, _OVERLAP_SIZE)
    mask_blocks, _ = splitimage.split_image_with_overlap(crop_mask, _TILE_SIZE, _TILE_SIZE, _OVERLAP_SIZE)
    predict_blocks = []
    for i, block_image in enumerate(image_blocks):
        block_mask = mask_blocks[i]
        if np.any(block_mask > 0):
            logging.info("Predicting block %s/%s...", i + 1, len(image_blocks))
            pre_image = predict(pipe, block_image, block_mask)
            if pre_image is None:
                predict_blocks.append(image_blocks[i])
                continue
            predict_blocks.append(pre_image)
        else:
            predict_blocks.append(block_image)

    # 分割した処理画像を結合
    combine = splitimage.combine_image_with_overlap(predict_blocks, split_info)

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

if __name__ == "__main__":
    # MPSフォールバックを有効化（未対応オペレータによるクラッシュを防止）
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

