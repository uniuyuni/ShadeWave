
import json
import numpy as np
import logging

from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, Device, HDStrategy, RealESRGANModel, InteractiveSegModel, RunPluginRequest
from iopaint.plugins import RealESRGANUpscaler
from iopaint.plugins import InteractiveSeg

def predict(image, mask, config=None, model="mat", device=Device.mps, resize_limit=1280, use_realesrgan=True):

    if config is None:
        inpaint_request = InpaintRequest()
        inpaint_request.hd_strategy = HDStrategy.RESIZE
        inpaint_request.hd_strategy_resize_use_realesrgan = use_realesrgan
        inpaint_request.hd_strategy_resize_limit = resize_limit
        logging.info(f"Using default config: {inpaint_request}")
    else:
        with open(config, "r", encoding="utf-8") as f:
            inpaint_request = InpaintRequest(**json.load(f))

        logging.info(f"Using config: {inpaint_request}")

    model_manager = ModelManager(name=model, device=device)

    inpaint_result = model_manager(image, mask, inpaint_request)

    return inpaint_result

plugins = {}
plugins[RealESRGANUpscaler.name] = RealESRGANUpscaler(RealESRGANModel.realesr_general_x4v3, Device.mps)
plugins[InteractiveSeg.name] = InteractiveSeg(InteractiveSegModel.sam_hq_vit_h, Device.mps)

def predict_plugin(image, name, click=(0, 0), device=Device.mps):
    global plugins

    req = RunPluginRequest(name=name, image=str(create_sampling_array_fast(image)), scale=4, clicks=([[int(click[0]), int(click[1]), 1]]))
    req.name = name
    if req.name == InteractiveSeg.name:
        result = plugins[req.name].gen_mask(image, req)
    else:
        result = plugins[req.name].gen_image(image, req)
    return result


def create_sampling_array_fast(img, sample_ratio=0.02, grid_ratio=0.7):
    """
    画像から特徴的なサンプリングポイントを抽出し、比較用の小さなNumPy配列を作成（高速版）
    """
    h, w = img.shape[:2]
    total_pixels = h * w

    np.random.seed(42)  # ランダムシードを同じにする


    # サンプルサイズ計算（最小値を保証）
    sample_size = max(int(total_pixels * sample_ratio), 100)
    
    # 1. 四隅のサンプリング（ベクトル化）
    corner_size = min(h, w) // 10
    corner_points = []
    
    # 四隅の座標を一度に計算
    corners = [
        (0, corner_size, 0, corner_size),                     # 左上
        (0, corner_size, w-corner_size, w),                   # 右上
        (h-corner_size, h, 0, corner_size),                   # 左下
        (h-corner_size, h, w-corner_size, w)                  # 右下
    ]
    
    corner_samples_per_region = sample_size // 20
    corner_indices = []
    
    for y1, y2, x1, x2 in corners:
        # 各コーナー領域のインデックスを生成
        ys = np.random.randint(y1, y2, size=corner_samples_per_region)
        xs = np.random.randint(x1, x2, size=corner_samples_per_region)
        corner_indices.append(np.column_stack((ys, xs)))
    
    # すべてのコーナーインデックスを結合
    corner_indices = np.vstack(corner_indices)
    
    # 2. 中心付近のサンプリング（ベクトル化）
    center_y, center_x = h//2, w//2
    center_range = min(h, w) // 8
    center_samples = sample_size // 10
    
    center_ys = np.random.randint(center_y-center_range, center_y+center_range, size=center_samples)
    center_xs = np.random.randint(center_x-center_range, center_x+center_range, size=center_samples)
    center_indices = np.column_stack((center_ys, center_xs))
    
    # 3. 均等グリッドサンプリング（ベクトル化）
    grid_sample_size = int(sample_size * grid_ratio)
    grid_steps = int(np.sqrt(grid_sample_size))
    
    # 均等なグリッドインデックスを生成
    grid_y = np.linspace(0, h-1, num=grid_steps, dtype=int)
    grid_x = np.linspace(0, w-1, num=grid_steps, dtype=int)
    
    # メッシュグリッドを作成
    grid_y, grid_x = np.meshgrid(grid_y, grid_x)
    grid_indices = np.column_stack((grid_y.flatten(), grid_x.flatten()))
    
    # 4. ランダムサンプリング（残り）
    remaining = sample_size - len(corner_indices) - len(center_indices) - len(grid_indices)
    
    if remaining > 0:
        random_ys = np.random.randint(0, h, size=remaining)
        random_xs = np.random.randint(0, w, size=remaining)
        random_indices = np.column_stack((random_ys, random_xs))
    else:
        random_indices = np.empty((0, 2), dtype=int)
    
    # すべてのインデックスを結合
    all_indices = np.vstack((corner_indices, center_indices, grid_indices, random_indices))
    
    # インデックスが画像の境界内にあることを確認
    all_indices[:, 0] = np.clip(all_indices[:, 0], 0, h-1)
    all_indices[:, 1] = np.clip(all_indices[:, 1], 0, w-1)
    
    # 座標を正規化（0-1の範囲に）
    normalized_coords = all_indices.astype(np.float32)
    normalized_coords[:, 0] /= h
    normalized_coords[:, 1] /= w
    
    # サンプリングされたピクセル値を取得（for文なし）
    y_indices = all_indices[:, 0]
    x_indices = all_indices[:, 1]
    
    # チャネル数に応じて処理
    if len(img.shape) == 3:
        # カラー画像
        sampled_pixels = img[y_indices, x_indices]
    else:
        # グレースケール画像
        sampled_pixels = img[y_indices, x_indices].reshape(-1, 1)
    
    # 座標と結合（高速）
    # 正規化された座標と画素値を結合
    if len(img.shape) == 3:
        # カラー画像の場合、結果の形状は [サンプル数, 2+チャネル数]
        result = np.hstack((normalized_coords, sampled_pixels))
    else:
        # グレースケール画像の場合
        result = np.hstack((normalized_coords, sampled_pixels))
    
    return result.astype(np.float32)
