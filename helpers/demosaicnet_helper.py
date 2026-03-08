import sys
import os
import torch as th
import numpy as np

# Add demosaicnet_torch to sys.path
helpers_dir = os.path.dirname(os.path.abspath(__file__))
platypus_dir = os.path.dirname(helpers_dir)
demosaicnet_dir = os.path.join(platypus_dir, 'demosaicnet_torch')

if demosaicnet_dir not in sys.path:
    sys.path.append(demosaicnet_dir)

from demosaicnet import demosaick_load_model, demosaick, xtrans_mosaic, bayer_mosaic

def init_demosaicnet(mosaic_type='bayer', noiselevel=0.0, tile_size=512, device='cuda'):
    """
    DemosaicNetモデルを初期化する
    
    Args:
        mosaic_type (str): 'bayer' または 'xtrans'
        noiselevel (float): ノイズレベル (0.0 ならノイズなしモデル)
        tile_size (int): 分割処理するタイルサイズ
        device (str): 'cpu', 'cuda', 'mps' のいずれか
        
    Returns:
        dict: 初期化されたモデルとパラメータを含む辞書
    """
    model_ref = demosaick_load_model(
        net_path=None, 
        noiselevel=noiselevel, 
        xtrans=(mosaic_type == 'xtrans')
    )
    
    # デバイスの可用性チェックとフォールバック
    if device == 'mps' and not th.backends.mps.is_available():
        print("Warning: MPS is not available. Falling back to CPU.")
        device = 'cpu'
    elif device == 'cuda' and not th.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU.")
        device = 'cpu'
        
    dev = th.device(device)
    model_ref.to(device=dev)
    
    return {
        'model': model_ref,
        'mosaic_type': mosaic_type,
        'noiselevel': noiselevel,
        'tile_size': tile_size,
        'device': device
    }

def inference_demosaicnet(model_info, raw, crop=48, out_dtype=None, offset_y=0, offset_x=0):
    """
    DemosaicNetでRAW画像をデモザイクする
    
    Args:
        model_info (dict): init_demosaicnetで取得した辞書
        raw (numpy.ndarray): 入力RAW画像 (H, W形式の2次元配列、または単一チャネルの3次元配列など)
        crop (int): エッジのクロップサイズ
        out_dtype (type, optional): 出力画像のnumpyデータ型 (np.uint16, np.float32等)。Noneの場合は入力画像と同じ型が使用されます。
        offset_y (int): X-Trans/BayerパターンのY方向オフセット (0-5)
        offset_x (int): X-Trans/BayerパターンのX方向オフセット (0-5)
        
    Returns:
        numpy.ndarray: デモザイクされた画像 (H, W, C形式)
    """
    net = model_info['model']
    mosaic_type = model_info['mosaic_type']
    noiselevel = model_info['noiselevel']
    tile_size = model_info['tile_size']
    
    dtype = raw.dtype
    if dtype not in [np.uint16, np.float32]:
        raise ValueError(f'Input type not handled: {dtype}')
        
    Iref = raw.copy()
                       
    # 入力が (H, W) または (H, W, 1) の場合
    if len(Iref.shape) == 2:
        # モザイクを適用するために3チャネルに複製する
        Iref = np.dstack((Iref, Iref, Iref))
    elif len(Iref.shape) == 3 and Iref.shape[2] == 1:
        Iref = Iref[:, :, 0]
        Iref = np.dstack((Iref, Iref, Iref))
        
    I = Iref

    # padding (demosaicnet.py と同等) 
    # シフト(offset)はパディング非対称化ではなく、純粋に np.roll でピクセルをずらすことで対応する
    # 非対称パディングは内部ストライドクロップで致命的な位相ズレ(紫化・格子)を引き起こすため。
    if offset_y > 0 or offset_x > 0:
        I = np.roll(I, shift=(offset_y, offset_x), axis=(0, 1))

    if crop > 0:
        c_y = crop
        c_x = crop
        if mosaic_type == 'bayer':
            c_y += (c_y % 2)
            c_x += (c_x % 2)
            I = np.pad(I, [(c_y, c_y), (c_x, c_x), (0, 0)], 'reflect')
        else: # xtrans
            c_y += (c_y % 6)
            c_x += (c_x % 6)
            I = np.pad(I, [(c_y, c_y), (c_x, c_x), (0, 0)], 'symmetric')

    if dtype == np.uint16:
        I = I.astype(np.float32) / 65535.0
    I = np.array(I).transpose(2, 0, 1)

    # モザイクパターンの適用 (マスクの適用)
    if mosaic_type == 'xtrans':
        M = xtrans_mosaic(I)
    else:
        M = bayer_mosaic(I)
    
    # bayer_mosaic / xtrans_mosaic の返り値は (mos*mask, mask) なので1要素目を取得
    M = np.array(M)[:1, :, :, :] 

    # GPU転送等は demosaick 関数内で行われる (dev=next(net.parameters()).device を使用)
    with th.no_grad():
        R, _ = demosaick(net, M, noiselevel, tile_size, crop)

    R = R.squeeze().transpose(1, 2, 0)

    # クロップおよびオフセット成分を除去して元の画像サイズに合わせる
    if crop > 0:
        R = R[c_y : R.shape[0] - c_y, c_x : R.shape[1] - c_x, :]
        
    if offset_y > 0 or offset_x > 0:
        R = np.roll(R, shift=(-offset_y, -offset_x), axis=(0, 1))

    # 出力型の決定
    target_dtype = out_dtype if out_dtype is not None else dtype

    # floatから指定された型に戻す
    out = R
        
    if target_dtype == np.uint16:
        out = (out * 65535.0 + 0.5)
        out = np.clip(out, 0, 65535).astype(np.uint16)
        
    elif target_dtype == np.float32:
        out = out.astype(target_dtype)
        
    return out

def find_xtrans_offset(model_info, raw, patch_size=256):
    """
    推論結果のChroma Noise（色差分散）をバッチテストし、最適なX-Transのオフセットを求めます。
    
    Args:
        model_info (dict): init_demosaicnetで取得した辞書
        raw (numpy.ndarray): 入力RAW画像
    
    Returns:
        tuple: (offset_y, offset_x)
    """
    h, w = raw.shape[:2]
    # 画像中心から指定サイズのパッチを取得。パターンが壊れないように開始座標は6の倍数にする
    cy, cx = h // 2 - patch_size // 2, w // 2 - patch_size // 2
    cy, cx = cy - (cy % 6), cx - (cx % 6)
    test_patch = raw[cy : cy + patch_size, cx : cx + patch_size].copy()
    
    min_noise = float('inf')
    best_offset = (0, 0)
    
    for oy in range(6):
        for ox in range(6):
            # 高速化のため、cropは48でパディングを正確にテストする
            res = inference_demosaicnet(
                model_info, test_patch, crop=48, out_dtype=np.float32,
                offset_y=oy, offset_x=ox
            )
            
            # (R - G) と (B - G) の色差の分散を求める。不適切なパターンの場合、色差の分散が激増する。
            r, g, b = res[..., 0], res[..., 1], res[..., 2]
            diff_rg = r - g
            diff_bg = b - g
            
            # 垂直/水平のエッジ量を計算して足し合わせる
            noise = (
                np.mean(np.abs(diff_rg[1:] - diff_rg[:-1])) +
                np.mean(np.abs(diff_rg[:, 1:] - diff_rg[:, :-1])) +
                np.mean(np.abs(diff_bg[1:] - diff_bg[:-1])) +
                np.mean(np.abs(diff_bg[:, 1:] - diff_bg[:, :-1]))
            )
            
            if noise < min_noise:
                min_noise = noise
                best_offset = (oy, ox)
                
    return best_offset


# 使用例・テスト用コード
if __name__ == '__main__':
    print("Testing DemosaicNet Helper...")
    try:
        # テスト用のダミーRAW画像(Bayer)を作成
        h, w = 128, 128
        dummy_raw = np.random.randint(0, 65535, (h, w), dtype=np.uint16)
        
        # モデル初期化 (Apple Silicon であれば 'mps' を優先的に使用してテスト)
        device = 'mps' if th.backends.mps.is_available() else ('cuda' if th.cuda.is_available() else 'cpu')
        print(f"Initializing with device: {device}")
        
        model_info = init_demosaicnet(mosaic_type='bayer', noiselevel=0.0, tile_size=64, device=device)
        print("Model initialized.")
        
        # 推論実行 (デフォルト挙動)
        print("Running inference (default output dtype)...")
        output_img = inference_demosaicnet(model_info, dummy_raw, crop=16)
        
        # 推論実行 (float32指定)
        print("Running inference (float32 output dtype)...")
        output_float = inference_demosaicnet(model_info, dummy_raw, crop=16, out_dtype=np.float32)
        
        print("Inference completed successfully!")
        print(f"Input shape: {dummy_raw.shape}, dtype: {dummy_raw.dtype}")
        print(f"Output (default) shape: {output_img.shape}, dtype: {output_img.dtype}")
        print(f"Output (float32) shape: {output_float.shape}, dtype: {output_float.dtype}")
        
    except Exception as e:
        print(f"Testing failed with error: {e}")
