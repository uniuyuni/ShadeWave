
import sys
import types
import logging

try:
    # Check if `torchvision.transforms.functional_tensor` and `rgb_to_grayscale` are missing
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    # Import `rgb_to_grayscale` from `functional` if it’s missing in `functional_tensor`
    from torchvision.transforms.functional import rgb_to_grayscale

    # Create a module for `torchvision.transforms.functional_tensor`
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale

    # Add this module to `sys.modules` so other imports can access it
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# GFPGANのキャッシュ用グローバル変数
_gfpgan_face_enhancer = None

def init_realesrgan(
    model_name='RealESRGAN_x4plus',
    denoise_strength=0.5,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    fp32=False,
    gpu_id=None
):
    """
    Real-ESRGANモデルを初期化する
    
    Args:
        model_name (str): 使用するモデル名
        denoise_strength (float): ノイズ除去強度 (realesr-general-x4v3モデルのみ有効)
        tile (int): タイルサイズ (メモリ不足時に使用)
        tile_pad (int): タイルのパディングサイズ
        pre_pad (int): 事前パディングサイズ
        fp32 (bool): Trueでfloat32精度を使用
        gpu_id (int): 使用するGPU ID
    
    Returns:
        RealESRGANer: 初期化されたアップサンプラーオブジェクト
    """
    # モデル名の拡張子を除去
    model_name = model_name.split('.')[0]
    
    # モデル設定
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # モデルパスの決定
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url, 
                model_dir=os.path.join(ROOT_DIR, 'weights'), 
                progress=True, 
                file_name=None
            )

    # DNIウェイトの設定
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # アップサンプラーの初期化
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id
    )
    
    return upsampler

def inference_realesrgan(
    upsampler,
    img,
    outscale=4.0,
    face_enhance=False
):
    """
    Real-ESRGANで画像を超解像する
    
    Args:
        upsampler (RealESRGANer): 初期化済みアップサンプラー
        img (numpy.ndarray): 入力画像 (OpenCV形式)
        outscale (float): 出力スケール係数
        face_enhance (bool): Trueで顔強調を有効化
    
    Returns:
        numpy.ndarray: 超解像された出力画像
    """
    global _gfpgan_face_enhancer
    
    # 画像モードの判定 (RGBAかどうか)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    try:
        if face_enhance:
            # GFPGANの初期化 (初回のみ)
            if _gfpgan_face_enhancer is None:
                from gfpgan import GFPGANer
                _gfpgan_face_enhancer = GFPGANer(
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    upscale=outscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler,
                    device=f'cuda:{upsampler.gpu_id}' if upsampler.gpu_id is not None else 'cpu'
                )
            # 顔強調処理
            _, _, output = _gfpgan_face_enhancer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
        else:
            # 通常の超解像処理
            output, _ = upsampler.enhance(img, outscale=outscale)
            
    except RuntimeError as error:
        logging.exception("Error during Real-ESRGAN processing")
        logging.warning("If you encounter CUDA out of memory, try reducing the tile size.")
        raise RuntimeError from error

    return output

# 使用例
if __name__ == '__main__':
    # 1. モデルの初期化
    upsampler = init_realesrgan(
        model_name='RealESRGAN_x4plus',
        gpu_id=0  # GPUを使用する場合
    )
    
    # 2. 画像の読み込み
    img = cv2.imread('input.jpg', cv2.IMREAD_UNCHANGED)
    
    # 3. 超解像実行
    try:
        output_img = inference_realesrgan(
            upsampler=upsampler,
            img=img,
            outscale=4.0,
            face_enhance=True
        )
        
        # 4. 結果の保存
        cv2.imwrite('output.jpg', output_img)
        logging.info("Processing completed successfully!")
        
    except RuntimeError:
        logging.exception("Processing failed due to a runtime error.")
