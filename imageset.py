
import cv2
import numpy as np
import core
import rawpy
import math
import logging
import io
import colour
from wand.image import Image as WandImage
from PIL import Image as PILImage, ImageOps as PILImageOps
import jax.numpy as jnp
from jax import jit
from functools import partial
from multiprocessing import shared_memory

#from dcp_profile import DCPReader, DCPProcessor
import config
import file_cache_system
import params
import util
import viewer_widget
import color
import bit_depth_expansion
import highlight_recovery

def imageset_to_shared_memory(imgset):
    """
    ImageSetを共有メモリに変換する
    """
    # 画像のサイズを取得
    width, height = imgset.img.shape[:2]
    # 共有メモリを作成
    shm = shared_memory.SharedMemory(create=True, size=imgset.img.nbytes)
    # 共有メモリに画像を書き込む
    shared_array = np.ndarray(imgset.img.shape, dtype=imgset.img.dtype, buffer=shm.buf)
    shared_array[:] = imgset.img[:]
    # 共有メモリのサイズを返す
    return (imgset.file_path, shm.name, imgset.img.shape, imgset.img.dtype)

def shared_memory_to_imageset(file_path, shm_name, shape, dtype):
    """
    共有メモリからImageSetを作成する
    """
    # 共有メモリを読み込む
    shm = shared_memory.SharedMemory(name=shm_name)
    #
    shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # 共有メモリから画像を読み込む
    img = np.ndarray(shape, dtype=dtype)
    img[:] = shared_array[:]
    # 共有メモリを閉じる
    shm.close()
    # 共有メモリを削除
    shm.unlink()
    # ImageSetを作成
    imgset = ImageSet()
    imgset.file_path = file_path
    imgset.img = img

    return imgset

class ImageSet:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.file_path = None
        self.img = None

    def _black(self, in_img, black_level):
        in_img[in_img < black_level] = black_level
        out_img = in_img - black_level
        return out_img
    
    def _apply_whitebalance(self, img_array, raw, exif_data, param):
        wb = raw.camera_whitebalance
        wb = np.array([wb[0], wb[1], wb[2]]).astype(np.float32)/1024.0
        """
        gl, rl, bl = exif_data.get('WB_GRBLevels', "1024 1024 1024").split(' ')
        gl, rl, bl = int(gl), int(rl), int(bl)
        wb = np.array([rl, gl, bl]).astype(np.float32)/1024.0
        """
        wb[1] = np.sqrt(wb[1])
        #img_array /= wb
        params.set_temperature_to_param(param, *core.invert_RGB2TempTint(wb))
        return img_array

    def _delete_exif_orientation(self, exif_data):
        top, left, width, height = core.get_exif_image_size_with_orientation(exif_data)
        core.set_exif_image_size(exif_data, top, left, width, height)
        if exif_data.get("Orientation", None) is not None:
            del exif_data["Orientation"]

        return (top, left, width, height)


    def _load_raw_preview(self, file_path, exif_data, param):
        try:
            # RAWで読み込んでみる
            raw = rawpy.imread(file_path)
            return raw

            # プレビューを読む
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                # JPEGフォーマットの場合
                with PILImage.open(io.BytesIO(thumb.data)).convert("RGB") as img:
                    img = PILImageOps.exif_transpose(img)
                    img_array = np.array(img)
                """
                # バイナリデータをNumPy配列に変換
                img_array = np.frombuffer(thumb.data, dtype=np.uint8)
                # OpenCVで画像としてデコード
                img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR cv2.IM)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                """
                """
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                # BITMAPフォーマットの場合
                """
            else:
                raise ValueError(f"Unsupported thumbnail format: {thumb.format}")

            # RGB画像初期設定
            img_array = util.convert_to_float32(img_array)

            # 色空間変換
            img_array = color.rgb_to_xyz(img_array, "sRGB", True)

            # GPU to CPU
            img_array = np.array(img_array)

            # ホワイトバランス定義
            img_array = self._apply_whitebalance(img_array, raw, exif_data, param)

            # GPU to CPU
            img_array = np.array(img_array)

            # クロップとexifデータの回転
            _, _, width, height = self._delete_exif_orientation(exif_data)

            # 情報の設定
            width, height = params.set_image_param(param, img_array)

            # RAW画像のサイズに合わせてリサイズ
            img_array = cv2.resize(img_array, (width, height))

            # 正方形にする
            img_array = core.adjust_shape(img_array)

            # 描画用に設定
            self.img = img_array

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            logging.warning("file is not supported " + file_path)
            raw.close()
            return False
        
        return raw

    def _load_raw_fast(self, raw, file_path, exif_data, param):
        file_path, imgset, exif_data, param = self._load_raw_process(raw, file_path, exif_data, param, True)
        return (file_path, imgset, exif_data, param, 0)

    def _load_raw(self, raw, file_path, exif_data, param):
        file_path, imgset, exif_data, param = self._load_raw_process(raw, file_path, exif_data, param, False)
        return (file_path, imageset_to_shared_memory(imgset), exif_data, param, 1)
                             
    def _load_raw_process(self, raw, file_path, exif_data, param, half=False):
        try:
            raw = rawpy.imread(file_path)
            img_array = raw.postprocess(output_color=rawpy.ColorSpace.sRGB, # どのRGBカラースペースを指定してもsRGBになっちゃう
                                        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                                        output_bps=16,
                                        no_auto_scale=False,
                                        use_camera_wb=True,
                                        user_wb = [1.0, 1.0, 1.0, 0.0],
                                        gamma=(1.0, 1.0),
                                        four_color_rgb=True,
                                        half_size=half,
                                        #user_black=0,
                                        no_auto_bright=True,
                                        highlight_mode=5,
                                        auto_bright_thr=0.0005)
                                        #fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full)
            """
            # ブラックレベル補正
            raw_image = self._black(raw.raw_image_visible, raw.black_level_per_channel[0])

            # float32へ
            raw_image = raw_image.astype(np.float32) / ((1<<exif_data.get('BitsPerSample', 14))-1)
            """

            # クロップとexifデータの回転
            top, left, width, height = self._delete_exif_orientation(exif_data)

            # サイズを整える
            """
            left = int(left * (img_array.shape[1] / width))
            if img_array.shape[1] < left + width:
                width = img_array.shape[1] - left
            top = int(top * (img_array.shape[0] / height))
            if img_array.shape[0] < top + height:
                height = img_array.shape[0] - top
            """
            img_array = img_array[top:top+height, left:left+width]

            # 下位2bit補完
            if config.get_config('raw_depth_expansion') == True:
                img_array = img_array >> 2
                img_array = bit_depth_expansion.process_rgb_image(img_array)

            # float32へ
            img_array = util.convert_to_float32(img_array)

            # 色空間変更
            img_array = colour.RGB_to_RGB(img_array, 'sRGB', 'ProPhoto RGB', 'CAT16',
                                          apply_cctf_encoding=False, apply_gamut_mapping=True).astype(np.float32)

            # プロファイルを適用
            #dcp_path = os.getcwd() + "/dcp/Fujifilm X-Pro3 Adobe Standard velvia.dcp"
            #reader = DCPReader(dcp_path)
            #profile = reader.read()
            #processor = DCPProcessor(profile)
            #img_array = processor.process(img_array, illuminant='1', use_look_table=True).astype(np.float32)
            
            # ホワイトバランス定義
            img_array = self._apply_whitebalance(img_array, raw, exif_data, param)

            # 明るさ補正
            if config.get_config('raw_auto_exposure') == True:
                source_ev, _ = core.calculate_ev_from_image(core.normalize_image(img_array))
                Av = exif_data.get('ApertureValue', 1.0)
                _, Tv = exif_data.get('ShutterSpeedValue', "1/100").split('/')
                Tv = float(_) / float(Tv)
                Ev = math.log2((Av**2)/Tv)
                Sv = math.log2(exif_data.get('ISO', 100)/100.0)
                Ev = Ev + Sv
                img_array = core.adjust_exposure(img_array, core.calculate_correction_value(source_ev, Ev, 4))

                # 超ハイライト領域のコントラストを上げてディティールをはっきりさせ、ついでにトーンマッピング
                img_array = highlight_recovery.reconstruct_highlight_details(img_array)

            # サイズを合わせる
            if img_array.shape[1] != width or img_array.shape[0] != height:
                img_array = cv2.resize(img_array, (width, height))

            # 情報の設定
            params.set_image_param(param, img_array)

            # 正方形にする
            img_array = core.adjust_shape(img_array)
            
            # 描画用に設定
            self.img = np.array(img_array)

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            logging.warning("file is not supported " + file_path)
        
        finally:
            raw.close()

        return (file_path, self, exif_data, param)

    def _load_rgb(self, file_path, exif_data, param):
        # RGB画像で読み込んでみる
        with PILImage.open(file_path) as img:
            img = PILImageOps.exif_transpose(img)
            img_array = np.array(img)

            # float32へ
            img_array = util.convert_to_float32(img_array)

            # 色空間変更
            img_array = colour.RGB_to_RGB(img_array, 'sRGB', 'ProPhoto RGB', 'CAT16',
                                          apply_cctf_encoding=False, apply_gamut_mapping=True).astype(np.float32)
            
            # 画像からホワイトバランスパラメータ取得
            params.set_temperature_to_param(param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0)))
            
            # クロップとexifデータの回転
            self._delete_exif_orientation(exif_data)

            # 情報の設定
            params.set_image_param(param, img_array)

            # 正方形へ変換
            img_array = core.adjust_shape(img_array)
            
        self.img = np.array(img_array)
        
        return 0

    class Result():
        def __init__(self, worker, source):
            self.worker = worker
            self.source = source

    def preload(self, file_path, exif_data, param):
        self.file_path = file_path

        if file_path.lower().endswith(viewer_widget.supported_formats_raw):
            raw = None #self._load_raw_preview(file_path, exif_data, param)
            
            result = []
            result.append(ImageSet.Result(worker="_load_raw_fast", source=raw))
            result.append(ImageSet.Result(worker="_load_raw", source=raw))

            return result
            
        elif file_path.lower().endswith(viewer_widget.supported_formats_rgb):
            return self._load_rgb(file_path, exif_data, param)
            
        logging.warning("file is not supported " + file_path)
        return 0

    def load(self, result, file_path, exif_data, param):
        if result == 0:
            return
        file_cache_system.run_method(self, result[len(result)-1].worker, None, file_path, exif_data, param)
