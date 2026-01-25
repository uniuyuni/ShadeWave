
import cv2
import time
import numpy as np
import rawpy
import logging
import io
from functools import partial
from PIL import Image as PILImage, ImageOps as PILImageOps, ImageCms
from multiprocessing import shared_memory
import base64
import pyvips

import libraw_enhanced as lre
import config
import define
import file_cache_system
import params
from cores.dcp_profile import DCPReader, DCPProcessor
import cores.bit_depth_expansion as bit_depth_expansion
import cores.core as core
import cores.highlight_recovery as highlight_recovery
import cores.local_contrast as local_contrast

#print(f"libraw version:{rawpy.libraw_version}")

def imageset_to_shared_memory(imgset):
    """
    ImageSetを共有メモリに変換する
    """
    # 共有メモリを作成
    shm = shared_memory.SharedMemory(create=True, size=imgset.img.nbytes)
    # 共有メモリに画像を書き込む
    shared_array = np.ndarray(imgset.img.shape, dtype=imgset.img.dtype, buffer=shm.buf)
    shared_array[:] = imgset.img[:]
    # 共有メモリのサイズを返す
    return (imgset.file_path, shm.name, imgset.img.shape, imgset.img.dtype, imgset.flag)

def shared_memory_to_imageset(file_path, shm_name, shape, dtype, flag):
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
    imgset.flag = flag

    return imgset

class ImageSet:
    FORWARDMATRIX1 = np.array([
        [0.429000, 0.447800, 0.087600],
        [0.174400, 0.804300, 0.021300],
        [0.048700, 0.000600, 0.775700],
    ])
    FORWARDMATRIX2 = np.array([
        [0.397000, 0.418000, 0.149300],
        [0.219000, 0.743600, 0.044100],
        [0.102100, 0.001700, 0.721300],
    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.file_path = None
        self.img = None
        self.flag = None
        self.color_space = 'ProPhoto RGB'

    def _black(self, in_img, black_level):
        in_img[in_img < black_level] = black_level
        out_img = in_img - black_level
        return out_img
    
    def _apply_whitebalance(self, img_array, raw, exif_data, param):
        #wb = raw.camera_whitebalance
        #wb = np.array([wb[0], wb[1], wb[2]], dtype=np.float32)/1024.0
        gl, rl, bl = exif_data.get('WB_GRBLevels', "1024 1024 1024").split(' ')
        wb = np.array([int(rl), int(gl), int(bl)], dtype=np.float32) / 1024.0

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


    def _load_raw_preview(self, raw, file_path, exif_data, param):
        t0 = time.perf_counter()
        try:
            # exifのプレビューを展開
            preview_base64 = exif_data.get('PreviewImage', None)
            if preview_base64 is not None:
                decode = base64.b64decode(preview_base64[7:])
                with PILImage.open(io.BytesIO(decode)) as img:
                    img = PILImageOps.exif_transpose(img)
                    img_array = np.array(img)
            else:
                raise ValueError(f"Unsupported thumbnail format.")

            t1 = time.perf_counter()
            logging.info(f"PERF: Preview image decoded. {t1-t0:.4f}s Size: {img.size}")


            # float32へ
            img_array = core.convert_to_float32(img_array)

            # 色空間変換
            #スキップ
            #import cores.colour_functions as colour_functions
            #img_array = colour_functions.RGB_to_RGB(img_array, 'sRGB', 'ProPhoto RGB', 'cat02',
            #                    apply_cctf_encoding=False, apply_cctf_decoding=True, apply_gamut_mapping=True).astype(np.float32)
            import cores.color as color
            img_array = color.rgb_gamma_decode(img_array, 'sRGB') # ガンマ補正だけは必須
            self.color_space = 'sRGB'
            t2 = time.perf_counter()
            logging.info(f"PERF: Color conversion took {t2-t1:.4f}s")
            
            # ホワイトバランス定義
            img_array = self._apply_whitebalance(img_array, raw, exif_data, param)

            # クロップとexifデータの回転
            _, _, width, height = self._delete_exif_orientation(exif_data)

            # RAW画像のサイズに合わせてリサイズ（INTER_AREAは縮小時にノイズを低減）
            img_array = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_AREA)
            t3 = time.perf_counter()
            logging.info(f"PERF: Resize took {t3-t2:.4f}s. Target size: {width}x{height}")

            # 自動露出調整値を適当に設定する
            param['rgb_or_raw'] = 'rgb'
            param['auto_exposure'] = -2.0

            # 情報の設定
            params.set_image_param(param, img_array)
            param['lens_modifier'] = False

            # 正方形にする
            #img_array = core.adjust_shape_to_square(img_array)

            # 描画用に設定
            self.img = img_array
            
            logging.info(f"PERF: _load_raw_preview finished. Shape: {img_array.shape}")

        except Exception as e:
            logging.error(f"raw error {file_path} {e}")
        
        return (file_path, self, exif_data, param, 0)

    def _load_raw_fast(self, raw, file_path, exif_data, param):
        file_path, imgset, exif_data, param = self._load_raw_process(raw, file_path, exif_data, param, True)
        #return (file_path, imgset, exif_data, param, 0)
        return (file_path, imageset_to_shared_memory(imgset), exif_data, param, 1)

    def _load_raw_full(self, raw, file_path, exif_data, param):
        file_path, imgset, exif_data, param = self._load_raw_process(raw, file_path, exif_data, param, False)
        return (file_path, imageset_to_shared_memory(imgset), exif_data, param, -1)
                             
    def _load_raw_process(self, raw, file_path, exif_data, param, half=False):
        try:
            raw = lre.imread(file_path)

            img_array = raw.postprocess(output_color=lre.ColorSpace.ProPhotoRGB,
                                        demosaic_algorithm=lre.DemosaicAlgorithm.AMaZE,
                                        output_bps=32,
                                        #no_auto_scale=False,
                                        use_camera_wb=True,
                                        #user_wb = [1.0, 1.0, 1.0, 0.0],
                                        gamma=(1.0, 1.0),
                                        #four_color_rgb=True if half == False else False,
                                        half_size=half,
                                        #user_black=0,
                                        #no_auto_bright=True,
                                        highlight_mode=5,
                                        use_gpu_acceleration=True)
                                        #auto_bright_thr=0.0005)
                                        #fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full)

            # クロップとexifデータの回転
            top, left, width, height = self._delete_exif_orientation(exif_data)

            # サイズを整える
            if half == True:
                cheight = height // 2
                cwidth = width // 2
            else:
                cheight = height
                cwidth = width
            if cwidth > cheight:
                img_array = img_array[:cheight, :cwidth]
            else:
                # 回転させる
                img_array = np.rot90(img_array)
                img_array = img_array[-cheight:, :cwidth]
            
            # float32へ
            #img_array = core.convert_to_float32(img_array)

            #img_array = img_array - raw.black_level_per_channel[0] / ((1<<14)-1)
            #img_array = np.clip(img_array, 0, 1)

            # 倍率色収差低減
            #if half == False:
            #    img_array = core.chromatic_aberration_correction(img_array)

            """
            if True:
                # プロファイルを適用
                reader = DCPReader("dcp/Fujifilm X-E3 Adobe Standard.dcp")
                profile = reader.read()
                processor = DCPProcessor(profile)
                img_array = processor.process(img_array,
                        partial(colour_functions.XYZ_to_RGB, colourspace='ProPhoto RGB', chromatic_adaptation_transform=config.get_config('cat')),
                        illuminant='1', use_look_table=True)
            else:            
                # プロファイルを使わない時用
                img_array = np.dot(img_array, self.FORWARDMATRIX1.T)
                img_array = colour_functions.XYZ_to_RGB(img_array, 'ProPhoto RGB', None, config.get_config('cat')).astype(np.float32)
            """
            # ホワイトバランス定義
            img_array = self._apply_whitebalance(img_array, raw, exif_data, param)

            # 明るさ補正
            Ev = 0
            if config.get_config('raw_auto_exposure') == True:
                
                Ev, _ = core.calc_ev_from_image(core.normalize_image(img_array))
                
                # ここで補正
                img_array = core.adjust_exposure(img_array, Ev)
                print(f"img_array range: [{img_array.min():.4f}, {img_array.max():.4f}]")
                img_array = core.adjust_tone(img_array, white_level=-100)
                print(f"img_array range: [{img_array.min():.4f}, {img_array.max():.4f}]")

            param['rgb_or_raw'] = 'raw'
            param['auto_exposure'] = -Ev # 補正は逆方向
            
            # サイズを合わせる（INTER_AREAは縮小時にノイズを低減）
            #if img_array.shape[1] != width or img_array.shape[0] != height:
            if half == True:
                img_array = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_AREA)        

            # 情報の設定
            params.set_image_param(param, img_array)
            param['lens_modifier'] = not half

            # 正方形にする
            #img_array = core.adjust_shape_to_square(img_array)
            
            # 描画用に設定
            self.img = np.array(img_array)
            self.flag = half

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            logging.warning("file is not supported " + file_path)
        
        except Exception as e:
            logging.error(e)

        finally:
            raw.close()

        return (file_path, self, exif_data, param)

    def _load_rgb(self, raw, file_path, exif_data, param):
        # RGB画像で読み込んでみる
        with pyvips.Image.new_from_file(file_path) as vips_image:
            img_array = np.array(vips_image)

            # float32へ
            img_array = core.convert_to_float32(img_array)

            # グレイ画像をカラーへ
            if img_array.ndim == 2 or img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

            # 色空間変更
            def get_icc_profile_name(vips_image):                
                # ICCプロファイルバイナリ取得
                try:
                    icc_data = vips_image.get("icc-profile-data")
                    if icc_data is None:
                        return "sRGB"
                    
                    # Pillowでパース
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_data))
                    profile_name = ImageCms.getProfileDescription(profile)
                    return profile_name.strip()
                    
                except Exception as e:
                    return "sRGB"

            src_icc_profile_name = get_icc_profile_name(vips_image)
            import cores.colour_functions as colour_functions
            img_array = colour_functions.RGB_to_RGB(img_array, core.ICC_PROFILE_TO_COLOR_SPACE.get(src_icc_profile_name, 'sRGB'), 'ProPhoto RGB', config.get_config('cat'),
                                            apply_cctf_decoding=True, apply_gamut_mapping=True).astype(np.float32)

            # 画像からホワイトバランスパラメータ取得
            params.set_temperature_to_param(param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0)))
            
            # クロップとexifデータの回転
            self._delete_exif_orientation(exif_data)

            # 自動露出調整値を適当に設定する
            param['rgb_or_raw'] = 'rgb'
            param['auto_exposure'] = -2.5

            # 情報の設定
            params.set_image_param(param, img_array)
            param['lens_modifier'] = False

            # 正方形へ変換
            #img_array = core.adjust_shape_to_square(img_array)
            
        self.img = np.array(img_array)
        
        return (file_path, self, exif_data, param, -2)

    class Result():
        def __init__(self, worker, source):
            self.worker = worker
            self.source = source

    def preload(self, file_path, exif_data, param):
        self.file_path = file_path

        if file_path.lower().endswith(define.SUPPORTED_FORMATS_RAW):            
            result = []
            result.append(ImageSet.Result(worker="_load_raw_preview", source=None))
            #result.append(ImageSet.Result(worker="_load_raw_fast", source=None))
            result.append(ImageSet.Result(worker="_load_raw_full", source=None))

            return result
            
        elif file_path.lower().endswith(define.SUPPORTED_FORMATS_RGB):
            result = []
            result.append(ImageSet.Result(worker="_load_rgb", source=None))
            return result
            
        logging.warning("file is not supported " + file_path)
        return None

    def load(self, preload_result, file_path, exif_data, param):
        if not isinstance(preload_result, list):
            return
        file_cache_system.run_method(self, preload_result[len(preload_result)-1].worker, config._config, None, file_path, exif_data, param)
