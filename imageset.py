
import cv2
import numpy as np
import rawpy
import logging
import io
import colour
import time
from PIL import Image as PILImage, ImageOps as PILImageOps
from multiprocessing import shared_memory

from dcp_profile import DCPReader, DCPProcessor
import config
import define
import file_cache_system
import params
import bit_depth_expansion
import highlight_recovery
import core

print(f"libraw version:{rawpy.libraw_version}")

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

    def _black(self, in_img, black_level):
        in_img[in_img < black_level] = black_level
        out_img = in_img - black_level
        return out_img
    
    def _apply_whitebalance(self, img_array, raw, exif_data, param):
        #wb = raw.camera_whitebalance
        #wb = np.array([wb[0], wb[1], wb[2]], dtype=np.float32)/1024.0
        gl, rl, bl = exif_data.get('WB_GRBLevels', "1024 1024 1024").split(' ')
        gl, rl, bl = int(gl), int(rl), int(bl)
        wb = np.array([rl, gl, bl], dtype=np.float32) / 1024.0

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
        try:
            # RAWで読み込んでみる
            raw = rawpy.imread(file_path)

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
            img_array = core.convert_to_float32(img_array)

            # 色空間変換
            img_array = colour.RGB_to_RGB(img_array, 'sRGB', 'ProPhoto RGB', 'XYZ Scaling', # 最速？
                                apply_cctf_decoding=True, apply_gamut_mapping=False).astype(np.float32)

            # ホワイトバランス定義
            img_array = self._apply_whitebalance(img_array, raw, exif_data, param)

            # クロップとexifデータの回転
            _, _, width, height = self._delete_exif_orientation(exif_data)

            # RAW画像のサイズに合わせてリサイズ
            img_array = cv2.resize(img_array, (width, height))

            # 情報の設定
            params.set_image_param(param, img_array)

            # 正方形にする
            #img_array = core.adjust_shape_to_square(img_array)

            # 描画用に設定
            self.img = img_array

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            logging.warning("file is not supported " + file_path)

        except rawpy.LibRawNoThumbnailError:
            logging.error('no thumbnail found')

        except rawpy.LibRawUnsupportedThumbnailError:
            logging.error('unsupported thumbnail')

        except Exception as e:
            logging.error(f"raw error {file_path} {e}")
        
        finally:
            raw.close()
        
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
            raw = rawpy.imread(file_path)
            logging.debug(raw.sizes)
            """
            crop_top_margin, crop_left_margin = exif_data.get("RawImageCropTopLeft").split(' ')
            crop_top_margin, crop_left_margin = int(crop_top_margin), int(crop_left_margin)
            crop_width, crop_height = exif_data.get("RawImageCroppedSize").split('x')
            crop_width, crop_height = int(crop_width), int(crop_height)
            
            raw.__setattr__('sizes', rawpy.ImageSizes(raw_height=exif_data.get("RawImageFullHeight"),
                                         raw_width=exif_data.get("RawImageFullWidth"),
                                         height=crop_height,
                                         width=crop_width,
                                         top_margin=crop_top_margin,
                                         left_margin=crop_left_margin,
                                         iheight=exif_data.get("RawImageHeight"),
                                         iwidth=exif_data.get("RawImageWidth"),
                                         pixel_aspect=raw.sizes.pixel_aspect,
                                         flip=raw.sizes.flip,
                                         crop_top_margin=crop_top_margin,
                                         crop_left_margin=crop_left_margin,
                                         crop_width=crop_width,
                                         crop_height=crop_height))
            """
            img_array = raw.postprocess(output_color=rawpy.ColorSpace.raw, # if half == False else rawpy.ColorSpace.sRGB, # どのRGBカラースペースを指定してもsRGBになっちゃう
                                        #demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
                                        output_bps=16,
                                        no_auto_scale=False,
                                        use_camera_wb=True,
                                        #user_wb = [1.0, 1.0, 1.0, 0.0],
                                        gamma=(1.0, 1.0),
                                        four_color_rgb=True if half == False else False,
                                        half_size=half,
                                        #user_black=0,
                                        no_auto_bright=True,
                                        highlight_mode=5,)
                                        #auto_bright_thr=0.0005)
                                        #fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full)
            logging.debug(raw.sizes)
            """
            # ブラックレベル補正
            raw_image = self._black(raw.raw_image_visible, raw.black_level_per_channel[0])

            # float32へ
            raw_image = raw_image.astype(np.float32) / ((1<<exif_data.get('BitsPerSample', 14))-1)
            """

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
                img_array = img_array[-cheight:, :cwidth]
            
            # 下位2bit補完
            if half == False and config.get_config('raw_depth_expansion') == True:
                img_array = img_array >> 2
                img_array = bit_depth_expansion.process_rgb_image(img_array)

            # float32へ
            img_array = core.convert_to_float32(img_array)

            #img_array = img_array - raw.black_level_per_channel[0] / ((1<<14)-1)
            #img_array = np.clip(img_array, 0, 1)

            # 色空間変更
            if False:
                t = time.time()
                img_array = colour.RGB_to_RGB(img_array, 'sRGB', 'ProPhoto RGB', config.get_config('cat'),
                                              apply_cctf_decoding=False, apply_gamut_mapping=True).astype(np.float32)
                logging.debug(f"XYZ_to_RGB: {time.time() - t}")
            else:
                # プロファイルを適用
                # RAW色空間からXYZ色空間への変換にしか使ってない
                """
                dcp_path = "dcp/Fujifilm X-Pro3 Adobe Standard classic chrome.dcp"
                reader = DCPReader(dcp_path)
                profile = reader.read()
                processor = DCPProcessor(profile)
                img_array = processor.process(img_array, illuminant='1', use_look_table=True).astype(np.float32)
                """
                t = time.time()
                img_array = np.dot(img_array, self.FORWARDMATRIX1.T)
                img_array = colour.XYZ_to_RGB(img_array, 'ProPhoto RGB', None, config.get_config('cat')).astype(np.float32)
                logging.debug(f"XYZ_to_RGB: {time.time() - t}")
                
            # ホワイトバランス定義
            img_array = self._apply_whitebalance(img_array, raw, exif_data, param)

            # 明るさ補正
            if config.get_config('raw_auto_exposure') == True:
                # RAWの明るさをとってくる
                source_ev = exif_data.get('LightValue', None)

                # とってこれなかったら計算する
                if source_ev is None:
                    source_ev, _ = core.calc_ev_from_image(core.normalize_image(img_array))
                    source_ev = float(source_ev)
                
                # 設定値から目標Evを計算
                Ev = core.calc_ev_from_exif(exif_data)

                # 適用
                img_array = core.adjust_exposure(img_array, core.calculate_correction_value(source_ev, Ev, 4))

                # 超ハイライト領域のコントラストを上げてディティールをはっきりさせ、ついでにトーンマッピング
                img_array = highlight_recovery.reconstruct_highlight_details(img_array)

            # サイズを合わせる
            #if img_array.shape[1] != width or img_array.shape[0] != height:
            if half == True:
                img_array = cv2.resize(img_array, (width, height))

            # 情報の設定
            params.set_image_param(param, img_array)

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
        with PILImage.open(file_path) as img:
            img = PILImageOps.exif_transpose(img)
            img_array = np.array(img)

            # float32へ
            img_array = core.convert_to_float32(img_array)

            # グレイ画像をカラーへ
            if img_array.ndim == 2 or img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                #cv2.imwrite("test.jpg", cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            # 色空間変更
            src_icc_profile_name = core.get_icc_profile_name(img)
            img_array = colour.RGB_to_RGB(img_array, core.ICC_PROFILE_TO_COLOR_SPACE[src_icc_profile_name], 'ProPhoto RGB', config.get_config('cat'),
                                            apply_cctf_decoding=True, apply_gamut_mapping=True).astype(np.float32)

            # 画像からホワイトバランスパラメータ取得
            params.set_temperature_to_param(param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0)))
            
            # クロップとexifデータの回転
            self._delete_exif_orientation(exif_data)

            # 情報の設定
            params.set_image_param(param, img_array)

            # 正方形へ変換
            #img_array = core.adjust_shape_to_square(img_array)
            
        self.img = np.array(img_array)
        
        return (file_path, self, exif_data, param, 0)

    class Result():
        def __init__(self, worker, source):
            self.worker = worker
            self.source = source

    def preload(self, file_path, exif_data, param):
        self.file_path = file_path

        if file_path.lower().endswith(define.SUPPORTED_FORMATS_RAW):            
            result = []
            result.append(ImageSet.Result(worker="_load_raw_preview", source=None))
            result.append(ImageSet.Result(worker="_load_raw_fast", source=None))
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
        file_cache_system.run_method(self, preload_result[len(preload_result)-1].worker, None, file_path, exif_data, param)
