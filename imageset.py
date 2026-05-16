
import cv2
import time
import numpy as np
import rawpy
import logging
import io
from functools import partial
import importlib.metadata
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
import cores.colour_functions as colour_functions
import cores.color as color
from enums import ImageFidelity, LoadStage

def _log_lre_info():
    try:
        # metadata からバージョンを取得 (pip install されている場合)
        v = importlib.metadata.version("libraw_enhanced")
        logging.info(f"libraw_enhanced version: {v} (from metadata)")
    except importlib.metadata.PackageNotFoundError:
        try:
            logging.info(f"libraw_enhanced version: {lre.__version__} (from module)")
        except AttributeError:
            logging.warning("libraw_enhanced version attribute not found. Installation might be corrupted.")
    logging.info(f"libraw_enhanced file path: {lre.__file__}")

_log_lre_info()

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
    fid = getattr(imgset, "fidelity", ImageFidelity.FULL)
    fid_val = fid.value if isinstance(fid, ImageFidelity) else str(fid)
    return (imgset.file_path, shm.name, imgset.img.shape, imgset.img.dtype, fid_val)

def shared_memory_to_imageset(file_path, shm_name, shape, dtype, *rest):
    """
    共有メモリからImageSetを作成する。
    rest[0]: fidelity 文字列（'preview' / 'full'）。
    旧6タプル (…, fidelity, raw_half_bool) の末尾 bool は無視する。
    旧 (raw_half_bool, fidelity) 順も rest の型で判別して吸収する。
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
    field_a = rest[0] if len(rest) > 0 else None
    field_b = rest[1] if len(rest) > 1 else None
    if isinstance(field_a, bool) and isinstance(field_b, str):
        field_a, field_b = field_b, None
    elif isinstance(field_b, bool):
        field_b = None

    fidelity_val = None
    if field_a is not None and isinstance(field_a, str) and field_a in (
        ImageFidelity.PREVIEW.value,
        ImageFidelity.FULL.value,
    ):
        fidelity_val = field_a
    elif field_b is not None and isinstance(field_b, str):
        fidelity_val = field_b
    elif field_a is not None or field_b is not None:
        fidelity_val = field_a if isinstance(field_a, str) else field_b

    # ImageSetを作成
    imgset = ImageSet()
    imgset.file_path = file_path
    imgset.img = img
    if fidelity_val is not None:
        try:
            imgset.fidelity = ImageFidelity(fidelity_val)
        except ValueError:
            imgset.fidelity = ImageFidelity.FULL
    else:
        imgset.fidelity = ImageFidelity.FULL

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
        self.fidelity = ImageFidelity.FULL
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

    def _override_exif_size_from_raw_geometry(self, raw, exif_data, half_size=False):
        """
        LibRaw側の最終出力ジオメトリで EXIF サイズ情報を補正する。
        まずは Fuji honeycomb のようなセンサー固有変換に対応。
        """
        try:
            geometry = raw.get_output_geometry(half_size=half_size)
            if not geometry.get("is_fuji_rotated_output", False):
                return

            width = int(geometry.get("width", 0))
            height = int(geometry.get("height", 0))
            if width <= 0 or height <= 0:
                return

            top, left, _, _ = core.get_exif_image_size(exif_data)
            core.set_exif_image_size(exif_data, top, left, width, height)
            logging.info(f"EXIF size overridden by raw geometry: {width}x{height}")
        except Exception as e:
            logging.warning(f"Failed to override EXIF size from raw geometry: {e}")

    def _delete_exif_orientation(self, exif_data):
        top, left, width, height = core.get_exif_image_size_with_orientation(exif_data)
        core.set_exif_image_size(exif_data, top, left, width, height)
        if exif_data.get("Orientation", None) is not None:
            del exif_data["Orientation"]

        return (top, left, width, height)


    def _load_raw_preview(self, raw, file_path, exif_data, param):
        t0 = time.perf_counter()
        try:
            # Fuji honeycomb などセンサー固有ジオメトリをプレビューにも反映
            # （後段は既存の EXIF サイズベース処理をそのまま利用）
            if raw is not None:
                self._override_exif_size_from_raw_geometry(raw, exif_data, half_size=False)
            else:
                try:
                    with lre.imread(file_path) as raw_for_geometry:
                        self._override_exif_size_from_raw_geometry(
                            raw_for_geometry, exif_data, half_size=False
                        )
                except Exception as e:
                    logging.warning(f"Failed to load raw for preview geometry: {e}")

            # exifのプレビューを展開
            preview_base64 = exif_data.get('PreviewImage', None)
            if preview_base64 is not None:
                decode = base64.b64decode(preview_base64[7:])
                with PILImage.open(io.BytesIO(decode)) as img:
                    img = PILImageOps.exif_transpose(img)
                    img = img.convert("RGB")
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
            self.fidelity = ImageFidelity.PREVIEW
            
            logging.info(f"PERF: _load_raw_preview finished. Shape: {img_array.shape}")

        except Exception as e:
            logging.error(f"raw error {file_path} {e}")
        
        return (file_path, self, exif_data, param, LoadStage.FIRST_PAINTABLE)

    def _load_raw_full(self, raw, file_path, exif_data, param):
        file_path, imgset, exif_data, param = self._load_raw_process(raw, file_path, exif_data, param)
        return (file_path, imageset_to_shared_memory(imgset), exif_data, param, LoadStage.FULL_DECODE)
                             
    def _load_raw_process(self, raw, file_path, exif_data, param):
        raw = None
        try:
            raw = lre.imread(file_path)
            self._override_exif_size_from_raw_geometry(raw, exif_data, half_size=False)

            # AI demosaicフラグ
            ai_demosaic = config.get_config('ai_demosaic')

            if ai_demosaic:
                # AIデモザイク時は preprocess=True となり、LibRawの内部処理が途中で終わるため、
                # LibRaw側の defringe は実質的に機能しません。
                logging.warning("AI Demosaic is ENABLED. LibRaw's internal defringe will be SKIPPED due to preprocess=True.")
            else:
                logging.info("LibRaw internal processing with defringe=True.")

            # デモザイク、AI demosaicするときは preprocessだけ
            img_array = raw.postprocess(output_color=lre.ColorSpace.ProPhotoRGB,
                                        demosaic_algorithm=lre.DemosaicAlgorithm.AMaZE,
                                        output_bps=32,
                                        #no_auto_scale=False,
                                        use_camera_wb=True,
                                        #user_wb = [1.0, 1.0, 1.0, 0.0],
                                        gamma=(1.0, 1.0),
                                        half_size=False,
                                        #user_black=0,
                                        #no_auto_bright=True,
                                        highlight_mode=5,
                                        defringe=True,
                                        defringe_green=True,
                                        lateral_ca_correction=True,
                                        axial_ca_correction=True,
                                        use_gpu_acceleration=True,
                                        preprocess=ai_demosaic)
            print(f"[DEBUG] postprocess min={img_array.min()} max={img_array.max()}")
            
            if ai_demosaic:
                from helpers.demosaicnet_helper import init_demosaicnet, inference_demosaicnet_with_log_tonemap, find_xtrans_offset

                # NOTE: noiselevel=0.0 を使用する。暗い画像でnoiselevel=0.3にすると
                # DemosaicNetがGチャンネルを負値にするため紫色になる。
                # (信号が0.01~0.02なのに対しnoise=0.3では信号対ノイズ比が極低になる)
                model_info = init_demosaicnet(mosaic_type='xtrans' if raw.is_xtrans else 'bayer',
                                              noiselevel=0.0, tile_size=512, device='mps')

                # オフセットを指定してlibrawのパターンと合わせる
                if raw.is_xtrans:
                    offset_y, offset_x = 2, 2
                    #offset_y, offset_x = find_xtrans_offset(model_info, img_array)
                else:
                    offset_y, offset_x = raw.get_bayer_pattern_offset()

                # NOTE: 暗部が破綻するので一旦増幅して、デモザイク後に復元する
                # k は utils.aiutils.LOG1P_TONEMAP_K_DEFAULT（demosaicnet_helper 内で使用）
                img_array = inference_demosaicnet_with_log_tonemap(
                    model_info,
                    img_array,
                    out_dtype=np.float32,
                    offset_y=offset_y,
                    offset_x=offset_x,
                )

                # ハイライト復元
                thr = raw.get_threshold()          # maximum / data_maximum の値を取得
                img_array = raw.recover_highlights(img_array)           # threshold=-1 で自動
                #img_array = raw.tone_mapping(img_array)                 # after_scale=1.0
                img_array = raw.enhance_micro_contrast(img_array, threshold=thr, strength=8.0, target_contrast=0.06)

                # 1. LibRaw のマトリクス(D65)から D50 へのブラッドフォード逆変換行列                
                # D65の影響を打ち消して元の E光源(1,1,1) 状態に戻す
                cmat_e = np.dot(color.M_D65_to_D50, raw.color_matrix[:3, :3])
                
                # 2. D50のホワイトポイント比率を各行に掛けて、D50基準の順行列にする
                D50_wp = np.array([0.96422, 1.00000, 0.82521])
                forward_matrix = cmat_e * (D50_wp / cmat_e.sum(axis=1))[:, np.newaxis]
                
                # 3. 変換した D50 ForwardMatrix を適用する
                img_array = np.dot(img_array, forward_matrix.T)

                # 色空間変換
                img_array = colour_functions.XYZ_to_RGB(img_array, colourspace='ProPhoto RGB', chromatic_adaptation_transform=config.get_config('cat')).astype(np.float32)

            # クロップとexifデータの回転
            top, left, width, height = self._delete_exif_orientation(exif_data)

            # サイズを整える（フル解像）
            cheight = height
            cwidth = width
            if cwidth > cheight:
                img_array = img_array[:cheight, :cwidth]
            else:
                # 回転させる
                img_array = np.rot90(img_array)
                img_array = img_array[-cheight:, :cwidth]
            
            # ホワイトバランス定義
            img_array = self._apply_whitebalance(img_array, raw, exif_data, param)

            # 明るさ補正
            Ev = 0
            if config.get_config('raw_auto_exposure') == True:
                
                Ev, _ = core.calc_ev_from_image(core.normalize_image(img_array))
                #Ev *= 0.75
                
                # ここで補正
                print(f"img_array range: [{img_array.min():.4f}, {img_array.max():.4f}]")
                img_array = core.adjust_exposure(img_array, Ev)
                #img_array = core.adjust_tone(img_array, white_level=-100)
                #img_array = core.apply_level_adjustment(img_array, 10)
                print(f"img_array range: [{img_array.min():.4f}, {img_array.max():.4f}]")

            param['rgb_or_raw'] = 'raw'
            param['auto_exposure'] = -Ev # 補正は逆方向
            
            # 情報の設定
            params.set_image_param(param, img_array)
            param['lens_modifier'] = True
            param['exif_data'] = exif_data

            # 正方形にする
            #img_array = core.adjust_shape_to_square(img_array)

            # 描画用に設定
            self.img = np.array(img_array)
            self.fidelity = ImageFidelity.FULL

        except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawIOError):
            logging.warning("file is not supported " + file_path)
        
        except Exception as e:
            logging.error(e)

        finally:
            if raw is not None:
                raw.close()

        return (file_path, self, exif_data, param)

    def _load_rgb(self, raw, file_path, exif_data, param):
        # RGB画像で読み込んでみる
        with pyvips.Image.new_from_file(file_path) as vips_image:
            img_array = np.array(vips_image)
            if img_array.ndim == 3 and img_array.shape[2] > 3:
                img_array = img_array[:, :, :3]

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
            # RAW 側 _load_raw_process と同様、以降の on_fcs / レーティング追読用に param に載せる
            param['exif_data'] = exif_data

            # 正方形へ変換
            #img_array = core.adjust_shape_to_square(img_array)
            
        self.img = np.array(img_array)
        self.fidelity = ImageFidelity.FULL
        
        return (file_path, self, exif_data, param, LoadStage.RGB_DONE)

    class Result():
        def __init__(self, worker, source):
            self.worker = worker
            self.source = source

    def preload(self, file_path, exif_data, param):
        self.file_path = file_path

        if file_path.lower().endswith(define.SUPPORTED_FORMATS_RAW):            
            result = []
            result.append(ImageSet.Result(worker="_load_raw_preview", source=None))
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
