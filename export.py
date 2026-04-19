
import os
import logging
import numpy as np
import json
import exiftool
import cores.colour_functions as colour_functions
import pyvips
import subprocess

import cores.core as core
import define
from enums import ImageFidelity
from imageset import ImageSet
import effects
import pipeline
import params
import effects
import config
from cores.mask2 import Mask2HeadlessPipeline


def _export_cancel_requested(cancel_event):
    return cancel_event is not None and cancel_event.is_set()


_SAFE_TAGS = [
    # EXIF（カメラと撮影設定）
    "EXIF:Make",
    "EXIF:Model",
    "EXIF:Software",
    "EXIF:ExposureTime",
    "EXIF:FNumber",
    "EXIF:ApertureValue",
    "EXIF:Aperture",
    "EXIF:ISO",
    "EXIF:ISOSpeedRatings",
    "EXIF:ShutterSpeedValue",
    "EXIF:ExposureProgram",
    "EXIF:ExposureCompensation",
    "EXIF:ExposureBiasValue",
    "EXIF:MeteringMode",
    "EXIF:Flash",
    "EXIF:FlashMode",
    "EXIF:WhiteBalance",
    "EXIF:FocalLength",
    "EXIF:FocalLengthIn35mmFormat",
    "EXIF:DigitalZoomRatio",
    "EXIF:LensModel",
    "EXIF:LensInfo",
    "EXIF:LensMake",
    "EXIF:LensSerialNumber",
    "EXIF:SceneCaptureType",
    "EXIF:Contrast",
    "EXIF:Saturation",
    "EXIF:Sharpness",
    "EXIF:SubjectDistance",
    "EXIF:SubjectDistanceRange",
    "EXIF:BrightnessValue",
    "EXIF:WhiteBalance",
    "EXIF:PictureMode",
    "EXIF:SubjectDistanceRange",
    
    # EXIF（基本情報）
    "EXIF:Artist",
    "EXIF:Copyright",
    "EXIF:ImageDescription",
    "EXIF:UserComment",
    "EXIF:XPTitle",
    "EXIF:XPComment",
    "EXIF:XPAuthor",
    "EXIF:XPKeywords",
    "EXIF:XPSubject",
    "EXIF:DocumentName",
    "EXIF:Orientation",
    #"EXIF:ImageWidth",
    #"EXIF:ImageHeight",
    #"EXIF:XResolution",
    #"EXIF:YResolution",
    #"EXIF:ResolutionUnit",
    
    # EXIF（日時情報）
    "EXIF:DateTimeOriginal",
    "EXIF:CreateDate",
    "EXIF:ModifyDate",
    "EXIF:DateTimeDigitized",
    
    # EXIF（GPS情報）
    "EXIF:GPSLatitude",
    "EXIF:GPSLongitude",
    "EXIF:GPSAltitude",
    "EXIF:GPSTimeStamp",
    "EXIF:GPSDateStamp",
    "EXIF:GPSProcessingMethod",
    "EXIF:GPSImgDirection",
    
    # IPTC（基本情報）
    "IPTC:ObjectName",
    "IPTC:Keywords",
    "IPTC:Caption-Abstract",
    "IPTC:Writer-Editor",
    "IPTC:Headline",
    "IPTC:SpecialInstructions",
    "IPTC:Byline",
    "IPTC:BylineTitle",
    "IPTC:Credit",
    "IPTC:Source",
    "IPTC:CopyrightNotice",
    "IPTC:Contact",
    
    # XMP（基本情報）
    "XMP:Title",
    "XMP:Description",
    "XMP:Creator",
    "XMP:Rights",
    "XMP:Subject",
    "XMP:Label",
    "XMP:Rating",
    "XMP:CreateDate",
    "XMP:ModifyDate"
]

def make_safe_metadata(exif_data, gpssw):
    safe_metadata = {}
    for tag in _SAFE_TAGS:
        group, field = tag.split(':')
        
        # タグが存在する場合のみ追加
        if field in exif_data:
            # gpsswがTrue、またはgpsswがFalseかつGPSタグではない場合に情報を保持する
            if gpssw == True or 'GPS' not in field:
                safe_metadata[field] = exif_data[field]
    return safe_metadata
  
class ExportFile():

    FORMAT = {
        '.JPG': 'JPEG',
        '.TIFF': 'TIFF',
        '.JXL': 'JPEG XL',
        '.HEIF': 'HEIF',
        '.PNG': 'PNG',
    }

    def __init__(self, file_path, exif_data):
        self.file_path = str(file_path)
        self.exif_data = exif_data.copy()

        self.ex_path = None
        self.quality = 100
        self.icc_profile = "sRGB"
        self.imgset = None
        self.effects = effects.create_effects()
        self.param = {}
        self.mask_editor2 = None

    def write_to_file(
        self,
        ex_path,
        quality,
        resize_str,
        sharpen,
        icc_profile,
        exifsw,
        gpssw,
        dithering,
        cancel_event=None,
    ):
        if _export_cancel_requested(cancel_event):
            return False

        self.quality = quality
        self.ex_path = ex_path
        self.icc_profile = icc_profile
        self.imgset = ImageSet()
        result = self.imgset.preload(self.file_path, self.exif_data, self.param)
        self.imgset.load(result, self.file_path, self.exif_data, self.param)

        if _export_cancel_requested(cancel_event):
            return False

        params.apply_original_geometry_if_missing(self.param, self.imgset.img)
        if not params.has_original_img_size(self.param):
            logging.error("エクスポート中止: original_img_size を確定できません")
            return False

        self.mask_editor2 = Mask2HeadlessPipeline()

        #self.mask_editor2.set_orientation(self.param.get('rotation', 0), self.param.get('rotation2', 0), self.param.get('flip_mode', 0))
        self.mask_editor2.set_texture_size(self.imgset.img.shape[1], self.imgset.img.shape[0])
        self.mask_editor2.set_primary_param(self.param, params.get_disp_info(self.param))
        self.mask_editor2.set_ref_image(self.imgset.img, self.imgset.img)
        #self.mask_editor2.update()

        params.load_json(self.file_path, self.param, self.mask_editor2, load_heavy=True)
        self.param['image_fidelity'] = getattr(self.imgset, 'fidelity', ImageFidelity.FULL).value

        if _export_cancel_requested(cancel_event):
            return False

        img = pipeline.export_pipeline(self.imgset.img, self.effects, self.param, self.mask_editor2)

        if _export_cancel_requested(cancel_event):
            return False

        img = colour_functions.RGB_to_RGB(img, 'ProPhoto RGB', core.ICC_PROFILE_TO_COLOR_SPACE[self.icc_profile], config.get_config('cat'),
                                apply_cctf_encoding=True, apply_gamut_mapping=True).astype(np.float32)
        img = np.clip(img, 0, 1) # ここじゃないとダメ

        if _export_cancel_requested(cancel_event):
            return False

        format = ex_ext = os.path.splitext(self.ex_path)[1]

        # Ditheringか単なるキャストか
        if dithering:
            match format:
                case '.JPG' | '.JXL' | '.HEIF' | '.PNG':
                    img = core.jjn_dither_uint8(img)
                case '.TIFF':
                    img = core.jjn_dither_uint16(img)
        else:
            match format:
                case '.JPG' | '.JXL' | '.HEIF' | '.PNG':
                    img = (img * 255).astype(np.uint8)
                case '.TIFF':
                    img = (img * 65535).astype(np.uint16)

        # Save options
        save_options = {}
        match format:
            case '.JPG' | '.HEIF' | '.JXL' | '.PNG':
                save_options['Q'] = self.quality
            case '.TIFF':
                save_options['compression'] = 'deflate'
                save_options['bitdepth'] = 16

        # ディレクトリがなかったら作成
        ex_dir = os.path.dirname(self.ex_path)
        os.makedirs(ex_dir, exist_ok=True)

        # VipsImage作成
        vips_image = pyvips.Image.new_from_array(img)

        # Resize
        if resize_str:
            try:
                scale = 1.0
                h, w = vips_image.height, vips_image.width
                
                parts = resize_str.lower().split('x')
                if len(parts) == 2:
                    tx_str, ty_str = parts
                    
                    target_w = int(tx_str) if tx_str else None
                    target_h = int(ty_str) if ty_str else None

                    if target_w and target_h:
                        scale = min(target_w / w, target_h / h)
                    elif target_w:
                        scale = target_w / w
                    elif target_h:
                        scale = target_h / h
                        
                    if scale != 1.0:
                        vips_image = vips_image.resize(scale, kernel=pyvips.enums.Kernel.LANCZOS3)
                        
            except ValueError:
                print(f"Export: Invalid resize string {resize_str}")
        
        # Sharpen
        if sharpen > 0:
            vips_image = vips_image.sharpen(sigma=sharpen)

        if _export_cancel_requested(cancel_event):
            return False

        # ICCプロファイルファイルを読み込む
        try:
            with open('icc/' + self.icc_profile + '.icc', 'rb') as f:
                icc_data = f.read()
            
            # 画像にICCプロファイルを設定
            vips_image = vips_image.copy()
            vips_image.set_type(pyvips.GValue.blob_type, 'icc-profile-data', icc_data)
            vips_image.set_type(pyvips.GValue.gstr_type, 'icc-profile-description', self.icc_profile)
            vips_image.set_type(pyvips.GValue.gstr_type, 'exif-ifd0-InterColorProfile', self.icc_profile)
            vips_image.set_type(pyvips.GValue.gint_type, 'exif-ifd0-ColorSpace', 0xfffe)

        except:
            logging.warning(f"ICC profile {self.icc_profile} not found")

        # ファイル書き込み
        if _export_cancel_requested(cancel_event):
            return False

        vips_image.write_to_file(self.ex_path, **save_options)

        # Exif書き込み
        if exifsw:
            if _export_cancel_requested(cancel_event):
                return False
            with exiftool.ExifToolHelper(common_args=['-P', '-overwrite_original']) as et:
                safe_metadata = make_safe_metadata(self.exif_data, gpssw)
                safe_metadata["Software"] = define.APPNAME + " " + define.VERSION
                #safe_metadata["ColorSpace"] = 0xfffe
                et.set_tags(self.ex_path, tags=safe_metadata)

        return True
