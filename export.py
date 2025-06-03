
import os
import numpy as np
from wand.image import Image as WandImage
from datetime import datetime as dt
import json
import exiftool
import colour

from imageset import ImageSet
import effects
import pipeline
import mask_editor2
import color
import crop_editor
import config
import effects

def get_version():
    """
    escargot.code-workspaceファイルからバージョン情報を取得します。
    バージョン情報が見つからない場合は「不明」を返します。
    
    Returns:
        str: バージョン文字列
    """
    try:
        # ワークスペースファイルのパスを取得
        workspace_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "escargot.code-workspace")
        
        # ファイルが存在するか確認
        if not os.path.exists(workspace_path):
            return "不明"
            
        # JSONファイルを読み込む
        with open(workspace_path, 'r', encoding='utf-8') as f:
            workspace_data = json.load(f)
            
        # バージョン情報を探す
        # 通常はsettingsやmetadataなどに格納されている可能性がある
        version = "不明"
        
        # 基本的な場所を確認
        if "version" in workspace_data:
            version = workspace_data["version"]
        elif "settings" in workspace_data and "version" in workspace_data["settings"]:
            version = workspace_data["settings"]["version"]
        elif "metadata" in workspace_data and "version" in workspace_data["metadata"]:
            version = workspace_data["metadata"]["version"]
        elif "launch" in workspace_data and "version" in workspace_data["launch"]:
            version = workspace_data["launch"]["version"]
            
        return version
    
    except Exception as e:
        print(f"バージョン情報の取得中にエラーが発生しました: {e}")
        return "不明"

VERSION = get_version()

SPECIAL_PARAM = [
    # for core.set_image_param
    #'original_img_size',
    'img_size',
    #'crop_rect',
    'disp_info',
    # for imageset._set_temperature
    #'color_temperature_switch',
    'color_temperature_reset',
    'color_tint_reset',
    'color_Y',
    # for effects.CropEffect
    'crop_enable',
    # for effecs.Inpaint
    'inpaint',
    'inpaint_predict',
]

def delete_special_param(param):
    p = param.copy()

    for key in SPECIAL_PARAM:
        try:
            val = p[key]
            del p[key]
        except KeyError:
            pass
    
    return p

def copy_special_param(tar, src):
    for key in SPECIAL_PARAM:
        try:
            val = src[key]
            tar[key] = val
        except KeyError:
            pass

def _serialize_param(param):
    effects.InpaintEffect.dump(param)

def _deserialize_param(param):
    param['disp_info'] = crop_editor.CropEditor.convert_rect_to_info(param['crop_rect'], param['original_img_size'], config.get_config('preview_size')/max(param['original_img_size']))
    effects.InpaintEffect.load(param)

def serialize(param, mask_editor2):
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y/%m/%d')
    mask_dict = mask_editor2.serialize()

    # セーブしないパラメータを削除
    param = delete_special_param(param)

    # 色々処理変換
    _serialize_param(param)

    # パラメータがないのでそもそもファイルを作らない
    if len(param) == 0 and (mask_dict is None or len(mask_dict) == 0):
        return None

    dict = {
        'make': "escargo",
        'date': tstr,
        'version': VERSION,
        'primary_param': param,
    }
    if mask_dict is not None:
        dict.update(mask_dict)

    return dict

def deserialize(dict, param, mask_editor2):
    param.update(dict['primary_param'])

    # 色々処理変換
    _deserialize_param(param)

    mask_editor2.clear_mask()
    mask_dict = dict.get('mask2', None)
    if mask_dict is not None:
        mask_editor2.deserialize(dict)

def save_json(file_path, param, mask_editor2):
    if file_path is not None:
        file_path = file_path + '.json'
        dict = serialize(param, mask_editor2)
        if dict is not None:
            with open(file_path, 'w') as f:
                json.dump(dict, f)

def load_json(file_path, param, mask_editor2):
    if file_path is not None:
        file_path = file_path + '.json'
        try:
            with open(file_path, 'r') as f:
                dict = json.load(f)
                deserialize(dict, param, mask_editor2)
                return dict
            
        except FileNotFoundError as e:
            pass
    
    return None

safe_tags = [
    # EXIF（カメラと撮影設定）
    "EXIF:Make",
    "EXIF:Model",
    "EXIF:Software",
    "EXIF:ExposureTime",
    "EXIF:FNumber",
    "EXIF:ApertureValue",
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
    "EXIF:XResolution",
    "EXIF:YResolution",
    "EXIF:ResolutionUnit",
    
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

def make_safe_metadata(exif_data):
    safe_metadata = {}
    for tag in safe_tags:
        group, field = tag.split(':')
        
        # タグが存在する場合のみ追加
        if field in exif_data:
            safe_metadata[field] = exif_data[field]
    return safe_metadata
  
class ExportFile():

    FORMAT = {
        '.JPG': 'JPEG',
        '.TIFF': 'TIFF',
        '.HEIC': 'HEIC',
    }

    def __init__(self, file_path, exif_data):
        self.file_path = str(file_path)
        self.exif_data = exif_data.copy()

        self.ex_path = None
        self.quality = 100
        self.color_space = "sRGB"
        self.imgset = None
        self.effects = effects.create_effects()
        self.param = {}
        self.mask_editor2 = mask_editor2.MaskEditor2()

    def write_to_file(self, ex_path, quality, resize_str, sharpen, color_space, exifsw):
        self.quality = quality
        self.ex_path = ex_path
        self.color_space = color_space
        self.imgset = ImageSet()
        result = self.imgset.load(self.file_path, self.exif_data, self.param)
        if result == False:
            return
        elif result == True:
            pass
        else:
            self.imgset.load(self.file_path, self.exif_data, self.param, result)
        #self.mask_editor2.set_orientation(self.param.get('rotation', 0), self.param.get('rotation2', 0), self.param.get('flip_mode', 0))
        self.imgset.img.shape[1], self.imgset.img.shape[0]
        self.mask_editor2.set_texture_size(self.imgset.img.shape[1], self.imgset.img.shape[0])
        self.mask_editor2.set_image(self.param['original_img_size'], self.param.get('disp_info', None))
        #self.mask_editor2.update()

        load_json(self.file_path, self.param, self.mask_editor2)
        img = pipeline.export_pipeline(self.imgset.img, self.effects, self.param, self.mask_editor2)
        img = colour.RGB_to_RGB(img, 'ProPhoto RGB', self.color_space, 'CAT16',
                                apply_cctf_encoding=True, apply_gamut_mapping=True).astype(np.float32)
        img = np.clip(img, 0, 1)

        ex_ext = os.path.splitext(self.ex_path)[1]
        try:
            format = ExportFile.FORMAT[ex_ext]
        except KeyError:
            return
        
        ex_dir = os.path.dirname(self.ex_path)
        os.makedirs(ex_dir, exist_ok=True)

        # ファイル書き込み
        with WandImage.from_array(img) as wi:
            wi.compression_quality = self.quality 
            wi.format = format
            wi.sharpen(sharpen)
            wi.transform(resize=resize_str)
            wi.save(filename=self.ex_path)

        # Exif書き込み
        if exifsw:
            with exiftool.ExifToolHelper(common_args=['-P', '-overwrite_original']) as et:
                safe_metadata = make_safe_metadata(self.exif_data)
                safe_metadata["Software"] = "escargot " + VERSION
                result = et.set_tags(self.ex_path, tags=safe_metadata)
                print(result)
