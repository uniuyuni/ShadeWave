
import os
import dis
import json
from datetime import datetime as dt

from numpy import diag, disp

import effects
import crop_editor
import config

SPECIAL_PARAM = [
    # for set_image_param
    #'original_img_size',
    'img_size',
    #'crop_rect',
    'disp_info',
    # for imageset._set_temperature
    'color_temperature_reset',
    'color_tint_reset',
    'color_Y',
    # for effects.CropEffect
    'crop_enable',
    # for effecs.Inpaint
    'inpaint',
    'inpaint_predict',
]

# 正規化込みの読み出し、設定
def get_crop_rect(param, none_value=None):
    crop_rect = param.get('crop_rect', none_value)
    if crop_rect is not None:
        if none_value is not None:
            crop_rect = (
                crop_rect[0] / param['original_img_size'][0],
                crop_rect[1] / param['original_img_size'][1],
                crop_rect[2] / param['original_img_size'][0],
                crop_rect[3] / param['original_img_size'][1],
            )
        
        crop_rect2 = (
            int(round(crop_rect[0] * param['original_img_size'][0])),
            int(round(crop_rect[1] * param['original_img_size'][1])),
            int(round(crop_rect[2] * param['original_img_size'][0])),
            int(round(crop_rect[3] * param['original_img_size'][1])),
        )
    else:
        crop_rect2 = None
    
    return crop_rect2

def set_crop_rect(param, crop_rect):
    if crop_rect is not None:
        crop_rect2 = (
            crop_rect[0] / param['original_img_size'][0],
            crop_rect[1] / param['original_img_size'][1],
            crop_rect[2] / param['original_img_size'][0],
            crop_rect[3] / param['original_img_size'][1],
        )
        param['crop_rect'] = crop_rect2

def get_disp_info(param, none_value=None):
    disp_info = param.get('disp_info', none_value)
    if disp_info is not None:
        if none_value is not None:
            disp_info = (
                disp_info[0] / param['original_img_size'][0],
                disp_info[1] / param['original_img_size'][1],
                disp_info[2] / param['original_img_size'][0],
                disp_info[3] / param['original_img_size'][1],
                disp_info[4],
            )

        disp_info2 = (
            int(round(disp_info[0] * param['original_img_size'][0])),
            int(round(disp_info[1] * param['original_img_size'][1])),
            int(round(disp_info[2] * param['original_img_size'][0])),
            int(round(disp_info[3] * param['original_img_size'][1])),
            disp_info[4],
        )
    else:
        disp_info2 = None

    return disp_info2

def set_disp_info(param, disp_info):
    if disp_info is not None:
        disp_info2 = (
            disp_info[0] / param['original_img_size'][0],
            disp_info[1] / param['original_img_size'][1],
            disp_info[2] / param['original_img_size'][0],
            disp_info[3] / param['original_img_size'][1],
            disp_info[4],
        )
        param['disp_info'] = disp_info2

#-------------------------------------------------
# 画像の初期設定を設定する
def set_image_param(param, img):
    height, width = img.shape[:2]

    # イメージサイズをパラメータに入れる
    param['original_img_size'] = (width, height)
    param['img_size'] = (width, height)
    set_crop_rect(param, get_crop_rect(param, crop_editor.CropEditor.get_initial_crop_rect(width, height)))
    set_disp_info(param, crop_editor.CropEditor.convert_rect_to_info(get_crop_rect(param), param['original_img_size'], config.get_config('preview_size')/max(param['original_img_size'])))

    return (width, height)

def set_temperature_to_param(param, temp, tint, Y):
    param['color_temperature_reset'] = temp
    param['color_temperature'] = temp
    param['color_tint_reset'] = tint
    param['color_tint'] = tint
    param['color_Y'] = Y

#-------------------------------------------------

def delete_special_param(param):
    p = param.copy()

    for key in SPECIAL_PARAM:
        try:
            del p[key]
        except KeyError:
            pass
    
    return p

def delete_not_special_param(param):
    p = param.copy()

    for key in not SPECIAL_PARAM:
        try:
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
    param2 = delete_special_param(param)

    # 色々処理変換
    _serialize_param(param2)

    # パラメータがないのでそもそもファイルを作らない
    if len(param2) == 0 and (mask_dict is None or len(mask_dict) == 0):
        return None

    dict = {
        'make': "Platypus",
        'date': tstr,
        'version': VERSION,
        'primary_param': param2,
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
    if file_path is not None and is_empty_param(param) == False:
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

def is_empty_param(param):
    param = delete_special_param(param)
    return len(param) == 0

def delete_empty_param_json(file_path, param):
    if file_path is not None:
        file_path = file_path + '.json'

        if is_empty_param(param) and os.path.exists(file_path):
            os.remove(file_path)
            return True

    return False

#-------------------------------------------------

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
                                     "platypus.code-workspace")
        
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

APPNAME = "Platypus"
VERSION = get_version()
