
import os
import json
from datetime import datetime as dt

import config
import define
import core

SPECIAL_PARAM = [
    # for set_image_param
    'original_img_size',
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
    # for effects.LUTEffect
    'lut_path',
]

REMAIN_PARAM = [
    'crop_rect',
]

# 正規化込みの読み出し、設定
def get_crop_rect(param, none_value=None):
    crop_rect = param.get('crop_rect', none_value)
    if crop_rect is not None:
        maxsize = max(param['original_img_size'])
        if crop_rect is none_value:
            crop_rect = (
                crop_rect[0] / maxsize,
                crop_rect[1] / maxsize,
                crop_rect[2] / maxsize,
                crop_rect[3] / maxsize,
            )
        
        crop_rect2 = (
            int(crop_rect[0] * maxsize),
            int(crop_rect[1] * maxsize),
            int(crop_rect[2] * maxsize),
            int(crop_rect[3] * maxsize),
        )
    else:
        crop_rect2 = None
    
    return crop_rect2

def set_crop_rect(param, crop_rect):
    if crop_rect is not None:
        maxsize = max(param['original_img_size'])
        crop_rect2 = (
            crop_rect[0] / maxsize,
            crop_rect[1] / maxsize,
            crop_rect[2] / maxsize,
            crop_rect[3] / maxsize,
        )
        param['crop_rect'] = crop_rect2

def get_disp_info(param, none_value=None):
    disp_info = param.get('disp_info', none_value)
    if disp_info is not None:
        maxsize = max(param['original_img_size'])
        if disp_info is none_value:
            disp_info = (
                disp_info[0] / maxsize,
                disp_info[1] / maxsize,
                disp_info[2] / maxsize,
                disp_info[3] / maxsize,
                disp_info[4],
            )

        disp_info2 = (
            int(disp_info[0] * maxsize),
            int(disp_info[1] * maxsize),
            int(disp_info[2] * maxsize),
            int(disp_info[3] * maxsize),
            disp_info[4],
        )
    else:
        disp_info2 = None

    return disp_info2

def set_disp_info(param, disp_info):
    if disp_info is not None:
        maxsize = max(param['original_img_size'])
        disp_info2 = (
            disp_info[0] / maxsize,
            disp_info[1] / maxsize,
            disp_info[2] / maxsize,
            disp_info[3] / maxsize,
            disp_info[4],
        )
        param['disp_info'] = disp_info2

def denorm_param(param, val):
    if val is not None:
        maxsize = max(param['original_img_size'])
        if type(val) == tuple:
            val = (v * maxsize for v in val)
        else:
            val = val * maxsize
    return val

def norm_param(param, val):
    if val is not None:
        maxsize = max(param['original_img_size'])
        if type(val) == tuple:
            val = (v / maxsize for v in val)
        else:
            val = val / maxsize
    return val

#-------------------------------------------------
# 画像の初期設定を設定する
def set_image_param(param, img):
    height, width = img.shape[:2]

    # イメージサイズをパラメータに入れる
    param['original_img_size'] = (width, height)
    param['img_size'] = (width, height)
    set_crop_rect(param, get_crop_rect(param, core.get_initial_crop_rect(width, height)))
    set_disp_info(param, core.convert_rect_to_info(get_crop_rect(param), config.get_config('preview_size')/max(param['original_img_size'])))

    return (width, height)

def set_image_param_for_mask2(param, size):
    width, height = size
    param['original_img_size'] = (width, height)

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

    for key in param.keys():
        if key not in SPECIAL_PARAM and key not in REMAIN_PARAM:
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

def _inpaint_dump(param):
    inpaint_diff_list = param.get('inpaint_diff_list', None)
    if inpaint_diff_list is not None:
        inpaint_diff_list_dumps = []
        for inpaint_diff in inpaint_diff_list:
            inpaint_diff.image2list()
            inpaint_diff_list_dumps.append((inpaint_diff.disp_info, inpaint_diff.image))
        param['inpaint_diff_list'] = inpaint_diff_list_dumps

def _inpaint_load(param):
    inpaint_diff_list_dumps = param.get('inpaint_diff_list', None)
    if inpaint_diff_list_dumps is not None:
        inpaint_diff_list = []
        for inpaint_diff_dump in inpaint_diff_list_dumps:
            inpaint_diff = InpaintDiff(disp_info=inpaint_diff_dump[0], image=inpaint_diff_dump[1])
            inpaint_diff.list2image()
            inpaint_diff_list.append(inpaint_diff)
        param['inpaint_diff_list'] = inpaint_diff_list

def _serialize_param(param):
    _inpaint_dump(param)

def _deserialize_param(param):
    param['disp_info'] = core.convert_rect_to_info(param['crop_rect'], config.get_config('preview_size')/max(param['original_img_size']))
    _inpaint_load(param)

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
        'version': define.VERSION,
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
    if file_path is not None and is_empty_param(param, mask_editor2) == False:
        file_path = file_path + '.json'
        dict = serialize(param, mask_editor2)
        if dict is not None:
            with open(file_path, 'w') as f:
                json.dump(dict, f, cls=core.CompactNumpyEncoder)
            return True
    return False

def load_json(file_path, param, mask_editor2):
    if file_path is not None:
        file_path = file_path + '.json'
        try:
            with open(file_path, 'r') as f:
                dict = json.load(f, object_hook=core.compact_numpy_decoder)
                # tupleがlistになってしまうのでtupleに戻す
                try:
                    dict['primary_param']['crop_rect'] = tuple(dict['primary_param']['crop_rect'])
                except:
                    pass

                deserialize(dict, param, mask_editor2)
                return dict
            
        except FileNotFoundError as e:
            pass
    
    return None

def is_empty_param(param, mask_editor2):
    param2 = delete_special_param(param)
    mask_list = mask_editor2.get_layers_list()
    if len(param2) == 0 and (mask_list is None or len(mask_list) == 0):
        return True

    return False
    

def delete_empty_param_json(file_path):
    if file_path is not None:
        file_path = file_path + '.json'

        if os.path.exists(file_path):
            os.remove(file_path)
            return True

    return False

