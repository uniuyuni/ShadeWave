

import numpy as np
import json
import math
from datetime import datetime as dt

import cores.core as core
import config
import define
import effects
import utils.utils as utils
import macos as device

SPECIAL_PARAM = [
    # for set_image_param
    'original_img_size',
    'img_size',
    #'crop_rect',
    'disp_info',
    # for effects.LensModifierEffect
    'lens_modifier',
    # for imageset._set_temperature
    'color_temperature_reset',
    'color_tint_reset',
    'color_Y',
    # for effects.CropEffect
    'matrix',
    'crop_enable',
    # for effecs.Inpaint
    'inpaint',
    'inpaint_predict',
    # for effects.LUTEffect
    'lut_path',
    # for effects.AutoExposureEffect
    'rgb_or_raw',
    'auto_exposure',
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
        if type(val) == tuple or type(val) == list:
            x = val[0] * param['original_img_size'][0]
            y = val[1] * param['original_img_size'][1]
            return (x, y)
        
        return val * max(param['original_img_size'])

    return None

    if val is not None:
        maxsize = max(param['original_img_size'])
        if type(val) == tuple:
            val = (v * maxsize for v in val)
        else:
            val = val * maxsize
    return val

def norm_param(param, val):
    if val is not None:
        if type(val) == tuple or type(val) == list:
            x = val[0] / param['original_img_size'][0]
            y = val[1] / param['original_img_size'][1]
            return (x, y)
        
        return val / max(param['original_img_size'])
    return None

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
            inpaint_diff_list_dumps.append((inpaint_diff.type, inpaint_diff.disp_info, inpaint_diff.image))
        param['inpaint_diff_list'] = inpaint_diff_list_dumps

def _inpaint_load(param):
    inpaint_diff_list_dumps = param.get('inpaint_diff_list', None)
    if inpaint_diff_list_dumps is not None:
        inpaint_diff_list = []
        for inpaint_diff_dump in inpaint_diff_list_dumps:
            inpaint_diff = effects.InpaintDiff(type=inpaint_diff_dump[0], disp_info=inpaint_diff_dump[1], image=inpaint_diff_dump[2])
            inpaint_diff.list2image()
            inpaint_diff_list.append(inpaint_diff)
        param['inpaint_diff_list'] = inpaint_diff_list

def _ai_noise_reduction_dump(param):
    ai_noise_reduction_result = param.get('ai_noise_reduction_result', None)
    if ai_noise_reduction_result is not None:
        ai_noise_reduction_result = utils.convert_image_to_list(ai_noise_reduction_result)
        param['ai_noise_reduction_result'] = ai_noise_reduction_result

def _ai_noise_reduction_load(param):
    ai_noise_reduction_result = param.get('ai_noise_reduction_result', None)
    if ai_noise_reduction_result is not None:
        ai_noise_reduction_result = utils.convert_image_from_list(ai_noise_reduction_result)
        param['ai_noise_reduction_result'] = ai_noise_reduction_result

def _serialize_param(param):
    _inpaint_dump(param)
    _ai_noise_reduction_dump(param)

def _deserialize_param(param):
    param['disp_info'] = core.convert_rect_to_info(param['crop_rect'], config.get_config('preview_size')/max(param['original_img_size']))
    _inpaint_load(param)
    _ai_noise_reduction_load(param)

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
    mask_list = mask_editor2.get_mask_list()
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


#-------------------------------------------------

def param_to_tcg_info(param):
    """
    パラメータをTCGパラメータに変換

    param: パラメータ辞書(Noneなら全て0に設定)
    戻り値: 情報辞書
    """
    tcg_info = {}
    tcg_info['original_img_size'] = param.get('original_img_size', (config.get_config('preview_size'), config.get_config('preview_size')))
    tcg_info['disp_info'] = param.get('disp_info', (0, 0, 1.0, 1.0, 1.0))
    tcg_info['rotation'] = math.radians(param.get('rotation', 0.0))
    tcg_info['rotation2'] = math.radians(param.get('rotation2', 0.0))
    tcg_info['flip_mode'] = param.get('flip_mode', 0)
    tcg_info['matrix'] = param.get('matrix', np.eye(3))

    return tcg_info

def window_to_tcg(cx, cy, widget, texture_size, tcg_info, normalize=True):
    """
    ウインドウ座標からTCG座標に変換
    cx, cy: TCG座標
    widget: 表示するウィジェット
    texture_size: テクスチャサイズ
    ref_image: 参照イメージ
    tcg_info: 回転情報
    normalize: 正規化するかどうか
    戻り値: TCG座標
    """
    disp_info = get_disp_info(tcg_info)
    imax = max(tcg_info['original_img_size'][0] / 2, tcg_info['original_img_size'][1] / 2)
    wx, wy = widget.to_window(*widget.pos)
    cx, cy = cx - wx, cy - wy
    cx, cy = cx / device.dpi_scale(), cy / device.dpi_scale()
    margin_x, margin_y = (widget.size[0] / device.dpi_scale() - texture_size[0])/2, (widget.size[1] / device.dpi_scale() - texture_size[1])/2
    cx, cy = cx - margin_x, cy - margin_y
    cx, cy = cx, texture_size[1] - cy
    _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*texture_size, disp_info)
    cx, cy = cx - offset_x, cy - offset_y
    cx, cy = cx / disp_info[4], cy / disp_info[4]
    cx, cy = cx + disp_info[0], cy + disp_info[1]
    cx, cy = cx - imax, cy - imax # ここで - (imax - self.current_image.shape[0] / 2)の分の計算もやってる
    cx, cy = center_rotate_invert(cx, cy, tcg_info)
    if normalize:
        cx, cy = norm_param(tcg_info, (cx, cy))
    return (cx, cy)

def tcg_to_window(cx, cy, widget, texture_size, tcg_info, normalize=True):
    """
    TCG座標をウインドウ座標に変換
    cx, cy: TCG座標
    widget: 表示するウィジェット
    texture_size: テクスチャサイズ
    ref_image: 参照イメージ
    tcg_info: 回転情報
    normalize: 正規化するかどうか
    戻り値: ウインドウ座標
    """
    disp_info = get_disp_info(tcg_info)
    imax = max(tcg_info['original_img_size'][0] / 2, tcg_info['original_img_size'][1] / 2)
    if normalize:
        cx, cy = denorm_param(tcg_info, (cx, cy))
    cx, cy = center_rotate(cx, cy, tcg_info)
    cx, cy = cx + imax, cy + imax
    cx, cy = cx - disp_info[0], cy - disp_info[1]
    cx, cy = cx * disp_info[4], cy * disp_info[4]
    _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*texture_size, disp_info)
    cx, cy = cx + offset_x, cy + offset_y
    cx, cy = cx, texture_size[1] - cy
    margin_x, margin_y = (widget.size[0] / device.dpi_scale() - texture_size[0])/2, (widget.size[1] / device.dpi_scale() - texture_size[1])/2
    cx, cy = cx + margin_x, cy + margin_y
    cx, cy = cx * device.dpi_scale(), cy * device.dpi_scale()
    wx, wy = widget.to_window(*widget.pos)
    cx, cy = cx + wx, cy + wy
    return (cx, cy)

def window_to_tcg_scale(x, tcg_info):
    # ワールド座標にスケーリングだけ適用する
    if isinstance(x, (tuple, list)):
        return (x[0] / tcg_info['disp_info'][4] / device.dpi_scale(), x[1] / tcg_info['disp_info'][4] / device.dpi_scale())
    return x / tcg_info['disp_info'][4] / device.dpi_scale()

def tcg_to_window_scale(x, tcg_info):
    # TCG座標にスケーリングだけ適用する
    if isinstance(x, (tuple, list)):
        return (x[0] * tcg_info['disp_info'][4] * device.dpi_scale(), x[1] * tcg_info['disp_info'][4] * device.dpi_scale())
    return x * tcg_info['disp_info'][4] * device.dpi_scale()

def tcg_to_image_scale(x, tcg_info):
    # TCG座標にスケーリングだけ適用する
    if isinstance(x, (tuple, list)):
        return (x[0] * tcg_info['disp_info'][4], x[1] * tcg_info['disp_info'][4])
    return x * tcg_info['disp_info'][4]

def tcg_to_ref_image(cx, cy, ref_img, tcg_info, apply_disp_info=False):
    """
    TCGから参照イメージの座標を得る

    cx, cy: TCG座標
    ref_img: 参照イメージ
    tcg_info: 回転情報
    apply_disp_info: disp_infoを適用するかどうか
    戻り値: 参照イメージ座標
    """
    cx, cy = denorm_param(tcg_info, (cx, cy))
    cx, cy = center_rotate(cx, cy, tcg_info)
    imax = max(tcg_info['original_img_size'][0] / 2, tcg_info['original_img_size'][1] / 2)
    cx, cy = cx + imax, cy + imax
    if apply_disp_info:
        disp_info = get_disp_info(tcg_info)
        if (   np.isclose(disp_info[2], disp_info[3])
            or not (    np.isclose(disp_info[2], tcg_info['original_img_size'][0])
                    and np.isclose(disp_info[3], tcg_info['original_img_size'][1]))
            or np.isclose(disp_info[4], 1.0)
           ):
            # Geometryモード時、クロップ時または拡大表示時
            cx, cy = cx - disp_info[0], cy - disp_info[1]
            # クロップ時の表示空白
            cx = cx + (ref_img.shape[1] / disp_info[4] - disp_info[2]) / 2
            cy = cy + (ref_img.shape[0] / disp_info[4] - disp_info[3]) / 2        
        cx, cy = cx * disp_info[4], cy * disp_info[4]
    return (cx, cy)

def ref_image_to_tcg(cx, cy, ref_img, tcg_info, apply_disp_info=False):
    """
    参照イメージの座標からTCGを得る

    cx, cy: 参照イメージ座標
    ref_img: 参照イメージ
    tcg_info: 回転情報
    apply_disp_info: disp_infoを適用するかどうか
    戻り値: TCG座標
    """
    if apply_disp_info:
        disp_info = get_disp_info(tcg_info)
        cx, cy = cx / disp_info[4], cy / disp_info[4]
        if (   np.isclose(disp_info[2], disp_info[3])
            or not (    np.isclose(disp_info[2], tcg_info['original_img_size'][0])
                    and np.isclose(disp_info[3], tcg_info['original_img_size'][1]))
            or np.isclose(disp_info[4], 1.0)
           ):
            # クロップ時の表示空白
            cx = cx - (ref_img.shape[1] / disp_info[4] - disp_info[2]) / 2
            cy = cy - (ref_img.shape[0] / disp_info[4] - disp_info[3]) / 2        
            # Geometryモード時、クロップ時または拡大表示時
            cx, cy = cx + disp_info[0], cy + disp_info[1]
    imax = max(tcg_info['original_img_size'][0] / 2, tcg_info['original_img_size'][1] / 2)
    cx, cy = cx - imax, cy - imax
    cx, cy = center_rotate_invert(cx, cy, tcg_info)
    cx, cy = norm_param(tcg_info, (cx, cy))
    return (cx, cy)
    

def apply_orientation(cx, cy, tcg_info):
    rad, flip = tcg_info['rotation2'], tcg_info['flip_mode']

    if (flip & 1) == 1:
        cx = -cx
    if (flip & 2) == 2:
        cy = -cy

    return cx, cy, rad

def center_rotate(cx, cy, tcg_info):
    cx, cy, rad = apply_orientation(cx, cy, tcg_info)
    rad = tcg_info['rotation'] + rad
    rad = -rad

    new_cx = cx * math.cos(rad) - cy * math.sin(rad)
    new_cy = cx * math.sin(rad) + cy * math.cos(rad)

    new_cx, new_cy = apply_matrix(new_cx, new_cy, tcg_info['matrix'])

    return (new_cx, new_cy)

def center_rotate_invert(cx, cy, tcg_info):
    rad = tcg_info['rotation'] + tcg_info['rotation2']
    rad = -rad

    cx, cy = apply_matrix_inverse(cx, cy, tcg_info['matrix'])

    new_cx = cx * math.cos(rad) + cy * math.sin(rad)
    new_cy = -cx * math.sin(rad) + cy * math.cos(rad)

    new_cx, new_cy, _ = apply_orientation(new_cx, new_cy, tcg_info)

    return (new_cx, new_cy)

def set_matrix(param, matrix):
    if matrix is None:
        param['matrix'] = np.eye(3)
    else:
        param['matrix'] = matrix.copy()

def add_matrix(param, matrix, offset=(0, 0)):
    if offset != (0, 0):
        # 座標系変換
        T = np.array([
            [1, 0, offset[0]],
            [0, 1, offset[1]],
            [0, 0, 1]
        ])
        T_inv = np.linalg.inv(T)
        matrix = T_inv @ matrix @ T
    
    if 'matrix' not in param:
        param['matrix'] = matrix.copy()
    else:
        param['matrix'] = np.dot(matrix, param['matrix'])

def apply_matrix(px, py, matrix):
    pt = np.array([px, py, 1.0])
    pt_transformed = matrix @ pt
    px_new = pt_transformed[0] / pt_transformed[2]
    py_new = pt_transformed[1] / pt_transformed[2]
    return (float(px_new), float(py_new))

def apply_matrix_inverse(px, py, matrix):
    H_inv = np.linalg.inv(matrix)
    return apply_matrix(px, py, H_inv)
