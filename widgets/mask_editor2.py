
import os
import sys
if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import cv2
import time
import uuid
from enum import Enum
import copy
import logging
import importlib
from functools import partial

from kivy.app import App as KVApp
from kivy.core.window import Window as KVWindow
from kivy.uix.widget import Widget as KVWidget
from kivy.uix.image import Image as KVImage
from kivy.uix.button import Button as KVButton
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.properties import (
    NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty, ListProperty as KVListProperty,
    StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty, Property as KVProperty
)
from kivy.graphics import (
    Color as KVColor, Ellipse as KVEllipse, Line as KVLine, PushMatrix as KVPushMatrix, PopMatrix as KVPopMatrix, Rotate as KVRotate, Translate as KVTranslate,
    Rectangle as KVRectangle, ScissorPush as KVScissorPush, ScissorPop as KVScissorPop,
)
from kivy.graphics.texture import Texture as KVTexture
from kivy.clock import Clock as KVClock
from kivy.uix.label import Label as KVLabel
from kivy.uix.popup import Popup as KVPopup
from kivy.uix.textinput import TextInput as KVTextInput

import cores.core as core
import cores.expand_mask as expand_mask
import cores.hlsrgb as hlsrgb
import params
import effects
import config
import utils.dialogutils as dialogutils
import utils.utils as utils
from processing_dialog import wait_prosessing
from history import LayerCtrl, get_history_ctrl
import macos as device

from cores.mask2 import elliptical_raster, freedraw_raster, gradient_raster, polyline_raster
from cores.mask2.cutout_guided import create_cutout_mask_guided


def _clip_mask_range(image, allow_over_one=False, allow_under_zero=False):
    min_value = None if allow_under_zero else 0
    max_value = None if allow_over_one else 1
    if min_value is None and max_value is None:
        return image
    return np.clip(image, min_value, max_value)


class TextInputDialog(KVPopup):
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.title = "Input Target Text in English"
        self.size_hint = (None, None)
        self.ref_width = 420
        self.ref_height = 240
        
        layout = KVBoxLayout(orientation='vertical')
        layout.ref_padding = 10
        layout.ref_spacing = 10
        self.text_input = KVTextInput(multiline=False, size_hint_y=None)
        self.text_input.ref_height = 50
        self.text_input.bind(on_text_validate=lambda x: self.save(callback))
        
        btn_layout = KVBoxLayout(orientation='horizontal', size_hint_y=None)
        btn_layout.ref_height = 40
        btn_layout.ref_spacing = 10
        save_button = KVButton(text='OK')
        save_button.bind(on_press=lambda x: self.save(callback))
        btn_layout.add_widget(save_button)
        
        layout.add_widget(self.text_input)
        layout.add_widget(btn_layout)
        self.content = layout
        dialogutils.install_ref_scaling(self)

    def save(self, callback):
        text = self.text_input.text
        if not text or text.isspace():
            text = "All"
        self.dismiss()
        KVClock.schedule_once(lambda dt: callback(text), 0.5)

class MaskType(str, Enum):
    COMPOSIT = 'composit'
    CIRCULAR = 'circular'
    GRADIENT = 'gradient'
    FULL = 'full'
    FREEDRAW = 'free_draw'
    POLYLINE = 'polyline'
    SEGMENT = 'segment'
    DEPTHMAP = 'depth_map'
    FACE = 'face'
    TARGET_TEXT = 'target_text'

# コントロールポイントのクラス
class ControlPoint(KVWidget):
    touching = KVBooleanProperty(False)
    is_center = KVBooleanProperty(False)  # 中心のコントロールポイントかどうか
    color = KVListProperty([0, 0, 0])  # デフォルトの色
    ctrl_center = KVListProperty([0, 0])
    type = KVListProperty(['c', 0])

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate()
            #self.rotate = KVRotate(angle=0, origin=(0, 0))            
            self.color_instruction = KVColor(*self.color)
            self.circle = KVEllipse(pos=(-10, -10), size=(20, 20))
            self.editor.pop_scissor()
            KVPopMatrix()
        self.center = (0, 0)
        #self.update_graphics()
        self.bind(center=self.update_graphics, color=self.update_color)

    def update_graphics(self, *args):
        cx, cy = self.editor.tcg_to_window(self.center_x, self.center_y)
        self.translate.x = cx
        self.translate.y = cy
        self.editor.set_scissor(self.scissor)
        #self.size = self.editor.window_to_tcg_scale(20, 20) # sizeをセットすると何故かcenterの値がおかしくなるのでコメントアウト

    def update_color(self, *args):
        self.color_instruction.rgb = self.color

    def on_touch_down(self, touch):
        self.touching = True
        return True

    def on_touch_move(self, touch):
        if self.touching:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.ctrl_center = [cx, cy]
            #self.cnter = (cx, cy)
            return True
        return False

    def on_touch_up(self, touch):
        if self.touching:
            self.touching = False
            return True
        return False

        return False

# マスクのベースクラス
class BaseMask(KVWidget):
    color = KVListProperty([1, 0, 0, 0.5])  # デフォルトの半透明赤色
    selected = KVBooleanProperty(False)
    active = KVBooleanProperty(False)
    name = KVStringProperty("Mask")
    mask_id = KVStringProperty(str(uuid.uuid4()))

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor  # MaskEditorのインスタンスへの参照
        self.control_points = []  # 標準のPythonリストで管理
        self.bind(active=self.on_active_changed)

        # エフェクトパラメータ保持
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, self.editor.get_image_size())
        params.set_temperature_to_param(self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0)))

        self.is_draw_mask = True
        self.image_mask_cache = None
        self.image_mask_cache_hash = None
        self.do_draw_composit_mask = True

    def clear(self):
        for cp in self.control_points:
            self.remove_widget(cp)
        self.control_points = []
        self.effects_param = params.delete_not_special_param(self.effects_param)
        effects.reeffect_all(self.effects)

    def start(self):
        pass

    def end(self):
        pass

    def is_composit(self):
        return isinstance(self, CompositMask)

    def on_active_changed(self, instance, value):
        if value:
            self.show_all_control_points()
        else:
            self.show_center_control_point_only()

    def show_all_control_points(self):
        self.opacity = 1
        for cp in self.control_points:
            cp.opacity = 1
            if cp.is_center:
                cp.color = [0, 0, 1]  # アクティブなマスクの中心点
            else:
                if cp.type[0] == 'r' or cp.type[0] == 's':
                    cp.color = [1, 1, 0]
                else:
                    cp.color = [1, 1, 1]  # 他のコントロールポイントは白色
        self.is_draw_mask = True
        self.update_mask()

    def show_center_control_point_only(self):
        self.opacity = 0.2
        for cp in self.control_points:
            if cp.is_center:
                cp.opacity = 2
                cp.color = [1, 0, 0]  # 非アクティブなマスクの中心点は赤色
            else:
                cp.opacity = 0  # 非表示
        self.is_draw_mask = False
        self.update_mask()

    def is_center_click(self, touch):
        for cp in self.control_points:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            if cp.collide_point(cx, cy):
                return cp.is_center
        return False

    def on_touch_down(self, touch):
        for cp in self.control_points:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            if cp.collide_point(cx, cy): #or (self.editor.collide_point(*touch.pos) and isinstance(self, FreeDrawMask)): # フリーだけコントロールポイント関係ない
                if cp.is_center:
                    self.editor.set_active_mask(self)
                    cp.on_touch_down(touch)
                    get_history_ctrl().begin_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self), None)
                    self.is_draw_mask = True
                    return True

                elif self.active:
                    cp.on_touch_down(touch)
                    get_history_ctrl().begin_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self), None)
                    self.is_draw_mask = True
                    return True

        return False

    def on_touch_move(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_move(touch)
                #self.is_draw_mask = True
                return True
        return False

    def on_touch_up(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_up(touch)
                get_history_ctrl().end_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self))
                return True
        return False

    def get_name(self):
        return self.name

    def update(self):
        if len(self.control_points) > 0:
            cp_center = self.control_points[0]
            cp_center.property('ctrl_center').dispatch(cp_center)
            self.is_draw_mask = True
            self.update_mask()

    def update_control_points(self):
        pass

    def on_center_control_point_move(self, instance, value):
        dx = instance.ctrl_center[0] - self.center_x
        dy = instance.ctrl_center[1] - self.center_y
        self.center = (self.center_x + dx, self.center_y + dy)
        for cp in self.control_points:
            #if cp != instance:
            center = (cp.center_x + dx, cp.center_y + dy)
            if cp.center[0] == center[0] and cp.center[1] == center[1]:
                cp.property('center').dispatch(cp) # 値が同じだとディスパッチされないから
            else:
                cp.center = center
        self.update_control_points()
        self.update_mask()
        self.editor.start_draw_image()
    
    def draw_mask_to_fbo(self, absolute=False):
        if self.active == True or absolute == True:
            mask_image = self.get_mask_image()
            # イメージを描画してもらう
            self.editor.draw_mask_image(mask_image)

    def _apply_extened_params(self, image):
        simg = self._apply_mask_space(image)
        dimg = self._apply_depth_mask(simg)
        himg = self._draw_hue_mask(dimg)
        limg = self._draw_lum_mask(himg)
        simg = self._draw_sat_mask(limg)
        bimg = self._apply_mask_blur(simg)

        return bimg

    def get_hash_items(self):
        return (effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_allow_over_one'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_allow_under_zero'),
                effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_depth'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_min'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_max'),
                effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_hue'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_hue_distance'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_hue_min'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_hue_max'),
                effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_lum'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_lum_distance'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_lum_min'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_lum_max'),
                effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_sat'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_sat_distance',),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_sat_min'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_sat_max'),
                effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_options'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_blur'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_open_space'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_close_space'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_freedraw_brush_hardness'))

    def _apply_mask_space(self, image):
        switch_mask2_options = effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_options')
        if switch_mask2_options == True:
            open_space = effects.Mask2Effect.get_param(self.effects_param, 'mask2_open_space')
            image = expand_mask.adjust_foreground_only(image, open_space * params.get_disp_info(self.editor.tcg_info)[4], False)

            close_space = effects.Mask2Effect.get_param(self.effects_param, 'mask2_close_space')
            image = expand_mask.adjust_holes_only(image, close_space * params.get_disp_info(self.editor.tcg_info)[4], False)
        
        return image

    def _apply_depth_mask(self, image):
        switch_mask2_depth = effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_depth')
        if switch_mask2_depth == True:
            dmin = effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_min') / 255
            dmax = effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_max') / 255
            if (dmin != 0) or (1 != dmax):
                image = np.where((image < dmin) | (dmax < image), 0, image)

        return image
    
    def _apply_mask_blur(self, image):
        switch_mask2_options = effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_options')
        blur = effects.Mask2Effect.get_param(self.effects_param, 'mask2_blur')
        if switch_mask2_options == True and blur != 0:
            ksize = int(max(0, blur*2-1))
            image = core.gaussian_blur_cv(image, (ksize, ksize))

        return image

    def _draw_hls_mask(self, mask, hls_str):
        HLS_NUM = {
            'hue': 0,
            'lum': 1,
            'sat': 2,
        }
        HLS_DIS_MAX = {
            'hue': 179,
            'lum': 127,
            'sat': 127,
        }
        HLS_MAX = {
            'hue': 359,
            'lum': 255,
            'sat': 255,
        }

        #original_image_hls = self.editor.get_original_image_hls()
        crop_image_hls = self.editor.get_crop_image_hls()
        #if original_image_hls is not None:            
        if crop_image_hls is not None:            
            #oimg = original_image_hls[..., HLS_NUM[hls_str]]
            cimg = crop_image_hls[..., HLS_NUM[hls_str]]
            dmax = HLS_DIS_MAX[hls_str]
            mmax = HLS_MAX[hls_str]
            
            ndis = effects.Mask2Effect.get_param(self.effects_param, f'mask2_{hls_str}_distance', dmax)
            if ndis != dmax:
                #cx, cy = self.editor.tcg_to_original_image(*self.center)
                #print(f"point: {cx}, {cy}, {oimg[int(cy), int(cx)]}")
                #center_n = oimg[int(cy), int(cx)]
                cx, cy = self.editor.tcg_to_crop_image(*self.center)
                print(f"point: {cx}, {cy}, {cimg[int(cy), int(cx)]}")
                center_n = cimg[int(cy), int(cx)] 
                
                if hls_str == 'hue':
                    # 色相の範囲チェック（0-360の円状ループを考慮）
                    _min = (center_n - ndis) % 360
                    _max = (center_n + ndis) % 360
                else:
                    ndis = ndis / 255
                    _min = (((center_n - ndis) * 65535) % 65536) / 65535
                    _max = (((center_n + ndis) * 65535) % 65536) / 65535
                
                if _min <= _max:
                    # 通常の範囲チェック
                    nimg = np.where((cimg < _min) | (_max < cimg), 0, mask)
                else:
                    # 0をまたぐ場合の範囲チェック
                    nimg = np.where(((cimg < _min) & (_max < cimg)), 0, mask)
            else:
                nimg = mask
            
            _min = effects.Mask2Effect.get_param(self.effects_param, f'mask2_{hls_str}_min')
            _max = effects.Mask2Effect.get_param(self.effects_param, f'mask2_{hls_str}_max', mmax)
            if _min != 0 or _max != mmax:
                if hls_str != 'hue':
                    _min = _min / mmax
                    _max = _max / mmax

                if _min <= _max:
                    # 通常の範囲チェック
                    nimg = np.where((cimg < _min) | (_max < cimg), 0, nimg)
                else:
                    # 0をまたぐ場合の範囲チェック
                    nimg = np.where(((cimg < _min) & (_max < cimg)), 0, nimg)

            return nimg
        
        return mask

    def _draw_hue_mask(self, mask):
        switch_mask2_hue = effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_hue')
        if switch_mask2_hue == True:
            return self._draw_hls_mask(mask, 'hue')
        
        return mask

    def _draw_lum_mask(self, mask):
        switch_mask2_lum = effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_lum')
        if switch_mask2_lum == True:
            return self._draw_hls_mask(mask, 'lum')
        
        return mask

    def _draw_sat_mask(self, mask):
        switch_mask2_sat = effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_sat')
        if switch_mask2_sat == True:
            return self._draw_hls_mask(mask, 'sat')
        
        return mask

# マスクの合成マスク
class CompositMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Composit"
        self.mask_list = list()
        self.initializing = False

    def add_mask(self, mask, maskop='Add', index=0):
        # 子マスクの追加
        self.mask_list.insert(index, (mask, maskop))
        #self.editor.dispatch('on_structure_change')

    def remove_mask(self, mask):
        # 子マスクの削除
        for item in self.mask_list:
            if item[0] is mask:
                mask.clear()
                self.mask_list.remove(item)
                break
        #self.editor.dispatch('on_structure_change')

    def get_mask_list(self):
        # 子マスクのリスト
        return self.mask_list

    def get_mask(self, index):
        # 子マスクの取得
        return self.mask_list[index]

    def find_mask_op(self, mask):
        # 登録されている子マスクのタイプを取得
        for cmask, maskop in self.mask_list:
            if cmask is mask:
                return maskop
        return None

    def clear(self):
        # 子マスクのクリア
        for mask, _ in self.mask_list:
            mask.clear()
        self.mask_list.clear()

    def on_touch_down(self, touch):
        """
        # 先にアクティブなマスクのイベントを処理する
        active_mask = self.editor.get_active_mask()
        if active_mask is not None and active_mask is not self:
            if active_mask.on_touch_down(touch):
                return True
        # 子マスクのイベント処理（逆順で、上に描画されたものを先に）
        for mask, _ in reversed(self.mask_list):
            if mask.active or mask.initializing:
                if mask.on_touch_down(touch):
                    return True
        """
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        """
        active_mask = self.editor.get_active_mask()
        if active_mask is not None and active_mask is not self:
            if active_mask.on_touch_move(touch):
                return True
        for mask, _ in reversed(self.mask_list):
            if mask.active or mask.initializing:
                if mask.on_touch_move(touch):
                    return True
        """
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """
        active_mask = self.editor.get_active_mask()
        if active_mask is not None and active_mask is not self:
            if active_mask.on_touch_up(touch):
                return True
        for mask, _ in reversed(self.mask_list):
            if mask.active or mask.initializing:
                if mask.on_touch_up(touch):
                    return True
        """
        return super().on_touch_up(touch)

    def serialize(self):
        # パラメータの余計なものを削除
        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        mdict = {
            'type': MaskType.COMPOSIT,
            'name': self.name,
            'effects_param': param,
            'mask_list': list(),
        }
        # 子マスクのシリアライズ
        for mask, maskop in self.mask_list:
            mdict['mask_list'].append((mask.serialize(), maskop))

        return mdict

    def deserialize(self, dict):
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])
        # 子マスクのデシリアライズ
        index = self.editor.mask_list.index(self)
        for i, mask_info in enumerate(dict['mask_list']):
            index += 1
            new_mask = self.editor._create_mask(mask_info[0]['type'], index)
            new_mask.deserialize(mask_info[0])
            self.add_mask(new_mask, mask_info[1], i)

    def update_mask(self):
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):
        # 合成マスクの画像作成
        composit = np.zeros((int(self.editor.texture_size[1]), int(self.editor.texture_size[0])), dtype=np.float32)
        allow_over_one = False
        allow_under_zero = False

        for mask, maskop in reversed(self.mask_list):
            mimage = mask.get_mask_image()
            mask_allow_over_one = False
            mask_allow_under_zero = False
            match(maskop):
                case 'Add':
                    composit = _clip_mask_range(composit + mimage, mask_allow_over_one, mask_allow_under_zero)
                case 'Subtract':
                    composit = _clip_mask_range(composit - mimage, mask_allow_over_one, mask_allow_under_zero)
                case _:
                    logger.error(f"Unknown mask operation: {maskop}")
                    assert False

        return composit


# 円形グラデーションマスクのクラス
class CircularGradientMask(BaseMask):
    inner_radius_x = KVNumericProperty(0)
    inner_radius_y = KVNumericProperty(0)
    outer_radius_x = KVNumericProperty(0)
    outer_radius_y = KVNumericProperty(0)
    rotate_rad = KVNumericProperty(0)

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Circle"
        self.initializing = True  # 初期配置中かどうか

        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate(*self.center)
            self.rotate = KVRotate(angle=0, origin=(0, 0))
            KVColor(*self.color)
            self.outer_line = KVLine(ellipse=(0, 0, 0, 0), width=2) # 外側の円
            self.inner_line = KVLine(ellipse=(0, 0, 0, 0), width=2) # 内側の円
            self.editor.pop_scissor()
            KVPopMatrix()

        #self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            self.inner_radius_x = 0
            self.inner_radius_y = 0
            self.outer_radius_x = 0
            self.outer_radius_y = 0
            return True
        else:            
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            dx = cx - self.center_x
            dy = cy - self.center_y
            self.outer_radius_x = ((dx**2 + dy**2) ** 0.5)
            self.outer_radius_y = ((dx**2 + dy**2) ** 0.5)
            self.inner_radius_x = self.outer_radius_x * 0.5  # 内側の半径を仮設定
            self.inner_radius_y = self.outer_radius_y * 0.5  # 内側の半径を仮設定
            self.update_mask()
            return True
        else:
            return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        # 8つのコントロールポイントを作成
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        types  = ['x', 'r', 'y', 'r', 'x', 'r', 'y', 'r']
        self.control_points = []
        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        for i, angle in enumerate(angles):
            # 内側のコントロールポイント
            cp_inner = ControlPoint(self.editor)
            cp_inner.type = [types[i], angle]
            cp_inner.center = self.calculate_point(self.inner_radius_x, self.inner_radius_y, angle)
            cp_inner.ctrl_center = cp_inner.center
            cp_inner.bind(ctrl_center=self.on_inner_control_point_move)
            self.control_points.append(cp_inner)
            self.add_widget(cp_inner)

            # 外側のコントロールポイント
            cp_outer = ControlPoint(self.editor)
            cp_outer.type = [types[i], angle]
            cp_outer.center = self.calculate_point(self.outer_radius_x, self.outer_radius_y, angle)
            cp_outer.ctrl_center = cp_outer.center
            cp_outer.bind(ctrl_center=self.on_outer_control_point_move)
            self.control_points.append(cp_outer)
            self.add_widget(cp_outer)

        if not self.active:
            self.show_center_control_point_only()
        else:
            self.show_all_control_points()  # アクティブなら全ポイントの色・表示を更新

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        ix, iy = params.norm_param(self.effects_param, (self.inner_radius_x, self.inner_radius_y))
        ox, oy = params.norm_param(self.effects_param, (self.outer_radius_x, self.outer_radius_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.CIRCULAR,
            'name': self.name,
            'center': [cx, cy],
            'inner_radius': [ix, iy],
            'outer_radius': [ox, oy],
            'rotate_rad': self.rotate_rad,
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']
        ix, iy = dict['inner_radius']
        ox, oy = dict['outer_radius']
        self.rotate_rad = dict['rotate_rad']
        self.effects_param.update(dict['effects_param'])

        self.center = params.denorm_param(self.effects_param, (cx, cy))
        self.inner_radius_x, self.inner_radius_y = params.denorm_param(self.effects_param, (ix, iy))
        self.outer_radius_x, self.outer_radius_y = params.denorm_param(self.effects_param, (ox, oy))

        self.create_control_points()
        #self.update_mask()
 
    def calculate_point(self, radius_x, radius_y, angle_deg):
        angle_rad = math.radians(angle_deg)
        radius_x = radius_x
        radius_y = radius_y
        dx = radius_x * math.cos(angle_rad)
        dy = radius_y * math.sin(angle_rad)
        new_r_x = dx * math.cos(-self.rotate_rad) - dy * math.sin(-self.rotate_rad)
        new_r_y = dx * math.sin(-self.rotate_rad) + dy * math.cos(-self.rotate_rad)
        return (self.center_x + new_r_x, self.center_y + new_r_y)

    def calculate_rotate(self, radius_x, radius_y, angle_deg, dx, dy):
        angle_rad = math.radians(angle_deg)
        px = radius_x * math.cos(angle_rad)
        py = radius_y * math.sin(angle_rad)
        rotate_rad = -math.atan2(dy, dx)
        new_rad = rotate_rad+math.atan2(py, px)
        return new_rad

    def update_ellipse(self, dx, dy):
        # 回転角の変化に応じて、半径を更新
        new_r_x = dx * math.cos(self.rotate_rad) - dy * math.sin(self.rotate_rad)
        new_r_y = dx * math.sin(self.rotate_rad) + dy * math.cos(self.rotate_rad)
        
        return (abs(new_r_x), abs(new_r_y))


    def on_outer_control_point_move(self, instance, value):
        if self.active:
            dx = instance.ctrl_center[0] - self.center_x
            dy = instance.ctrl_center[1] - self.center_y
            sx = self.inner_radius_x / self.outer_radius_x
            sy = self.inner_radius_y / self.outer_radius_y
            if instance.type[0] == 'x':
                self.outer_radius_x, _ = self.update_ellipse(dx, dy)
                self.inner_radius_x = self.outer_radius_x * sx
                self.outer_radius_x = max(10, max(self.outer_radius_x, self.inner_radius_x))
            elif instance.type[0] == 'y':
                _, self.outer_radius_y = self.update_ellipse(dx, dy)
                self.inner_radius_y = self.outer_radius_y * sy
                self.outer_radius_y = max(10, max(self.outer_radius_y, self.inner_radius_y))
            elif instance.type[0] == 'r':
                self.rotate_rad = self.calculate_rotate(self.outer_radius_x, self.outer_radius_y, instance.type[1], dx, dy)
            self.update_control_points()
            self.update_mask()
            self.editor.start_draw_image()

    def on_inner_control_point_move(self, instance, value):
        if self.active:
            dx = instance.ctrl_center[0] - self.center_x
            dy = instance.ctrl_center[1] - self.center_y
            sx = self.inner_radius_x / self.outer_radius_x
            sy = self.inner_radius_y / self.outer_radius_y
            if instance.type[0] == 'x':
                self.inner_radius_x, _ = self.update_ellipse(dx, dy)
                self.inner_radius_x = max(5, min(self.inner_radius_x, self.outer_radius_x-10))
                #self.inner_radius_y = self.outer_radius_y * sx
            elif instance.type[0] == 'y':
                _, self.inner_radius_y = self.update_ellipse(dx, dy)
                self.inner_radius_y = max(5, min(self.inner_radius_y, self.outer_radius_y-10))
                #self.inner_radius_x = self.outer_radius_x * sy
            elif instance.type[0] == 'r':
                self.rotate_rad = self.calculate_rotate(self.inner_radius_x, self.inner_radius_y, instance.type[1], dx, dy)
            self.update_control_points()
            self.update_mask()
            self.editor.start_draw_image()

    def update_control_points(self):
        # コントロールポイントの位置を更新
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        cp_center = self.control_points[0]
        cp_center.center = self.center
        index = 1  # 0は中心点
        for angle in angles:
            cp_inner = self.control_points[index]
            cp_inner.center_x, cp_inner.center_y = self.calculate_point(self.inner_radius_x, self.inner_radius_y, angle)
            index += 1
            cp_outer = self.control_points[index]
            cp_outer.center_x, cp_outer.center_y = self.calculate_point(self.outer_radius_x, self.outer_radius_y, angle)
            index += 1

    def _matrix_transformed_ellipse(self, rx_tcg, ry_tcg):
        """Jacobian-at-ellipse-center: 楕円中心位置での center_rotate Jacobian を
        軸ベクトルに線形適用し、SVD で (rx, ry, rotate_rad) を再パラメータ化。

        center_rotate = apply_orientation + R(-(rotation+rotation2)) + apply_matrix。
        apply_matrix の Jacobian を中心 (cx_pre, cy_pre) で評価し、軸方向には同じ線形
        変形を掛ける (位置依存性は中心のみに反映、軸方向では一定)。
        これにより、強 projective (Four Points 等) でも ellipse 形状が弧上で不均一に
        歪まず、中心位置で評価された perspective が ellipse 全体に均一に効く。

        Returns: (new_rx_tcg, new_ry_tcg, new_rotate_rad_for_rasterizer)
        matrix = identity 時、affine 時は既存 / sample-and-fit と同一の結果。
        """
        tcg_info = self.editor.tcg_info
        theta = self.rotate_rad
        cx, cy = self.center
        c, s = math.cos(theta), math.sin(theta)

        try:
            # apply_orientation + R(-(rotation+rotation2)) で中心を pre-matrix coord に運ぶ
            cx_o, cy_o, rot2 = params.apply_orientation(cx, cy, tcg_info)
            rad = -(tcg_info['rotation'] + rot2)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            cx_pre = cx_o * cos_r - cy_o * sin_r
            cy_pre = cx_o * sin_r + cy_o * cos_r

            # apply_matrix の解析的 Jacobian at (cx_pre, cy_pre)
            # apply_matrix(x, y) = ((ax+by+e)/w, (cx+dy+f)/w), w = gx+hy+i
            M = np.asarray(tcg_info['matrix'], dtype=np.float64)
            a, b, e = M[0]
            c_m, d, f = M[1]
            g, h, i = M[2]
            denom = g * cx_pre + h * cy_pre + i
            if abs(denom) < 1e-12:
                return rx_tcg, ry_tcg, self.editor.get_rotate_rad(self.rotate_rad)
            num_x = a * cx_pre + b * cy_pre + e
            num_y = c_m * cx_pre + d * cy_pre + f
            d2 = denom * denom
            J_mat = np.array([
                [(a * denom - num_x * g) / d2, (b * denom - num_x * h) / d2],
                [(c_m * denom - num_y * g) / d2, (d * denom - num_y * h) / d2],
            ])

            # 全 Jacobian = J_mat @ R(rad) @ F (= linearized center_rotate at TCG center)
            R_rad = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            flip = tcg_info['flip_mode']
            F = np.array([
                [-1.0 if (flip & 1) else 1.0, 0.0],
                [0.0, -1.0 if (flip & 2) else 1.0],
            ])
            full_jacobian = J_mat @ R_rad @ F
        except Exception:
            return rx_tcg, ry_tcg, self.editor.get_rotate_rad(self.rotate_rad)

        # 楕円軸ベクトル (TCG image-coord, Y-down)
        # x 軸方向 (rx 倍): (cos θ, -sin θ) · rx
        # y 軸方向 (ry 倍): (sin θ, cos θ) · ry
        ax_image = np.array([rx_tcg * c, -rx_tcg * s])
        ay_image = np.array([ry_tcg * s, ry_tcg * c])

        ax_post = full_jacobian @ ax_image
        ay_post = full_jacobian @ ay_image

        Mat = np.column_stack([ax_post, ay_post])
        try:
            U, S, _ = np.linalg.svd(Mat)
        except np.linalg.LinAlgError:
            return rx_tcg, ry_tcg, self.editor.get_rotate_rad(self.rotate_rad)

        new_rx = float(S[0]) if len(S) > 0 else rx_tcg
        new_ry = float(S[1]) if len(S) > 1 else ry_tcg
        # U は image-coord (Y-down) standard rotation。Kivy/画面 CCW positive 規約に negate
        new_angle_image = math.atan2(float(U[1, 0]), float(U[0, 0]))
        new_rot = -new_angle_image
        return new_rx, new_ry, new_rot

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        # matrix 追従の SVD 再パラメータ化 (inner と outer は同じ rotation を共有させる)
        new_inner_rx, new_inner_ry, new_rotate_rad = self._matrix_transformed_ellipse(
            self.inner_radius_x, self.inner_radius_y)
        new_outer_rx, new_outer_ry, _ = self._matrix_transformed_ellipse(
            self.outer_radius_x, self.outer_radius_y)

        with self.canvas:
            self.editor.set_scissor(self.scissor)
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
            self.rotate.angle = math.degrees(new_rotate_rad)
            ix, iy = self.editor.tcg_to_window_scale(new_inner_rx, new_inner_ry)
            self.inner_line.ellipse = (-ix, -iy, ix*2, iy*2)
            ox, oy = self.editor.tcg_to_window_scale(new_outer_rx, new_outer_ry)
            self.outer_line.ellipse = (-ox, -oy, ox*2, oy*2)

        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)
        # matrix 追従の SVD 再パラメータ化 (inner と outer の rotation を共有)
        new_inner_rx, new_inner_ry, new_rotate_rad = self._matrix_transformed_ellipse(
            self.inner_radius_x, self.inner_radius_y)
        new_outer_rx, new_outer_ry, _ = self._matrix_transformed_ellipse(
            self.outer_radius_x, self.outer_radius_y)
        inner_axes = self.editor.tcg_to_image_scale(new_inner_rx, new_inner_ry)
        outer_axes = self.editor.tcg_to_image_scale(new_outer_rx, new_outer_ry)
        rotate_rad = new_rotate_rad
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
            invert = not effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')
        else:
            invert = False

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center, inner_axes, outer_axes, rotate_rad, invert))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:

            # グラデーションを描画
            gradient_image = self.draw_elliptical_gradient(image_size, center, inner_axes, outer_axes, rotate_rad, invert, 1.5)

            # ルミノシティマスクを作成
            gradient_image = self._apply_extened_params(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_elliptical_gradient(self, image_size, center, inner_axes, outer_axes, angle_rad, invert=False, smoothness=1):
        return elliptical_raster.draw_elliptical_gradient(
            image_size, center, inner_axes, outer_axes, angle_rad, invert, smoothness
        )

# GradientMask クラス
class GradientMask(BaseMask):
    start_point = KVListProperty([0, 0])    # グラデーションの開始点
    end_point = KVListProperty([0, 0])      # グラデーションの終点
    
    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Line"
        self.initializing = True  # 初期配置中かどうか

        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate(*self.center)
            self.rotate = KVRotate(angle=0, origin=(0, 0))
            KVColor(*self.color)
            self.start_line = KVLine(points=(0, 0, 0, 0), width=2)
            self.center_line = KVLine(points=(0, 0, 0, 0), width=2)
            self.end_line = KVLine(points=(0, 0, 0, 0), width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

        self.rotate_rad = 0
        #self.update_mask()
    
    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center = (cx, cy)
            self.start_point = [cx, cy]
            return True
        else:
            return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.end_point = [cx, cy]
            self.center = [(self.start_point[0] + self.end_point[0]) / 2,
                           (self.start_point[1] + self.end_point[1]) / 2]
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            self.rotate_rad = math.atan2(dy, dx)
            self.update_mask()
            return True
        else:
            return super().on_touch_move(touch)
    
    def on_touch_up(self, touch):
        if self.initializing:
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)
    
    def serialize(self):
        sx, sy = params.norm_param(self.effects_param, (self.start_point[0], self.start_point[1]))
        ex, ey = params.norm_param(self.effects_param, (self.end_point[0], self.end_point[1]))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
         
        dict = {
            'type': MaskType.GRADIENT,
            'name': self.name,
            'start_point': [sx, sy],
            'end_point': [ex, ey],
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        sx, sy = dict['start_point']
        ex, ey = dict['end_point']
        self.effects_param.update(dict['effects_param'])

        self.start_point = params.denorm_param(self.effects_param, (sx, sy))
        self.end_point = params.denorm_param(self.effects_param, (ex, ey))

        self.center = [(self.start_point[0] + self.end_point[0]) / 2,
                       (self.start_point[1] + self.end_point[1]) / 2]
        
        self.create_control_points()
        #self.update_mask()

    def create_control_points(self):
        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = self.center
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)
    
        # グラデーションの開始点と終点のコントロールポイント
        cp_start = ControlPoint(self.editor)
        cp_start.center = self.start_point
        cp_start.ctrl_center = cp_start.center
        cp_start.type = ['s', 0]
        cp_start.bind(ctrl_center=self.on_control_point_move)
        self.control_points.append(cp_start)
        self.add_widget(cp_start)
    
        cp_end = ControlPoint(self.editor)
        cp_end.center = self.end_point
        cp_end.ctrl_center = cp_end.center
        cp_end.type = ['e', 0]
        cp_end.bind(ctrl_center=self.on_control_point_move)
        self.control_points.append(cp_end)
        self.add_widget(cp_end)
    
        if not self.active:
            self.show_center_control_point_only()
        else:
            self.show_all_control_points()  # アクティブなら全ポイントの色・表示を更新
    
    def calculate_point(self, point, dir):
        r = np.sqrt((point[0]-self.center_x)**2+(point[1]-self.center_y)**2)
        dx = dir * r
        dy = 0.
        new_r_x = dx * np.cos(-self.rotate_rad) + dy * np.sin(-self.rotate_rad)
        new_r_y = dy * np.cos(-self.rotate_rad) - dx * np.sin(-self.rotate_rad)
        return (float(self.center_x + new_r_x), float(self.center_y + new_r_y))

    def on_center_control_point_move(self, instance, value):
        dx = instance.ctrl_center[0] - self.center[0]
        dy = instance.ctrl_center[1] - self.center[1]
        self.start_point = [self.start_point[0] + dx, self.start_point[1] + dy]
        self.end_point = [self.end_point[0] + dx, self.end_point[1] + dy]
        self.center = [self.center[0] + dx, self.center[1] + dy]
        for cp in self.control_points:
            #if cp != instance:
            center = (cp.center_x + dx, cp.center_y + dy)
            if cp.center[0] == center[0] and cp.center[1] == center[1]:
                cp.property('center').dispatch(cp) # 値が同じだとディスパッチされないから
            else:
                cp.center = center
        self.update_control_points()
        self.update_mask()
        self.editor.start_draw_image()        
    
    def on_control_point_move(self, instance, value):
        if self.active:
            if instance == self.control_points[1]:
                self.start_point = [instance.ctrl_center[0], instance.ctrl_center[1]]
                dx = self.center_x - self.start_point[0]
                dy = self.center_y - self.start_point[1]
                self.end_point[0] = self.center_x + dx
                self.end_point[1] = self.center_y + dy
            elif instance == self.control_points[2]:
                self.end_point = [instance.ctrl_center[0], instance.ctrl_center[1]]
                dx = self.center_x - self.end_point[0]
                dy = self.center_y - self.end_point[1]
                self.start_point[0] = self.center_x + dx
                self.start_point[1] = self.center_y + dy
            # 再計算
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            self.rotate_rad = math.atan2(dy, dx)
            self.update_control_points()
            self.update_mask()
            self.editor.start_draw_image()        

    def update_control_points(self):
        # コントロールポイントの位置を更新
        cp_center = self.control_points[0]
        cp_center.center = self.center
        cp_start = self.control_points[1]
        cp_start.center = self.start_point
        cp_end = self.control_points[2]
        cp_end.center = self.end_point
    
    def calculate_line(self, point1, point2, dir):
        p1x, p1y = self.editor.tcg_to_window(*point1)
        p2x, p2y = self.editor.tcg_to_window(*point2)
        r = math.sqrt((p1x-p2x)**2+(p1y-p2y)**2)
        dx = dir * r
        dy = -self.editor.width
        new_dx1 = dx
        new_dy1 = dy
        dx = dir * r
        dy = self.editor.width
        new_dx2 = dx
        new_dy2 = dy
        dx = p1x-p2x
        dy = p1y-p2y
        rad = 0 if dx == 0 else math.atan2(dy, dx)
        return (new_dx1, new_dy1, new_dx2, new_dy2), rad

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            self.editor.set_scissor(self.scissor)
            if self.initializing:
                tx, ty = self.editor.tcg_to_window(*self.start_point)
                #self.line_color.rgba = self.color
                self.translate.x, self.translate.y = tx, ty
                self.start_line.points, _ = self.calculate_line(self.start_point, self.start_point, 0)
                self.center_line.points, _ = self.calculate_line(self.center, self.start_point, +1)
                self.end_line.points, rad = self.calculate_line(self.end_point, self.start_point, +1)
            else:
                tx, ty = self.editor.tcg_to_window(*self.center)
                self.translate.x, self.translate.y = tx, ty
                self.start_line.points, rad = self.calculate_line(self.start_point, self.center, -1)
                self.center_line.points, _ = self.calculate_line(self.center, self.center, 0)
                self.end_line.points, _ = self.calculate_line(self.end_point, self.center, +1)
            
            self.rotate.angle = math.degrees(rad)

        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()
    
    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)
        start_point = self.editor.tcg_to_texture(*self.start_point)
        end_point = self.editor.tcg_to_texture(*self.end_point)
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert') == True:
                start_point, end_point = end_point, start_point

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center, start_point, end_point))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            # グラデーションを描画
            gradient_image = self.draw_gradient(image_size, center, start_point, end_point, 1)
            
            # ルミノシティマスクを作成
            gradient_image = self._apply_extened_params(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    def draw_gradient(self, image_size, center, start_point, end_point, smoothness=1):
        return gradient_raster.draw_linear_gradient(
            image_size, center, start_point, end_point, smoothness
        )

# 全体マスクのクラス
class FullMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Full"
        self.initializing = True  # 初期配置中かどうか

        self.center = (0, 0)

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate(*self.center)
            KVPopMatrix()

        #self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else: 
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        self.control_points = []

        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.FULL,
            'name': self.name,
            'center': [cx, cy],
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])

        self.center = params.denorm_param(self.effects_param, (cx, cy))

        # 描き直し
        self.create_control_points()
        #self.update_mask()    

    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            # 描画
            gradient_image = self.draw_full(image_size, center)

            # ルミノシティマスクを作成
            gradient_image = self._apply_extened_params(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
        
        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_full(self, image_size, center):
        # 画像の初期化
        image = np.ones((image_size[1], image_size[0]), dtype=np.float32)

        return image

# 自由描画マスクのクラス
class FreeDrawMask(BaseMask):

    class Line:
        def __init__(self, is_erasing=False, size=10, soft=100):
            self.is_erasing = is_erasing
            self.size = size
            self.soft = soft
            self.points = []

        def add_point(self, x, y):
            self.points.append((x, y))

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Draw"
        self.initializing = True

        self.lines = []  # 複数の線を保持
        self.current_line = None
        self.brush_size = 300
        self._stroke_history_started = False

        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate(0, 0)
            self.rotate = KVRotate(angle=0, origin=(0, 0))
            self.brush_color = KVColor((0, 1, 1, 1))
            self.brush_cursor = KVLine(ellipse=(0, 0, self.brush_size, self.brush_size), width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

        KVWindow.bind(mouse_pos=self.on_mouse_pos)

    def start(self):
        self.brush_color.rgba = (1, 1, 1, 1)
        KVWindow.bind(mouse_pos=self.on_mouse_pos)

    def end(self):
        self.brush_color.rgba = (0, 0, 0, 0)
        KVWindow.unbind(mouse_pos=self.on_mouse_pos)

    def clear(self):
        self.lines = []
        self.current_line = None
        super().clear()

    def serialize(self):
        """マスクの状態をシリアライズ"""
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        
        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)

        lines = []
        for line in self.lines:
            lines.append({
                'is_erasing': line.is_erasing,
                'size': line.size,
                'soft': line.soft,
                'points': copy.deepcopy(line.points)
            })
        
        dict = {
            'type': MaskType.FREEDRAW,
            'name': self.name,
            'center': [cx, cy],
            'lines': lines,
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']

        lines = []
        for line in dict['lines']:
            lineobj = FreeDrawMask.Line(
                is_erasing=line['is_erasing'],
                size=line['size'],
                soft=line['soft'],
            )
            for point in line['points']:
                lineobj.add_point(*point)
            lines.append(lineobj)
        self.lines = lines

        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))

        self.create_control_points()

    def on_mouse_pos(self, window, pos):
        self.update_brush_cursor(pos[0], pos[1])

    def _pan_mode_active(self):
        """スペースキー押下中はパン優先。Draw 描画系の touch を完全に無効化する。"""
        root = getattr(self.editor, 'root', None)
        return bool(getattr(root, 'is_press_space', False))

    def on_touch_down(self, touch):
        if self._pan_mode_active():
            # マスク側はイベントを掴まない。親 (preview_widget) の panning handler に流す。
            return False
        if self.editor.get_active_mask() != self and self.editor.get_created_mask() != self:
            return super().on_touch_down(touch)
        
        if self.editor.is_center_click_anyone(touch, self):
            return False

        if touch.is_mouse_scrolling:
            if self.editor.collide_point(*touch.pos):
                # 描画中または消去中はブラシサイズを変更できない
                if self.current_line is None:
                    if touch.button == 'scrollup':
                        self.brush_size = max(10, self.brush_size - 10)
                    elif touch.button == 'scrolldown':
                        self.brush_size = min(2000, self.brush_size + 10)
                        
                    self.update_brush_cursor(touch.pos[0], touch.pos[1])

                    return super().on_touch_down(touch)

        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            self.create_control_points()
            self.editor.set_active_mask(self)            

        # 右クリックで消去モード、左クリックで描画モード
        is_erasing = (touch.button == 'right')            
        if not self.initializing:
            get_history_ctrl().begin_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self), None)
            self._stroke_history_started = True

        hardness = effects.Mask2Effect.get_param(self.effects_param, 'mask2_freedraw_brush_hardness')
        self.current_line = FreeDrawMask.Line(is_erasing, self.brush_size, hardness)
        self.current_line.add_point(*self.editor.window_to_tcg(*touch.pos))
        self.editor.set_active_mask(self)
        self.lines.append(self.current_line)

        self.update_mask()
        self.editor.start_draw_image()
        
        # 初期化時はBaseMaskの方を呼び出さない
        if self.initializing:
            self.initializing = False
            return True

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._pan_mode_active():
            return False
        if self.current_line is not None:
            self.current_line.add_point(*self.editor.window_to_tcg(*touch.pos))
            self.update_mask()
            self.editor.start_draw_image()
            return True

        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self._pan_mode_active():
            # パンモード抜け遅延（touch_up が先に来る場合）に備え、
            # ストロークの後始末は通常パスでも行う。ここでは描画動作だけ抑止する。
            if self.current_line is not None:
                self.current_line = None
                self.update_mask()
                self.editor.start_draw_image()
                if self._stroke_history_started:
                    get_history_ctrl().end_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self))
                    self._stroke_history_started = False
            return False
        if self.current_line is not None:
            self.current_line = None
            # マスクを更新
            self.update_mask()
            self.editor.start_draw_image()
            if self._stroke_history_started:
                get_history_ctrl().end_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self))
                self._stroke_history_started = False
            return True

        return super().on_touch_up(touch)

    def create_control_points(self):
        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

    def update_brush_cursor(self, x, y):
        brush_size = self.editor.tcg_to_window_scale(self.brush_size, 0)[0]
        self.translate.x, self.translate.y = x - brush_size / 2, y - brush_size / 2
        self.brush_cursor.ellipse = (0, 0, brush_size, brush_size)

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            return
        
        self.editor.set_scissor(self.scissor)
        self.rotate.angle = math.degrees(self.editor.get_rotate_rad(0))

        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        copy_lines = []
        for i, src_line in enumerate(self.lines):
            copy_line = FreeDrawMask.Line(src_line.is_erasing, self.editor.tcg_to_image_scale(src_line.size, 0)[0], src_line.soft)
            for point in src_line.points:
                copy_line.add_point(*self.editor.tcg_to_texture(*point))
            copy_lines.append(copy_line)

        line_hash = tuple(
            (line.is_erasing, line.size, line.soft, tuple(line.points))
            for line in self.lines
        )
        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, line_hash))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            allow_over_one = False
            allow_under_zero = False
            mask = freedraw_raster.draw_line_texture(
                image_size,
                copy_lines,
                allow_over_one=allow_over_one,
                allow_under_zero=allow_under_zero,
            )

            # ルミナンスとマスクを作成
            mask = self._apply_extened_params(mask)

            self.image_mask_cache = mask
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)


# 折れ線マスクのクラス
class PolylineMask(BaseMask):
    """頂点をクリックして折れ線を描き、閉じれば塗りつぶせるマスク。

    各 polyline は確定後に頂点 ControlPoint で再編集できる。
    """

    # 始点と現在地が十分近いと判定する画面上のピクセル距離 (TCG 換算は描画時に動的計算)
    _CLOSE_HIT_RADIUS_PX = 16.0

    class Polyline:
        def __init__(self, is_erasing=False, size=10, soft=100,
                     is_closed=False, is_filled=True):
            self.is_erasing = bool(is_erasing)
            self.size = float(size)
            self.soft = float(soft)
            self.is_closed = bool(is_closed)
            self.is_filled = bool(is_filled)
            self.points = []

        def add_point(self, x, y):
            self.points.append((float(x), float(y)))

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Polyline"
        self.initializing = True

        self.polylines = []          # 確定済み polyline のリスト
        self.current_polyline = None # 描画中の polyline
        self.brush_size = 300        # 線幅 (TCG)
        self._stroke_history_started = False

        # コントロールポイント描画状態
        self._vertex_control_points = []  # [(polyline_idx, point_idx, ControlPoint)]

        with self.canvas:
            # スコープ A: ラバーバンドと始点ハイライト (ウィンドウ座標、translate なし)
            KVPushMatrix()
            self._overlay_scissor = self.editor.push_scissor()
            self.preview_color = KVColor((1, 1, 0, 0))   # 初期不可視
            self.preview_line = KVLine(points=[], width=1)
            self.start_color = KVColor((0, 1, 1, 0))
            self.start_indicator = KVLine(ellipse=(0, 0, 0, 0), width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

            # スコープ B: ブラシカーソル (translate でカーソル位置に移動)
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate(0, 0)
            self.rotate = KVRotate(angle=0, origin=(0, 0))
            self.brush_color = KVColor((0, 1, 1, 0))     # 初期不可視
            self.brush_cursor = KVLine(ellipse=(0, 0, self.brush_size, self.brush_size), width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

        KVWindow.bind(mouse_pos=self.on_mouse_pos)

    # ---- ライフサイクル ----
    def start(self):
        self.brush_color.rgba = (1, 1, 1, 1)
        KVWindow.bind(mouse_pos=self.on_mouse_pos)

    def end(self):
        # 描画中の polyline は終了扱い (開いた折れ線として確定 or 破棄)
        self.commit_in_progress()
        self.brush_color.rgba = (0, 0, 0, 0)
        self.preview_color.rgba = (1, 1, 0, 0)
        self.start_color.rgba = (0, 1, 1, 0)
        KVWindow.unbind(mouse_pos=self.on_mouse_pos)

    def clear(self):
        self.polylines = []
        self.current_polyline = None
        self._clear_vertex_control_points()
        super().clear()

    # ---- シリアライズ ----
    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)

        polys = []
        for p in self.polylines:
            polys.append({
                'is_erasing': p.is_erasing,
                'size': p.size,
                'soft': p.soft,
                'is_closed': p.is_closed,
                'is_filled': p.is_filled,
                'points': copy.deepcopy(p.points),
            })

        return {
            'type': MaskType.POLYLINE,
            'name': self.name,
            'center': [cx, cy],
            'polylines': polys,
            'effects_param': param,
        }

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']

        polys = []
        for p in dict.get('polylines', []):
            polyobj = PolylineMask.Polyline(
                is_erasing=p.get('is_erasing', False),
                size=p.get('size', 10),
                soft=p.get('soft', 100),
                is_closed=p.get('is_closed', False),
                is_filled=p.get('is_filled', True),
            )
            for point in p.get('points', []):
                polyobj.add_point(*point)
            polys.append(polyobj)
        self.polylines = polys

        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))

        self.create_control_points()
        # 確定 polyline の頂点 ControlPoint も復元
        self._rebuild_vertex_control_points()

    # ---- マウス入力 ----
    def on_mouse_pos(self, window, pos):
        self.update_brush_cursor(pos[0], pos[1])
        self._update_preview_line(pos)

    def _pan_mode_active(self):
        root = getattr(self.editor, 'root', None)
        return bool(getattr(root, 'is_press_space', False))

    def _close_hit_distance_tcg(self):
        """始点との距離判定用しきい値 (TCG 単位)。"""
        return self.editor.window_to_tcg_scale(self._CLOSE_HIT_RADIUS_PX, 0)[0]

    def _is_near_first_point(self, tcg_x, tcg_y):
        if self.current_polyline is None or len(self.current_polyline.points) < 2:
            return False
        sx, sy = self.current_polyline.points[0]
        thr = self._close_hit_distance_tcg()
        return (tcg_x - sx) ** 2 + (tcg_y - sy) ** 2 <= thr * thr

    def consumes_double_tap(self, touch):
        """ダブルタップを polyline 確定として消費するかどうか。
        プレビュー領域内で描画中の場合 True を返し、preview_widget のズーム切替を抑制する。"""
        if self.current_polyline is None:
            return False
        try:
            return bool(self.editor.collide_point(*touch.pos))
        except Exception:
            return False

    def on_touch_down(self, touch):
        if self._pan_mode_active():
            return False

        # アクティブマスクが自分でないかつ作成中でもないなら標準 ControlPoint 経路
        if self.editor.get_active_mask() != self and self.editor.get_created_mask() != self:
            return super().on_touch_down(touch)

        # preview_widget (= self.editor) の外側 (パラメータパネル等) のクリックは無視。
        # editor 内なら image 範囲外 (レターボックス部分) でも頂点設定 OK。
        if not self.editor.collide_point(*touch.pos):
            return False

        if self.editor.is_center_click_anyone(touch, self):
            return False

        # スクロールで線幅 (描画中以外のみ)
        if touch.is_mouse_scrolling:
            if self.editor.collide_point(*touch.pos):
                if self.current_polyline is None:
                    if touch.button == 'scrollup':
                        self.brush_size = max(2, self.brush_size - 10)
                    elif touch.button == 'scrolldown':
                        self.brush_size = min(2000, self.brush_size + 10)
                    self.update_brush_cursor(touch.pos[0], touch.pos[1])
                    return super().on_touch_down(touch)

        # 確定 polyline の頂点 ControlPoint クリックを最優先で処理する。
        # 描画中 (current_polyline is not None) はラバーバンドと衝突するので無効化。
        if self.current_polyline is None and not self.initializing:
            for cp in self._iter_vertex_control_points():
                cx, cy = self.editor.window_to_tcg(*touch.pos)
                if cp.collide_point(cx, cy):
                    return super().on_touch_down(touch)

        tcg_x, tcg_y = self.editor.window_to_tcg(*touch.pos)

        # 初期化時 (最初の左クリックでのみマスク中心を確定)。
        # 右クリックでの初期化は中心だけ残って polyline が始まらないので不可。
        if self.initializing and touch.button == 'left' and not getattr(touch, 'is_double_tap', False):
            self.center_x = tcg_x
            self.center_y = tcg_y
            self.create_control_points()
            self.editor.set_active_mask(self)

        # 右クリック: 描画中のみ直近頂点を取消 (idle 右クリックは何もしない)
        if touch.button == 'right':
            if self.current_polyline is not None:
                if self.current_polyline.points:
                    self.current_polyline.points.pop()
                if len(self.current_polyline.points) <= 0:
                    self.current_polyline = None
                    if self._stroke_history_started:
                        get_history_ctrl().end_history_layer_ctrl(
                            self.editor, "Update", self.editor.get_mask_list().index(self))
                        self._stroke_history_started = False
                self.update_mask()
                self.editor.start_draw_image()
                return True
            # idle 右クリックは消費せず親に流す
            return super().on_touch_down(touch)

        # 左クリック (描画/閉じる/ダブルクリック開放確定)
        if touch.button == 'left':
            # ダブルタップ: 直近で追加した点を pop して開放確定
            if getattr(touch, 'is_double_tap', False):
                if self.current_polyline is not None:
                    # is_double_tap の前段 down で頂点を 1 つ余計に追加していることが多いので pop
                    if self.current_polyline.points:
                        self.current_polyline.points.pop()
                    self._finish_current_polyline(is_closed=False)
                    self.update_mask()
                    self.editor.start_draw_image()
                    if self.initializing:
                        self.initializing = False
                    return True
                return super().on_touch_down(touch)

            # 描画中で始点付近をクリックしたら閉じて確定
            if self.current_polyline is not None and self._is_near_first_point(tcg_x, tcg_y):
                self._finish_current_polyline(is_closed=True)
                self.update_mask()
                self.editor.start_draw_image()
                if self.initializing:
                    self.initializing = False
                return True

            # 通常の頂点追加
            if self.current_polyline is None:
                self._begin_new_polyline(tcg_x, tcg_y, is_erasing=False)
            else:
                self.current_polyline.add_point(tcg_x, tcg_y)
                self.update_mask()
                self.editor.start_draw_image()
            if self.initializing:
                self.initializing = False
                return True
            return super().on_touch_down(touch)

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._pan_mode_active():
            return False
        # 描画中のラバーバンドは on_mouse_pos で更新するため move は不要
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self._pan_mode_active():
            return False
        return super().on_touch_up(touch)

    # ---- 描画中 polyline 制御 ----
    def _begin_new_polyline(self, tcg_x, tcg_y, is_erasing):
        if not self._stroke_history_started:
            get_history_ctrl().begin_history_layer_ctrl(
                self.editor, "Update", self.editor.get_mask_list().index(self), None)
            self._stroke_history_started = True
        hardness = effects.Mask2Effect.get_param(self.effects_param, 'mask2_freedraw_brush_hardness')
        self.current_polyline = PolylineMask.Polyline(
            is_erasing=is_erasing,
            size=self.brush_size,
            soft=hardness,
            is_closed=False,
            is_filled=True,
        )
        self.current_polyline.add_point(tcg_x, tcg_y)
        self.editor.set_active_mask(self)
        self.update_mask()
        self.editor.start_draw_image()

    def _finish_current_polyline(self, is_closed: bool):
        if self.current_polyline is None:
            return
        # 頂点 2 個未満なら破棄
        if len(self.current_polyline.points) < 2:
            self.current_polyline = None
        else:
            self.current_polyline.is_closed = bool(is_closed)
            # 開いた場合は塗りつぶさない
            if not is_closed:
                self.current_polyline.is_filled = False
            self.polylines.append(self.current_polyline)
            self.current_polyline = None
            # 確定 polyline の頂点 ControlPoint を生成
            self._rebuild_vertex_control_points()
        if self._stroke_history_started:
            get_history_ctrl().end_history_layer_ctrl(
                self.editor, "Update", self.editor.get_mask_list().index(self))
            self._stroke_history_started = False

    def commit_in_progress(self):
        """描画中の polyline があれば「開いた折れ線」として確定する。
        タブ切替やマスク非アクティブ化など、描画コンテキストを抜けるときに呼ぶ。"""
        if self.current_polyline is not None:
            self._finish_current_polyline(is_closed=False)
            self.update_mask()
            self.editor.start_draw_image()
        # 残留オーバーレイをリセット
        self.preview_line.points = []
        self.preview_color.rgba = (1, 1, 0, 0)
        self.start_indicator.ellipse = (0, 0, 0, 0)
        self.start_color.rgba = (0, 1, 1, 0)

    def _update_preview_line(self, window_pos):
        """ラバーバンド (直近頂点 → カーソル) と始点強調を更新。"""
        if self.current_polyline is None or len(self.current_polyline.points) == 0:
            self.preview_color.rgba = (1, 1, 0, 0)
            self.start_color.rgba = (0, 1, 1, 0)
            return
        # 直近頂点 → カーソル 線分
        last_tcg = self.current_polyline.points[-1]
        last_wx, last_wy = self.editor.tcg_to_window(*last_tcg)
        self.preview_line.points = [last_wx, last_wy, window_pos[0], window_pos[1]]
        self.preview_color.rgba = (1, 1, 0, 0.8)

        # 始点強調 (頂点が 2 個以上、つまり閉じ判定可能なときのみ表示)
        if len(self.current_polyline.points) >= 2:
            sx, sy = self.editor.tcg_to_window(*self.current_polyline.points[0])
            r_px = self._CLOSE_HIT_RADIUS_PX
            self.start_indicator.ellipse = (sx - r_px, sy - r_px, r_px * 2, r_px * 2)
            tcg_x, tcg_y = self.editor.window_to_tcg(*window_pos)
            if self._is_near_first_point(tcg_x, tcg_y):
                self.start_color.rgba = (0, 1, 0, 1)  # 閉じる予告
            else:
                self.start_color.rgba = (0, 1, 1, 0.6)
        else:
            self.start_color.rgba = (0, 1, 1, 0)

    # ---- ControlPoint ----
    def create_control_points(self):
        # 中心 ControlPoint (FreeDrawMask に倣う)
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

    def _iter_vertex_control_points(self):
        for _, _, cp in self._vertex_control_points:
            yield cp

    def _vertex_cp_set(self):
        return {cp for _, _, cp in self._vertex_control_points}

    def show_all_control_points(self):
        """BaseMask の実装は非中心 CP を全部赤に塗るが、Polyline では頂点 CP は青を維持する。"""
        self.opacity = 1.0
        vertex_cps = self._vertex_cp_set()
        for cp in self.control_points:
            cp.opacity = 1
            if cp.is_center:
                cp.color = [0, 1, 0]
            elif cp in vertex_cps:
                cp.color = [0.2, 0.6, 1.0]  # 青
            else:
                cp.color = [1, 0, 0]
        self.is_draw_mask = True
        self.update_mask()

    def show_center_control_point_only(self):
        """非アクティブ時: 中心 CP のみ表示 (頂点 CP は隠す)。"""
        self.opacity = 0.2
        vertex_cps = self._vertex_cp_set()
        for cp in self.control_points:
            if cp.is_center:
                cp.opacity = 2
                cp.color = [1, 0, 0]
            else:
                # 頂点 CP もそれ以外も非表示
                cp.opacity = 0
                if cp in vertex_cps:
                    cp.color = [0.2, 0.6, 1.0]  # 復帰時用に色だけ保持
        self.is_draw_mask = False
        self.update_mask()

    def _clear_vertex_control_points(self):
        for _, _, cp in self._vertex_control_points:
            self.remove_widget(cp)
            if cp in self.control_points:
                self.control_points.remove(cp)
        self._vertex_control_points = []

    def _rebuild_vertex_control_points(self):
        """確定 polyline の各頂点に ControlPoint を生成する。"""
        self._clear_vertex_control_points()
        for pi, poly in enumerate(self.polylines):
            for vi, (px, py) in enumerate(poly.points):
                cp = ControlPoint(self.editor)
                cp.center = (px, py)
                cp.ctrl_center = cp.center
                cp.is_center = False
                cp.color = [0.2, 0.6, 1.0]  # 青系
                cp.bind(ctrl_center=self._make_vertex_callback(pi, vi))
                self.control_points.append(cp)
                self.add_widget(cp)
                self._vertex_control_points.append((pi, vi, cp))
        # アクティブ/非アクティブの表示状態を最新化
        if self.active:
            self.show_all_control_points()
        else:
            self.show_center_control_point_only()

    def _make_vertex_callback(self, pi, vi):
        def _cb(instance, value):
            try:
                self.polylines[pi].points[vi] = (instance.ctrl_center[0], instance.ctrl_center[1])
            except (IndexError, AttributeError):
                return
            # ControlPoint の見た目位置も更新 (ctrl_center だけでは center が動かない)
            new_center = (instance.ctrl_center[0], instance.ctrl_center[1])
            if instance.center[0] == new_center[0] and instance.center[1] == new_center[1]:
                instance.property('center').dispatch(instance)
            else:
                instance.center = new_center
            self.update_mask()
            self.editor.start_draw_image()
        return _cb

    def on_center_control_point_move(self, instance, value):
        # 中心移動: 全 polyline の全頂点を平行移動 (FreeDrawMask 同様の流儀)
        dx = instance.ctrl_center[0] - self.center_x
        dy = instance.ctrl_center[1] - self.center_y
        self.center = (self.center_x + dx, self.center_y + dy)
        # 確定 polyline の頂点を平行移動
        for poly in self.polylines:
            poly.points = [(p[0] + dx, p[1] + dy) for p in poly.points]
        # 描画中 polyline も追従させる
        if self.current_polyline is not None:
            self.current_polyline.points = [(p[0] + dx, p[1] + dy) for p in self.current_polyline.points]
        # 頂点 ControlPoint の center 値も同期 (super の制御点移動は中心のみ動かす)
        for pi, vi, cp in self._vertex_control_points:
            try:
                px, py = self.polylines[pi].points[vi]
                if cp.center[0] == px and cp.center[1] == py:
                    cp.property('center').dispatch(cp)
                else:
                    cp.center = (px, py)
            except (IndexError, AttributeError):
                pass
        # 中心 ControlPoint の center も追従
        super_cp_iter = (cp for cp in self.control_points if cp.is_center)
        for cp in super_cp_iter:
            center = (cp.center_x + dx, cp.center_y + dy)
            if cp.center[0] == center[0] and cp.center[1] == center[1]:
                cp.property('center').dispatch(cp)
            else:
                cp.center = center
        self.update_mask()
        self.editor.start_draw_image()

    # ---- カーソル ----
    def update_brush_cursor(self, x, y):
        brush_size = self.editor.tcg_to_window_scale(self.brush_size, 0)[0]
        self.translate.x, self.translate.y = x - brush_size / 2, y - brush_size / 2
        self.brush_cursor.ellipse = (0, 0, brush_size, brush_size)

    # ---- マスク描画 ----
    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            return
        self.editor.set_scissor(self.scissor)
        self.rotate.angle = math.degrees(self.editor.get_rotate_rad(0))

        if self.is_draw_mask:
            if self.do_draw_composit_mask:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def _build_render_polylines(self, image_size):
        """確定 polyline + 描画中 polyline をテクスチャ座標に変換して返す。"""
        render = list(self.polylines)
        if self.current_polyline is not None and len(self.current_polyline.points) >= 1:
            render.append(self.current_polyline)
        result = []
        for src in render:
            tex_poly = polyline_raster.Polyline(
                is_erasing=src.is_erasing,
                size=self.editor.tcg_to_image_scale(src.size, 0)[0],
                soft=src.soft,
                is_closed=src.is_closed,
                # 描画中は仮で fill=False, 確定後のみ fill 適用
                is_filled=src.is_filled and src.is_closed,
            )
            for point in src.points:
                tex_poly.add_point(*self.editor.tcg_to_texture(*point))
            result.append(tex_poly)
        return result

    def get_mask_image(self):
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        copy_polys = self._build_render_polylines(image_size)

        poly_hash = tuple(
            (p.is_erasing, p.size, p.soft, p.is_closed, p.is_filled, tuple(p.points))
            for p in (self.polylines + ([self.current_polyline] if self.current_polyline else []))
        )
        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, poly_hash))

        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            mask = polyline_raster.draw_polyline_texture(
                image_size,
                copy_polys,
                allow_over_one=False,
                allow_under_zero=False,
            )
            mask = self._apply_extened_params(mask)
            self.image_mask_cache = mask
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)


# セグメントマスクのクラス
class SegmentMask(BaseMask):
    corner = KVListProperty([0, 0])

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Segment"
        self.initializing = True  # 初期配置中かどうか

        self.center = (0, 0)
        self.corner = (0, 0)

        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None

        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            # center位置への移動
            self.translate = KVTranslate(0, 0)
            KVColor(*self.color)
            self.rect_line = KVLine(points=[], close=True, width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

        #self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            self.corner = [cx, cy]
            #self.update_mask()
            return True
        else: 
            self.is_draw_mask = False
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.corner = [cx, cy]
            self.update_mask()
            return True
        else:
            self.is_draw_mask = False
            self.update_mask()
            return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            self.get_mask_image() # 即座に計算開始
            #self.update_mask()
            self.update_draw_mask()
            return True
        else:
            self.is_draw_mask = True
            self.update_mask()
            self.update_draw_mask()
            return super().on_touch_up(touch)

    def create_control_points(self):
        # 中心のコントロールポイント（始点）
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        # コーナーのコントロールポイント（終点）
        cp_corner = ControlPoint(self.editor)
        cp_corner.center = self.corner
        cp_corner.ctrl_center = cp_corner.center
        cp_corner.type = ['corner', 0]
        # コーナーもコントロールポイントとして独立して動かせるようにする
        cp_corner.bind(ctrl_center=self.on_corner_control_point_move)
        self.control_points.append(cp_corner)
        self.add_widget(cp_corner)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        crx, cry = params.norm_param(self.effects_param, (self.corner[0], self.corner[1]))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.SEGMENT,
            'name': self.name,
            'center': [cx, cy],
            'corner': [crx, cry],
            'effects_param': param
        }
        # マスクデータ保存
        if self.image_mask_cache is not None:
            dict['image_mask_cache'] = utils.convert_image_to_list(self.image_mask_cache)
            dict['image_mask_cache_hash'] = self.image_mask_cache_hash

        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        crx, cry = dict.get('corner', [cx, cy]) # 後方互換性
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        self.corner = params.denorm_param(self.effects_param, (crx, cry))

        # マスクデータ展開
        self.image_mask_cache = dict.get('image_mask_cache', None)
        if self.image_mask_cache is not None:
            self.image_mask_cache = utils.convert_image_from_list(self.image_mask_cache)
            self.image_mask_cache_hash = dict.get('image_mask_cache_hash', None)

        # 描き直し
        self.create_control_points()
        #self.update_mask()     

    def update_control_points(self):
        if len(self.control_points) > 0:
            cp_center = self.control_points[0]
            cp_center.center = self.center
        if len(self.control_points) > 1:
            cp_corner = self.control_points[1]
            cp_corner.center = self.corner

    def on_center_control_point_move(self, instance, value):
        # 始点移動：コーナーは動かさない（ボックスの形が変わる）
        self.center = value
        self.update_control_points()

        #super().on_center_control_point_move(instance, value)
        # update_maskはsuper()の中で呼ばれる

    def on_corner_control_point_move(self, instance, value):
        self.corner = value
        self.update_control_points()
#        instance.center = value
        #self.update_mask()

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            self.editor.set_scissor(self.scissor)
            cx, cy = self.center
            crx, cry = self.corner

            # 4隅の座標を計算（TCG座標系）
            p1 = (cx, cy)
            p2 = (crx, cy)
            p3 = (crx, cry)
            p4 = (cx, cry)
            
            # ウィンドウ座標系に変換
            wp1 = self.editor.tcg_to_window(*p1)
            wp2 = self.editor.tcg_to_window(*p2)
            wp3 = self.editor.tcg_to_window(*p3)
            wp4 = self.editor.tcg_to_window(*p4)
            
            # BaseMaskの仕組みでTranslateされているが、回転に対応するためTranslateを無効化（0,0）して絶対座標で描く
            self.translate.x, self.translate.y = 0, 0
            
            self.rect_line.points = [*wp1, *wp2, *wp3, *wp4]
        
    def update_draw_mask(self):
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_original_image(*self.center)
        corner = self.editor.tcg_to_original_image(*self.corner)
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
            invert = effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')
        else:
            invert = False
        segment_mask = None

        # _draw_segmentを呼び出さなければならない用
        newhash = hash((image_size, center, corner))
        if (self.image_mask_cache_hash != newhash) and self.initializing == False:
            self.image_mask_cache_hash = newhash

            # 描画
            cx, cy = center
            crx, cry = corner
            
            # 2点からバウンディングボックスを計算 (XYWH)
            min_x = min(cx, crx)
            min_y = min(cy, cry)
            w = abs(cx - crx)
            h = abs(cy - cry)
            
            # predict_sam3 に渡す box = [x, y, w, h]
            segment_mask = wait_prosessing(self._draw_segment, image_size, [min_x, min_y, w, h], invert)
            #segment_mask = self._draw_segment(image_size, [min_x, min_y, w, h])

            # SegmentMask用のキャッシュ
            self.image_mask_cache = segment_mask

        # その他更新用
        newhash = hash((self.get_hash_items(), self.editor.get_hash_items()))
        if self.image_mask_cache is not None and (self.image_mask_cache is segment_mask or self.segment_mask_cache is None or self.segment_mask_cache_hash != newhash) and self.initializing == False:
            self.segment_mask_cache_hash = newhash

            # SegmentMask用のキャッシュ
            segment_mask = self.image_mask_cache

            # パラメータに従って画像を変形
            disp_info, rotate_rad, flip, matrix = self.editor.get_hash_items()
            segment_mask = core.rotation(segment_mask, np.rad2deg(rotate_rad), flip, np.array(matrix).reshape(3, 3))
            #segment_mask = core.crop_image_with_disp_info(segment_mask, disp_info)

            nw, nh, ox, oy = core.crop_size_and_offset_from_texture(*self.editor.texture_size, disp_info)
            cx, cy ,cw, ch, scale = disp_info
            #cx, cy, cw, ch = int(cx * scale), int(cy * scale), int(cw * scale), int(ch * scale)
            segment_mask = cv2.resize(segment_mask[cy:cy+ch, cx:cx+cw], (nw, nh))
            segment_mask = np.pad(segment_mask, ((oy, self.editor.texture_size[1]-(oy+nh)), (ox, self.editor.texture_size[0]-(ox+nw))), constant_values=0)

            # ルミノシティマスクを作成
            segment_mask = self._apply_extened_params(segment_mask)

            self.segment_mask_cache = segment_mask

        if segment_mask is None:
            segment_mask = self.segment_mask_cache

        return segment_mask if segment_mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def _draw_segment(self, image_size, bbox, invert):
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        img = self.editor.get_original_image_rgb()
        return mask2_inference_runtime.predict_sam3_bbox(img, bbox, invert)

class DepthMapMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Depth Map"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.depth_map_mask_cache = None
        self.depth_map_mask_cache_hash = None

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate(*self.center)
            KVPopMatrix()

        #self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else: 
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        self.control_points = []

        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)

        dict = {
            'type': MaskType.DEPTHMAP,
            'name': self.name,
            'center': [cx, cy],
            'effects_param': param
        }
        # マスクデータ保存
        if self.image_mask_cache is not None:
            dict['image_mask_cache'] = utils.convert_image_to_list(self.image_mask_cache)
            dict['image_mask_cache_hash'] = self.image_mask_cache_hash

        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        # マスクデータ展開
        self.image_mask_cache = dict.get('image_mask_cache', None)
        if self.image_mask_cache is not None:
            self.image_mask_cache = utils.convert_image_from_list(self.image_mask_cache)
            self.image_mask_cache_hash = dict.get('image_mask_cache_hash', None)

        # 描き直し
        self.create_control_points()
        #self.update_mask()
     
    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_original_image(*self.center)
        depth_map_mask = None

        newhash = hash((image_size))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            self.image_mask_cache_hash = newhash

            depth_map_mask = wait_prosessing(self.draw_depth_map, image_size)
            #depth_map_mask = self.draw_depth_map(image_size)

            self.image_mask_cache = depth_map_mask

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items()))
        if self.image_mask_cache is not None and (self.image_mask_cache is depth_map_mask or self.depth_map_mask_cache is None or self.depth_map_mask_cache_hash != newhash) and self.initializing == False:
            self.depth_map_mask_cache_hash = newhash

            depth_map_mask = self.image_mask_cache

            # パラメータに従って画像を変形
            disp_info, rotate_rad, flip, matrix = self.editor.get_hash_items()
            depth_map_mask = core.rotation(depth_map_mask, rotate_rad, flip, np.array(matrix).reshape(3, 3))
            depth_map_mask = core.crop_image_with_disp_info(depth_map_mask, disp_info)

            nw, nh, ox, oy = core.crop_size_and_offset_from_texture(self.editor.texture_size[0], self.editor.texture_size[1], disp_info)
            cx, cy ,cw, ch, scale = disp_info
            cx, cy, cw, ch = int(cx * scale), int(cy * scale), int(cw * scale), int(ch * scale)
            depth_map_mask = cv2.resize(depth_map_mask[cy:cy+ch, cx:cx+cw], (nw, nh))
            depth_map_mask = np.pad(depth_map_mask, ((oy, self.editor.texture_size[0]-(oy+nh)), (ox, self.editor.texture_size[1]-(ox+nw))), constant_values=0)

            # ルミノシティマスクを作成
            depth_map_mask = self._apply_extened_params(depth_map_mask)

            self.depth_map_mask_cache = depth_map_mask

        if depth_map_mask is None:
            depth_map_mask = self.depth_map_mask_cache

        return depth_map_mask if depth_map_mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_depth_map(self, image_size):
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        return mask2_inference_runtime.predict_depth_map(self.editor.get_original_image_rgb())

class FaceMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Face"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.faces_mask_cache = None
        self.faces_mask_cache_hash = None

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate(*self.center)
            KVPopMatrix()

        #self.update_mask()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else: 
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        self.control_points = []

        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)

        dict = {
            'type': MaskType.FACE,
            'name': self.name,
            'center': [cx, cy],
            'effects_param': param
        }
        # マスクデータ保存
        if self.image_mask_cache is not None:
            dict['image_mask_cache'] = utils.convert_image_to_list(self.image_mask_cache)
            dict['image_mask_cache_hash'] = self.image_mask_cache_hash

        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        # マスクデータ展開
        self.image_mask_cache = dict.get('image_mask_cache', None)
        if self.image_mask_cache is not None:
            self.image_mask_cache = utils.convert_image_from_list(self.image_mask_cache)
            self.image_mask_cache_hash = dict.get('image_mask_cache_hash', None)

        # 描き直し
        self.create_control_points()     

    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_original_image(*self.center)
        exclude_names = []
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_face') == True:
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_face') == False:
                exclude_names.append('face')
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_brows') == False:
                exclude_names.extend(['rb', 'lb'])
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_eyes') == False:
                exclude_names.extend(['re', 'le'])
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_nose') == False:
                exclude_names.append('nose')
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_mouth') == False:
                exclude_names.append('imouth')
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_lips') == False:
                exclude_names.extend(['ulip', 'llip'])
        faces_mask = None

        newhash = hash((image_size, tuple(exclude_names)))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            self.image_mask_cache_hash = newhash

            # 描画
            faces_mask = wait_prosessing(self.draw_face, image_size, exclude_names)
            #faces_mask = self.draw_face(image_size, exclude_names)

            self.image_mask_cache = faces_mask

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items()))
        if self.image_mask_cache is not None and (self.image_mask_cache is faces_mask or self.faces_mask_cache is None or self.faces_mask_cache_hash != newhash) and self.initializing == False:
            self.faces_mask_cache_hash = newhash

            faces_mask = self.image_mask_cache

            # パラメータに従って画像を変形
            disp_info, rotate_rad, flip, matrix = self.editor.get_hash_items()
            faces_mask = core.rotation(faces_mask, np.rad2deg(rotate_rad), flip, np.array(matrix).reshape(3, 3))
            #faces_mask = core.crop_image_with_disp_info(faces_mask, disp_info)

            nw, nh, ox, oy = core.crop_size_and_offset_from_texture(*self.editor.texture_size, disp_info)
            cx, cy ,cw, ch, scale = disp_info
            #cx, cy, cw, ch = int(cx * scale), int(cy * scale), int(cw * scale), int(ch * scale)
            faces_mask = cv2.resize(faces_mask[cy:cy+ch, cx:cx+cw], (nw, nh))
            faces_mask = np.pad(faces_mask, ((oy, self.editor.texture_size[1]-(oy+nh)), (ox, self.editor.texture_size[0]-(ox+nw))), constant_values=0)

            # ルミノシティマスクを作成
            faces_mask = self._apply_extened_params(faces_mask)

            self.faces_mask_cache = faces_mask

        if faces_mask is None:
            faces_mask = self.faces_mask_cache

        return faces_mask if faces_mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_face(self, image_size, exclude_names):
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        return mask2_inference_runtime.predict_face_mask(
            self.editor.get_original_image_rgb(), exclude_names
        )

    @staticmethod
    def delete_faces():
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        mask2_inference_runtime.delete_faces()

# セグメントマスクのクラス
class TargetTextMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Target Text"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None
        
        self.target_text = ""

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate(*self.center)
            KVPopMatrix()

    def on_touch_down(self, touch):
        if self.initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else:
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            #self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            
            # text inout dialog
            dialog = TextInputDialog(self.on_text_entered)
            dialog.open()
            
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        # 中心のコントロールポイント（始点）
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def on_text_entered(self, text):
        self.target_text = text
        self.initializing = False
        
        self.update_mask()
        
        self.editor._create_end_new_mask()
        self.editor.created_mask = None

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.TARGET_TEXT,
            'name': self.name,
            'center': [cx, cy],
            'target_text': self.target_text,
            'effects_param': param
        }
        # マスクデータ保存
        if self.image_mask_cache is not None:
            dict['image_mask_cache'] = utils.convert_image_to_list(self.image_mask_cache)
            dict['image_mask_cache_hash'] = self.image_mask_cache_hash

        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.target_text = dict.get('target_text', "All")
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        # マスクデータ展開
        self.image_mask_cache = dict.get('image_mask_cache', None)
        if self.image_mask_cache is not None:
            self.image_mask_cache = utils.convert_image_from_list(self.image_mask_cache)
            self.image_mask_cache_hash = dict.get('image_mask_cache_hash', None)

        # 描き直し
        self.create_control_points()
        #self.update_mask()     

    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_original_image(*self.center)
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
            invert = effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')
        else:
            invert = False
        text = self.target_text
        segment_mask = None

        # _draw_segmentを呼び出さなければならない用
        newhash = hash((image_size, text))
        if (self.image_mask_cache_hash != newhash) and self.initializing == False:
            self.image_mask_cache_hash = newhash
            
            # predict_sam3 に渡す box = [x, y, w, h]
            segment_mask = wait_prosessing(self._draw_segment, image_size, text, invert)
            #segment_mask = self._draw_segment(image_size, text)

            # SegmentMask用のキャッシュ
            self.image_mask_cache = segment_mask

        # その他更新用
        newhash = hash((self.get_hash_items(), self.editor.get_hash_items()))
        if self.image_mask_cache is not None and (self.image_mask_cache is segment_mask or self.segment_mask_cache is None or self.segment_mask_cache_hash != newhash) and self.initializing == False:
            self.segment_mask_cache_hash = newhash

            # SegmentMask用のキャッシュ
            segment_mask = self.image_mask_cache

            # パラメータに従って画像を変形
            disp_info, rotate_rad, flip, matrix = self.editor.get_hash_items()
            segment_mask = core.rotation(segment_mask, np.rad2deg(rotate_rad), flip, np.array(matrix).reshape(3, 3))
            #segment_mask = core.crop_image_with_disp_info(segment_mask, disp_info)

            nw, nh, ox, oy = core.crop_size_and_offset_from_texture(*self.editor.texture_size, disp_info)
            cx, cy ,cw, ch, scale = disp_info
            #cx, cy, cw, ch = int(cx * scale), int(cy * scale), int(cw * scale), int(ch * scale)
            segment_mask = cv2.resize(segment_mask[cy:cy+ch, cx:cx+cw], (nw, nh))
            segment_mask = np.pad(segment_mask, ((oy, self.editor.texture_size[1]-(oy+nh)), (ox, self.editor.texture_size[0]-(ox+nw))), constant_values=0)

            # ルミノシティマスクを作成
            segment_mask = self._apply_extened_params(segment_mask)

            self.segment_mask_cache = segment_mask

        if segment_mask is None:
            segment_mask = self.segment_mask_cache

        return segment_mask if segment_mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def _draw_segment(self, image_size, text, invert):
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        img = self.editor.get_original_image_rgb()
        return mask2_inference_runtime.predict_sam3_text(img, text, invert)


# メインのエディタークラス
class MaskEditor2(KVFloatLayout, LayerCtrl):
    mask_list = KVListProperty([])
    active_mask = KVObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event_type('on_structure_change')

        self.mask_container = KVWidget()
        self.add_widget(self.mask_container)
        self.rectangle = None

        self.created_mask = None
        self.texture_size = (0, 0)

        self.crop_image_rgb = None
        self.crop_image_hls = None
        self.original_image_rgb = None
        self.original_image_hls = None

        logging.info("MaskEditor: 初期化完了")

    def on_structure_change(self, *args):
        pass

    # 終了処理
    def end(self):
        if self.active_mask is not None:
            self.active_mask.end()

    def push_scissor(self):
        scissor = KVScissorPush()
        self.set_scissor(scissor)
        return scissor

    def set_scissor(self, scissor):
        scissor.x = int(self.pos[0])
        scissor.y = int(self.pos[1])
        scissor.width = int(self.size[0])
        scissor.height = int(self.size[1])

    def pop_scissor(self):
        KVScissorPop()
    
    def set_ref_image(self, crop_image, original_image=None):
        if self.crop_image_rgb is not crop_image:
            self.crop_image_rgb = crop_image
            self.crop_image_hls = None

        if self.original_image_rgb is not original_image:
            self.original_image_rgb = original_image
            self.original_image_hls = None

    def get_crop_image_hls(self):
        if self.crop_image_hls is None and self.crop_image_rgb is not None:
            self.crop_image_hls = hlsrgb.rgb_to_hlc_gain(self.crop_image_rgb)
            self.crop_image_rgb = None
        return self.crop_image_hls

    def get_original_image_rgb(self):
        return self.original_image_rgb

    def get_original_image_hls(self):
        if self.original_image_hls is None and self.original_image_rgb is not None:
            self.original_image_hls = hlsrgb.rgb_to_hlc_gain(self.original_image_rgb)
        return self.original_image_hls

    def set_texture_size(self, tx, ty):
        self.texture_size = (tx, ty)

    def set_primary_param(self, primary_param, disp_info):

        # TCG情報を設定
        self.tcg_info = params.param_to_tcg_info(primary_param)
        params.set_disp_info(self.tcg_info, disp_info) # これだけ引数の値を設定

        self.__set_image_info()
        #self.update()

    def refresh_active_mask_overlay(self):
        mask = self.get_active_mask()
        if mask is not None and mask.is_draw_mask == True:
            mask.update_mask()

    def _get_mask_image_rect(self):
        scale = device.dpi_scale()
        px, py = self.to_window(*self.pos)
        marginx = (self.size[0] - self.texture_size[0] * scale) / 2
        marginy = (self.size[1] - self.texture_size[1] * scale) / 2
        return (px + marginx, py + marginy), (self.texture_size[0] * scale, self.texture_size[1] * scale)

    def reposition_mask_image(self):
        if self.rectangle is not None:
            self.rectangle.pos, self.rectangle.size = self._get_mask_image_rect()

    def get_hash_items(self):
        return (params.get_disp_info(self.tcg_info), self.tcg_info['rotation'] + self.tcg_info['rotation2'], self.tcg_info['flip_mode'], tuple(self.tcg_info['matrix'].flatten()))

    def __set_image_info(self):
        for mask in reversed(self.mask_list):
            #pass    # 無限ループ対策
            effects.reeffect_all(mask.effects)
        
    def update(self):
        KVClock.schedule_once(self._update, 0)

    def _update(self, dt=0):
        # 既存のマスクに対する更新を処理
        for mask in reversed(self.mask_list):
            mask.update()

    def serialize(self):
        list = []
        for mask in reversed(self.mask_list):
            parent = self.find_composit_mask(mask)
            if parent is not None and parent != mask:
                continue
            list.append(mask.serialize())
        if len(list) <= 0:
            return None

        dict = {
            'mask2': list,
        }
        return dict

    def deserialize(self, dict):
        list = dict['mask2']

        for dict in list:
            type = dict.get('type', None)
            mask = self._create_mask(type)
            mask.deserialize(dict)
            #mask.update()

        self.dispatch('on_structure_change')

    def is_center_click_anyone(self, touch, self_mask):
        for mask in reversed(self.mask_list):
            if mask != self_mask and mask.is_center_click(touch):
                return True
        return False

    def get_created_mask(self):
        return self.created_mask

    def get_active_mask(self):
        if self.disabled == True:
            return None

        return self.active_mask

    def commit_in_progress(self):
        """アクティブマスクが描画中の操作 (Polyline 描画など) を持っていれば確定させる。"""
        mask = self.active_mask
        if mask is None:
            return
        committer = getattr(mask, 'commit_in_progress', None)
        if callable(committer):
            try:
                committer()
            except Exception:
                logging.exception("MaskEditor: commit_in_progress 中に例外")
    
    def find_mask(self, mask_id):
        for mask in reversed(self.mask_list):
            if mask.mask_id == mask_id:
                return mask
        return None
        
    # LayerCtrl用
    def update_layer(self, op, index, op_type, dict):
        match op:
            case "Create":
                mask = self._create_mask(dict['type'], index, dict)
                # 通常マスクなら親にくっつける
                if op_type != "Composit":
                    # なんでもいいから親探す
                    composit_mask = self.find_composit_mask(mask, index)
                    if composit_mask is None:
                        logging.error("Composit mask not found")
                        assert False

                    # インデクスがコンポジットマスクの中の何番目かを調べる
                    composit_mask_index = 0
                    for i in range(index-1, -1, -1):
                        composit_mask = self.mask_list[i]
                        if composit_mask.is_composit():
                            composit_mask_index = index - 1 - i
                            break
                    composit_mask.add_mask(mask, op_type, composit_mask_index)
                self.set_active_mask(mask)
                mask.update_mask()

            case "Delete":
                self._remove_mask(self.get_mask(index))

            case "Update":
                mask = self.get_mask(index)
                mask.clear()
                mask.deserialize(dict)
                self.set_active_mask(mask)
                mask.update_mask()

            case _:
                logging.error("Invalid operation: " + op)
                assert False

        self.dispatch('on_structure_change')
    
    # LayerCtrl用
    def get_layer(self, index):
        return self.get_mask(index)
    
    def get_mask(self, index):
        return self.mask_list[index]
    
    def get_mask_list(self):
        return self.mask_list

    # mask2_content用
    def add_mask(self, mask_type, op_type, index):
        return self._create_start_new_mask(mask_type, op_type, index)

    # mask2_content用
    def add_composit_mask(self, instance):
        self._create_start_new_mask(MaskType.COMPOSIT, "Composit")
    
    # mask2_content用
    def del_mask(self, mask):
        index = self.get_mask_list().index(mask)
        is_composit = mask.is_composit()
        if is_composit:
            maskop = 'Composit'
        else:
            composit_mask = self.find_composit_mask(mask, index)
            if composit_mask:
                maskop = composit_mask.find_mask_op(mask)

        get_history_ctrl().begin_history_layer_ctrl(self, "Create", index, maskop)
        get_history_ctrl().end_history_layer_ctrl(self, "Delete", index)
        self._remove_mask(mask)

    def set_draw_mask(self, is_draw_mask):
        if is_draw_mask == False:
            if self.rectangle is not None:
                try:
                    self.mask_container.canvas.before.remove(self.rectangle)
                except:
                    pass
                self.rectangle = None
        mask = self.get_active_mask()
        if mask is not None:
            mask.is_draw_mask = is_draw_mask
            if is_draw_mask == True:
                mask.update()
    
    def start_draw_image(self):
        if self.root is not None:
            self.root.start_draw_image()

    def draw_mask_image(self, glayimg):
        if self.rectangle is not None:
            self.mask_container.canvas.before.remove(self.rectangle)
            self.rectangle = None

        if glayimg is not None:
            with self.mask_container.canvas.before:
                # マスクをアルファとして扱い、ルミナンスを白(1.0)にする
                glayimg = np.clip(glayimg, 0, 1)
                h, w = glayimg.shape[:2]
                la_img = np.empty((h, w, 2), dtype=np.float32)
                la_img[..., 0] = 1.0  # Luminance = White
                la_img[..., 1] = glayimg  # Alpha = Mask Value
                texture = KVTexture.create(size=(w, h), colorfmt='luminance_alpha', bufferfmt='float')
                texture.blit_buffer(la_img.tobytes(), colorfmt='luminance_alpha', bufferfmt='float')
                texture.flip_vertical()
                pos, size = self._get_mask_image_rect()
                KVColor(1, 0, 0, 0.4)
                self.rectangle = KVRectangle(texture=texture, pos=pos, size=size)

                # cv2.imwrite('combined_mask.png', (glayimg*255).astype(np.uint8))

    def _create_start_new_mask(self, type, op_type, index=0):
        # 画像サイズがまだ設定されていない場合、マスクの作成をスキップ

        mask = self._create_mask(type, index)
        self.set_active_mask(None)
        self.created_mask = mask
        if self.root is not None:
            self.root.update_mask2_options_enabled()

        # ここで履歴の更新を始める
        get_history_ctrl().begin_history_layer_ctrl(self, "Delete", self.get_mask_list().index(self.created_mask), op_type)
        
        # CompositMaskなど初期化が不要な場合は即座に終了処理を行う
        if mask.initializing == False:
            self._create_end_new_mask()
            self.created_mask = None
        
        return self.created_mask

    def _create_end_new_mask(self):
        self.set_active_mask(self.created_mask)
        
        # 履歴記録。 create_maskがレイヤーリストにある場合のみ
        if self.created_mask in self.get_mask_list():
            get_history_ctrl().end_history_layer_ctrl(self, "Create", self.get_mask_list().index(self.created_mask))

    def on_touch_down(self, touch):
        if self.disabled == True:
            return False
      
        # アクティブなマスクを先に処理
        if self.created_mask is not None:
            if self.created_mask.on_touch_down(touch):
                return True
        """    
        # 既存のマスクに対するタッチイベントを処理（新しい方から）
        for mask in self.mask_list:
            if mask.on_touch_down(touch):
                return True
        """
        return KVFloatLayout.on_touch_down(self, touch)
        
    def on_touch_up(self, touch):
        if self.disabled == True:
            return False
        
        result = KVFloatLayout.on_touch_up(self, touch)

        # こっちを後でやらないとまだコントロールポイントが作られてない
        if self.created_mask is not None:
            if self.created_mask.initializing == False:
                self._create_end_new_mask()        
                self.created_mask = None
        
        return result

    def _create_mask_object(self, mask_type):
        # マスクオブジェクト作成のみを行う
        match mask_type:
            case MaskType.CIRCULAR:
                mask = CircularGradientMask(editor=self)
            case MaskType.GRADIENT:
                mask = GradientMask(editor=self)
            case MaskType.FULL:
                mask = FullMask(editor=self)
            case MaskType.FREEDRAW:
                mask = FreeDrawMask(editor=self)
            case MaskType.POLYLINE:
                mask = PolylineMask(editor=self)
            case MaskType.SEGMENT:
                mask = SegmentMask(editor=self)
            case MaskType.DEPTHMAP:
                mask = DepthMapMask(editor=self)
            case MaskType.FACE:
                mask = FaceMask(editor=self)
            case MaskType.TARGET_TEXT:
                mask = TargetTextMask(editor=self)
            case MaskType.COMPOSIT:
                mask = CompositMask(editor=self)
            case _:
                logging.error(f"MaskEditor: 不明なマスクタイプ: {mask_type}")
                assert False

        return mask

    def _create_mask(self, mask_type, index=0, dict=None):
        # マスクオブジェクト作成
        mask = self._create_mask_object(mask_type)

        # コンテナに追加
        self.mask_container.add_widget(mask, index)
        self.mask_list.insert(index, mask)

        # デシリアライズ
        if dict is not None:
            mask.deserialize(dict)

        # パラメータをウィジェットに反映        
        if self.root is not None:
            self.root.set2widget_all(mask.effects, mask.effects_param)

        #self.dispatch('on_structure_change')
        return mask

    def _remove_mask(self, mask):
        # 削除する前にアクティブなものを移動する
        if len(self.mask_list) <= 1:
            self.draw_mask_image(None)
            self.set_active_mask(None)
        else:
            i = self.mask_list.index(mask)
            i = i+1 if i+1 < len(self.mask_list) else i-1
            self.set_active_mask(self.mask_list[i])

        # 親探す
        composit_mask = self.find_composit_mask(mask)
        if composit_mask is mask:
            # Compositなら子をすべて削除
            for child, _ in list(composit_mask.get_mask_list()):
                self._remove_mask(child)
            composit_mask.clear()
        elif composit_mask is not None:
            # Compositでないなら親から削除
            composit_mask.remove_mask(mask)
        else:
            logging.error(f"MaskEditor: 親が見つかりませんでした。マスクを削除できません。")
            assert False

        # コンテナから削除
        self.mask_container.remove_widget(mask)
        self.mask_list.remove(mask)

        # 再描画
        if self.active_mask:
            self.active_mask.update_mask()
        #self.dispatch('on_structure_change')

    def clear_mask(self):
        self.set_active_mask(None)
        self.draw_mask_image(None)
        self.mask_container.clear_widgets()
        self.mask_list.clear()
        FaceMask.delete_faces()
        self.dispatch('on_structure_change')

    def find_composit_mask(self, mask, index=0):
        # 自分の親（コンポジット）を探す
        if mask.is_composit():
            return mask     # 自分がコンポジット

        # 自分がコンポジットでない場合、コンポジットを探す
        for composit_mask in self.mask_list:
            if composit_mask.is_composit():
                if composit_mask.find_mask_op(mask) is not None:
                    return composit_mask
        
        # リスト内の直前の親にする
        for i in range(index-1, -1, -1):
            composit_mask = self.mask_list[i]
            if composit_mask.is_composit():
                return composit_mask
            
        return None

    def set_active_mask(self, mask):
        if self.active_mask is mask:
            return

        if self.active_mask is not None:
            self.active_mask.active = False
            self.active_mask.end()

        self.active_mask = mask
        if mask is not None:
            mask.active = True
            if mask.is_composit():
                # コンポジットなら通常属性のみ反映
                self.root.set2widget_all(mask.effects, mask.effects_param)
            else:
                # コンポジットでないならコンポジットの属性と合わせて反映
                composit_mask = self.find_composit_mask(mask)
                if composit_mask is not None:
                    marge_param = composit_mask.effects_param.copy()
                    marge_param.update(mask.effects_param)
                    self.root.set2widget_all(composit_mask.effects, marge_param)
                else:
                    logging.error(f"MaskEditor: 親が見つかりませんでした。マスクを反映できません。")

            mask.start()
            #mask.update()
        else:
            self.draw_mask_image(None)
            if self.root is not None:
                self.root.set2widget_all(None, None)

        self.start_draw_image()

        # Mask2パネルのON / OFF
        if self.root is not None:
            self.root.update_mask2_options_enabled()
 
    def get_rotate_rad(self, rotate_rad):
        # 画像の回転角度を取得する
        rad, flip = self.tcg_info['rotation2'], self.tcg_info['flip_mode']
        
        angle_rad = rotate_rad + rad
        match flip:
            case 0: # 0: normal
                pass
            case 1: # 1: horizontal flip
                angle_rad = -angle_rad
            case 2: # 2: vertical flip
                angle_rad = angle_rad + np.radians(90)
            case 3: # 3: horizontal and vertical flip
                angle_rad = angle_rad - np.radians(180)
        
        return self.tcg_info['rotation'] + angle_rad

    def get_image_size(self):
        return self.tcg_info['original_img_size']
    
    def window_to_tcg_scale(self, x, y):
        # ワールド座標にスケーリングだけ適用する
        return params.window_to_tcg_scale((x, y), self.tcg_info)
    
    def tcg_to_window_scale(self, x, y):
        # TCG座標にスケーリングだけ適用する
        return params.tcg_to_window_scale((x, y), self.tcg_info)

    def tcg_to_image_scale(self, x, y):
        # TCG座標にスケーリングだけ適用する
        return params.tcg_to_image_scale((x, y), self.tcg_info)

    def window_to_tcg(self, cx, cy):
        # ワールド座標からTCG座標に変換する
        cx, cy = params.window_to_tcg(cx, cy, self, self.texture_size, self.tcg_info, normalize=False)
        return (cx, cy)

    def tcg_to_window(self, cx, cy):
        # TCG座標をウィンドウ座標に変換する

        return params.tcg_to_window(cx, cy, self, self.texture_size, self.tcg_info, normalize=False)

    def tcg_to_texture(self, cx, cy):
        #cx, cy = cx * device.dpi_scale(), cy * device.dpi_scale()
        #return params.tcg_to_ref_image(cx, cy, self.original_image_rgb, self.tcg_info, apply_disp_info=True)
        # TCG座標をテクスチャ座標に変換する
        #cx, cy = cx * device.dpi_scale(), cy * device.dpi_scale()
        disp_info = params.get_disp_info(self.tcg_info)
        imax = max(self.tcg_info['original_img_size'][0]/2, self.tcg_info['original_img_size'][1]/2)
        cx, cy = params.center_rotate(cx, cy, self.tcg_info)
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - disp_info[0], cy - disp_info[1]
        cx, cy = cx * disp_info[4], cy * disp_info[4]        
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*self.texture_size, disp_info)
        cx, cy = cx + offset_x, cy + offset_y
        return (cx, cy)

    def tcg_to_full_image(self, cx, cy):
        # TCG座標をフル画像（pipeline0処理後画像）座標に変換する
        imax = max(self.tcg_info['original_img_size'][0]/2, self.tcg_info['original_img_size'][1]/2)
        cx, cy = params.center_rotate(cx, cy, self.tcg_info)
        cx, cy = cx + imax, cy + imax
        return (cx, cy)

    def tcg_to_crop_image(self, cx, cy):
        # TCG座標をクロップ（pipeline0処理後のクロップ画像）画像座標に変換する
        cx, cy = self.tcg_to_full_image(cx, cy)
        shape_max = max(self.original_image_rgb.shape[0], self.original_image_rgb.shape[1])
        cx = cx * (self.crop_image_hls.shape[1] / shape_max)
        cy = cy * (self.crop_image_hls.shape[0] / shape_max)
        return (cx, cy)

    def tcg_to_original_image(self, cx, cy):
        # 座標変換：TCG座標（回転後） -> Original座標（回転前）
        # 1. TCG座標は元画像の中心を原点とした、回転・反転のない座標系
        # なので、単に左上原点に戻すだけでよい
        h, w = self.get_original_image_rgb().shape[:2]
        cx, cy = cx + w * 0.5, cy + h * 0.5
        cx, cy = min(max(cx, 0), w), min(max(cy, 0), h) # クリップ (範囲外に出ないように)
        return (cx, cy)

# アプリケーションクラス
class MaskEditor2App(KVApp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main_widget = self

    def begin_history_layer_ctrl(self, layer_ctrl, op, index):
        pass

    def end_history_layer_ctrl(self, layer_ctrl, op, index):
        pass

    def build(self):
        # 画像ファイルのパスを正しく設定してください
        image_path = 'your_image.JPG'
        if not os.path.exists(image_path):
             image_path = 'your_image.jpg'

        # KVファイルをロード
        from kivy.lang import Builder as KVBuilder
        KVBuilder.load_file(os.path.join(os.path.dirname(__file__), 'mask2_content.kv'))

        box0 = KVBoxLayout(orientation='horizontal') # 全体を横並びに
        
        # エディタ部
        editor = MaskEditor2()
        box0.add_widget(editor)

        # サイドパネル部
        from widgets import mask2_content
        side_panel = mask2_content.create_mask2_content_panel(editor)
        # サイドパネルの幅を制限
        side_panel.size_hint_x = 0.3
        box0.add_widget(side_panel)

        KVClock.schedule_once(partial(editor.imread, image_path), 0.5)

        return box0

if __name__ == '__main__':
    MaskEditor2App().run()
