
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

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (
    NumericProperty, ObjectProperty, ListProperty,
    StringProperty, BooleanProperty, Property
)
from kivy.graphics import (
    Color, Ellipse, Line, PushMatrix, PopMatrix, Rotate, Translate,
    Rectangle, ScissorPush, ScissorPop,
)
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput

import cores.core as core
import cores.expand_mask as expand_mask
import cores.hlsrgb as hlsrgb
import params
import effects
import config
import utils.utils as utils
from processing_dialog import wait_prosessing
from history import LayerCtrl, get_history_ctrl
import macos as device
 
class TextInputDialog(Popup):
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.title = "Input Target Text in English"
        self.size_hint = (0.4, None)
        self.height = 240
        self.ref_height = 240
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.text_input = TextInput(multiline=False, size_hint_y=None, height=50)
        self.ref_height = 50
        self.text_input.bind(on_text_validate=lambda x: self.save(callback))
        
        btn_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=10)
        btn_layout.ref_height = 40
        save_button = Button(text='OK')
        save_button.bind(on_press=lambda x: self.save(callback))
        btn_layout.add_widget(save_button)
        
        layout.add_widget(self.text_input)
        layout.add_widget(btn_layout)
        self.content = layout

    def save(self, callback):
        text = self.text_input.text
        if not text or text.isspace():
            text = "All"
        self.dismiss()
        Clock.schedule_once(lambda dt: callback(text), 0.5)

class MaskType(str, Enum):
    COMPOSIT = 'composit'
    CIRCULAR = 'circular'
    GRADIENT = 'gradient'
    FULL = 'full'
    FREEDRAW = 'free_draw'
    SEGMENT = 'segment'
    DEPTHMAP = 'depth_map'
    FACE = 'face'
    TARGET_TEXT = 'target_text'

# コントロールポイントのクラス
class ControlPoint(Widget):
    touching = BooleanProperty(False)
    is_center = BooleanProperty(False)  # 中心のコントロールポイントかどうか
    color = ListProperty([0, 0, 0])  # デフォルトの色
    ctrl_center = ListProperty([0, 0])
    type = ListProperty(['c', 0])

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        with self.canvas:
            PushMatrix()
            self.editor.push_scissor()
            self.translate = Translate()
            #self.rotate = Rotate(angle=0, origin=(0, 0))            
            self.color_instruction = Color(*self.color)
            self.circle = Ellipse(pos=(-10, -10), size=(20, 20))
            self.editor.pop_scissor()
            PopMatrix()
        self.center = (0, 0)
        #self.update_graphics()
        self.bind(center=self.update_graphics, color=self.update_color)

    def update_graphics(self, *args):
        cx, cy = self.editor.tcg_to_window(self.center_x, self.center_y)
        self.translate.x = cx
        self.translate.y = cy
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
class BaseMask(Widget):
    color = ListProperty([1, 0, 0, 0.5])  # デフォルトの半透明赤色
    selected = BooleanProperty(False)
    active = BooleanProperty(False)
    name = StringProperty("Mask")
    mask_id = StringProperty(str(uuid.uuid4()))

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
            #cp_center.ctrl_center[0] += float(np.finfo(np.float32).eps)
            #cp_center.ctrl_center[0] -= float(np.finfo(np.float32).eps)

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
        return (effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_open_space'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_close_space'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_min'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_max'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_blur'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_hue_distance'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_hue_min'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_hue_max'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_lum_distance'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_lum_min'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_lum_max'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_sat_distance',),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_sat_min'),
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_sat_max'))

    def _apply_mask_space(self, image):
        open_space = effects.Mask2Effect.get_param(self.effects_param, 'mask2_open_space')
        image = expand_mask.adjust_foreground_only(image, open_space * params.get_disp_info(self.editor.tcg_info)[4], False)

        close_space = effects.Mask2Effect.get_param(self.effects_param, 'mask2_close_space')
        image = expand_mask.adjust_holes_only(image, close_space * params.get_disp_info(self.editor.tcg_info)[4], False)
        
        return image

    def _apply_depth_mask(self, image):
        dmin = effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_min') / 255
        dmax = effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_max') / 255
        if (dmin != 0) or (1 != dmax):
            dimg = np.where((image < dmin) | (dmax < image), 0, image)
        else:
            dimg = image

        return dimg
    
    def _apply_mask_blur(self, image):
        blur = effects.Mask2Effect.get_param(self.effects_param, 'mask2_blur')
        if blur != 0:
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
        return self._draw_hls_mask(mask, 'hue')

    def _draw_lum_mask(self, mask):
        return self._draw_hls_mask(mask, 'lum')

    def _draw_sat_mask(self, mask):
        return self._draw_hls_mask(mask, 'sat')

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

        for mask, maskop in reversed(self.mask_list):
            mimage = mask.get_mask_image()
            match(maskop):
                case 'Add':
                    composit = np.clip(composit + mimage, 0, 1)
                case 'Subtract':
                    composit = np.clip(composit - mimage, 0, 1)
                case _:
                    logger.error(f"Unknown mask operation: {maskop}")
                    assert False
                    
        return composit


# 円形グラデーションマスクのクラス
class CircularGradientMask(BaseMask):
    inner_radius_x = NumericProperty(0)
    inner_radius_y = NumericProperty(0)
    outer_radius_x = NumericProperty(0)
    outer_radius_y = NumericProperty(0)
    rotate_rad = NumericProperty(0)

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Circle"
        self.initializing = True  # 初期配置中かどうか

        with self.canvas:
            PushMatrix()
            self.editor.push_scissor()
            self.translate = Translate(*self.center)
            self.rotate = Rotate(angle=0, origin=(0, 0))
            Color(*self.color)
            self.outer_line = Line(ellipse=(0, 0, 0, 0), width=2) # 外側の円
            self.inner_line = Line(ellipse=(0, 0, 0, 0), width=2) # 内側の円
            self.editor.pop_scissor()
            PopMatrix()

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

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:            
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
            self.rotate.angle = math.degrees(self.editor.get_rotate_rad(self.rotate_rad))
            ix, iy = self.editor.tcg_to_window_scale(self.inner_radius_x, self.inner_radius_y)
            self.inner_line.ellipse = (-ix, -iy, ix*2, iy*2)
            ox, oy = self.editor.tcg_to_window_scale(self.outer_radius_x, self.outer_radius_y)
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
        inner_axes = self.editor.tcg_to_window_scale(self.inner_radius_x, self.inner_radius_y)
        outer_axes = self.editor.tcg_to_window_scale(self.outer_radius_x, self.outer_radius_y)
        rotate_rad = self.editor.get_rotate_rad(self.rotate_rad)
        invert = not effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')

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

        width, height = image_size
        
        # 0. 極小サイズのチェック
        if width <= 0 or height <= 0:
            return np.zeros((height, width), dtype=np.float32)

        # 回転方向の修正 (以前のロジックと合わせるため反転)
        angle_rad = -angle_rad

        # 1. パラメータ計算 (Radius)
        rx_in, ry_in = inner_axes
        rx_out, ry_out = outer_axes
        
        # 中間の半径（ステップ位置）
        rx_mid = (rx_in + rx_out) / 2.0
        ry_mid = (ry_in + ry_out) / 2.0
        
        # Sigma計算 (距離の1/4程度がErfの自然な遷移幅)
        sigma_x = abs(rx_out - rx_in) * 0.25 * smoothness
        sigma_y = abs(ry_out - ry_in) * 0.25 * smoothness
        
        # Sigmaが小さすぎる場合はブラーなし
        if sigma_x < 0.1 and sigma_y < 0.1:
            sigma_x = 0.1
            sigma_y = 0.1

        # ダウンサンプリングスケールの計算 (Adaptive Downscaling for Warp Destination)
        # Warp処理後の出力サイズを小さくすることで、WarpAffineの負荷を下げる
        # ジッターを防ぐため、ターゲットSigmaを少し大きめ(4.0)に設定
        target_sigma = 4.0
        
        # Sigmaの小さい方に合わせて全体をスケールする（安全策）
        min_sigma = min(sigma_x, sigma_y)
        dest_scale = target_sigma / min_sigma if min_sigma > target_sigma else 1.0
        dest_scale = min(dest_scale, 1.0)
        
        # 最終出力（Full Res）へのWarp行列を計算してから、Scaleを適用する方が簡単
        
        # 2. 必要なキャンバスサイズの計算 (Rectified Space - Unrotated)
        # 最終画像の四隅を逆変換して、未回転空間でのバウンディングボックスを求める
        corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        
        corners_centered = corners - center
        x_rot = corners_centered[:, 0] * cos_a - corners_centered[:, 1] * sin_a
        y_rot = corners_centered[:, 0] * sin_a + corners_centered[:, 1] * cos_a
        
        # Unrotated空間でのBounding Box
        min_x = np.min(x_rot)
        max_x = np.max(x_rot)
        min_y = np.min(y_rot)
        max_y = np.max(y_rot)
        
        # ソースキャンバス（Unrotated）の解像度
        # ここも sigma に応じて小さくても良いが、Blurによる劣化を防ぐため
        # Unrotated空間では「Warp後のScale」と同程度か、あるいは Blur自体を行うのでここでもDownscale可能
        # 今回は Warp先が小さいので、ソースもそれに準じた解像度で十分
        
        # ソース側の解像度も dest_scale に合わせる
        # Sigmaもスケールされる
        src_scale = dest_scale
        eff_sigma_x = sigma_x * src_scale
        eff_sigma_y = sigma_y * src_scale
        
        # Padding
        pad_x = int(math.ceil(3.0 * eff_sigma_x))
        pad_y = int(math.ceil(3.0 * eff_sigma_y))

        # Unrotated空間での座標 (Full Res)
        unrot_w = max_x - min_x
        unrot_h = max_y - min_y
        
        # ソース画像サイズ (Scaled)
        src_w = int(math.ceil(unrot_w * src_scale)) + 2 * pad_x
        src_h = int(math.ceil(unrot_h * src_scale)) + 2 * pad_y
        
        # ソース画像の原点オフセット (Scaled coords)
        # src_imgの(0,0) は Unrotated空間の (min_x * s - pad, min_y * s - pad)
        src_origin_x = min_x * src_scale - pad_x
        src_origin_y = min_y * src_scale - pad_y
        
        # 楕円中心 (Scaled coords, relative to src_img origin)
        # Unrotated Center is (0,0). Scaled is (0,0).
        # In src_img: (0 - src_origin_x, 0 - src_origin_y)
        ell_cx = -src_origin_x
        ell_cy = -src_origin_y
        
        # ソース画像作成
        src_img = np.zeros((src_h, src_w), dtype=np.float32)
        
        if invert == False:
            bg_color = 1.0
            fg_color = 0.0
        else:
            bg_color = 0.0
            fg_color = 1.0
            
        src_img.fill(bg_color)
        
        # 楕円描画 (Scaled)
        cv2.ellipse(src_img, (int(ell_cx), int(ell_cy)), 
                    (int(rx_mid * src_scale), int(ry_mid * src_scale)), 
                    0, 0, 360, color=fg_color, thickness=-1)

        # ガウシアンブラー (Scaled Anisotropic)
        src_img = cv2.GaussianBlur(src_img, (0, 0), sigmaX=eff_sigma_x, sigmaY=eff_sigma_y)
        
        # 5. Warp to Downscaled Destination
        dest_w = int(width * dest_scale)
        dest_h = int(height * dest_scale)
        
        if dest_w <= 0 or dest_h <= 0:
             return np.zeros((height, width), dtype=np.float32)
        
        # Matrix Construction: Source(Scaled) -> Dest(Scaled)
        # Source Pixel (u,v) -> Dest Pixel (dx, dy)
        
        # Flow:
        # P_src(u,v) 
        # -> Unrotated_Full(X, Y) = (u + src_origin_x)/src_scale ? No. 
        #    P_src coords are Scaled.
        #    Unrotated_Scaled = (u + src_origin_x, v + src_origin_y)
        #    Unrotated_Full = Unrotated_Scaled / src_scale
        # -> Rotated_Full = Rotate(angle) * Unrotated_Full
        # -> Dest_Full = Rotated_Full + Center
        # -> Dest_Scaled = Dest_Full * dest_scale
        
        # Since src_scale == dest_scale (we chose them same for simplicity):
        # Dest_Scaled = (Rotate(angle) * (Unrotated_Scaled / s) + Center) * s
        #             = Rotate(angle) * Unrotated_Scaled + Center * s
        # Dest_Scaled = Rotate(angle) * (P_src + Origin_Scaled) + Center * s
        
        # M = R(a) * T(Origin_Scaled) + shift(Center*s)
        
        cos_v = np.cos(angle_rad)
        sin_v = np.sin(angle_rad)
        
        # R terms (Applied to P_src)
        a00 = cos_v
        a01 = -sin_v
        a10 = sin_v
        a11 = cos_v
        
        # Translation
        # R * Origin
        ox = src_origin_x
        oy = src_origin_y
        
        # Center * scale
        cx = center[0] * dest_scale
        cy = center[1] * dest_scale
        
        tx = ox * cos_v - oy * sin_v + cx
        ty = ox * sin_v + oy * cos_v + cy
        
        M = np.array([
            [a00, a01, tx],
            [a10, a11, ty]
        ], dtype=np.float32)

        border_val = bg_color
        
        # Warp to Small Destination
        dst_small = cv2.warpAffine(src_img, M, (dest_w, dest_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(border_val))
        
        # 6. Setup Final Result
        if dest_scale < 1.0:
            # Upscale
            dst_img = cv2.resize(dst_small, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            dst_img = dst_small

        return dst_img
    
# GradientMask クラス
class GradientMask(BaseMask):
    start_point = ListProperty([0, 0])    # グラデーションの開始点
    end_point = ListProperty([0, 0])      # グラデーションの終点
    
    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Line"
        self.initializing = True  # 初期配置中かどうか

        with self.canvas:
            PushMatrix()
            self.editor.push_scissor()
            self.translate = Translate(*self.center)
            self.rotate = Rotate(angle=0, origin=(0, 0))
            Color(*self.color)
            self.start_line = Line(points=(0, 0, 0, 0), width=2)
            self.center_line = Line(points=(0, 0, 0, 0), width=2)
            self.end_line = Line(points=(0, 0, 0, 0), width=2)
            self.editor.pop_scissor()
            PopMatrix()

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

        width, height = image_size
        
        # ベクトル計算
        start_x, start_y = end_point # Swap to match gradient direction
        end_x, end_y = start_point
        vec_start_end = np.array([end_x - start_x, end_y - start_y])
        length_start_end = np.linalg.norm(vec_start_end)
        
        if length_start_end == 0:
            return np.zeros((height, width), dtype=np.float32)

        # Sigma計算 (距離の1/4程度)
        sigma = (length_start_end * 0.25) * smoothness
        if sigma < 0.1:
            # Hard Edge
            img = np.zeros((height, width), dtype=np.float32)
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            unit_vec = vec_start_end / length_start_end
            y_coords, x_coords = np.indices((height, width))
            projected = (x_coords - mid_x) * unit_vec[0] + (y_coords - mid_y) * unit_vec[1]
            img[projected >= 0] = 1.0
            return img

        # Downscaling Strategy
        # ターゲットSigma (4.0 ~ 5.0) になるように縮小
        target_sigma = 4.0
        scale = target_sigma / sigma if sigma > target_sigma else 1.0
        scale = min(scale, 1.0)
        
        small_w = int(math.ceil(width * scale))
        small_h = int(math.ceil(height * scale))
        
        if small_w <= 0 or small_h <= 0:
             return np.zeros((height, width), dtype=np.float32)

        # Small Image Generation
        img_small = np.zeros((small_h, small_w), dtype=np.float32)
        
        # Scale Points
        start_x_s = start_x * scale
        start_y_s = start_y * scale
        end_x_s = end_x * scale
        end_y_s = end_y * scale
        mid_x_s = (start_x_s + end_x_s) / 2
        mid_y_s = (start_y_s + end_y_s) / 2
        
        vec_s = np.array([end_x_s - start_x_s, end_y_s - start_y_s])
        len_s = np.linalg.norm(vec_s)
        if len_s == 0:
             return np.zeros((height, width), dtype=np.float32)
        unit_vec_s = vec_s / len_s
        
        y_coords_s, x_coords_s = np.indices((small_h, small_w))
        projected_s = (x_coords_s - mid_x_s) * unit_vec_s[0] + (y_coords_s - mid_y_s) * unit_vec_s[1]
        
        img_small[projected_s >= 0] = 1.0
        
        # Blur (Sigma is scaled)
        eff_sigma = sigma * scale
        img_small = cv2.GaussianBlur(img_small, (0, 0), sigmaX=eff_sigma, sigmaY=eff_sigma)
        
        # Upscale
        if scale < 1.0:
            img = cv2.resize(img_small, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            img = img_small

        return img

# 全体マスクのクラス
class FullMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Full"
        self.initializing = True  # 初期配置中かどうか

        self.center = (0, 0)

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            PopMatrix()

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
        def __init__(self, is_eracing=False, size=10, soft=1.5, **kwargs):
            self.size = size
            self.soft = soft
            self.points = []
            self.is_erasing = is_eracing

        def add_point(self, x, y):
            self.points.append((x, y))

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Draw"
        self.initializing = True

        self.lines = []  # 複数の線を保持
        self.current_line = None
        self.brush_size = 100

        with self.canvas:
            PushMatrix()
            self.editor.push_scissor()
            self.translate = Translate(0, 0)
            self.rotate = Rotate(angle=0, origin=(0, 0))
            self.brush_color = Color((0, 1, 1, 1))
            self.brush_cursor = Line(ellipse=(0, 0, self.brush_size, self.brush_size), width=2)
            self.editor.pop_scissor()
            PopMatrix()

    def start(self):
        Window.bind(mouse_pos=self.on_mouse_pos)

    def end(self):
        self.brush_color.rgba = (0, 0, 0, 0)
        Window.unbind(mouse_pos=self.on_mouse_pos)

    def clear(self):
        self.lines = []
        self.current_line = None
        super().clear()

    def serialize(self):
        """マスクの状態をシリアライズ"""
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        
        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.FREEDRAW,
            'name': self.name,
            'center': [cx, cy],
            'lines': copy.deepcopy(self.lines),
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']
        self.lines = dict['lines']
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))

        self.create_control_points()

    def on_mouse_pos(self, window, pos):
        self.update_brush_cursor(pos[0], pos[1])

    def on_touch_down(self, touch):
        if self.editor.get_active_mask() != self and self.editor.get_created_mask() != self:
            return super().on_touch_down(touch)
        
        if self.editor.is_center_click_anyone(touch, self):
            return False

        if touch.is_mouse_scrolling:
            if self.editor.collide_point(*touch.pos):
                # 描画中または消去中はブラシサイズを変更できない
                if self.current_line is None:
                    if touch.button == 'scrolldown':
                        self.brush_size = max(10, self.brush_size - 10)
                    elif touch.button == 'scrollup':
                        self.brush_size = min(100, self.brush_size + 10)
                        
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
        cx, cy = self.editor.window_to_tcg(*touch.pos)
        self.current_line = FreeDrawMask.Line(is_erasing, self.brush_size)
        self.current_line.add_point(cx, cy)
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
        if self.current_line is not None:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.current_line.add_point(cx, cy)

            self.update_mask()
            self.editor.start_draw_image()        

        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.current_line is not None:
            self.current_line = None
            # マスクを更新
            self.update_mask()
            self.editor.start_draw_image()        
        
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
        self.translate.x, self.translate.y = x - self.brush_size / 2, y - self.brush_size / 2
        self.brush_cursor.ellipse = (0, 0, self.brush_size, self.brush_size)

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            return
        
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
        nline = len(self.lines)
        npoint = 0
        for line in self.lines:
            npoint += len(line.points)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, nline, npoint))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
             
            mask = self.draw_line(image_size, self.lines)

            # ルミナンスとマスクを作成
            mask = self._apply_extened_params(mask)

            self.image_mask_cache = mask
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    def create_natural_brush(self, size, softness=1.2):
        """自然なブラシを作成"""
        brush_size = int(size * 2)  # 直径
        brush_radius = size
        center = (brush_size // 2, brush_size // 2)
        
        # 基本の円形ブラシ
        y, x = np.ogrid[:brush_size, :brush_size]
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # ガウシアンっぽい自然な減衰
        brush = np.zeros((brush_size, brush_size), dtype=np.float32)
        
        # 中心から外側への自然な減衰
        mask = distances <= brush_radius
        normalized_dist = distances / brush_radius
        
        # より自然なフォールオフ（ガウシアン＋べき乗の組み合わせ）
        intensity = np.exp(-2.0 * normalized_dist**2)  # ガウシアン成分
        intensity *= (1 - normalized_dist**(1/softness))  # ソフトエッジ成分
        intensity = np.maximum(0, intensity)
        
        brush[mask] = intensity[mask]
        return brush
    
    def safe_array_slice(self, array, y_min, y_max, x_min, x_max):
        """安全な配列スライス（境界チェック付き）"""
        h, w = array.shape[:2]
        
        # 境界を画像サイズに制限
        y_min = max(0, min(h-1, y_min))
        y_max = max(y_min+1, min(h, y_max))
        x_min = max(0, min(w-1, x_min))
        x_max = max(x_min+1, min(w, x_max))
        
        return array[y_min:y_max, x_min:x_max], (y_min, y_max, x_min, x_max)
    
    def apply_brush_at_point(self, image, x, y, brush, is_erasing=False, opacity=1.0):
        """指定位置にブラシを適用（安全な境界チェック付き）"""
        if brush.size == 0:
            return
            
        brush_h, brush_w = brush.shape
        brush_center_x, brush_center_y = brush_w // 2, brush_h // 2
        
        # 画像上の適用範囲を計算
        img_y_min = int(y - brush_center_y)
        img_y_max = int(y - brush_center_y + brush_h)
        img_x_min = int(x - brush_center_x)
        img_x_max = int(x - brush_center_x + brush_w)
        
        # 画像の境界内に制限
        img_h, img_w = image.shape
        img_y_min_clipped = max(0, img_y_min)
        img_y_max_clipped = min(img_h, img_y_max)
        img_x_min_clipped = max(0, img_x_min)
        img_x_max_clipped = min(img_w, img_x_max)
        
        # 適用範囲が有効かチェック
        if (img_y_min_clipped >= img_y_max_clipped or 
            img_x_min_clipped >= img_x_max_clipped):
            return
        
        # ブラシの対応部分を計算
        brush_y_min = img_y_min_clipped - img_y_min
        brush_y_max = brush_y_min + (img_y_max_clipped - img_y_min_clipped)
        brush_x_min = img_x_min_clipped - img_x_min
        brush_x_max = brush_x_min + (img_x_max_clipped - img_x_min_clipped)
        
        # 境界チェック
        brush_y_min = max(0, min(brush_h-1, brush_y_min))
        brush_y_max = max(brush_y_min+1, min(brush_h, brush_y_max))
        brush_x_min = max(0, min(brush_w-1, brush_x_min))
        brush_x_max = max(brush_x_min+1, min(brush_w, brush_x_max))
        
        try:
            # ブラシ部分を取得
            brush_part = brush[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
            if brush_part.size == 0:
                return
                
            # 不透明度を適用
            brush_part = brush_part * opacity
            
            # 画像に適用
            target_region = image[img_y_min_clipped:img_y_max_clipped, 
                                img_x_min_clipped:img_x_max_clipped]
            
            if is_erasing:
                # 消しゴムモード
                image[img_y_min_clipped:img_y_max_clipped, 
                     img_x_min_clipped:img_x_max_clipped] = np.maximum(0, target_region - brush_part)
            else:
                # 描画モード
                image[img_y_min_clipped:img_y_max_clipped, 
                     img_x_min_clipped:img_x_max_clipped] = np.minimum(1, target_region + brush_part)
        except (IndexError, ValueError) as e:
            # エラーが発生した場合は無視して続行
            pass
    
    def draw_smooth_line(self, image, points, brush_size, softness, is_erasing=False):
        """滑らかな線を描画"""
        if len(points) == 0:
            return
        
        # ブラシを作成
        brush = self.create_natural_brush(brush_size / 2, softness)
        
        # 単一点の場合
        if len(points) == 1:
            p = self.editor.tcg_to_texture(*points[0])
            self.apply_brush_at_point(image, int(p[0]), int(p[1]), brush, is_erasing)
            return
        
        # 複数点の場合は補間して滑らかに
        texture_points = [self.editor.tcg_to_texture(*p) for p in points]
        
        for i in range(len(texture_points) - 1):
            p1 = texture_points[i]
            p2 = texture_points[i + 1]
            
            # 2点間の距離を計算
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # 補間点数を距離に応じて調整（密度を一定に保つ）
            steps = max(1, int(distance / (brush_size * 0.2)))
            
            for j in range(steps + 1):
                t = j / max(1, steps)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                
                # 速度に基づく不透明度調整（速く動かすと薄くなる）
                speed_factor = min(1.0, 10.0 / max(1.0, distance))
                opacity = 0.3 + 0.7 * speed_factor
                
                self.apply_brush_at_point(image, int(x), int(y), brush, is_erasing, opacity)
    
    def draw_line(self, image_size, lines):
        """改良された線描画メソッド"""
        try:
            # 画像の初期化（透明背景）
            width, height = image_size
            if width <= 0 or height <= 0:
                return np.zeros((100, 100), dtype=np.float32)
                
            image = np.zeros((height, width), dtype=np.float32)
            
            # 各線を描画
            for line in lines:
                if not hasattr(line, 'points') or len(line.points) == 0:
                    continue
                
                try:
                    # 線のパラメータを安全に取得
                    brush_size = getattr(line, 'size', 50)
                    brush_soft = getattr(line, 'soft', 1.2)
                    is_erasing = getattr(line, 'is_erasing', False)
                    
                    # パラメータの範囲チェック
                    brush_size = max(1, min(200, brush_size))
                    brush_soft = max(0.1, min(5.0, brush_soft))
                    
                    # 滑らかな線を描画
                    self.draw_smooth_line(image, line.points, brush_size, brush_soft, is_erasing)
                    
                except Exception as e:
                    # 個別の線でエラーが発生しても他の線は描画を続ける
                    continue
            
            return image
            
        except Exception as e:
            # 全体的なエラーの場合は空の画像を返す
            return np.zeros((max(1, image_size[1]), max(1, image_size[0])), dtype=np.float32)


# セグメントマスクのクラス
class SegmentMask(BaseMask):
    __processor = None
    corner = ListProperty([0, 0])

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Segment"
        self.initializing = True  # 初期配置中かどうか

        self.center = (0, 0)
        self.corner = (0, 0)

        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None

        with self.canvas:
            PushMatrix()
            # center位置への移動
            self.translate = Translate(0, 0)
            self.editor.push_scissor()
            Color(*self.color)
            self.rect_line = Line(points=[], close=True, width=2)
            self.editor.pop_scissor()
            PopMatrix()

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
        invert = effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')
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
            segment_mask = core.rotation(segment_mask, np.rad2deg(rotate_rad), flip)
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
        import helpers.sam3_helper as sam3_helper
        if SegmentMask.__processor is None:
            SegmentMask.__processor = sam3_helper.setup_sam3(config.get_config('gpu_device'))
        
        # 画像の取得
        img = self.editor.get_original_image_rgb()
        
        # バウンディングボックスの検証
        if bbox[0] == bbox[0] + bbox[2] or bbox[1] == bbox[1] + bbox[3]:
            return np.zeros((self.editor.texture_size[1], self.editor.texture_size[0]), dtype=np.float32)
        
        # 推論実行 (Original画像に対して)
        mask_original = sam3_helper.predict_sam3_for_bbox(SegmentMask.__processor, img, bbox)
        
        if invert:
            mask_original = 1 - mask_original   
        
        return mask_original

class DepthMapMask(BaseMask):
    __model = None

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Depth Map"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.depth_map_mask_cache = None
        self.depth_map_mask_cache_hash = None

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            PopMatrix()

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
            depth_map_mask = core.rotation(depth_map_mask, rotate_rad, flip)
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
        import depth_pro
        if DepthMapMask.__model is None:
            DepthMapMask.__model = depth_pro.setup_model(device=config.get_config('gpu_device'))

        # 画像の取得
        img = self.editor.get_original_image_rgb()

        mask = depth_pro.predict_model(DepthMapMask.__model, img)

        return mask

class FaceMask(BaseMask):
    __faces = None

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Face"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.faces_mask_cache = None
        self.faces_mask_cache_hash = None

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            PopMatrix()

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
            faces_mask = core.rotation(faces_mask, np.rad2deg(rotate_rad), flip)
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
        import helpers.facer_helper as facer_helper

        # 画像の取得
        img = self.editor.get_original_image_rgb()

        if FaceMask.__faces is None:
            FaceMask.__faces = facer_helper.create_faces(img, device='cpu')
        
        # マスク画像を作成
        if FaceMask.__faces == 0:
            return np.zeros((image_size[1], image_size[0]), dtype=np.float32)

        result = facer_helper.draw_face_mask(FaceMask.__faces, exclude_names)

        return result

    @staticmethod
    def delete_faces():
        if FaceMask.__faces is not None:
            FaceMask.__faces = None

# セグメントマスクのクラス
class TargetTextMask(BaseMask):
    __processor = None

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Target Text"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None
        
        self.target_text = ""

        with self.canvas:
            PushMatrix()
            self.translate = Translate(*self.center)
            PopMatrix()

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
        invert = effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')
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
            segment_mask = core.rotation(segment_mask, np.rad2deg(rotate_rad), flip)
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
        import helpers.sam3_helper as sam3_helper
        if TargetTextMask.__processor is None:
            TargetTextMask.__processor = sam3_helper.setup_sam3(config.get_config('gpu_device'))
        
        # 画像の取得
        img = self.editor.get_original_image_rgb()
        
        # 推論実行 (Original画像に対して)
        mask_original = sam3_helper.predict_sam3_for_text(TargetTextMask.__processor, img, text)
        
        if invert:
            mask_original = 1 - mask_original
        
        return mask_original


# メインのエディタークラス
class MaskEditor2(FloatLayout, LayerCtrl):
    mask_list = ListProperty([])
    active_mask = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event_type('on_structure_change')

        self.mask_container = Widget()
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
        ScissorPush(x=int(self.pos[0]), y=int(self.pos[1]), width=int(self.size[0]), height=int(self.size[1]))

    def pop_scissor(self):
        ScissorPop()
    
    def set_ref_image(self, crop_image, original_image=None):
        if self.crop_image_rgb is None:
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
        self.update()

    def get_hash_items(self):
        return (params.get_disp_info(self.tcg_info), self.tcg_info['rotation'] + self.tcg_info['rotation2'], self.tcg_info['flip_mode'], (0, 0)) # self.tcg_info['matrix'])

    def __set_image_info(self):
        for mask in reversed(self.mask_list):
            #pass    # 無限ループ対策
            effects.reeffect_all(mask.effects)
        
    def update(self):
        Clock.schedule_once(self._update, 0)

    def _update(self, dt=0):
        # 既存のマスクに対する更新を処理
        for mask in reversed(self.mask_list):
            #pass    # 無限ループ対策
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
            mask.update()

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
                h, w = glayimg.shape[:2]
                la_img = np.empty((h, w, 2), dtype=np.float32)
                la_img[..., 0] = 1.0  # Luminance = White
                la_img[..., 1] = glayimg  # Alpha = Mask Value
                texture = Texture.create(size=(w, h), colorfmt='luminance_alpha', bufferfmt='float')
                texture.blit_buffer(la_img.tobytes(), colorfmt='luminance_alpha', bufferfmt='float')
                texture.flip_vertical()
                px, py = self.to_window(*self.pos)
                scale = device.dpi_scale()
                marginx, marginy = (self.size[0]-self.texture_size[0]*scale)/2, (self.size[1]-self.texture_size[1]*scale)/2
                px, py = px+marginx, py+marginy
                Color(1, 0, 0, 0.4)
                self.rectangle = Rectangle(texture=texture, pos=(px, py), size=(self.texture_size[0]*scale, self.texture_size[1]*scale))

                # cv2.imwrite('combined_mask.png', (glayimg*255).astype(np.uint8))

    def _create_start_new_mask(self, type, op_type, index=0):
        # 画像サイズがまだ設定されていない場合、マスクの作成をスキップ
        
        mask = self._create_mask(type, index)
        self.set_active_mask(None)
        self.created_mask = mask

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
        return FloatLayout.on_touch_down(self, touch)
        
    def on_touch_up(self, touch):
        if self.disabled == True:
            return False
        
        result = FloatLayout.on_touch_up(self, touch)

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
            self.root.ids['mask2_panel'].disabled = self.active_mask is None or self.active_mask.is_composit()
 
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

    def window_to_tcg(self, cx, cy):
        # ワールド座標からTCG座標に変換する
        return params.window_to_tcg(cx, cy, self, self.texture_size, self.tcg_info, normalize=False)

    def tcg_to_window(self, cx, cy):
        # TCG座標をウィンドウ座標に変換する
        return params.tcg_to_window(cx, cy, self, self.texture_size, self.tcg_info, normalize=False)

    def tcg_to_texture(self, cx, cy):
        # TCG座標をテクスチャ座標に変換する
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
class MaskEditor2App(App):

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
        from kivy.lang import Builder
        Builder.load_file(os.path.join(os.path.dirname(__file__), 'mask2_content.kv'))

        box0 = BoxLayout(orientation='horizontal') # 全体を横並びに
        
        # エディタ部
        editor = MaskEditor2()
        box0.add_widget(editor)

        # サイドパネル部
        from widgets import mask2_content
        side_panel = mask2_content.create_mask2_content_panel(editor)
        # サイドパネルの幅を制限
        side_panel.size_hint_x = 0.3
        box0.add_widget(side_panel)

        Clock.schedule_once(partial(editor.imread, image_path), 0.5)

        return box0

if __name__ == '__main__':
    MaskEditor2App().run()
