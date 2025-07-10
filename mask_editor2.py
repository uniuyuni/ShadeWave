
import os
import numpy as np
import math
import cv2
import time

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
from kivy.logger import Logger
from functools import partial
import importlib

import core
import params
import effects
import facer_util
import expand_mask
import config
from processing_dialog import wait_prosessing

MASKTYPE_CIRCULAR = 'circular'
MASKTYPE_GRADIENT = 'gradient'
MASKTYPE_FULL = 'full'
MASKTYPE_FREEDRAW = 'free_draw'
MASKTYPE_SEGMENT = 'segment'
MASKTYPE_DEPTHMAP = 'depth_map'
MASKTYPE_FACE = 'face'
MASKTYPE_SCENE = 'scene'

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
        #self.size = self.editor.world_to_tcg_scale(20, 20) # sizeをセットすると何故かcenterの値がおかしくなるのでコメントアウト

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

# マスクのベースクラス
class BaseMask(Widget):
    color = ListProperty([1, 0, 0, 0.5])  # デフォルトの半透明赤色
    selected = BooleanProperty(False)
    active = BooleanProperty(False)
    name = StringProperty("Mask")

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor  # MaskEditorのインスタンスへの参照
        self.control_points = []  # 標準のPythonリストで管理
        self.bind(active=self.on_active_changed)

        # エフェクトパラメータ保持
        self.effects = effects.create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, self.editor.image_size)
        params.set_temperature_to_param(self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0)))

        self.is_draw_mask = True
        self.image_mask_cache = None
        self.image_mask_cache_hash = None

    def start(self):
        pass

    def end(self):
        pass

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

    def on_touch_down(self, touch):
        for cp in self.control_points:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            if cp.collide_point(cx, cy):
                if cp.is_center:
                    self.editor.set_active_mask(self)
                    cp.on_touch_down(touch)
                    self.is_draw_mask = True
                    return True
                elif self.active:
                    cp.on_touch_down(touch)
                    self.is_draw_mask = True
                    return True
        return False

    def on_touch_move(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_move(touch)
                self.is_draw_mask = True
                return True
        return False

    def on_touch_up(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_up(touch)
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
    
    def draw_mask_to_fbo(self):
        if not self.editor.disp_info:
            Logger.warning(f"{self.__class__.__name__}: disp_infoが未設定。")
            return

        if self.active == True:
            mask_image = self.get_mask_image()

            # イメージを描画してもらう
            self.editor.draw_mask_image(mask_image)

    def draw_hls_mask(self, image):
        simg = self.apply_mask_space(image)
        dimg = self.apply_depth_mask(simg)
        himg = self.draw_hue_mask(dimg)
        limg = self.draw_lum_mask(himg)
        result = self.draw_sat_mask(limg)
        
        return result

    def get_hash_items(self):
        return (self.effects_param.get('mask2_open_space', 0),
                self.effects_param.get('mask2_close_space', 0),
                self.effects_param.get('mask2_depth_min', 0),
                self.effects_param.get('mask2_depth_max', 255),
                self.effects_param.get('mask2_blur', 0),
                self.effects_param.get('mask2_hue_distance', 179),
                self.effects_param.get('mask2_hue_min', 0),
                self.effects_param.get('mask2_hue_max', 359),
                self.effects_param.get('mask2_lum_distance', 127),
                self.effects_param.get('mask2_lum_min', 0),
                self.effects_param.get('mask2_lum_max', 255),
                self.effects_param.get('mask2_sat_distance', 127),
                self.effects_param.get('mask2_sat_min', 0),
                self.effects_param.get('mask2_sat_max', 255))

    def apply_mask_space(self, image):
        open_space = self.effects_param.get('mask2_open_space', 0)
        image = expand_mask.adjust_foreground_only(image, open_space * self.editor.disp_info[4], False)

        close_space = self.effects_param.get('mask2_close_space', 0)
        image = expand_mask.adjust_holes_only(image, close_space * self.editor.disp_info[4], False)
        
        return image

    def apply_depth_mask(self, image):
        dmin = self.effects_param.get('mask2_depth_min', 0) / 255
        dmax = self.effects_param.get('mask2_depth_max', 255) / 255
        if (dmin != 0) or (1 != dmax):
            dimg = np.where((image < dmin) | (dmax < image), 0, image)
        else:
            dimg = image

        return dimg
    
    def apply_mask_blur(self, image):
        ksize = int(max(0, self.effects_param.get('mask2_blur', 0)*2-1))
        img2 = core.gaussian_blur_cv(image, (ksize, ksize))
        return img2

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

        if self.editor.full_image_hls is not None:            
            fimg = self.editor.full_image_hls[..., HLS_NUM[hls_str]]
            cimg = self.editor.crop_image_hls[..., HLS_NUM[hls_str]]
            dmax = HLS_DIS_MAX[hls_str]
            mmax = HLS_MAX[hls_str]
            
            ndis = self.effects_param.get(f'mask2_{hls_str}_distance', dmax)
            if ndis != dmax:
                cx, cy = self.editor.tcg_to_full_image(*self.center)
                print(f"point: {cx}, {cy}, {fimg[int(cy), int(cx)]}")
                center_n = fimg[int(cy), int(cx)]
                
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
            
            _min = self.effects_param.get(f'mask2_{hls_str}_min', 0)
            _max = self.effects_param.get(f'mask2_{hls_str}_max', mmax)
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

    def draw_hue_mask(self, mask):
        return self._draw_hls_mask(mask, 'hue')

    def draw_lum_mask(self, mask):
        return self._draw_hls_mask(mask, 'lum')

    def draw_sat_mask(self, mask):
        return self._draw_hls_mask(mask, 'sat')

# 円形グラデーションマスクのクラス
class CircularGradientMask(BaseMask):
    inner_radius_x = NumericProperty(0)
    inner_radius_y = NumericProperty(0)
    outer_radius_x = NumericProperty(0)
    outer_radius_y = NumericProperty(0)
    rotate_rad = NumericProperty(0)
    invert = NumericProperty(0)

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
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            if touch.is_double_tap and self.control_points[0].collide_point(cx, cy):
                self.invert = 1-self.invert
                self.update_control_points()
                self.update_mask()
                return True
            
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

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        ix, iy = params.norm_param(self.effects_param, (self.inner_radius_x, self.inner_radius_y))
        ox, oy = params.norm_param(self.effects_param, (self.outer_radius_x, self.outer_radius_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MASKTYPE_CIRCULAR,
            'name': self.name,
            'center': [cx, cy],
            'inner_radius': [ix, iy],
            'outer_radius': [ox, oy],
            'rotate_rad': self.rotate_rad,
            'invert': self.invert,
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
        self.invert = dict['invert']
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
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:            
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
            self.rotate.angle = math.degrees(self.editor.get_rotate_rad(self.rotate_rad))
            ix, iy = self.editor.tcg_to_world_scale(self.inner_radius_x, self.inner_radius_y)
            self.inner_line.ellipse = (-ix, -iy, ix*2, iy*2)
            ox, oy = self.editor.tcg_to_world_scale(self.outer_radius_x, self.outer_radius_y)
            self.outer_line.ellipse = (-ox, -oy, ox*2, oy*2)
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)
        inner_axes = self.editor.tcg_to_world_scale(self.inner_radius_x, self.inner_radius_y)
        outer_axes = self.editor.tcg_to_world_scale(self.outer_radius_x, self.outer_radius_y)
        rotate_rad = self.editor.get_rotate_rad(self.rotate_rad)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center, inner_axes, outer_axes, rotate_rad, self.invert))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:

            # グラデーションを描画
            gradient_image = self.draw_elliptical_gradient(image_size, center, inner_axes, outer_axes, rotate_rad, self.invert)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_elliptical_gradient(self, image_size, center, inner_axes, outer_axes, angle_rad, invert=0, smoothness=2.0):
    
        # 座標グリッドの作成
        y_indices, x_indices = np.indices((image_size[1], image_size[0]))
        x = x_indices - center[0]
        y = y_indices - center[1]

        # 回転角をラジアンに変換
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 座標の回転
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle

        # 内側と外側の楕円の値を計算
        e_inner = (x_rot / inner_axes[0])**2 + (y_rot / inner_axes[1])**2 - 1
        e_outer = (x_rot / outer_axes[0])**2 + (y_rot / outer_axes[1])**2 - 1

        # グラデーションの初期化
        gradient = np.ones((image_size[1], image_size[0]), dtype=np.float32)

        # 内側の楕円内のピクセルを設定
        gradient[e_inner <= 0] = 0.0

        # 外側の楕円の外側のピクセル
        #gradient[e_outer >= 0] = 1.0

        # 内側と外側の楕円の間のピクセルに対してグラデーションを計算
        mask_between = (e_inner > 0) & (e_outer <= 0)

        # グラデーション値の計算（線形補間）
        t = e_inner[mask_between] / (e_inner[mask_between] - e_outer[mask_between])
        #t = np.clip(t, 0.0, 1.0)

        # スムーズネスを適用
        t = np.power(t, smoothness)

        # グラデーションを適用
        gradient[mask_between] = t

        # 反転オプションの適用
        if invert == 0:
            gradient = 1.0 - gradient

        return gradient
    
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
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            if touch.is_double_tap and self.control_points[0].collide_point(cx, cy):
                self.start_point, self.end_point = self.end_point, self.start_point
                self.update_control_points()
                self.update_mask()
                return True
            
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
            'type': MASKTYPE_GRADIENT,
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
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
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
            self.draw_mask_to_fbo()
    
    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)
        start_point = self.editor.tcg_to_texture(*self.start_point)
        end_point = self.editor.tcg_to_texture(*self.end_point)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center, start_point, end_point))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            # グラデーションを描画
            gradient_image = self.draw_gradient(image_size, center, start_point, end_point)
            
            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    def draw_gradient(self, image_size, center, start_point, end_point, smoothness=1):

        width, height = image_size
        img = np.zeros((height, width), dtype=np.float32)  # 開始点前はすべて(0, 0, 0, 0)

        # ベクトル計算のための設定
        start_x, start_y = end_point # 開始点と入れ替える
        end_x, end_y = start_point   #
        vec_start_end = np.array([end_x - start_x, end_y - start_y])
        length_start_end = np.linalg.norm(vec_start_end)
        if length_start_end == 0:
            return img  # 開始点と終了点が同じ場合はそのまま透明を返す

        # 正規化した方向ベクトル
        unit_vec_start_end = vec_start_end / length_start_end

        # ピクセルの座標
        y_coords, x_coords = np.indices((height, width))
        pixel_vectors = np.stack((x_coords - start_x, y_coords - start_y), axis=-1)

        # 各ピクセルの射影を計算（開始点からの距離をグラデーションの方向に投影）
        projection_lengths = np.dot(pixel_vectors, unit_vec_start_end)

        # 射影の割合tを計算し、範囲を制限
        t = np.clip(projection_lengths / length_start_end, 0, 1)

        # グラデーションの急激さを調整（逆数のべき乗による非線形変換）
        t = t ** (1 / smoothness)

        # tが0より大きい場所にのみ色を設定（開始点以前は透明）
        mask = projection_lengths >= 0
        img[mask] = t[mask]

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
            'type': MASKTYPE_FULL,
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
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            # 描画
            gradient_image = self.draw_full(image_size, center)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

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

    def serialize(self):
        """マスクの状態をシリアライズ"""
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        
        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MASKTYPE_FREEDRAW,
            'name': self.name,
            'center': [cx, cy],
            'lines': self.lines,
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
            self.initializing = False
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
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            return
        
        self.rotate.angle = math.degrees(self.editor.get_rotate_rad(0))

        if self.is_draw_mask == True:
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
            mask = self.draw_hls_mask(mask)

            # マスクぼかし
            mask = self.apply_mask_blur(mask)

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
    __iopaint_plugins = None
    __iopaint_predict = None

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Segment"
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
            'type': MASKTYPE_SEGMENT,
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
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_full_image(*self.center)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            # 描画
            gradient_image = self.draw_segment(image_size, center)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
            
        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_segment(self, image_size, center):
        if SegmentMask.__iopaint_plugins is None:
            SegmentMask.__iopaint_plugins = importlib.import_module('iopaint.plugins')
        if SegmentMask.__iopaint_predict is None:
            SegmentMask.__iopaint_predict = importlib.import_module('iopaint.predict')
        
        img = self.editor.full_image_rgb
        result = SegmentMask.__iopaint_predict.predict_plugin(img, SegmentMask.__iopaint_plugins.InteractiveSeg.name, click=center)
        result = ((result > 0) * 1).astype(np.float32)

        nw, nh, ox, oy = core.crop_size_and_offset_from_texture(self.editor.texture_size[0], self.editor.texture_size[1], self.editor.disp_info)
        cx, cy ,cw, ch, scale = self.editor.disp_info
        result = cv2.resize(result[cy:cy+ch, cx:cx+cw], (nw, nh))
        result = np.pad(result, ((oy, self.editor.texture_size[0]-(oy+nh)), (ox, self.editor.texture_size[1]-(ox+nw))), constant_values=0)

        return result

class DepthMapMask(BaseMask):
    __depth_pro = None
    __depth_pro_mt = None

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Depth Map"
        self.initializing = True  # 初期配置中かどうか

        self.depth_map = None

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
            'type': MASKTYPE_DEPTHMAP,
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
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_full_image(*self.center)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
           # 描画
            gradient_image = self.draw_depth_map(image_size, center)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
            
        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_depth_map(self, image_size, center):
        if DepthMapMask.__depth_pro is None:
            DepthMapMask.__depth_pro = importlib.import_module('depth_pro')
            DepthMapMask.__depth_pro_mt = DepthMapMask.__depth_pro.setup_model(device=config.get_config('gpu_device'))

        if self.depth_map is None or self.editor.rotation_changed_flag:
            self.depth_map = DepthMapMask.__depth_pro.predict_model(DepthMapMask.__depth_pro_mt, self.editor.full_image_rgb)

        nw, nh, ox, oy = core.crop_size_and_offset_from_texture(self.editor.texture_size[0], self.editor.texture_size[1], self.editor.disp_info)
        cx, cy ,cw, ch, scale = self.editor.disp_info
        result = cv2.resize(self.depth_map[cy:cy+ch, cx:cx+cw], (nw, nh))
        result = np.pad(result, ((oy, self.editor.texture_size[0]-(oy+nh)), (ox, self.editor.texture_size[1]-(ox+nw))), constant_values=0)

        return result

class FaceMask(BaseMask):
    __faces = None

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Face"
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
            'type': MASKTYPE_FACE,
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
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_full_image(*self.center)
        exclude_names = []
        if self.effects_param.get('mask2_face_face', True) == False:
            exclude_names.append('face')
        if self.effects_param.get('mask2_face_brows', True) == False:
            exclude_names.extend(['rb', 'lb'])
        if self.effects_param.get('mask2_face_eyes', True) == False:
            exclude_names.extend(['re', 'le'])
        if self.effects_param.get('mask2_face_nose', True) == False:
            exclude_names.append('nose')
        if self.effects_param.get('mask2_face_mouth', True) == False:
            exclude_names.append('imouth')
        if self.effects_param.get('mask2_face_lips', True) == False:
            exclude_names.extend(['ulip', 'llip'])

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center, tuple(exclude_names)))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            # 描画
            gradient_image = self.draw_face(image_size, center, exclude_names)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
            
        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_face(self, image_size, center, exclude_names):
        if FaceMask.__faces is None or self.editor.rotation_changed_flag:
            FaceMask.__faces = facer_util.create_faces(self.editor.full_image_rgb, device='cpu')
        
        # マスク画像を作成
        if FaceMask.__faces == 0:
            return np.zeros((image_size[1], image_size[0]), dtype=np.float32)

        result = facer_util.draw_face_mask(FaceMask.__faces, exclude_names)

        nw, nh, ox, oy = core.crop_size_and_offset_from_texture(self.editor.texture_size[0], self.editor.texture_size[1], self.editor.disp_info)
        cx, cy ,cw, ch, scale = self.editor.disp_info
        result = cv2.resize(result[cy:cy+ch, cx:cx+cw], (nw, nh))
        result = np.pad(result, ((oy, self.editor.texture_size[0]-(oy+nh)), (ox, self.editor.texture_size[1]-(ox+nw))), constant_values=0)

        return result

    @staticmethod
    def delete_faces():
        if FaceMask.__faces is not None:
            FaceMask.__faces = None

# セグメントマスクのクラス
class SceneMask(BaseMask):
    __model = None

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Scene"
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
            'type': MASKTYPE_SCENE,
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
        if not self.editor or self.editor.image_size[0] == 0 or self.editor.image_size[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            Logger.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_full_image(*self.center)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            # 描画
            gradient_image = self.draw_scene(image_size, center)

            # ルミナんとマスクを作成
            gradient_image = self.draw_hls_mask(gradient_image)

            # マスクぼかし
            gradient_image = self.apply_mask_blur(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
            
        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)


    def draw_scene(self, image_size, center):

        def _process(full_image):
            import detectron2_helper

            if SceneMask.__model is None:
                config_file = 'detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml'
                checkpoint_file = 'checkpoints/model_final_c10459.pkl'
                #config_file = 'kmax_deeplab/configs/ade20k/panoptic_segmentation/kmax_convnext_large.yaml'
                #checkpoint_file = 'checkpoints/kmax_convnext_large_ade20k.pth'
                #config_file = 'kmax_deeplab/configs/coco/panoptic_segmentation/kmax_r50.yaml'
                #checkpoint_file = 'checkpoints/kmax_r50.pth'
                SceneMask.__model = detectron2_helper.setup_model(config_file, checkpoint_file, 'cpu', 0.35)
            
            height, width = full_image.shape[:2]
            img = cv2.resize(full_image, (width//2, height//2))
            result = detectron2_helper.run_inference(SceneMask.__model, img)
            result = detectron2_helper.create_mask(result)
            result = result * 0.5 + 0.5
            result = cv2.resize(result, (width, height))

            return result

        # 裏でやらせているつもり（ダイアログ表示あり）
        result = wait_prosessing(_process, self.editor.full_image_rgb)

        nw, nh, ox, oy = core.crop_size_and_offset_from_texture(self.editor.texture_size[0], self.editor.texture_size[1], self.editor.disp_info)
        cx, cy ,cw, ch, scale = self.editor.disp_info
        result = cv2.resize(result[cy:cy+ch, cx:cx+cw], (nw, nh))
        result = np.pad(result, ((oy, self.editor.texture_size[0]-(oy+nh)), (ox, self.editor.texture_size[1]-(ox+nw))), constant_values=0)

        return result

# マスクレイヤーの管理クラス
class MaskLayer(BoxLayout):
    mask = ObjectProperty(None)
    mask_name = StringProperty('')

    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.root = None

        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = 30
        self.mask = mask
        self.mask_name = mask.get_name()

        btn_delete = Button(text='Del', size_hint=(0.4, 1))
        btn_delete.bind(on_press=self.delete_mask)

        label = Button(text=self.mask_name, size_hint=(0.8, 1), background_normal='', background_color=(0,0,0,1))
        label.bind(on_press=self.set_active)

        self.add_widget(label)
        self.add_widget(btn_delete)

    def delete_mask(self, instance):
        self.parent.remove_widget(self)
        self.mask.parent.remove_widget(self.mask)
        self.mask.editor._delete_mask(self.mask)

    def set_active(self, instance):
        self.mask.editor.set_active_mask(self.mask)

# メインのエディタークラス
class MaskEditor2(FloatLayout):
    mask_layers = ListProperty([])
    active_mask = ObjectProperty(None, allownone=True)
    image_size = ListProperty([0, 0])  # 画像のサイズを保持
    disp_info = Property((0, 0, 0, 0, 1))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.image_widget = Image() #Image(allow_stretch=False, keep_ratio=True)
        self.image_widget.pos_hint = {"x":0, "top":1}
        self.add_widget(self.image_widget)

        self.mask_container = Widget()
        self.add_widget(self.mask_container)
        self.rectangle = None

        self.ui_layout = BoxLayout(orientation='vertical', size_hint=(0.1, 1))
        self.ui_layout.pos_hint = {"x":0, "top":1}
        self.add_widget(self.ui_layout)
        self.create_ui()

        self.create_mask = None
        self.center_rotate_rad = 0
        self.orientation = (0, 0)
        self.margin = (0, 0)
        self.texture_size = (0, 0)
        self.rotation_changed_flag = False

        self.full_image_rgb = None
        self.full_image_hls = None
        self.crop_image_rgb = None
        self.crop_image_hls = None

        #self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        #self._keyboard.bind(on_key_down=self._on_keyboard_down)

        Logger.info("MaskEditor: 初期化完了")

    # 終了処理
    def end(self):
        if self.active_mask is not None:
            self.active_mask.end()

    def push_scissor(self):
        ScissorPush(x=int(self.pos[0]), y=int(self.pos[1]), width=int(self.size[0]), height=int(self.size[1]))

    def pop_scissor(self):
        ScissorPop()

    def imread(self, image_source, dt):
        # 画像の読み込みとサイズの取得
        if not os.path.isfile(image_source):
            Logger.error(f"MaskEditor: 画像ファイルが見つかりません: {image_source}")
            return False
        
        self.image_widget.source = image_source
        self.image_size = self.image_widget.texture.size
        self.image_widget.opacity = 1

        if self.image_size[0] >= self.image_size[1]:
            scale = self.size[0] / self.image_size[0]
        else:
            scale = self.size[1] / self.image_size[1]
        self.texture_size = (self.size[0], self.size[0])
        self.disp_info = (0, 0, self.size[0], self.size[1], scale)
        self.__set_image_info()
        return True
    
    def set_ref_image(self, crop_image, full_image):
        self.crop_image_rgb = crop_image
        self.crop_image_hls = cv2.cvtColor(crop_image, cv2.COLOR_RGB2HLS_FULL)
        if self.full_image_rgb is not full_image:
            self.full_image_rgb = full_image
            self.full_image_hls = cv2.cvtColor(full_image, cv2.COLOR_RGB2HLS_FULL)

    def set_texture_size(self, tx, ty):
        self.texture_size = (tx, ty)

    def set_primary_param(self, primary_param, disp_info):
        self.image_widget.source = None
        self.image_widget.opacity = 0

        self.image_size[0], self.image_size[1] = primary_param['original_img_size']
        self.disp_info = disp_info

        new_center_rotate_rad = math.radians(primary_param.get('rotation', 0))
        new_orientation = (math.radians(primary_param.get('rotation2', 0)), primary_param.get('flip_mode', 0))
        if new_center_rotate_rad != self.center_rotate_rad or new_orientation != self.orientation:
            self.set_rotation_changed_flag(True)
        self.center_rotate_rad = new_center_rotate_rad
        self.orientation = new_orientation

        self.__set_image_info()

    """ 
    def set_orientation(self, rotation, rotation2, flip):
        self.center_rotate_rad = math.radians(rotation)
        self.orientation = (math.radians(rotation2), flip)
    """

    def set_rotation_changed_flag(self, flag):
        self.rotation_changed_flag = flag

    def get_hash_items(self):
        return (self.disp_info, self.center_rotate_rad, self.orientation)

    def __set_image_info(self):
        self.margin = ((self.size[0]-self.texture_size[0])/2, (self.size[1]-self.texture_size[1])/2)
        for mask in reversed(self.mask_layers):
            #pass    # 無限ループ対策
            effects.reeffect_all(mask.effects)
        
    def update(self):
        # 既存のマスクに対する更新を処理
        for mask in reversed(self.mask_layers):
            #pass    # 無限ループ対策
            mask.update()

    def serialize(self):
        list = []
        for mask in reversed(self.mask_layers):
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

    def get_active_mask(self):
        if self.disabled == True:
            return None
        
        return self.active_mask
    
    def get_layers_list(self):
        return self.mask_layers
    
    def create_ui(self):
        # マスクタイプ選択ボタン
        btn_circular = Button(text='Circle', size_hint=(1, 0.05))
        btn_circular.bind(on_release=self.select_circular_gradient_mask)
        self.ui_layout.add_widget(btn_circular)

        btn_gradient = Button(text='Line', size_hint=(1, 0.05))
        btn_gradient.bind(on_release=self.select_gradient_mask)
        self.ui_layout.add_widget(btn_gradient)

        btn_full = Button(text='Full', size_hint=(1, 0.05))
        btn_full.bind(on_release=self.select_full_mask)
        self.ui_layout.add_widget(btn_full)

        btn_free_draw = Button(text='Draw', size_hint=(1, 0.05))
        btn_free_draw.bind(on_release=self.select_free_draw_mask)
        self.ui_layout.add_widget(btn_free_draw)

        btn_segment = Button(text='Segment', size_hint=(1, 0.05))
        btn_segment.bind(on_release=self.select_segment_mask)
        self.ui_layout.add_widget(btn_segment)

        btn_depth_map = Button(text='Depth', size_hint=(1, 0.05))
        btn_depth_map.bind(on_release=self.select_depth_map_mask)
        self.ui_layout.add_widget(btn_depth_map)

        btn_face = Button(text='Face', size_hint=(1, 0.05))
        btn_face.bind(on_release=self.select_face_mask)
        self.ui_layout.add_widget(btn_face)

        btn_scene = Button(text='Scene', size_hint=(1, 0.05))
        btn_scene.bind(on_release=self.select_scene)
        self.ui_layout.add_widget(btn_scene)

        # マスクレイヤー表示
        self.layer_list = BoxLayout(orientation='vertical', size_hint=(2, 0.7))
        self.ui_layout.add_widget(self.layer_list)

    def set_draw_mask(self, is_draw_mask):
        if is_draw_mask == False:
            if self.rectangle is not None:
                self.mask_container.canvas.before.remove(self.rectangle)
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
                texture = Texture.create(size=(glayimg.shape[1], glayimg.shape[0]), colorfmt='luminance', bufferfmt='float')
                texture.blit_buffer(glayimg.tobytes(), colorfmt='luminance', bufferfmt='float')
                texture.flip_vertical()
                px, py = self.to_window(*self.pos)
                px, py = px+self.margin[0], py+self.margin[1]
                Color(1, 0, 0, 0.3)
                self.rectangle = Rectangle(texture=texture, pos=(px, py), size=self.texture_size)

        # cv2.imwrite('combined_mask.png', (glayimg*255).astype(np.uint8))

    def select_circular_gradient_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_CIRCULAR)

    def select_gradient_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_GRADIENT)

    def select_full_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_FULL)

    def select_free_draw_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_FREEDRAW)

    def select_segment_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_SEGMENT)

    def select_depth_map_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_DEPTHMAP)

    def select_face_mask(self, instance):
        self._create_start_new_mask(MASKTYPE_FACE)

    def select_scene(self, instance):
        self._create_start_new_mask(MASKTYPE_SCENE)

    def _create_start_new_mask(self, type):
        # 画像サイズがまだ設定されていない場合、マスクの作成をスキップ
        if self.image_size == [0, 0]:
            Logger.warning("MaskEditor: 画像がまだロードされていません。マスクを追加できません。")
            return
        
        self.ui_layout.disabled = True
        mask = self._create_mask(type)
        self.set_active_mask(None)
        self.create_mask = mask

    def _create_end_new_mask(self):
        self.ui_layout.disabled = False
        self.set_active_mask(self.create_mask)

    def on_touch_down(self, touch):
        if self.disabled == True:
            return False
        
        # 既存のマスクに対するタッチイベントを処理
        for mask in reversed(self.mask_layers):
            if mask.on_touch_down(touch):
                return True

        return super().on_touch_down(touch)
        
    def on_touch_up(self, touch):
        if self.disabled == True:
            return False
        
        result = super().on_touch_up(touch)

        # こっちを後でやらないとまだコントロールポイントが作られてない
        if self.create_mask is not None:
            self._create_end_new_mask()        
            self.create_mask = None
        
        return result

    def _create_mask(self, mask_type):
        # マスク作成
        if mask_type == MASKTYPE_CIRCULAR:
            mask = CircularGradientMask(editor=self)
        elif mask_type == MASKTYPE_GRADIENT:
            mask = GradientMask(editor=self)
        elif mask_type == MASKTYPE_FULL:
            mask = FullMask(editor=self)
        elif mask_type == MASKTYPE_FREEDRAW:
            mask = FreeDrawMask(editor=self)
        elif mask_type == MASKTYPE_SEGMENT:
            mask = SegmentMask(editor=self)
        elif mask_type == MASKTYPE_DEPTHMAP:
            mask = DepthMapMask(editor=self)
        elif mask_type == MASKTYPE_FACE:
            mask = FaceMask(editor=self)
        elif mask_type == MASKTYPE_SCENE:
            mask = SceneMask(editor=self)
        else:
            Logger.error(f"MaskEditor: 不明なマスクタイプ: {self.current_mask_type}")
            return None
            
        self.mask_container.add_widget(mask)
        self.mask_layers.append(mask)

        # レイヤーUIに追加
        layer = MaskLayer(mask=mask)
        self.layer_list.add_widget(layer)
        if self.root is not None:
            self.root.set2widget_all(mask.effects, mask.effects_param)

        return mask

    def _delete_mask(self, mask):
        if len(self.mask_layers) <= 1:
            self.draw_mask_image(None)
            self.set_active_mask(None)
        else:
            i = self.mask_layers.index(mask)
            i = i+1 if i+1 < len(self.mask_layers) else i-1
            self.set_active_mask(self.mask_layers[i])
 
        self.mask_container.remove_widget(mask)
        self.mask_layers.remove(mask)

    def clear_mask(self):
        self.set_active_mask(None)
        self.draw_mask_image(None)
        self.mask_container.clear_widgets()
        self.mask_layers.clear()
        self.layer_list.clear_widgets()
        FaceMask.delete_faces()

    def set_active_mask(self, mask):
        if self.active_mask and self.active_mask != mask:
            self.active_mask.active = False
            self.active_mask.end()

        self.active_mask = mask
        if mask is not None:
            mask.active = True
            self.root.set2widget_all(mask.effects, mask.effects_param)
            mask.start()
            #mask.update()
        else:
            self.draw_mask_image(None)
            if self.root is not None:
                self.root.set2widget_all(None, None)
        self.start_draw_image()

    def get_rotate_rad(self, rotate_rad):
        rad, flip = self.orientation
        """
        if flip == 0:
            rad = rotate_rad + rad
        if flip == 1:
            rad = -rotate_rad + rad
        if flip == 2:
            rad = -rotate_rad + rad
        """
        return self.center_rotate_rad + rad + rotate_rad
    
    def world_to_tcg_scale(self, x, y):
        return (x / self.disp_info[4], y / self.disp_info[4])
    
    def tcg_to_world_scale(self, x, y):
        return (x * self.disp_info[4], y * self.disp_info[4])

    # ワールド座標からテクスチャのグローバル座標に
    def window_to_tcg(self, cx, cy):
        wx, wy = self.to_window(*self.pos)
        cx, cy = cx - wx, cy - wy
        cx, cy = cx - self.margin[0], cy - self.margin[1]
        cx, cy = cx, self.texture_size[1] - cy
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*self.texture_size, self.disp_info)
        cx, cy = cx - offset_x, cy - offset_y
        cx, cy = cx / self.disp_info[4], cy / self.disp_info[4]
        cx, cy = cx + self.disp_info[0], cy + self.disp_info[1]
        imax = max(self.image_size[0]/2, self.image_size[1]/2)
        cx, cy = cx - imax, cy - imax
        cx, cy = self.center_rotate_invert(cx, cy, self.center_rotate_rad)
        return (cx, cy)

    def tcg_to_window(self, cx, cy):
        imax = max(self.image_size[0]/2, self.image_size[1]/2)
        cx, cy = self.center_rotate(cx, cy, self.center_rotate_rad)
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - self.disp_info[0], cy - self.disp_info[1]
        cx, cy = cx * self.disp_info[4], cy * self.disp_info[4]        
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*self.texture_size, self.disp_info)
        cx, cy = cx + offset_x, cy + offset_y
        cx, cy = cx, self.texture_size[1] - cy
        cx, cy = cx + self.margin[0], cy + self.margin[1]
        wx, wy = self.to_window(*self.pos)
        cx, cy = cx + wx, cy + wy
        return (cx, cy)

    def tcg_to_texture(self, cx, cy):
        imax = max(self.image_size[0]/2, self.image_size[1]/2)
        cx, cy = self.center_rotate(cx, cy, self.center_rotate_rad)
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - self.disp_info[0], cy - self.disp_info[1]
        cx, cy = cx * self.disp_info[4], cy * self.disp_info[4]        
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*self.texture_size, self.disp_info)
        cx, cy = cx + offset_x, cy + offset_y
        return (cx, cy)

    def tcg_to_full_image(self, cx, cy):
        imax = max(self.image_size[0]/2, self.image_size[1]/2)
        cx, cy = self.center_rotate(cx, cy, self.center_rotate_rad)
        cx, cy = cx + imax, cy + imax
        return (cx, cy)

    def tcg_to_crop_image(self, cx, cy):
        cx, cy = self.tcg_to_full_image(cx, cy)
        cx = cx * (self.crop_image_hls.shape[1] / self.full_image_rgb.shape[1])
        cy = cy * (self.crop_image_hls.shape[0] / self.full_image_rgb.shape[0])
        return (cx, cy)

    def apply_orientation(self, cx, cy):
        rad, flip = self.orientation

        if (flip & 1) == 1:
            cx = -cx
        if (flip & 2) == 2:
            cy = -cy

        return cx, cy, rad

    def center_rotate(self, cx, cy, rotation_rad):
        cx, cy, rad = self.apply_orientation(cx, cy)
        rad = rotation_rad + rad
        rad = -rad

        new_cx = cx * math.cos(rad) - cy * math.sin(rad)
        new_cy = cx * math.sin(rad) + cy * math.cos(rad)

        return (new_cx, new_cy)

    def center_rotate_invert(self, cx, cy, rotation_rad):
        rad, _ = self.orientation
        rad = rotation_rad + rad
        rad = -rad

        new_cx = cx * math.cos(rad) + cy * math.sin(rad)
        new_cy = -cx * math.sin(rad) + cy * math.cos(rad)

        new_cx, new_cy, _ = self.apply_orientation(new_cx, new_cy)

        return (new_cx, new_cy)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'z':
            if self.disp_info[4] == 1.0:
                self.disp_info[0], self.disp_info[1] = 0, 0
                if self.image_size[0] >= self.image_size[1]:
                    scale = self.size[0] / self.image_size[0]
                else:
                    scale = self.size[1] / self.image_size[1]
                self.disp_info[4] = scale
            else:
                self.disp_info[0], self.disp_info[1] = self.image_size[0]/2, self.image_size[1]/2
                self.disp_info[4] = 1.0
        elif keycode[1] == 'up':
            self.disp_info[1] -= 100
        elif keycode[1] == 'down':
            self.disp_info[1] += 100
        elif keycode[1] == 'left':
            self.disp_info[0] -= 100
        elif keycode[1] == 'right':
            self.disp_info[0] += 100
        elif keycode[1] == 't':
            for mask in reversed(self.mask_layers):
                dict = mask.serialize()        
                mask.deserialize(dict)
            return True

        elif keycode[1] == 'r':
            self.center_rotate_rad += math.radians(5)
            for mask in reversed(self.mask_layers):
                mask.update()
            return True

        elif keycode[1] == 'a':
            self.root.force_full_redraw()
            
            return True
        
        elif keycode[1] == 's':
            self.root.save_json()
            return True
        
        # 既存のマスクに対するタッチイベントを処理
        for mask in reversed(self.mask_layers):
            mask.update()  
    
        return True

# アプリケーションクラス
class MaskEditor2App(App):
    def build(self):
        # 画像ファイルのパスを正しく設定してください
        image_path = 'your_image.jpg'

        box0 = BoxLayout()
        box0.orientation = 'vertical'
        box0.size = (Window.width, Window.height)
        box0.size_hint = (None, None)

        box1 = FloatLayout()
        box1.size_hint_y = 2
        box0.add_widget(box1)
        box2 = BoxLayout()
        box2.orientation = 'horizontal'
        box2.size_hint_y = 6
        box0.add_widget(box2)
        box3 = FloatLayout()
        box3.size_hint_y = 2
        box0.add_widget(box3)

        box4 = FloatLayout()
        box4.size_hint_x = 2
        box2.add_widget(box4)
        float = FloatLayout()
        float.size_hint_x = 6
        editor = MaskEditor2()
        editor.pos_hint = {'x': 0, 'top': 1}
        #editor.size_hint_x = 6
        float.add_widget(editor)
        box2.add_widget(float)
        box5 = FloatLayout()
        box5.size_hint_x = 2
        box2.add_widget(box5)

        Clock.schedule_once(partial(editor.imread, image_path), -1)

        return box0

if __name__ == '__main__':
    MaskEditor2App().run()
