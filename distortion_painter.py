
from kivy.core.window import Window
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import (
    Color, Line, PushMatrix, PopMatrix, Translate,
    ScissorPush, ScissorPop,
)
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty, ListProperty
from kivy.clock import Clock
import cv2
import numpy as np
import json
import os
import time
import numba as nb

import core

class DistortionEngine:
    def __init__(self, image_size, grid_size=30):
        self.width, self.height = image_size
        self.grid_size = grid_size
        
        # メッシュグリッドの生成
        self.original_grid = self._create_initial_grid()
        self.current_grid = self.original_grid.copy()
        
        # 変形マップ
        self.map_x = None
        self.map_y = None
        self.dirty = True
        
        # 前回の変形状態を保持
        self.last_warped_image = None
    
    def _create_initial_grid(self):
        """ 初期メッシュグリッドを生成 """
        cols = int(np.ceil(self.width / self.grid_size)) + 1
        rows = int(np.ceil(self.height / self.grid_size)) + 1
        
        grid = np.zeros((rows, cols, 2), dtype=np.float32)
        
        for i in range(rows):
            y = min(i * self.grid_size, self.height - 1)
            for j in range(cols):
                x = min(j * self.grid_size, self.width - 1)
                grid[i, j] = [x, y]
                
        return grid
    
    def apply_effect(self, center, radius, strength, effect_type, direction=(0,0), original_image=None):
        """ ブラシ効果を適用し、ROI領域を更新して返す """
        # 効果半径を計算 (安全マージン追加)
        effect_radius = radius * 1.5 + 10
        
        # ROI領域を正確に計算
        x_min = max(0, int(center[0] - effect_radius))
        y_min = max(0, int(center[1] - effect_radius))
        x_max = min(self.width, int(center[0] + effect_radius))
        y_max = min(self.height, int(center[1] + effect_radius))
        
        # 領域サイズが0の場合は処理しない
        if x_min >= x_max or y_min >= y_max:
            return image
        
        # Numbaで高速化された処理を実行
        if effect_type == 'forward_warp':
            self.current_grid = DistortionEngine._apply_forward_warp_numba(
                self.current_grid, center, radius, strength, direction,
                x_min, x_max, y_min, y_max, effect_radius
            )
        elif effect_type == 'bulge':
            self.current_grid = DistortionEngine._apply_bulge_numba(
                self.current_grid, center, radius, -strength,
                x_min, x_max, y_min, y_max, effect_radius
            )
        elif effect_type == 'pinch':
            self.current_grid = DistortionEngine._apply_bulge_numba(
                self.current_grid, center, radius, strength,
                x_min, x_max, y_min, y_max, effect_radius
            )
        elif effect_type == 'swirl':
            self.current_grid = DistortionEngine._apply_swirl_numba(
                self.current_grid, center, radius, strength,
                x_min, x_max, y_min, y_max, effect_radius
            )
        elif effect_type == 'restore':
            self.current_grid = DistortionEngine._apply_restore_numba(
                self.current_grid, self.original_grid, center, radius, strength,
                x_min, x_max, y_min, y_max, effect_radius
            )
        
        self.dirty = True
        
        # 画像が提供されている場合、ROI領域のみを更新して返す
        if original_image is not None:

            # 前回の変形結果をベースに使用
            base_image = self.last_warped_image if self.last_warped_image is not None else original_image.copy()

            # 変形マップを高速更新
            self._update_deformation_map_fast()
            
            # ROI領域のみを変形
            roi = original_image[y_min:y_max, x_min:x_max].copy()
            
            # マップのROI部分を切り出し
            map_x_roi = self.map_x[y_min:y_max, x_min:x_max].copy()
            map_y_roi = self.map_y[y_min:y_max, x_min:x_max].copy()
            
            # オフセット調整
            map_x_roi -= x_min
            map_y_roi -= y_min
            
            warped_roi = cv2.remap(
                roi, 
                map_x_roi, 
                map_y_roi, 
                cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_REFLECT
            )
            
            # 結果をコピー
            base_image[y_min:y_max, x_min:x_max] = warped_roi
            
            # 更新結果を保持
            self.last_warped_image = base_image
            return base_image
        
        return None
    
    def _update_deformation_map_fast(self):
        """ 高速な変形マップ更新 """
        if not self.dirty:
            return
        
        # Numbaで高速に変形マップを生成
        self.map_x, self.map_y = DistortionEngine._generate_deformation_map_numba(
            self.original_grid, 
            self.current_grid,
            self.width,
            self.height,
            self.grid_size
        )
        
        self.dirty = False
    
    def warp_image(self, image):
        """ 画像全体を変形 """
        self._update_deformation_map_fast()
        
        # 全体を変形
        result = cv2.remap(
            image, 
            self.map_x, 
            self.map_y, 
            cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT
        )
        
        # 結果を保持
        self.last_warped_image = result
        return result
    
    def reset(self):
        """ 変形をリセット """
        self.current_grid = self.original_grid.copy()
        self.dirty = True
        self.last_warped_image = None

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _generate_deformation_map_numba(original_grid, current_grid, width, height, grid_size):
        """ Numbaで高速化された変形マップ生成 """
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        
        rows, cols, _ = original_grid.shape
        displacement_x = current_grid[:, :, 0] - original_grid[:, :, 0]
        displacement_y = current_grid[:, :, 1] - original_grid[:, :, 1]
        
        grid_step_x = grid_size
        grid_step_y = grid_size
        
        for y in nb.prange(height):
            for x in nb.prange(width):
                # グリッドインデックス計算
                grid_i = y // grid_step_y
                grid_j = x // grid_step_x
                
                # グリッドセル内の位置
                dy = y % grid_step_y
                dx = x % grid_step_x
                
                # グリッドセルの4点を取得
                i0 = min(grid_i, rows-1)
                j0 = min(grid_j, cols-1)
                i1 = min(grid_i+1, rows-1)
                j1 = min(grid_j+1, cols-1)
                
                # バイリニア補間の重み
                wx = dx / grid_step_x
                wy = dy / grid_step_y
                
                # 4点の変位
                d00_x = displacement_x[i0, j0]
                d00_y = displacement_y[i0, j0]
                d01_x = displacement_x[i0, j1]
                d01_y = displacement_y[i0, j1]
                d10_x = displacement_x[i1, j0]
                d10_y = displacement_y[i1, j0]
                d11_x = displacement_x[i1, j1]
                d11_y = displacement_y[i1, j1]
                
                # X方向の補間
                top_x = (1 - wx) * d00_x + wx * d01_x
                bottom_x = (1 - wx) * d10_x + wx * d11_x
                dx_interp = (1 - wy) * top_x + wy * bottom_x
                
                # Y方向の補間
                top_y = (1 - wx) * d00_y + wx * d01_y
                bottom_y = (1 - wx) * d10_y + wx * d11_y
                dy_interp = (1 - wy) * top_y + wy * bottom_y
                
                # 変形マップに設定
                map_x[y, x] = x + dx_interp
                map_y[y, x] = y + dy_interp
        
        return map_x, map_y

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _apply_forward_warp_numba(grid, center, radius, strength, direction, 
                                x_min, x_max, y_min, y_max, effect_radius):
        new_grid = grid.copy()
        center_x, center_y = center
        dir_x, dir_y = direction
        dir_norm = max(1e-5, np.sqrt(dir_x*dir_x + dir_y*dir_y))
        dir_x /= -dir_norm
        dir_y /= -dir_norm
        
        for i in nb.prange(grid.shape[0]):
            for j in nb.prange(grid.shape[1]):
                px, py = grid[i, j]
                if not (x_min <= px <= x_max and y_min <= py <= y_max):
                    continue
                    
                dx = px - center_x
                dy = py - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist >= effect_radius:
                    continue
                    
                weight = np.exp(-(dist*dist) / (2 * (radius/3)**2))
                move = strength * radius * weight * 0.5
                new_grid[i, j, 0] = px + dir_x * move
                new_grid[i, j, 1] = py + dir_y * move
                
        return new_grid

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _apply_bulge_numba(grid, center, radius, strength, 
                        x_min, x_max, y_min, y_max, effect_radius):
        new_grid = grid.copy()
        center_x, center_y = center
        
        for i in nb.prange(grid.shape[0]):
            for j in nb.prange(grid.shape[1]):
                px, py = grid[i, j]
                if not (x_min <= px <= x_max and y_min <= py <= y_max):
                    continue
                    
                dx = px - center_x
                dy = py - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist >= effect_radius or dist < 1e-5:
                    continue
                    
                weight = np.exp(-(dist*dist) / (2 * (radius/3)**2))
                move = strength * radius * weight * 0.5
                dir_x = dx / dist
                dir_y = dy / dist
                new_grid[i, j, 0] = px + dir_x * move
                new_grid[i, j, 1] = py + dir_y * move
                
        return new_grid

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _apply_swirl_numba(grid, center, radius, strength, 
                        x_min, x_max, y_min, y_max, effect_radius):
        new_grid = grid.copy()
        center_x, center_y = center
        
        for i in nb.prange(grid.shape[0]):
            for j in nb.prange(grid.shape[1]):
                px, py = grid[i, j]
                if not (x_min <= px <= x_max and y_min <= py <= y_max):
                    continue
                    
                dx = px - center_x
                dy = py - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist >= effect_radius or dist < 1e-5:
                    continue
                    
                weight = np.exp(-(dist*dist) / (2 * (radius/3)**2))
                angle = weight * strength * 0.5
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                x_rot = dx * cos_a - dy * sin_a
                y_rot = dx * sin_a + dy * cos_a
                new_grid[i, j, 0] = center_x + x_rot
                new_grid[i, j, 1] = center_y + y_rot
                
        return new_grid

    @staticmethod
    #@nb.njit(parallel=True, fastmath=True, cache=True)
    def _apply_restore_numba(grid, original_grid, center, radius, strength, 
                        x_min, x_max, y_min, y_max, effect_radius):
        new_grid = grid.copy()
        center_x, center_y = center

        # 事前計算
        radius_sq = (radius / 3) ** 2
        effect_radius_sq = effect_radius ** 2
                
        for i in nb.prange(grid.shape[0]):
            for j in nb.prange(grid.shape[1]):
                px, py = grid[i, j]
                
                # ROIチェック
                if px < x_min or px > x_max or py < y_min or py > y_max:
                    continue

                # 元の位置
                orig_px, orig_py = original_grid[i, j]

                # 距離の二乗を計算
                dx = px - center_x
                dy = py - center_y
                dist_sq = dx*dx + dy*dy
                
                # 効果半径チェック
                if dist_sq > effect_radius_sq:
                    continue
                    
                # 重み計算 (指数関数の引数を制限)
                weight = strength * np.exp(-dist_sq / (2 * radius_sq))
                weight = min(max(weight, 0.0), 1.0)
                                
                # 復元処理
                new_grid[i, j, 0] = (1 - weight) * px + weight * orig_px
                new_grid[i, j, 1] = (1 - weight) * py + weight * orig_py
                
        return new_grid

class DistortionCanvas(FloatLayout):
    STRENGTH_SCALE = 0.0025

    brush_size = NumericProperty(100)
    strength = NumericProperty(50)
    effect_type = StringProperty('forward_warp')
    last_touch_pos = ListProperty([0, 0])
    
    def __init__(self, image_widget=None, recorded=[], callback=None, **kwargs):
        super().__init__(**kwargs)

        self.original_image = None
        self.current_image = None
        self.recorded = recorded
        self.is_recording = False
        self.last_touch_time = 0
        self.points_buffer = []  # 補間用ポイントバッファ
        self.update_event = None
        self.preview_texture = None
        self.full_quality_texture = None
        self.needs_full_update = False
        self.image_widget = image_widget if image_widget is not None else self.ids.image_widget
        self.callback = callback

        Clock.schedule_once(self._set_brush_cursor, -1)

    def on_size(self, *args):
        self._set_brush_cursor(0)

    def on_parent(self, instance, parent):
        if parent is not None:
            Window.bind(mouse_pos=self.on_mouse_pos)
        else:
            self.brush_color.rgba = (0, 0, 0, 0)
            Window.unbind(mouse_pos=self.on_mouse_pos)

    def load_image(self, path):
        if not os.path.exists(path):
            print(f"Error: File not found - {path}")
            return
            
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Failed to load image - {path}")
            return

        # イメージを正方形にする
        imax = max(img.shape[1], img.shape[0])
        offset_y = (imax-img.shape[0])//2
        offset_x = (imax-img.shape[1])//2
        img = np.pad(img, ((offset_y, imax-(offset_y+img.shape[0])), (offset_x, imax-(offset_x+img.shape[1])), (0, 0)))
        
        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.current_image = self.original_image.copy()
        self.is_update_texture = True        
        self.update_texture(full_quality=True)
        print(f"Loaded image: {path}")

        # サンプル
        iw = self.image_widget
        self.disp_info = (0, 0, iw.texture_size[0], iw.texture_size[1], min(iw.width, iw.height) / max(iw.texture_size))
        self.margin = ((iw.width-iw.texture_size[0]*self.disp_info[4])/2, (iw.height-iw.texture_size[1]*self.disp_info[4])/2)
        self.tcg_info = core.param_to_tcg_info(None)

    def set_ref_image(self, ref_image, engine_recreate=True):
        self.original_image = ref_image
        self.current_image = ref_image.copy()
        self.is_update_texture = False
        self.is_recording = True

        if engine_recreate:
            h, w = ref_image.shape[:2]
            self.engine = DistortionEngine((w, h))

    def set_primary_param(self, primary_param):
        self.tcg_info = core.param_to_tcg_info(primary_param)

    def on_mouse_pos(self, window, pos):
        #print(f"Mouse position: {pos}")
        self._update_brush_cursor(pos[0], pos[1])

    def _set_brush_cursor(self, dt):
        if isinstance(self.parent, Widget):
            self.pos = self.parent.pos
            self.size = self.parent.size

        self.canvas.after.clear()
        with self.canvas.after:
            PushMatrix()
            ScissorPush(x=int(self.pos[0]), y=int(self.pos[1]), width=int(self.size[0]), height=int(self.size[1]))
            self.translate = Translate(0, 0)
            self.brush_color = Color((0, 1, 1, 1))
            self.brush_cursor = Line(ellipse=(0, 0, self.brush_size, self.brush_size), width=2)
            ScissorPop()
            PopMatrix()

    def _update_brush_cursor(self, x, y):
        self.translate.x, self.translate.y = x - self.brush_size / 2, y - self.brush_size / 2
        self.brush_cursor.ellipse = (0, 0, self.brush_size, self.brush_size)

    def update_texture(self, full_quality=False):
        if self.current_image is None:
            return

        if self.is_update_texture:
            # フルクオリティ更新が必要な場合
            if full_quality or self.needs_full_update:
                # OpenCV画像をKivyテクスチャに変換
                self.full_quality_texture = Texture.create(size=(self.current_image.shape[1], self.current_image.shape[0]))
                self.full_quality_texture.flip_vertical()
                self.full_quality_texture.blit_buffer(self.current_image.tobytes(), colorfmt='rgb', bufferfmt='float')
                self.needs_full_update = False
            
            # プレビューテクスチャを使用
            self.ids.image_widget.texture = self.full_quality_texture
        
        if self.callback is not None:
            self.callback()

    def on_touch_down(self, touch):
        if self.image_widget.collide_point(*touch.pos) and self.current_image is not None:
            
            # 座標変換 (Widget座標 → 画像座標)
            tcg_x, tcg_y = self._window_to_tcg(touch.x, touch.y)
            self.last_touch_pos = [tcg_x, tcg_y]
            self.last_touch_time = time.time()  # 現在のシステム時間を使用
            self.points_buffer = []
           
            strength = -self.strength if self.effect_type == 'swirl' and 'meta' in Window.modifiers else self.strength

            # 記録開始
            if self.is_recording:
                self.recorded.append({
                    "x": tcg_x, "y": tcg_y,
                    "size": self.brush_size,
                    "strength": strength,
                    "effect": self.effect_type,
                    "time": self.last_touch_time  # タイムスタンプ記録
                })

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.image_widget.collide_point(*touch.pos) and self.current_image is not None:
             
            # 座標変換 (Widget座標 → 画像座標)
            tcg_x, tcg_y = self._window_to_tcg(touch.x, touch.y)
            
            # 移動方向ベクトルを計算
            direction = (tcg_x - self.last_touch_pos[0], tcg_y - self.last_touch_pos[1])
            
            # 現在の時間を取得
            current_time = time.time()
            
            # ポイントをバッファに追加
            self.points_buffer.append((tcg_x, tcg_y, direction, current_time))
            
            # 前回の更新から一定時間経過したら処理
            if current_time - self.last_touch_time > 0.05:  # 20fps (0.05秒間隔)
                self.process_buffer()
                self.last_touch_time = current_time

            strength = -self.strength if self.effect_type == 'swirl' and 'meta' in Window.modifiers else self.strength

            # 記録
            if self.is_recording:
                self.recorded.append({
                    "x": tcg_x, "y": tcg_y,
                    "size": self.brush_size,
                    "strength": strength,
                    "effect": self.effect_type,
                    "time": current_time
                })
                
            self.last_touch_pos = [tcg_x, tcg_y]

        return True

    def convert_to_image_coords(self, touch_x, touch_y):
        img_widget = self.image_widget
        if img_widget.texture_size[0] == 0 or img_widget.texture_size[1] == 0:
            return 0, 0
            
        # 座標変換 (Widget座標 → 画像座標)
        img_x = int((touch_x - img_widget.x) * img_widget.texture_size[0] / img_widget.width)
        img_y = int((img_widget.height - (touch_y - img_widget.y)) * 
                   img_widget.texture_size[1] / img_widget.height)
        
        # 境界チェック
        img_x = max(0, min(img_x, img_widget.texture_size[0] - 1))
        img_y = max(0, min(img_y, img_widget.texture_size[1] - 1))
        
        return img_x, img_y

    def process_buffer(self):
        if not self.points_buffer:
            return
            
        # バッファ内のポイントを処理
        for tcg_x, tcg_y, direction, _ in self.points_buffer:
            img_x, img_y = core.tcg_to_ref_image(tcg_x, tcg_y, self.current_image, self.tcg_info)
            try:
                strength = -self.strength if self.effect_type == 'swirl' and 'meta' in Window.modifiers else self.strength

                # 変形適用
                self.current_image = self.engine.apply_effect(
                    center=(img_x, img_y),
                    radius=self.brush_size / self.tcg_info['disp_info'][4],
                    strength=strength * (DistortionCanvas.STRENGTH_SCALE if self.effect_type != 'restore' else 0.01),
                    effect_type=self.effect_type,
                    direction=direction,
                    original_image=self.original_image
                )
            except Exception as e:
                print(f"Error applying distortion: {e}")
        
        # テクスチャ更新をスケジュール（パフォーマンスのため遅延処理）
        if self.update_event is None:
            self.update_event = Clock.schedule_once(self.delayed_texture_update, 0.05)
        
        # バッファクリア
        self.points_buffer = []

    def delayed_texture_update(self, dt):
        self.needs_full_update = True
        self.update_texture()
        self.update_event = None

    def on_touch_up(self, touch):
        # バッファに残っているポイントを処理
        if self.points_buffer:
            self.process_buffer()
        return super().on_touch_up(touch)

    def serialize(self):

        if len(self.recorded) <= 0:
            return None

        dict = {
            'distortion_points': self.recorded,
        }
        return dict

    def deserialize(self, dict):
        self.recorded = dict['distortion_points']
        #self.replay_recorded()
        
    def save_recorded(self, path="distortion_record.json"):
        try:
            with open(path, 'w') as f:
                json.dump(self.recorded, f, indent=2)
            print(f"Recording saved to {path}")
        except Exception as e:
            print(f"Error saving recorded: {e}")

    def load_recorded(self, path="distortion_record.json"):
        try:
            with open(path, 'r') as f:
                self.recorded = json.load(f)
            #self.replay_recorded()
            print(f"Recording loaded from {path}")
        except Exception as e:
            print(f"Error loading recorded: {e}")

    def remap_image(self):
        self.current_image = self.engine.warp_image(self.original_image)
        #DistortionCanvas.replay_recorded_with(self.current_image, self.recorded, self.tcg_info, self.engine)

    @staticmethod
    def replay_recorded(original_image, recorded, tcg_info, engine=None):
        if original_image is None:
            print("No image loaded for replay")
            return original_image

        if recorded is None or len(recorded) == 0:
            print("No recorded to replay")
            return original_image

        h, w = original_image.shape[:2]
        if engine is None:
            engine = DistortionEngine((w, h))
        
        # 記録再生
        for index in range(len(recorded)):
            action = recorded[index]
            try:
                # 方向ベクトルを計算（次の点がある場合）
                direction = (0, 0)
                if index < len(recorded) - 1:
                    next_action = recorded[index + 1]
                    dx = next_action['x'] - action['x']
                    dy = next_action['y'] - action['y']
                    direction = (dx, dy)
                
                engine.apply_effect(
                    center=core.tcg_to_ref_image(action['x'], action['y'], original_image, tcg_info),
                    radius=action['size'] / tcg_info['disp_info'][4],
                    strength=action['strength'] * (DistortionCanvas.STRENGTH_SCALE if action.get('effect', 'forward_warp') != 'restore' else 0.01),
                    effect_type=action.get('effect', 'forward_warp'),
                    direction=direction,
                    original_image=None,
                )
                                
            except Exception as e:
                print(f"Error replaying action {index}: {e}")

        img = engine.warp_image(original_image)

        return img

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.needs_full_update = True
            self.recorded = []
            self.update_texture()
            print("Image reset")

    def set_effect(self, effect_type):
        self.effect_type = effect_type
        print(f"Effect set to: {effect_type}")

    def set_brush_size(self, size):
        self.brush_size = size
        print(f"Brush size set to: {size}")

    def set_strength(self, strength):
        self.strength = strength
        print(f"Strength set to: {strength}")

    def get_current_image(self):
        return self.current_image

    def get_recorded(self):
        return self.recorded

    # ワールド座標からテクスチャのグローバル座標に
    def _window_to_tcg(self, cx, cy):
        return core.window_to_tcg(cx, cy, self, self.image_widget.texture_size, self.tcg_info)

    
class Distortion_PainterApp(App):
    def build(self):
        widget = DistortionCanvas()

        return widget        

if __name__ == '__main__':
    # テスト用画像が存在しない場合に作成
    if not os.path.exists("input.jpg"):
        # 単純なテスト画像を作成
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(img, "Distortion Test", (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
        cv2.imwrite("input.jpg", img)
        print("Created test image: input.jpg")
    
    Distortion_PainterApp().run()