
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
import math

import params
import core


class DistortionEngine:
    @staticmethod
    def apply_warp(image, center, radius, strength, effect_type, direction=(1,0), original_image=None):
        """ 選択された効果タイプに基づいて歪みを適用 """
        if effect_type == 'forward_warp':
            return DistortionEngine.forward_warp(image, center, radius, strength, direction)
        elif effect_type == 'bulge':
            return DistortionEngine.bulge(image, center, radius, strength)
        elif effect_type == 'pinch':
            return DistortionEngine.pinch(image, center, radius, strength)
        elif effect_type == 'swirl':
            return DistortionEngine.swirl(image, center, radius, strength)
        return image

    @staticmethod
    def forward_warp(image, center, radius, strength, direction):
        """ 前方ワープ効果 (ブラシの移動方向に沿って変形) """
        h, w = image.shape[:2]
        
        # ROI(関心領域)を設定して処理範囲を限定
        x1 = max(0, int(center[0] - radius * 1.5))
        y1 = max(0, int(center[1] - radius * 1.5))
        x2 = min(w, int(center[0] + radius * 1.5))
        y2 = min(h, int(center[1] + radius * 1.5))
        
        if x1 >= x2 or y1 >= y2:
            return image
            
        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        
        # ROI内の中心座標
        center_roi = (center[0] - x1, center[1] - y1)
        
        # 座標グリッド生成
        y, x = np.indices((roi_h, roi_w))
        dx = x - center_roi[0]
        dy = y - center_roi[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # 中心からの距離に基づく重み (ガウス分布)
        weight = np.exp(-(dist**2) / (2 * (radius/3)**2))
        
        # 方向ベクトルの正規化
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction = (direction[0]/dir_norm, direction[1]/dir_norm)
        
        # 移動ベクトル (方向に沿って)
        move_x = strength * radius * weight * direction[0]
        move_y = strength * radius * weight * direction[1]
        
        # 変形後の座標
        new_x = x + move_x
        new_y = y + move_y
        
        # 境界内に制限
        new_x = np.clip(new_x, 0, roi_w - 1)
        new_y = np.clip(new_y, 0, roi_h - 1)
        
        # 画像リマップ
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        distorted_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        
        # ROIを元の画像に戻す
        image[y1:y2, x1:x2] = distorted_roi
        return image

    @staticmethod
    def bulge(image, center, radius, strength):
        """ 膨張効果 (中心から外側に押し出す) """
        return DistortionEngine._radial_effect(image, center, radius, strength, outward=False)

    @staticmethod
    def pinch(image, center, radius, strength):
        """ 縮小効果 (中心に向かって引き込む) """
        return DistortionEngine._radial_effect(image, center, radius, strength, outward=True)

    @staticmethod
    def _radial_effect(image, center, radius, strength, outward=True):
        """ 放射状効果の共通処理 """
        h, w = image.shape[:2]

        if strength < 0:
            outward = not outward
            strength = -strength
        
        # ROI(関心領域)を設定して処理範囲を限定
        x1 = max(0, int(center[0] - radius * 1.5))
        y1 = max(0, int(center[1] - radius * 1.5))
        x2 = min(w, int(center[0] + radius * 1.5))
        y2 = min(h, int(center[1] + radius * 1.5))
        
        if x1 >= x2 or y1 >= y2:
            return image
            
        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        
        # ROI内の中心座標
        center_roi = (center[0] - x1, center[1] - y1)
        
        # 座標グリッド生成
        y, x = np.indices((roi_h, roi_w))
        dx = x - center_roi[0]
        dy = y - center_roi[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # 中心からの距離に基づく重み (ガウス分布)
        weight = np.exp(-(dist**2) / (2 * (radius/3)**2))
        
        # 変形計算
        sign = 1 if outward else -1
        new_dist = dist + sign * strength * radius * weight
        new_dist = np.maximum(new_dist, 0)
        
        # 中心点以外の処理
        mask = (dist > 0)
        new_x = center_roi[0] + dx * (new_dist / dist) * mask
        new_y = center_roi[1] + dy * (new_dist / dist) * mask
        
        # 中心点処理
        new_x[~mask] = center_roi[0]
        new_y[~mask] = center_roi[1]
        
        # 境界内に制限
        new_x = np.clip(new_x, 0, roi_w - 1)
        new_y = np.clip(new_y, 0, roi_h - 1)
        
        # 画像リマップ
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        distorted_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        
        # ROIを元の画像に戻す
        image[y1:y2, x1:x2] = distorted_roi
        return image

    @staticmethod
    def swirl(image, center, radius, strength):
        """ 渦巻き効果 (回転変形) 
        strengthの符号で回転方向を制御:
         正の値: 時計回り
         負の値: 反時計回り
        """
        h, w = image.shape[:2]
        
        # ROI(関心領域)を設定して処理範囲を限定
        x1 = max(0, int(center[0] - radius * 1.5))
        y1 = max(0, int(center[1] - radius * 1.5))
        x2 = min(w, int(center[0] + radius * 1.5))
        y2 = min(h, int(center[1] + radius * 1.5))
        
        if x1 >= x2 or y1 >= y2:
            return image
            
        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        
        # ROI内の中心座標
        center_roi = (center[0] - x1, center[1] - y1)
        
        # 座標グリッド生成
        y, x = np.indices((roi_h, roi_w))
        dx = x - center_roi[0]
        dy = y - center_roi[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # 距離に基づく重み計算
        weight = np.exp(-(dist**2) / (2 * (radius/3)**2))
        angle = weight * strength * 5  # 回転角度（strengthの符号で方向が変わる）
        
        # 変形後の座標計算
        new_x = center_roi[0] + dx * np.cos(angle) - dy * np.sin(angle)
        new_y = center_roi[1] + dx * np.sin(angle) + dy * np.cos(angle)
        
        # 境界内に制限
        new_x = np.clip(new_x, 0, roi_w - 1)
        new_y = np.clip(new_y, 0, roi_h - 1)
        
        # 画像リマップ
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        distorted_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        
        # ROIを元の画像に戻す
        image[y1:y2, x1:x2] = distorted_roi
        return image
    
class DistortionCanvas(FloatLayout):
    STRENGTH_SCALE = 0.0005

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

    def set_ref_image(self, ref_image):
        self.original_image = ref_image
        self.current_image = ref_image.copy()
        self.is_update_texture = False
        self.is_recording = True

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
                self.current_image = DistortionEngine.apply_warp(
                    self.current_image,
                    center=(img_x, img_y),
                    radius=self.brush_size / self.tcg_info['disp_info'][4],
                    strength=strength * (DistortionCanvas.STRENGTH_SCALE if self.effect_type != 'restore' else 0.01),
                    effect_type=self.effect_type,
                    direction=direction,
                    original_image=self.original_image  # 元の画像を渡す
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

    def replay_recorded(self):
        self.current_image = DistortionCanvas.replay_recorded_with(self.current_image, self.recorded, self.tcg_info)

    @staticmethod
    def replay_recorded_with(original_image, recorded, tcg_info):
        if original_image is None:
            print("No image loaded for replay")
            return original_image

        if recorded is None or len(recorded) == 0:
            print("No recorded to replay")
            return original_image
            
        img = original_image.copy()
        
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
                
                img = DistortionEngine.apply_warp(
                    img,
                    center=core.tcg_to_ref_image(action['x'], action['y'], img, tcg_info),
                    radius=action['size'] / tcg_info['disp_info'][4],
                    strength=action['strength'] * (DistortionCanvas.STRENGTH_SCALE if action.get('effect', 'forward_warp') != 'restore' else 0.01),
                    effect_type=action.get('effect', 'forward_warp'),
                    direction=direction,
                    original_image=original_image
                )
                                
            except Exception as e:
                print(f"Error replaying action {index}: {e}")

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