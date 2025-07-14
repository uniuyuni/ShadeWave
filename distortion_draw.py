
from kivy.core.window import Window
from kivy.app import App
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

class DistortionEngine:
    @staticmethod
    def apply_warp(image, center, radius, strength, effect_type, direction=(1,0)):
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
        move_x = strength * radius * weight * -direction[0]
        move_y = strength * radius * weight * -direction[1]
        
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
        """ 渦巻き効果 (回転変形) """
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
        angle = weight * strength * 5  # 回転角度
        
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
    brush_size = NumericProperty(50)
    strength = NumericProperty(1.0)
    effect_type = StringProperty('forward_warp')
    last_touch_pos = ListProperty([0, 0])
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_image = None
        self.current_image = None
        self.recording = []
        self.is_recording = False
        self.last_touch_time = 0
        self.points_buffer = []  # 補間用ポイントバッファ
        self.update_event = None
        self.preview_texture = None
        self.full_quality_texture = None
        self.needs_full_update = False

    def on_size(self, *args):
        with self.canvas.after:
            PushMatrix()
            ScissorPush(x=int(self.pos[0]), y=int(self.pos[1]), width=int(self.size[0]), height=int(self.size[1]))
            self.translate = Translate(0, 0)
            self.brush_color = Color((0, 1, 1, 1))
            self.brush_cursor = Line(ellipse=(0, 0, self.brush_size, self.brush_size), width=2)
            ScissorPop()
            PopMatrix()

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
        self.update_texture(full_quality=True)
        print(f"Loaded image: {path}")

        # サンプル
        iw = self.ids.image_widget
        self.disp_info = (0, 0, iw.texture_size[0], iw.texture_size[1], min(iw.width, iw.height) / max(iw.texture_size))
        self.margin = ((iw.width-iw.texture_size[0]*self.disp_info[4])/2, (iw.height-iw.texture_size[1]*self.disp_info[4])/2)
        pass

    def set_primary_param(self, primary_param):
        #self.disp_info = primary_param['disp_info']
        self.center_rotate_rad = math.radians(primary_param.get('rotation', 0))
        self.orientation_rad = (math.radians(primary_param.get('rotation2', 0)), primary_param.get('flip_mode', 0))

    def on_mouse_pos(self, window, pos):
        self.update_brush_cursor(pos[0], pos[1])

    def update_brush_cursor(self, x, y):
        self.translate.x, self.translate.y = x - self.brush_size / 2, y - self.brush_size / 2
        self.brush_cursor.ellipse = (0, 0, self.brush_size, self.brush_size)

    def update_texture(self, full_quality=False):
        if self.current_image is None:
            return
            
        # フルクオリティ更新が必要な場合
        if full_quality or self.needs_full_update:
            # OpenCV画像をKivyテクスチャに変換
            self.full_quality_texture = Texture.create(size=(self.current_image.shape[1], self.current_image.shape[0]))
            self.full_quality_texture.flip_vertical()
            self.full_quality_texture.blit_buffer(self.current_image.tobytes(), colorfmt='rgb', bufferfmt='float')
            self.needs_full_update = False
        
        # プレビューテクスチャを使用
        self.ids.image_widget.texture = self.full_quality_texture

    def on_touch_down(self, touch):
        if self.ids.image_widget.collide_point(*touch.pos) and self.current_image is not None:
            
            # 座標変換 (Widget座標 → 画像座標)
            tcg_x, tcg_y = self.window_to_tcg(touch.x, touch.y)
            self.last_touch_pos = [tcg_x, tcg_y]
            self.last_touch_time = time.time()  # 現在のシステム時間を使用
            self.points_buffer = []
            
            # 記録開始
            if self.is_recording:
                self.recording.append({
                    "x": tcg_x, "y": tcg_y,
                    "size": self.brush_size,
                    "strength": self.strength,
                    "effect": self.effect_type,
                    "time": self.last_touch_time  # タイムスタンプ記録
                })

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.ids.image_widget.collide_point(*touch.pos) and self.current_image is not None:
             
            # 座標変換 (Widget座標 → 画像座標)
            tcg_x, tcg_y = self.window_to_tcg(touch.x, touch.y)
            
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
            
            # 記録
            if self.is_recording:
                self.recording.append({
                    "x": tcg_x, "y": tcg_y,
                    "size": self.brush_size,
                    "strength": self.strength,
                    "effect": self.effect_type,
                    "time": current_time
                })
                
            self.last_touch_pos = [tcg_x, tcg_y]

        return True

    def convert_to_image_coords(self, touch_x, touch_y):
        img_widget = self.ids.image_widget
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
            img_x, img_y = self.tcg_to_full_image(tcg_x, tcg_y)
            try:
                # 変形適用
                self.current_image = DistortionEngine.apply_warp(
                    self.current_image,
                    center=(img_x, img_y),
                    radius=self.brush_size * 5,  # ブラシサイズ調整
                    strength=self.strength * 0.025,
                    effect_type=self.effect_type,
                    direction=direction
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

        if len(self.recording) <= 0:
            return None

        dict = {
            'distortion_points': self.recording,
        }
        return dict

    def deserialize(self, dict):
        self.recording = dict['distortion_points']
        self.replay_recording()
        
    def save_recording(self, path="distortion_record.json"):
        try:
            with open(path, 'w') as f:
                json.dump(self.recording, f, indent=2)
            print(f"Recording saved to {path}")
        except Exception as e:
            print(f"Error saving recording: {e}")

    def load_recording(self, path="distortion_record.json"):
        try:
            with open(path, 'r') as f:
                self.recording = json.load(f)
            self.replay_recording()
            print(f"Recording loaded from {path}")
        except Exception as e:
            print(f"Error loading recording: {e}")

    def replay_recording(self):
        if self.original_image is None:
            print("No image loaded for replay")
            return

        if self.recording is None or len(self.recording) == 0:
            print("No recording to replay")
            return
            
        self.current_image = self.original_image.copy()
        self.needs_full_update = True
        self.update_texture()
        
        # 記録再生
        for index in range(len(self.recording)):
            action = self.recording[index]
            try:
                # 方向ベクトルを計算（次の点がある場合）
                direction = (0, 0)
                if index < len(self.recording) - 1:
                    next_action = self.recording[index + 1]
                    dx = next_action['x'] - action['x']
                    dy = next_action['y'] - action['y']
                    direction = (dx, dy)
                
                self.current_image = DistortionEngine.apply_warp(
                    self.current_image,
                    center=(action['x'], action['y']),
                    radius=action['size'] * 5,
                    strength=action['strength'] * 0.025,
                    effect_type=action.get('effect', 'forward_warp'),
                    direction=direction
                )
                                
            except Exception as e:
                print(f"Error replaying action {index}: {e}")

        # 最終更新
        self.needs_full_update = True
        self.update_texture()
        

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.needs_full_update = True
            self.update_texture()
            print("Image reset")

    def set_effect(self, effect_type):
        self.effect_type = effect_type
        print(f"Effect set to: {effect_type}")

    # ワールド座標からテクスチャのグローバル座標に
    def window_to_tcg(self, cx, cy):
        wx, wy = self.to_window(*self.pos)
        cx, cy = cx - wx, cy - wy
        cx, cy = cx - self.margin[0], cy - self.margin[1]
        cx, cy = cx, self.ids['image_widget'].size[1] - cy
        cx, cy = cx / self.disp_info[4], cy / self.disp_info[4]
        cx, cy = cx + self.disp_info[0], cy + self.disp_info[1]
        imax = max(self.current_image.shape[1]/2, self.current_image.shape[0]/2)
        cx, cy = cx - imax, cy - imax
        cx, cy = self.center_rotate_invert(cx, cy, self.center_rotate_rad)
        return (cx, cy)

    def tcg_to_window(self, cx, cy):
        imax = max(self.current_image.shape[1]/2, self.current_image.shape[0]/2)
        cx, cy = self.center_rotate(cx, cy, self.center_rotate_rad)
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - self.disp_info[0], cy - self.disp_info[1]
        cx, cy = cx * self.disp_info[4], cy * self.disp_info[4]        
        cx, cy = cx, self.ids['image_widget'].size[1] - cy
        cx, cy = cx + self.margin[0], cy + self.margin[1]
        wx, wy = self.to_window(*self.pos)
        cx, cy = cx + wx, cy + wy
        return (cx, cy)

    def tcg_to_full_image(self, cx, cy):
        imax = max(self.current_image.shape[1]/2, self.current_image.shape[0]/2)
        cx, cy = self.center_rotate(cx, cy, self.center_rotate_rad)
        cx, cy = cx + imax, cy + imax
        return (cx, cy)

    def apply_orientation(self, cx, cy):
        rad, flip = self.orientation_rad

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
        rad, _ = self.orientation_rad
        rad = rotation_rad + rad
        rad = -rad

        new_cx = cx * math.cos(rad) + cy * math.sin(rad)
        new_cy = -cx * math.sin(rad) + cy * math.cos(rad)

        new_cx, new_cy, _ = self.apply_orientation(new_cx, new_cy)

        return (new_cx, new_cy)
    
class Distortion_DrawApp(App):
    def build(self):
        widget = DistortionCanvas()

        param = {}
        widget.set_primary_param(param)

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
    
    Distortion_DrawApp().run()