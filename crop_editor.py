
import numpy as np
from kivy.app import App as KVApp
from kivy.graphics import Color as KVColor, Line as KVLine, PushMatrix as KVPushMatrix, PopMatrix as KVPopMatrix, Translate as KVTranslate, Rotate as KVRotate
from kivy.properties import NumericProperty as KVNumericProperty, ListProperty as KVListProperty
from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.uix.label import Label as KVLabel
from kivy.metrics import dp
from kivy.clock import Clock
from enum import Enum
from typing import List, Tuple
import math

import core

class CropEditor(KVFloatLayout):
    input_width = KVNumericProperty(dp(400))
    input_height = KVNumericProperty(dp(300))
    input_angle = KVNumericProperty(0)
    scale = KVNumericProperty(1.0)
    crop_rect = KVListProperty([0, 0, 0, 0])
    corner_threshold = KVNumericProperty(dp(10))
    edge_threshold = KVNumericProperty(dp(10))  # 辺のドラッグ判定用の閾値を追加
    aspect_ratio = KVNumericProperty(0)

    def __init__(self, **kwargs):
        super(CropEditor, self).__init__(**kwargs)
        self.corner_dragging = None
        self.edge_dragging = None
        self.moving = False
        self.last_touch_pos = None
        self.callback = None

        self.set_aspect_ratio(self.aspect_ratio)

        # スケール座標をローカル座標に変換
        self._set_to_local_crop_rect(self.crop_rect)

        scaled_width = self.input_width * self.scale
        scaled_height = self.input_height * self.scale
        with self.canvas:
            KVPushMatrix()
            self.input_translate = KVTranslate(scaled_width/2, scaled_height/2)
            self.input_rotate = KVRotate(angle=self.input_angle)
            KVColor(0.5, 0.5, 0.5, 1)
            self.input_line = KVLine(rectangle=(-scaled_width/2, -scaled_height/2, scaled_width, scaled_height), width=1)
            KVPopMatrix()

            KVPushMatrix()
            self.translate = KVTranslate()
            # グリッド線用の色と線を追加
            KVColor(1, 1, 1, 0.5)
            self.grid_lines = []
            for _ in range(4):  # 縦横2本ずつ
                self.grid_lines.append(KVLine(points=[], dash_length=5, dash_offset=5))
            
            KVColor(1, 1, 1, 1)
            self.white_line = KVLine(rectangle=(0, 0, 0, 0), width=2)
            KVColor(0, 0, 0, 1)
            self.black_line = KVLine(rectangle=(0, 0, 0, 0), width=1)
            KVPopMatrix()
        
        self.label = KVLabel(font_size=20, bold=True, halign='left')
        self.add_widget(self.label)

        self.bind(crop_rect=self.update_rect,
                  input_width=self.update_crop_size,
                  input_height=self.update_crop_size,
                  scale=self.update_crop_size,
                  size=self.update_centering,
                  aspect_ratio=self.update_crop_size,
                  input_angle=self.update_crop_size)

        Clock.schedule_once(self.create_ui, -1)

    def create_ui(self, dt):
        self.pos = self.parent.pos
        
        # 初期設定の反映
        self.update_crop_size()

    def set_aspect_ratio(self, aspect_ratio):
        # アスペクト比変換
        if aspect_ratio != 0 and self.crop_rect[2] - self.crop_rect[0] < self.crop_rect[3] - self.crop_rect[1]:
            self.aspect_ratio = 1 / aspect_ratio
        else:
            self.aspect_ratio = aspect_ratio

    def _set_to_local_crop_rect(self, crop_rect):

        # 矩形のサイズを設定 (初期値は画像のサイズと同じ)
        if crop_rect == (0, 0, 0, 0):
            if int(round(self.input_angle)) // 90 % 2 != 1:
                w = self.input_width #* self.scale
                h = self.input_height #* self.scale
            else:
                h = self.input_width #* self.scale
                w = self.input_height #* self.scale
            crop_rect = core.get_initial_crop_rect(w, h)

        crop_x, crop_y, cx, cy = crop_rect
        crop_width = cx - crop_x
        crop_height = cy - crop_y

        # 最大サイズとパディング計算
        maxsize = max(self.input_width, self.input_height)
        padw = 0#(maxsize - self.input_width) // 2
        padh = 0#(maxsize - self.input_height) // 2
        
        # Y座標の変換（Y軸反転）
        x1 = (crop_x + padw) * self.scale
        y1 = ((maxsize - (crop_y + crop_height)) + padh) * self.scale
        x2 = x1 + crop_width * self.scale
        y2 = y1 + crop_height * self.scale
                    
        self.crop_rect = (x1, y1, x2, y2)

    def update_crop_size(self, *args):

        # 縦横比補正
        self.corner_dragging = None
        self.__resize_crop(None)

        # 中心にシフトするためのトランスレーションを設定
        self.update_centering()

    def update_rect(self, *args):
        x1, y1, x2, y2 = self.crop_rect
        width = x2 - x1
        height = y2 - y1
        
        # メインの矩形を更新
        self.white_line.rectangle = (x1, y1, width, height)
        self.black_line.rectangle = (x1, y1, width, height)
        
        # グリッド線を更新
        # 縦線
        for i, line in enumerate(self.grid_lines[:2]):
            third = width * (i + 1) / 3
            line.points = [x1 + third, y1, x1 + third, y2]
        
        # 横線
        for i, line in enumerate(self.grid_lines[2:]):
            third = height * (i + 1) / 3
            line.points = [x1, y1 + third, x2, y1 + third]
        
        # 大きさ表示
        self.label.x, self.label.y = 0, 0 #int(self.translate.x), int(self.translate.y)
        w = abs(int(round(width / self.scale)))
        h = abs(int(round(height / self.scale)))
        gcd = math.gcd(w, h)
        self.label.text = str(w) + " x " + str(h) + "  " + str(w / gcd) + ":" + str(h / gcd)

        if self.callback is not None:
            self.callback()

    def update_centering(self, *args):
        # 中心に移動するためのトランスレーションを設定
        inwidth = self.input_width * self.scale
        inheight = self.input_height * self.scale
        inm = max(inwidth, inheight)
        self.translate.x = self.pos[0] + (self.width - inm) / 2
        self.translate.y = self.pos[1] + (self.height - inm) / 2
        self.input_translate.x = self.translate.x + inm / 2
        self.input_translate.y = self.translate.y + inm / 2
        self.input_rotate.angle = self.input_angle

        self.update_rect()

    def on_touch_down(self, touch):
        self.corner_dragging = self.__get_dragging_corner(touch)
        if self.corner_dragging is not None:
            return True

        self.edge_dragging = self.__get_dragging_edge(touch)
        if self.edge_dragging is not None:
            return True

        if self.__is_inside_rect(touch):
            self.moving = True
            self.last_touch_pos = touch.pos
            return True
            
        return super(CropEditor, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.corner_dragging is not None:
            self.__resize_by_corner(touch)
            return True

        if self.edge_dragging is not None:
            self.__resize_by_edge(touch)
            return True

        if self.moving and self.last_touch_pos:
            dx = touch.x - self.last_touch_pos[0]
            dy = touch.y - self.last_touch_pos[1]
            self.__move_rect(dx, dy)
            self.last_touch_pos = touch.pos
            return True

        return super(CropEditor, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        self.corner_dragging = None
        self.edge_dragging = None
        self.moving = False
        self.last_touch_pos = None

        # 反転処理はここでやる
        x1, y1, x2, y2 = self.crop_rect
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        self.crop_rect = (x1, y1, x2, y2)
        
        return super(CropEditor, self).on_touch_up(touch)

    def __get_dragging_corner(self, touch):
        x, y = touch.pos
        x1, y1, x2, y2 = self.crop_rect
        cx1 = x1 + self.translate.x
        cy1 = y1 + self.translate.y
        cx2 = x2 + self.translate.x
        cy2 = y2 + self.translate.y

        if abs(x - cx1) < self.corner_threshold and abs(y - cy1) < self.corner_threshold:
            return 'top_left'
        if abs(x - cx2) < self.corner_threshold and abs(y - cy1) < self.corner_threshold:
            return 'top_right'
        if abs(x - cx1) < self.corner_threshold and abs(y - cy2) < self.corner_threshold:
            return 'bottom_left'
        if abs(x - cx2) < self.corner_threshold and abs(y - cy2) < self.corner_threshold:
            return 'bottom_right'
        return None

    def __get_dragging_edge(self, touch):
        x, y = touch.pos
        x1, y1, x2, y2 = self.crop_rect
        cx1 = x1 + self.translate.x
        cy1 = y1 + self.translate.y
        cx2 = x2 + self.translate.x
        cy2 = y2 + self.translate.y

        # 各辺の判定
        if abs(x - cx1) < self.edge_threshold and y1 + self.translate.y < y < y2 + self.translate.y:
            return 'left'
        if abs(x - cx2) < self.edge_threshold and y1 + self.translate.y < y < y2 + self.translate.y:
            return 'right'
        if abs(y - cy1) < self.edge_threshold and x1 + self.translate.x < x < x2 + self.translate.x:
            return 'top'
        if abs(y - cy2) < self.edge_threshold and x1 + self.translate.x < x < x2 + self.translate.x:
            return 'bottom'
        return None

    def __is_inside_rect(self, touch):
        x, y = touch.pos
        x1, y1, x2, y2 = self.crop_rect
        cx1 = x1 + self.translate.x
        cy1 = y1 + self.translate.y
        cx2 = x2 + self.translate.x
        cy2 = y2 + self.translate.y

        return (cx1 < x < cx2 and cy1 < y < cy2)

    def __move_rect(self, dx, dy):
        x1, y1, x2, y2 = self.crop_rect
        width = x2 - x1
        height = y2 - y1

        # 移動を試みる
        test_x1 = x1 + dx
        test_y1 = y1 + dy
        test_x2 = test_x1 + width
        test_y2 = test_y1 + height

        # 四隅の点それぞれを確認
        corners = [
            (test_x1, test_y1),  # 左上
            (test_x2, test_y1),  # 右上
            (test_x1, test_y2),  # 左下
            (test_x2, test_y2)   # 右下
        ]

        # すべての点が範囲内に収まるかチェック
        valid_corners = []
        aflag = False
        for test_x, test_y in corners:
            valid_x, valid_y, flag = rotate_and_correct_point(
                test_x, test_y,
                test_x, test_y,  # 古い位置は現在のテスト位置を使用
                self.input_width * self.scale,
                self.input_height * self.scale,
                self.input_angle
            )
            aflag |= flag
            valid_corners.append((valid_x, valid_y))

        # 補正された四隅の点から新しい矩形を計算
        new_x1 = min(x[0] for x in valid_corners[:2])  # 上辺のx座標の最小値
        new_y1 = min(y[1] for y in valid_corners[::2])  # 左辺のy座標の最小値
        new_x2 = max(x[0] for x in valid_corners[2:])  # 下辺のx座標の最大値
        new_y2 = max(y[1] for y in valid_corners[1::2])  # 右辺のy座標の最大値

        # 矩形のサイズを保持
        if aflag == True:
            return  # サイズが変わる場合は移動をキャンセル

        self.crop_rect = (new_x1, new_y1, new_x2, new_y2)

    def __resize_by_edge(self, touch):
        x1, y1, x2, y2 = self.crop_rect
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

        if self.edge_dragging == 'left':
            new_x1 = touch.x - self.translate.x
            fix_corners = ['top_right', 'bottom_right']

        elif self.edge_dragging == 'right':
            new_x2 = touch.x - self.translate.x
            fix_corners = ['top_left', 'bottom_left']

        elif self.edge_dragging == 'top':
            new_y1 = touch.y - self.translate.y
            fix_corners = ['bottom_left', 'bottom_right']

        elif self.edge_dragging == 'bottom':
            new_y2 = touch.y - self.translate.y
            fix_corners = ['top_left', 'top_right']

        else:
            return self.__resize_crop(touch)

        for i in range(5):

            new_x1, new_y1, new_x2, new_y2, carf = correct_aspect_ratio(new_x1, new_y1, new_x2, new_y2, self.aspect_ratio, fix_corners)

            # 位置を補正
            _, _, f = rotate_and_correct_point(new_x1, new_y1, x1, y1, self.input_width * self.scale, self.input_height * self.scale, self.input_angle)
            if f == True:
                self.__resize_by_corner2(new_x1, new_y1, new_x2, new_y2, x1, y1, x2, y2, 'top_left', ['bottom_right'])
                return

            _, _, f = rotate_and_correct_point(new_x2, new_y1, x2, y1, self.input_width * self.scale, self.input_height * self.scale, self.input_angle)
            if f == True:
                self.__resize_by_corner2(new_x1, new_y1, new_x2, new_y2, x1, y1, x2, y2, 'top_right', ['bottom_left'])
                return

            _, _, f = rotate_and_correct_point(new_x1, new_y2, x1, y2, self.input_width * self.scale, self.input_height * self.scale, self.input_angle)
            if f == True:
                self.__resize_by_corner2(new_x1, new_y1, new_x2, new_y2, x1, y1, x2, y2, 'bottom_left', ['top_right'])
                return

            _, _, f = rotate_and_correct_point(new_x2, new_y2, x2, y2, self.input_width * self.scale, self.input_height * self.scale, self.input_angle)
            if f == True:
                self.__resize_by_corner2(new_x1, new_y1, new_x2, new_y2, x1, y1, x2, y2, 'bottom_right', ['top_left'])
                return
            
            if carf == False:
                break

        self.crop_rect = (new_x1, new_y1, new_x2, new_y2)


    def __resize_by_corner(self, touch):
        x1, y1, x2, y2 = self.crop_rect
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

        if self.corner_dragging == 'top_left':
            new_x1 = touch.x - self.translate.x
            new_y1 = touch.y - self.translate.y
            fix_corners = ['bottom_right']
            
        elif self.corner_dragging == 'top_right':
            new_x2 = touch.x - self.translate.x
            new_y1 = touch.y - self.translate.y
            fix_corners = ['bottom_left']

        elif self.corner_dragging == 'bottom_left':
            new_x1 = touch.x - self.translate.x
            new_y2 = touch.y - self.translate.y
            fix_corners = ['top_right']

        elif self.corner_dragging == 'bottom_right':
            new_x2 = touch.x - self.translate.x
            new_y2 = touch.y - self.translate.y
            fix_corners = ['top_left']
        else:
            return self.__resize_crop(touch)

        return self.__resize_by_corner2(new_x1, new_y1, new_x2, new_y2, x1, y1, x2, y2, self.corner_dragging, fix_corners)

    def __resize_by_corner2(self, new_x1, new_y1, new_x2, new_y2, old_x1, old_y1, old_x2, old_y2, corner_dragging, fix_corners):
        # 収束するまでループする（最大10回まで試行）
        max_iterations = 10
        iterations = 0
        was_corrected = True
        
        while was_corrected and iterations < max_iterations:
            was_corrected = False
            iterations += 1    

            f1, f2, f3, f4 = False, False, False, False
            
            # 縦横入れ替え？
            if self.aspect_ratio > 0:
                current_ratio = abs(new_x2 - new_x1) / max(abs(new_y2 - new_y1), 0.001)  # ゼロ除算防止

                # アスペクト比が1以上か未満かに関わらず同じロジックで処理
                if (self.aspect_ratio >= 1.0 and current_ratio >= 1) or (self.aspect_ratio < 1.0 and current_ratio <= 1):
                    target_aspect_ratio = self.aspect_ratio
                else:
                    target_aspect_ratio = 1 / self.aspect_ratio
            else:
                target_aspect_ratio = self.aspect_ratio
            
            # クリップ先の画像の中に収める - 各角を個別に補正
            if not 'top_left' in fix_corners:
                new_x1, new_y1, f1 = rotate_and_correct_point(
                    new_x1, new_y1, old_x1, old_y1, 
                    self.input_width * self.scale, 
                    self.input_height * self.scale, 
                    self.input_angle
                )
                was_corrected |= f1
                
            if not 'top_right' in fix_corners:
                new_x2, new_y1, f2 = rotate_and_correct_point(
                    new_x2, new_y1, old_x2, old_y1, 
                    self.input_width * self.scale, 
                    self.input_height * self.scale, 
                    self.input_angle
                )
                was_corrected |= f2
                
            if not 'bottom_left' in fix_corners:
                new_x1, new_y2, f3 = rotate_and_correct_point(
                    new_x1, new_y2, old_x1, old_y2, 
                    self.input_width * self.scale, 
                    self.input_height * self.scale, 
                    self.input_angle
                )
                was_corrected |= f3
                
            if not 'bottom_right' in fix_corners:
                new_x2, new_y2, f4 = rotate_and_correct_point(
                    new_x2, new_y2, old_x2, old_y2, 
                    self.input_width * self.scale, 
                    self.input_height * self.scale, 
                    self.input_angle
                )
                was_corrected |= f4

            # 固定点を保持
            if corner_dragging == 'top_left':
                if f1:  # 左上が補正された場合
                    new_x2, new_y2 = old_x2, old_y2  # 右下を元に戻す
            elif corner_dragging == 'top_right':
                if f2:  # 右上が補正された場合
                    new_x1, new_y2 = old_x1, old_y2  # 左下を元に戻す
            elif corner_dragging == 'bottom_left':
                if f3:  # 左下が補正された場合
                    new_x2, new_y1 = old_x2, old_y1  # 右上を元に戻す
            elif corner_dragging == 'bottom_right':
                if f4:  # 右下が補正された場合
                    new_x1, new_y1 = old_x1, old_y1  # 左上を元に戻す

            # 縦横比を考慮
            old_x1, old_y1, old_x2, old_y2 = new_x1, new_y1, new_x2, new_y2
            new_x1, new_y1, new_x2, new_y2, carf = correct_aspect_ratio(
                new_x1, new_y1, new_x2, new_y2, 
                target_aspect_ratio, 
                fix_corners
            )
            was_corrected |= carf
            
            # 変化が小さい場合は収束したとみなす
            if (abs(old_x1 - new_x1) < 0.02 and 
                abs(old_y1 - new_y1) < 0.02 and 
                abs(old_x2 - new_x2) < 0.02 and 
                abs(old_y2 - new_y2) < 0.02):
                break

        self.crop_rect = (new_x1, new_y1, new_x2, new_y2)


    def __resize_crop(self, touch):
        """
        アスペクト比と画像の境界に従ってクロップ矩形全体をリサイズする
        """
        x1, y1, x2, y2 = self.crop_rect
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        
        # 収束するまでループする（最大5回まで試行）
        max_iterations = 5
        iterations = 0
        was_corrected = True
        
        mm = max(self.input_width * self.scale, self.input_height * self.scale) / 2
        old_x1, old_y1, old_x2, old_y2 = mm, mm, mm, mm

        while was_corrected and iterations < max_iterations:
            was_corrected = False
            iterations += 1
                        
            # 位置補正 - 各角を画像の範囲内に収める
            new1_x1, new1_y1, f1 = rotate_and_correct_point(
                new_x1, new_y1, old_x1, old_y1, 
                self.input_width * self.scale, 
                self.input_height * self.scale, 
                self.input_angle
            )
            new2_x2, new2_y1, f2 = rotate_and_correct_point(
                new_x2, new_y1, old_x2, old_y1, 
                self.input_width * self.scale, 
                self.input_height * self.scale, 
                self.input_angle
            )
            new3_x1, new3_y2, f3 = rotate_and_correct_point(
                new_x1, new_y2, old_x1, old_y2, 
                self.input_width * self.scale, 
                self.input_height * self.scale, 
                self.input_angle
            )
            new4_x2, new4_y2, f4 = rotate_and_correct_point(
                new_x2, new_y2, old_x2, old_y2, 
                self.input_width * self.scale, 
                self.input_height * self.scale, 
                self.input_angle
            )
            was_corrected = f1 | f2 | f3 | f4

            new_x1 = max(new1_x1, new3_x1, new_x1)
            new_y1 = max(new1_y1, new2_y1, new_y1)
            new_x2 = min(new2_x2, new4_x2, new_x2)
            new_y2 = min(new3_y2, new4_y2, new_y2)

            # 縦横比を考慮
            old_x1, old_y1, old_x2, old_y2 = new_x1, new_y1, new_x2, new_y2
            
            new_x1, new_y1, new_x2, new_y2, carf = correct_aspect_ratio(
                new_x1, new_y1, new_x2, new_y2, 
                self.aspect_ratio, 
                []  # 固定点なし - 中心を維持
            )
            was_corrected |= carf
            
            # 変化が小さい場合は収束したとみなす
            if (abs(old_x1 - new_x1) < 0.02 and 
                abs(old_y1 - new_y1) < 0.02 and 
                abs(old_x2 - new_x2) < 0.02 and 
                abs(old_y2 - new_y2) < 0.02):
                break

        self.crop_rect = (new_x1, new_y1, new_x2, new_y2)
    
    def get_crop_rect(self):
        # 上下反転させて返す、パディング削除
        x1, y1, x2, y2 = self.crop_rect
        x1, y1 = int(round(x1 / self.scale)), int(round(y1 / self.scale))
        x2, y2 = int(round(x2 / self.scale)), int(round(y2 / self.scale))
        maxsize = max(self.input_width, self.input_height)
        cx1 = x1
        cy1 = maxsize - (y1 + (y2-y1))
        cx2 = x2
        cy2 = cy1 + (y2 - y1)
        return (cx1, cy1, cx2, cy2)
    
    def get_disp_info(self):
        # 上下反転させて返す。グローバル座標へも変換
        x1, y1, x2, y2 = self.crop_rect
        x1, y1 = int(round(x1 / self.scale)), int(round(y1 / self.scale))
        x2, y2 = int(round(x2 / self.scale)), int(round(y2 / self.scale))
        maxsize = max(self.input_width, self.input_height)
        crop_x = x1
        crop_y = maxsize - (y1 + (y2-y1))
        crop_width = x2 - x1
        crop_height = y2 - y1
        return (crop_x, crop_y, crop_width, crop_height, self.scale)
    
    def set_editing_callback(self, callback):
        self.callback = callback

class PointPosition(Enum):
    INSIDE = "inside"       # 内側
    OUTSIDE = "outside"     # 外側
    ON_BORDER = "on_border" # 線上

def get_point_position_in_rotated_rectangle(
    point_x: float, 
    point_y: float, 
    rect_width: float, 
    rect_height: float, 
    angle_degrees: float,
    tolerance: float = 1e-10
) -> Tuple[PointPosition, float]:
    """
    点が回転した四角形に対してどの位置にあるかを判定する
    
    Parameters:
    point_x, point_y: 判定する点の座標
    rect_width, rect_height: 四角形のサイズ
    angle_degrees: 回転角度（度数法）
    tolerance: 点が線上にあるとみなす許容誤差
    
    Returns:
    Tuple[PointPosition, float]: 
        - 点の位置（INSIDE, OUTSIDE, ON_BORDER）
        - 最も近い境界までの符号付き距離（内側が正、外側が負）
    """
    # 度数法からラジアンに変換
    angle_rad = np.radians(angle_degrees)
    
    # 四角形の中心を原点とした場合の頂点座標
    half_width = rect_width / 2
    half_height = rect_height / 2
    corners = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    
    # 回転行列
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # 頂点を回転
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    def cross_product(p1, p2, p3):
        """三点から符号付き面積（の2倍）を計算"""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    def point_to_line_distance(p, line_start, line_end):
        """点と線分の距離を計算"""
        # 線分の長さの2乗
        line_length_sq = np.sum((line_end - line_start) ** 2)
        
        if line_length_sq == 0:
            return np.linalg.norm(p - line_start)
            
        # 線分上の最近点のパラメータ t
        t = max(0, min(1, np.dot(p - line_start, line_end - line_start) / line_length_sq))
        
        # 最近点
        projection = line_start + t * (line_end - line_start)
        
        return np.linalg.norm(p - projection)
    
    point = np.array([point_x, point_y])
    cross_products = []
    min_distance = float('inf')
    
    # 各辺について処理
    for i in range(4):
        j = (i + 1) % 4
        # 符号付き面積を計算
        cp = cross_product(rotated_corners[i], rotated_corners[j], point)
        cross_products.append(cp)
        
        # 点と辺の距離を計算
        distance = point_to_line_distance(point, rotated_corners[i], rotated_corners[j])
        min_distance = min(min_distance, distance)
    
    # すべての符号が正か負か確認
    all_positive = all(cp > tolerance for cp in cross_products)
    all_negative = all(cp < -tolerance for cp in cross_products)
    
    # 点の位置を判定
    if min_distance <= tolerance:
        position = PointPosition.ON_BORDER
        signed_distance = 0.0
    elif all_positive or all_negative:
        position = PointPosition.INSIDE
        signed_distance = min_distance
    else:
        position = PointPosition.OUTSIDE
        signed_distance = -min_distance
        
    return position, signed_distance

def find_nearest_point_on_edge(point_x, point_y, edge_start, edge_end):
    """
    線分上の最近接点を求める
    """
    edge_vector = edge_end - edge_start
    point_vector = np.array([point_x, point_y]) - edge_start
    
    # 線分の長さの二乗
    edge_length_sq = np.sum(edge_vector ** 2)
    
    # 線分上での位置（0から1の間）
    t = max(0, min(1, np.dot(point_vector, edge_vector) / edge_length_sq))
    
    # 最近接点
    return edge_start + t * edge_vector

def line_intersection(line1_start, line1_end, line2_start, line2_end):
    """
    2本の線分の交点を計算する関数（改善版）
    
    Args:
        line1_start (tuple): 線分1の開始点 (x, y)
        line1_end (tuple): 線分1の終了点 (x, y)
        line2_start (tuple): 線分2の開始点 (x, y)
        line2_end (tuple): 線分2の終了点 (x, y)
    
    Returns:
        tuple or None: 交点の座標、または交差しない場合はNone
    """
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end
    
    # クラメールの公式を使用して交点を計算
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # 数値的安定性のための許容値
    EPSILON = 1e-10
    
    # 平行または重なる場合
    if abs(denominator) < EPSILON:
        # 垂直線が重なる場合の特殊処理
        if abs(x1 - x2) < EPSILON and abs(x3 - x4) < EPSILON and abs(x1 - x3) < EPSILON:
            # y座標の重なりをチェック
            y_min = max(min(y1, y2), min(y3, y4))
            y_max = min(max(y1, y2), max(y3, y4))
            if y_min <= y_max:
                return (x1, (y_min + y_max) / 2)
        return None
    
    # 交点のパラメータ t1, t2 を計算
    t1 = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    t2 = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denominator
    
    # パラメータが [0,1] の範囲内にあるかチェック
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        # 交点の座標を計算
        x = x1 + t1 * (x2 - x1)
        y = y1 + t1 * (y2 - y1)
        return (x, y)
    
    return None

def rotate_and_correct_point(point_x, point_y, old_px, old_py, rect_width, rect_height, angle_degrees):
    """
    四角形を中心で回転させた時、与えられた点が四角形の内部に収まるように補正する
    回転後の斜めの辺を正確に考慮する
    
    Parameters:
    point_x, point_y: 補正する点の座標
    old_px, old_py: 移動する前の座標
    rect_width, rect_height: 四角形のサイズ
    angle_degrees: 回転角度（度数法）
    
    Returns:
    tuple: 補正後の (x, y) 座標
    Bool: 補正されたかフラグ
    """
    rect_mm = max(rect_height, rect_width)
    
    px = point_x - rect_mm/2
    py = point_y - rect_mm/2

    # 点が既に内部にある場合は補正不要
    position, _ = get_point_position_in_rotated_rectangle(px, py, rect_width, rect_height, angle_degrees)
    if position == PointPosition.INSIDE:
        return point_x, point_y, False
    
    elif position == PointPosition.ON_BORDER:
        return point_x, point_y, False

    # 回転後の頂点を計算
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    half_width = rect_width / 2
    half_height = rect_height / 2
    corners = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # 移動ベクトルを計算し十分な長さまで延長
    ox = old_px - rect_mm/2
    oy = old_py - rect_mm/2
    move_vector = np.array([px - ox, py - oy])
    if np.all(move_vector == 0):
        # 移動がない場合は最近接点を探す
        min_dist = float('inf')
        nearest = None
        for i in range(4):
            edge_point = find_nearest_point_on_edge(px, py, 
                rotated_corners[i], rotated_corners[(i+1)%4])
            dist = np.sum((edge_point - np.array([px, py])) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = edge_point
        return nearest[0] + rect_mm/2, nearest[1] + rect_mm/2, True

    # 移動ベクトルを正規化して延長
    length = np.sqrt(np.sum(move_vector ** 2))
    #if length > 0:
    #    move_vector = move_vector / length
    # 対角線の長さの2倍程度まで延長
    extension = np.sqrt(rect_mm**2 + rect_mm**2)
    extended_point = np.array([ox, oy]) + move_vector * 2#extension
    #extended_point = np.array([px, py])

    # 延長した線分と各辺との交点を探す
    min_distance = extension #float('inf')
    intersection_point = None
    
    for i in range(4):
        j = (i + 1) % 4
        lipoint = line_intersection((ox, oy), (extended_point[0], extended_point[1]),
                                  rotated_corners[i], rotated_corners[j])
        if lipoint is not None:
            # 元の点に最も近い交点を選択
            dist = np.sqrt((lipoint[0] - px)**2 + (lipoint[1] - py)**2)
            if dist < min_distance:
                min_distance = dist
                intersection_point = lipoint

    if intersection_point is not None:
        return intersection_point[0] + rect_mm/2, intersection_point[1] + rect_mm/2, True

    # それ以外の場合は最近接点を探す
    min_dist = float('inf')
    nearest = None
    for i in range(4):
        edge_point = find_nearest_point_on_edge(px, py, 
            rotated_corners[i], rotated_corners[(i+1)%4])
        dist = np.sum((edge_point - np.array([px, py])) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest = edge_point

    return nearest[0] + rect_mm/2, nearest[1] + rect_mm/2, True


def correct_aspect_ratio(
    x1: float, y1: float, x2: float, y2: float,
    target_aspect_ratio: float,
    fixed_points: List[str] = []
) -> Tuple[float, float, float, float, bool]:
    """
    四角形の頂点座標をアスペクト比に合わせて補正する関数

    Parameters:
        x1: 左上のX座標
        y1: 左上のY座標
        x2: 右下のX座標
        y2: 右下のY座標
        target_aspect_ratio: 目標のアスペクト比 (幅/高さ)
        fixed_points: 固定する頂点のリスト
            選択肢: 'top_left', 'top_right', 'bottom_left', 'bottom_right'

    Returns:
        Tuple[float, float, float, float, bool]: (x1, y1, x2, y2, was_corrected)
        - x1, y1, x2, y2: 補正後の座標
        - was_corrected: 補正が行われた場合True
    """
    # すべての点が固定されている場合は補正を行わない
    all_points = {'top_left', 'top_right', 'bottom_left', 'bottom_right'}
    if all_points.issubset(set(fixed_points)):
        return x1, y1, x2, y2, False
    
    # 縦横比補正なし
    if target_aspect_ratio <= 0:
        return x1, y1, x2, y2, False

    # 最小サイズを適用（0除算防止）
    min_width = 1
    min_height = 1

    # 現在のサイズを取得
    current_width = abs(x2 - x1)
    current_height = abs(y2 - y1)

    # 方向を保持
    sign_x = np.sign(x2 - x1)
    sign_y = np.sign(y2 - y1)

    # 現在のアスペクト比を計算
    current_ratio = max(current_width, min_width) / max(current_height, min_height)
    if current_ratio > target_aspect_ratio:
        min_width = min_height * target_aspect_ratio
    else:
        min_height = min_width / target_aspect_ratio

    # 補正が必要かどうかの判定
    needs_correction = (
        abs(current_ratio - target_aspect_ratio) > 0.0001 or
        current_width < min_width or
        current_height < min_height
    )
    if not needs_correction:
        return x1, y1, x2, y2, False

    def flip_fixed_points(fixed_points, x=True):
        if x == True:
            for i, fp in enumerate(fixed_points):
                if "left" in fp:
                    fixed_points[i] = fixed_points[i].replace("left", "right")
                if "right" in fp:
                    fixed_points[i] = fixed_points[i].replace("right", "left")
        else:
            for i, fp in enumerate(fixed_points):
                if "top" in fp:
                    fixed_points[i] = fixed_points[i].replace("top", "bottom")
                if "bottom" in fp:
                    fixed_points[i] = fixed_points[i].replace("bottom", "top")

    # 反転チェック
    """
    if x1 > x2:
        x1, x2 = x2, x1
        #x1 = x2 - min_width
        flip_fixed_points(fixed_points, x=True)
        sign_x = np.sign(x2 - x1)
    if y1 > y2:
        y1, y2 = y2, y1
        #y1 = y2 - min_height
        flip_fixed_points(fixed_points, x=False)    
        sign_y = np.sign(y2 - y1)
    """
    # 新しいサイズを計算
    if current_ratio > target_aspect_ratio:
        new_height = current_height
        new_width = new_height * target_aspect_ratio
    else:
        new_width = current_width
        new_height = new_width / target_aspect_ratio

    # 最小サイズを確保
    new_width = max(new_width, min_width)
    new_height = max(new_height, min_height)

    # 固定点なしの場合
    if not fixed_points:
        # 中心を維持しながらアスペクト比を調整
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        half_width = new_width / 2
        half_height = new_height / 2

        return (
            center_x - half_width * sign_x,
            center_y - half_height * sign_y,
            center_x + half_width * sign_x,
            center_y + half_height * sign_y,
            True
        )


    # 1点が固定の場合
    if len(fixed_points) == 1:
        fixed_point = fixed_points[0]
        
        # 固定点を基準に他の点を配置
        if fixed_point == 'top_left':
            return x1, y1, x1 + new_width * sign_x, y1 + new_height * sign_y, True
        elif fixed_point == 'top_right':
            return x2 - new_width * sign_x, y1, x2, y1 + new_height * sign_y, True
        elif fixed_point == 'bottom_left':
            return x1, y2 - new_height * sign_y, x1 + new_width * sign_x, y2, True
        else:  # bottom_right
            return x2 - new_width * sign_x, y2 - new_height * sign_y, x2, y2, True

    # 2点が固定（辺が固定）の場合
    if len(fixed_points) == 2:

        if 'top_left' in fixed_points and 'top_right' in fixed_points:
            # 上辺が固定 - 高さを維持して幅を中心から均等に調整
            current_height = abs(y2 - y1)
            center_x = (x1 + x2) / 2
            new_width = current_height * target_aspect_ratio
            new_width = max(new_width, min_width)
            half_width = new_width / 2
            return center_x - half_width * sign_x, y1, center_x + half_width * sign_x, y1 + current_height * sign_y, True

        elif 'bottom_left' in fixed_points and 'bottom_right' in fixed_points:
            # 下辺が固定 - 高さを維持して幅を中心から均等に調整
            current_height = abs(y2 - y1)
            center_x = (x1 + x2) / 2
            new_width = current_height * target_aspect_ratio
            new_width = max(new_width, min_width)
            half_width = new_width / 2
            return center_x - half_width * sign_x, y2 - current_height * sign_y, center_x + half_width * sign_x, y2, True

        elif 'top_left' in fixed_points and 'bottom_left' in fixed_points:
            # 左辺が固定 - 幅を維持して高さを中心から均等に調整
            current_width = abs(x2 - x1)
            center_y = (y1 + y2) / 2
            new_height = current_width / target_aspect_ratio
            new_height = max(new_height, min_height)
            half_height = new_height / 2
            return x1, center_y - half_height * sign_y, x1 + current_width * sign_x, center_y + half_height * sign_y, True

        elif 'top_right' in fixed_points and 'bottom_right' in fixed_points:
            # 右辺が固定 - 幅を維持して高さを中心から均等に調整
            current_width = abs(x2 - x1)
            center_y = (y1 + y2) / 2
            new_height = current_width / target_aspect_ratio
            new_height = max(new_height, min_height)
            half_height = new_height / 2
            return x2 - current_width * sign_x, center_y - half_height * sign_y, x2, center_y + half_height * sign_y, True


    # 対角の2点が固定の場合は、最初の固定点を基準に処理（対角でない一点）
    unfixed = list(all_points - set(fixed_points))[0]
    #first_point = fixed_points[0]

    if unfixed == 'bottom_right':
        return x1, y1, x1 + new_width * sign_x, y1 + new_height * sign_y, True

    elif unfixed == 'bottom_left':
        return x2 - new_width * sign_x, y1, x2, y1 + new_height * sign_y, True

    elif unfixed == 'top_right':
        return x1, y2 - new_height * sign_y, x1 + new_width * sign_x, y2, True

    elif unfixed == 'top_left':
        return x2 - new_width * sign_x, y2 - new_height * sign_y, x2, y2, True

    return x1, y1, x2, y2, False

class CropApp(KVApp):

    def build(self):
        root = KVFloatLayout()
        # ここで縦横サイズとスケールを指定
        crop_editor = CropEditor(input_width=dp(800), input_height=dp(600), input_angle=30, scale=0.8, aspect_ratio=0, crop_rect=(0, dp(100), dp(800), (dp(100)+dp(400))))
        crop_editor.pos = (dp(100), dp(100))
        root.add_widget(crop_editor)
        return root

if __name__ == '__main__':
    CropApp().run()