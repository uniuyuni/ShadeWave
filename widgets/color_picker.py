
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.slider import MDSlider
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivy.properties import ListProperty as KVListProperty, NumericProperty as KVNumericProperty
from kivy.graphics import Color, Ellipse, Quad
from kivy.clock import Clock as KVClock
from kivy.lang import Builder as KVBuilder
from kivy.config import Config as KVConfig
from colorsys import rgb_to_hls, hls_to_rgb
import math
import logging
import threading

import macos as device
import widgets.param_slider
import utils.kvutils as kvutils

class CWColorPreview(MDCard):
    color = KVListProperty([0.5, 0.5, 0.5, 1])

    # 親 CWColorPicker への参照（on_kv_post で設定）
    picker = None

    def on_touch_down(self, touch):
        # ダブルクリックで「文字列→色」入力ダイアログを開く
        if self.collide_point(*touch.pos) and touch.is_double_tap:
            self._open_text_color_dialog()
            return True
        return super().on_touch_down(touch)

    def _open_text_color_dialog(self):
        if self.picker is None:
            return
        # Native macOS prompt. Color descriptions can use non-ASCII text.
        try:
            text = device.prompt_native(
                message="Describe a color (for example: sunset orange)",
                title="Enter Color",
                default="",
                show_cancel=True,
                ascii_only=False,
            )
        except Exception as e:
            logging.warning(f"color text prompt failed: {e}")
            return
        # キャンセル(None)・空入力は何もしない
        if not text or not text.strip():
            return

        picker = self.picker

        # color_resolver は LLM フォールバックで重くなり得るのでバックグラウンドで解決し、
        # 結果（RGB 0-255）をメインスレッドへ戻して反映する。
        def worker():
            try:
                from cores import color_resolver
                rgb = color_resolver.resolve_color(text)
            except Exception as e:
                logging.warning(f"color_resolver failed for {text!r}: {e}")
                return
            if rgb is not None:
                KVClock.schedule_once(lambda dt: picker.set_color_from_rgb(rgb), 0)

        threading.Thread(target=worker, daemon=True).start()

class CWColorWheel(MDBoxLayout):
    selected_color = KVListProperty([0, 0.5, 0])
    hue = KVNumericProperty(0)
    lightness = KVNumericProperty(0.5)
    saturation = KVNumericProperty(0)  # 初期値を0に変更
    before_edit = KVNumericProperty(0)
    after_edit = KVNumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.after:
            # 選択位置のマーカー
            self.marker_color = Color(1, 1, 1, 1)
            self.marker_size = self._marker_size()
            self.marker = Ellipse(size=(self.marker_size, self.marker_size))
        
        KVClock.schedule_once(self.draw_wheel)
        self.bind(pos=self.update_wheel, size=self.update_wheel)
        self._ignore_next_up = False
        self._single_tap_event = None
    
    def draw_wheel(self, dt):
        self.wheel_radius = min(self.size[0], self.size[1]) / 2
        self.center_x = self.pos[0] + self.size[0] / 2
        self.center_y = self.pos[1] + self.size[1] / 2
        
        #self.canvas.clear()
        with self.canvas.after:
            """
            segments = 360  # 円周方向の分割数
            radial_steps = 25  # 半径方向のステップ数
            
            for r in range(radial_steps):
                inner_dist = r / radial_steps
                outer_dist = (r + 1) / radial_steps
                
                for angle in range(segments):
                    current_angle = math.radians(angle)
                    next_angle = math.radians((angle + 1) % segments)
                    
                    # 内側と外側の頂点座標を計算
                    inner_x1 = self.center_x + inner_dist * self.wheel_radius * math.cos(current_angle)
                    inner_y1 = self.center_y + inner_dist * self.wheel_radius * math.sin(current_angle)
                    inner_x2 = self.center_x + inner_dist * self.wheel_radius * math.cos(next_angle)
                    inner_y2 = self.center_y + inner_dist * self.wheel_radius * math.sin(next_angle)
                    outer_x1 = self.center_x + outer_dist * self.wheel_radius * math.cos(next_angle)
                    outer_y1 = self.center_y + outer_dist * self.wheel_radius * math.sin(next_angle)
                    outer_x2 = self.center_x + outer_dist * self.wheel_radius * math.cos(current_angle)
                    outer_y2 = self.center_y + outer_dist * self.wheel_radius * math.sin(current_angle)
                    
                    # 色の計算
                    hue = angle / segments
                    saturation = outer_dist
                    lightness = 0.5
                    r, g, b = hls_to_rgb(hue, lightness, saturation)
                    Color(r, g, b, 1)
                    
                    # 三角形を描画
                    Quad(points=[
                        inner_x1, inner_y1,  # 内側の1点
                        inner_x2, inner_y2,  # 内側の2点
                        outer_x1, outer_y1,  # 外側の1点
                        outer_x2, outer_y2   # 外側の次の点
                    ])
            
            # 選択位置のマーカー
            self.marker_color = Color(1, 1, 1, 1)
            self.marker_size = self._marker_size()
            self.marker = Ellipse(size=(self.marker_size, self.marker_size))
            """
            self.update_marker()    
            
    def update_wheel(self, *args):
        KVClock.schedule_once(self.draw_wheel)
    
    def update_marker(self):
        marker_size = self._marker_size()
        if marker_size != self.marker_size:
            self.marker_size = marker_size
            self.marker.size = (marker_size, marker_size)
        angle = self.hue * 2 * math.pi
        radius = self.saturation * self.wheel_radius
        x = self.center_x + radius * math.cos(angle)
        y = self.center_y + radius * math.sin(angle)
        self.marker.pos = (x - self.marker_size / 2, y - self.marker_size / 2)

    def _marker_size(self):
        return kvutils.dpi_scale_width(10)
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Check if touch is inside the wheel radius
            dx = touch.pos[0] - self.center_x
            dy = touch.pos[1] - self.center_y
            if math.sqrt(dx*dx + dy*dy) <= self.wheel_radius:
                touch.ud['cw_active'] = True
                self.before_edit += 1
                if touch.is_double_tap:
                    if self._single_tap_event:
                        self._single_tap_event.cancel()
                    self._ignore_next_up = True
                    self._reset_color()
                else:
                    self._ignore_next_up = True
                    touch_info = {'x': touch.pos[0], 'y': touch.pos[1]}
                    self._single_tap_event = KVClock.schedule_once(
                        lambda dt: self.on_single_tap(touch_info),
                        KVConfig.getint('postproc', 'double_tap_time') * 0.001 + 0.1
                    )
                return True
        return super().on_touch_down(touch)
    
    def on_single_tap(self, touch_info):
        self._update_color_from_touch(touch_info)
        self.after_edit += 1
    
    def on_touch_move(self, touch):
        if touch.ud.get('cw_active'):
            touch_info = {'x': touch.pos[0], 'y': touch.pos[1]}
            self._update_color_from_touch(touch_info)
            # return True  # Consume the event if desired, but updating color is key.
            # Original didn't prevent propagation, but we grabbed it in down?
            # If we returned True in down, we grabbed it. 
            # Kivy docs: "When a widget grabs a touch, it will receive the touch move/up events...".
            # It doesn't strictly stop others unless we return True here too? 
            # Actually return value of move stops propagation to other widgets? 
            # Let's keep consistent with valid behavior: return True if we handled it.
            return True 
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.ud.get('cw_active'):
            if self._ignore_next_up:
                self._ignore_next_up = False
                return True
            self.after_edit += 1
            return True
        return super().on_touch_up(touch)

    def _reset_color(self):
        self.hue = 0
        self.lightness = 0.5
        self.saturation = 0
        self.selected_color = [0, 0.5, 0]
        self.update_marker()
    
    def _update_color_from_touch(self, touch_info):
        dx = touch_info['x'] - self.center_x
        dy = touch_info['y'] - self.center_y
        
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance >= self.wheel_radius:
            self.saturation = 1.0
        else:
            self.saturation = distance / self.wheel_radius
        
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        self.hue = angle / (2 * math.pi)
        
        self.selected_color = [self.hue, self.lightness, self.saturation]
        self.update_marker()

    def _set_color_from_hls(self, hls):
        hue, lightness, saturation = hls
        
        # プロパティを更新
        self.hue = hue
        self.lightness = lightness
        self.saturation = saturation
        
        # 選択色を更新
        self.selected_color = [hue, lightness, saturation]
        
        # マーカーの位置を更新
        self.update_marker()

class CWColorPicker(MDCard):
    current_color = KVListProperty([0, 0.5, 0])
    before_edit = KVNumericProperty(0)
    after_edit = KVNumericProperty(0)

    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.ids.color_wheel.bind(selected_color=self.on_wheel_color)
        self.ids.color_wheel.bind(before_edit=self.on_wheel_before_edit)
        self.ids.color_wheel.bind(after_edit=self.on_wheel_after_edit)
        self.bind(current_color=self.on_current_color)

        # プレビューのダブルクリック（文字列→色）から色を反映できるよう参照を渡す
        self.ids.preview.picker = self

    def on_current_color(self, instance, value):
        h, l, s = value
        self.ids.slider_hue.set_slider_value(h * 360)
        self.ids.slider_lum.set_slider_value(l * 100)
        self.ids.slider_sat.set_slider_value(s * 100)

        r, g, b = hls_to_rgb(h, l, s)
        self.ids.slider_red.set_slider_value(r * 255)
        self.ids.slider_green.set_slider_value(g * 255)
        self.ids.slider_blue.set_slider_value(b * 255)

        self.ids.color_wheel._set_color_from_hls(value)
        self.ids.preview.color = self.get_current_color_rgb()

    def on_wheel_color(self, instance, value):
        self.current_color = value
    
    def on_slider_change_rgb(self):
        r = self.ids.slider_red.value / 255
        g = self.ids.slider_green.value / 255
        b = self.ids.slider_blue.value / 255
        h, l, s = rgb_to_hls(r, g, b)
        self.current_color = [h, l, s]

    def on_slider_change_hls(self):
        h = self.ids.slider_hue.value / 360
        l = self.ids.slider_lum.value / 100
        s = self.ids.slider_sat.value / 100
        self.current_color = [h, l, s]

    def set_slider_value(self, value):
        h, l, s = value
        self.current_color = [h / 360, l / 100, s / 100]

    def get_current_color_rgb(self):
        r, g, b = hls_to_rgb(*self.current_color)
        return [r, g, b, 1]

    def set_color_from_rgb(self, rgb):
        """color_resolver 由来の RGB(0-255) を現在色へ反映する。
        ホイール編集と同様に before_edit/after_edit を増分して編集履歴に乗せる。"""
        r, g, b = [max(0, min(255, int(c))) / 255.0 for c in rgb[:3]]
        h, l, s = rgb_to_hls(r, g, b)
        self.before_edit += 1            # 履歴開始（変更前の状態をスナップ）
        self.current_color = [h, l, s]   # on_current_color → スライダー/ホイール/preview/効果適用
        self.after_edit += 1             # 履歴確定

    def on_wheel_before_edit(self, instance, value):
        self.before_edit += 1

    def on_wheel_after_edit(self, instance, value):
        self.after_edit += 1

    def on_slider_before_edit(self):
        self.before_edit += 1

    def on_slider_after_edit(self):
        self.after_edit += 1

class MainScreen(MDScreen):
    pass

class CustomColorPickerApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        KVBuilder.load_file('color_picker.kv')
        screen = MainScreen()
        screen.add_widget(CWColorPicker(id='color_picker'))
        return screen

if __name__ == '__main__':
    CustomColorPickerApp().run()
