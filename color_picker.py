
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.slider import MDSlider
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivy.properties import ListProperty, NumericProperty
from kivy.graphics import Color, Ellipse, Quad
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.lang import Builder as KVBuilder
from colorsys import rgb_to_hls, hls_to_rgb
import math

import param_slider

class CWColorPreview(MDCard):
    color = ListProperty([0.5, 0.5, 0.5, 1])

class CWColorWheel(MDBoxLayout):
    selected_color = ListProperty([0.5, 0.5, 0.5, 1])
    hue = NumericProperty(0)
    lightness = NumericProperty(0.5)
    saturation = NumericProperty(0)  # 初期値を0に変更
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.after:
            # 選択位置のマーカー
            self.marker_color = Color(1, 1, 1, 1)
            self.marker_size = dp(10)
            self.marker = Ellipse(size=(self.marker_size, self.marker_size))
        
        Clock.schedule_once(self.draw_wheel)
        self.bind(pos=self.update_wheel, size=self.update_wheel)
    
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
            self.marker_size = dp(10)
            self.marker = Ellipse(size=(self.marker_size, self.marker_size))
            """
            self.update_marker()    
            
    def update_wheel(self, *args):
        Clock.schedule_once(self.draw_wheel)
    
    def update_marker(self):
        angle = self.hue * 2 * math.pi
        radius = self.saturation * self.wheel_radius
        x = self.center_x + radius * math.cos(angle)
        y = self.center_y + radius * math.sin(angle)
        self.marker.pos = (x - self.marker_size / 2, y - self.marker_size / 2)
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_double_tap:
                self._reset_color()
                return True
            else:
                self._update_color_from_touch(touch)
                return True
        return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            self._update_color_from_touch(touch)
            return True
        return super().on_touch_move(touch)

    def _reset_color(self):
        self.hue = 0
        self.lightness = 0.5
        self.saturation = 0
        self.selected_color = [0.5, 0.5, 0.5, 1]
        self.update_marker()
    
    def _update_color_from_touch(self, touch):
        dx = touch.pos[0] - self.center_x
        dy = touch.pos[1] - self.center_y
        
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance >= self.wheel_radius:
            self.saturation = 1.0
        else:
            self.saturation = distance / self.wheel_radius
        
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        self.hue = angle / (2 * math.pi)
        
        r, g, b = hls_to_rgb(self.hue, self.lightness, self.saturation)
        self.selected_color = [r, g, b, 1]
        self.update_marker()

    def _set_color_from_rgb(self, rgba):
        # RGBをHLSに変換
        r, g, b, a = rgba
        hue, lightness, saturation = rgb_to_hls(r, g, b)
        
        # プロパティを更新
        self.hue = hue
        self.lightness = lightness
        self.saturation = saturation
        
        # 選択色を更新
        self.selected_color = [r, g, b, 1]
        
        # マーカーの位置を更新
        self.update_marker()

class CWColorPicker(MDCard):
    current_color = ListProperty([0.5, 0.5, 0.5, 1])

    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.ids.color_wheel.bind(selected_color=self.on_wheel_color)
        self.bind(current_color=self.on_current_color)

    def update_preview(self, instance, value):
        self.ids.preview.color = value

    def on_current_color(self, instance, value):
        r, g, b, _ = self.current_color
        self.ids.slider_red.set_slider_value(r * 255)
        self.ids.slider_green.set_slider_value(g * 255)
        self.ids.slider_blue.set_slider_value(b * 255)

        h, l, s = rgb_to_hls(r, g, b)
        self.ids.slider_hue.set_slider_value(h * 360)
        self.ids.slider_lum.set_slider_value(l * 100)
        self.ids.slider_sat.set_slider_value(s * 100)

        self.ids.color_wheel._set_color_from_rgb(self.current_color)
        self.update_preview(instance, value)

    def on_wheel_color(self, instance, value):
        self.current_color = value
    
    def on_slider_change_rgb(self):
        r = self.ids.slider_red.value / 255
        g = self.ids.slider_green.value / 255
        b = self.ids.slider_blue.value / 255
        self.current_color = [r, g, b, 1]

    def on_slider_change_hls(self):
        h = self.ids.slider_hue.value / 360
        l = self.ids.slider_lum.value / 100
        s = self.ids.slider_sat.value / 100
        r, g, b = hls_to_rgb(h, l, s)
        self.current_color = [r, g, b, 1]

    def get_current_color_hls(self):
        return rgb_to_hls(*self.current_color)

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
