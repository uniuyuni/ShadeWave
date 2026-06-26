
import numpy as np
from scipy.interpolate import splprep, splev
from kivy.app import App as KVApp
from kivy.uix.widget import Widget as KVWidget
from kivy.uix.label import Label as KVLabel
from kivy.graphics import Color, Line, Ellipse, Rectangle, Translate, PushMatrix, PopMatrix
from kivy.graphics.texture import Texture as KVTexture
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.properties import NumericProperty as KVNumericProperty
from kivy.properties import ObjectProperty as KVObjectProperty
from kivy.properties import StringProperty as KVStringProperty
import logging

import utils.kvutils as kvutils

KR = 0.2126
KG = 0.7152
KB = 0.0722

_BACKGROUND_TEXTURE_CACHE = {}


def _linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


def _make_hlc_hue_background_texture(width=1024, height=1):
    cache_key = ("hlc_hue", width, height)
    texture = _BACKGROUND_TEXTURE_CACHE.get(cache_key)
    if texture is not None:
        return texture

    hue = np.linspace(0.0, 360.0, width, endpoint=False, dtype=np.float32)
    hue_rad = np.deg2rad(hue)
    cb = np.cos(hue_rad)
    cr = np.sin(hue_rad)

    rgb_line = np.stack(
        (
            cr,
            -(KR / KG) * cr - (KB / KG) * cb,
            cb,
        ),
        axis=1,
    )
    rgb_min = np.min(rgb_line, axis=1, keepdims=True)
    rgb_max = np.max(rgb_line, axis=1, keepdims=True)
    rgb_line = (rgb_line - rgb_min) / np.maximum(rgb_max - rgb_min, 1e-8)
    rgb_line = _linear_to_srgb(rgb_line)

    img = np.round(rgb_line * 255.0).astype(np.uint8)[np.newaxis, :, :]
    if height > 1:
        img = np.repeat(img, height, axis=0)

    texture = KVTexture.create(size=(width, height), colorfmt='rgb', bufferfmt='ubyte')
    texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
    texture.mag_filter = 'linear'
    texture.min_filter = 'linear'
    _BACKGROUND_TEXTURE_CACHE[cache_key] = texture
    return texture


_BACKGROUND_RENDERERS = {
    "hlc_hue": _make_hlc_hue_background_texture,
}


class DraggablePoint():

    def __init__(self, **kwargs):
        self.x = kwargs.get('x', 0.0)
        self.y = kwargs.get('y', 0.0)

    def get_width(self):
        return kvutils.dpi_scale_width(10)

    def get_height(self):
        return kvutils.dpi_scale_height(10)

    def collide_point(self, x, y, w, h):
        collide_width =  self.get_width() / w
        collide_height = self.get_height() / h

        if abs(self.x-x) <= collide_width and abs(self.y-y) <= collide_height:
            return True
        return False


class CurveWidget(KVBoxLayout):
    curve = KVNumericProperty(0)
    start_x = KVNumericProperty(0.0)
    start_y = KVNumericProperty(0.0)
    end_x = KVNumericProperty(1.0)
    end_y = KVNumericProperty(1.0)
    before_edit = KVNumericProperty(-1)
    after_edit = KVNumericProperty(-1)
    background_renderer = KVStringProperty("")
    background_source = KVStringProperty("")
    background_texture = KVObjectProperty(None, allownone=True)
    background_opacity = KVNumericProperty(1.0)

    def __init__(self, **kwargs):
        super(CurveWidget, self).__init__(**kwargs)
        self.touch_self = False

    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.set_point_list(None)
        
        self.bind(size=self.update_grid)
        self.bind(pos=self.update_grid)
        self.bind(background_renderer=self.update_grid)
        self.bind(background_source=self.update_grid)
        self.bind(background_texture=self.update_grid)
        self.bind(background_opacity=self.update_grid)
        #self.update_grid()

    def __update_points(self, *args):
        pass

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

        self.touch_self = True
        
        local_x = (touch.x - self.x)/self.width
        local_y = (touch.y - self.y)/self.height

        if touch.button == 'right':
            for point in self.points:
                if point not in [self.start_point, self.end_point] and point.collide_point(local_x, local_y, self.width, self.height):
                    self.points.remove(point)
                    self.__update_curve()
                    self.before_edit = self.curve
                    self.curve += 1
                    return True
        else:
            for point in self.points:
                if point.collide_point(local_x, local_y, self.width, self.height):
                    self.selected_point = point
                    self.before_edit = self.curve
                    touch.grab(self)
                    return True # Select existing point
                
            point = DraggablePoint()
            point.x, point.y = local_x, local_y
            self.points.append(point)
            self.selected_point = point
            self.__update_curve()
            self.before_edit = self.curve
            self.curve += 1
            touch.grab(self)
            return True
        
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        dragging_self = self.selected_point is not None and (
            self.touch_self or getattr(touch, "grab_current", None) is self
        )
        if not dragging_self and not self.collide_point(*touch.pos):
            return False
        
        local_x = (touch.x - self.x)/self.width
        local_y = (touch.y - self.y)/self.height

        if self.selected_point:
            new_x = float(np.clip(local_x, 0.0, 1.0))
            new_y = float(np.clip(local_y, 0.0, 1.0))
            self.selected_point.x, self.selected_point.y = new_x, new_y
            self.__update_curve()
            self.curve += 1

        return True

    def on_touch_up(self, touch):
        if self.touch_self:
            if getattr(touch, "grab_current", None) is self:
                touch.ungrab(self)
            self.selected_point = None
            if self.before_edit != self.curve:
                self.after_edit = self.curve
            self.touch_self = False
            return True

        return super().on_touch_up(touch)

    def update_grid(self, instance, size):
        self.__update_points()
        self.__update_curve()

    def __draw_background(self, width, height):
        if self.background_texture is not None:
            Color(1, 1, 1, self.background_opacity)
            Rectangle(texture=self.background_texture, pos=(0, 0), size=(width, height))
            return

        if self.background_source:
            Color(1, 1, 1, self.background_opacity)
            Rectangle(source=self.background_source, pos=(0, 0), size=(width, height))
            return

        if self.background_renderer:
            renderer = _BACKGROUND_RENDERERS.get(self.background_renderer)
            if renderer is None:
                logging.warning(f"Unknown curve background renderer: {self.background_renderer}")
                return

            texture = renderer()
            Color(1, 1, 1, self.background_opacity)
            Rectangle(texture=texture, pos=(0, 0), size=(width, height))

    def __update_curve(self):
        # 背景・グリッドは canvas.before（リセットボタンより奥）、カーブと制御点は
        # canvas.after（リセットボタンより手前）に描画する。子ウィジェットである
        # リセットボタンは canvas.before と canvas.after の間に描かれるため、
        # 結果としてグリッドとカーブの間にボタンが表示される。
        self.canvas.before.clear()  # Clear the canvas before redrawing
        self.canvas.after.clear()

        width, height = self.size   # 0になる場合があるのをなんとかしたい
        if width <= 0 or height <= 0:
            return

        with self.canvas.before:
            # ローカル座標系で描画するために変換行列をプッシュ
            PushMatrix()
            Translate(self.x, self.y)

            self.__draw_background(width, height)

            # Draw the grid
            Color(0.5, 0.5, 0.5)
            for i in range(0, 5):
                Line(points=[width / 4 * i, 0, width / 4 * i, height], width=1)
                Line(points=[0, height / 4 * i, width, height / 4 * i], width=1)

            # 変換行列をポップして元に戻す
            PopMatrix()

        with self.canvas.after:
            # ローカル座標系で描画するために変換行列をプッシュ
            PushMatrix()
            Translate(self.x, self.y)

            pts = sorted([(p.x, p.y) for p in self.points])

            Color(1, 1, 1)
            try:
                x, y = zip(*pts)
                x = np.array(x, dtype=np.float32)*width
                y = np.array(y, dtype=np.float32)*height
                tck, u = splprep([x, y], k=min(3, len(x)-1), s=0)  # Adjust `k` appropriately
                unew = np.linspace(0, 1.0, 100)
                out = splev(unew, tck)

                # Clipping processing
                points = []
                for i in range(len(out[0])):
                    x_coord, y_coord = out[0][i], out[1][i]
                    x_coord = np.clip(x_coord, 0, width)
                    y_coord = np.clip(y_coord, 0, height)
                    points.append((x_coord, y_coord))

                points_flat = [coord for point in points for coord in point]  # Flatten points
                if points_flat:
                    Line(points=points_flat, width=1.5)

            except ValueError as e:
                logging.error(f"Error during spline math: {e}")

            for point in self.points:
                Ellipse(pos=(point.x*width - point.get_width()/2, point.y*height - point.get_height()/2), size=(point.get_width(), point.get_height()))

            # 変換行列をポップして元に戻す
            PopMatrix()
    
    def get_point_list(self, flag=False):
        point_list = [(p.x, p.y) for p in self.points]
        if flag == False:
            if self.is_init_curve(point_list) == True:
                return None
        return point_list
    
    def is_init_curve(self, point_list):
        return True if len(point_list) == 2 and point_list[0][0] == self.start_x and point_list[0][1] == self.start_y and point_list[1][0] == self.end_x and point_list[1][1] == self.end_y else False

    def set_point_list(self, point_list, history=False):
        if history == True:
            self.before_edit = self.curve

        if point_list is not None:
            point_list = sorted((pl[0], pl[1]) for pl in point_list)
            self.points = [DraggablePoint(x=pl[0], y=pl[1]) for pl in point_list]
            self.start_point = self.points[0]
            self.end_point = self.points[len(self.points)-1]
            self.selected_point = None
        else:
            self.points = []
            self.selected_point = None

            # Add start and end points
            self.start_point = DraggablePoint(x=self.start_x, y=self.start_y)
            self.end_point = DraggablePoint(x=self.end_x, y=self.end_y)
            self.points.append(self.start_point)
            self.points.append(self.end_point)
        
        if history == True:
            self.curve += 1
            self.after_edit = self.curve
            
        self.__update_curve()

class ToneCurveApp(KVApp):

    def build(self):
        root = KVBoxLayout()
        label = KVLabel()
        label.text = "Tone Curve"
        root.add_widget(label)
        tone_curve_widget = CurveWidget(size_hint=(1, 1))
        root.add_widget(tone_curve_widget)
        return root


if __name__ == '__main__':
    ToneCurveApp().run()
