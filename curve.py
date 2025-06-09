
import numpy as np
from scipy.interpolate import splprep, splev
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Ellipse, Translate, PushMatrix, PopMatrix
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import NumericProperty
from kivy.metrics import dp
import logging

import utils

from kivy.config import Config
Config.set('input', 'mouse', 'mouse,disable_multitouch')  # 右クリック赤丸消去
Config.set('kivy', 'exit_on_escape', '0')  # kivy ESC無効
#Config.set('graphics', 'width', 1200)
#Config.set('graphics', 'height', 800)

class DraggablePoint():

    def __init__(self, **kwargs):
        self.x = kwargs.get('x', 0.0)
        self.y = kwargs.get('y', 0.0)

    def get_width(self):
        return utils.dpi_scale_width(10)

    def get_height(self):
        return utils.dpi_scale_height(10)

    def collide_point(self, x, y, w, h):
        collide_width =  self.get_width() / w
        collide_height = self.get_height() / h

        if abs(self.x-x) <= collide_width and abs(self.y-y) <= collide_height:
            return True
        return False


class CurveWidget(Widget):
    curve = NumericProperty(0)
    start_x = NumericProperty(0.0)
    start_y = NumericProperty(0.0)
    end_x = NumericProperty(1.0)
    end_y = NumericProperty(1.0)

    def __init__(self, **kwargs):
        super(CurveWidget, self).__init__(**kwargs)

    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)

        self.set_point_list(None)
        
        self.bind(size=self.update_grid)
        self.bind(pos=self.update_grid)
        self.update_grid()

    def __update_points(self, *args):
        pass

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        
        local_x = (touch.x - self.x)/self.width
        local_y = (touch.y - self.y)/self.height

        if touch.button == 'right':
            for point in self.points:
                if point not in [self.start_point, self.end_point] and point.collide_point(local_x, local_y, self.width, self.height):
                    self.points.remove(point)
                    self.__update_curve()
                    self.curve += 1
                    return
        else:
            for point in self.points:
                if point.collide_point(local_x, local_y, self.width, self.height):
                    self.selected_point = point
                    return  # Select existing point
                
            point = DraggablePoint()
            point.x, point.y = local_x, local_y
            self.points.append(point)
            self.selected_point = point
            self.__update_curve()
            self.curve += 1

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        
        local_x = (touch.x - self.x)/self.width
        local_y = (touch.y - self.y)/self.height

        if self.selected_point:
            min_x, max_x = 0, self.width - self.selected_point.get_width()
            min_y, max_y = 0, self.height - self.selected_point.get_height()
            if self.selected_point in [self.start_point, self.end_point]:
                min_x, min_y, max_x, max_y = 0, 0, self.width, self.height
            new_x = min(max(local_x, min_x), max_x)  # Clamp x within appropriate boundaries
            new_y = min(max(local_y, min_y), max_y)  # Clamp y within appropriate boundaries
            self.selected_point.x, self.selected_point.y = new_x, new_y
            self.__update_curve()
            self.curve += 1

    def on_touch_up(self, touch):
        self.selected_point = None

    def update_grid(self, *args):
        self.__update_points()
        self.__update_curve()

    def __update_curve(self):
        self.canvas.clear()  # Clear the canvas before redrawing
        with self.canvas:
            # ローカル座標系で描画するために変換行列をプッシュ
            PushMatrix()
            Translate(self.x, self.y)
            
            # Draw the grid
            Color(0.5, 0.5, 0.5)
            for i in range(0, 5):
                Line(points=[self.width / 4 * i, 0, self.width / 4 * i, self.height], width=1)
                Line(points=[0, self.height / 4 * i, self.width, self.height / 4 * i], width=1)

            pts = sorted([(p.x, p.y) for p in self.points])

            Color(1, 1, 1)
            try:
                x, y = zip(*pts)
                x = np.array(x)*self.width
                y = np.array(y)*self.height
                tck, u = splprep([x, y], k=min(3, len(x)-1), s=0)  # Adjust `k` appropriately
                unew = np.linspace(0, 1.0, 1000)
                out = splev(unew, tck)

                # Clipping processing
                points = []
                for i in range(len(out[0])):
                    x_coord, y_coord = out[0][i], out[1][i]
                    x_coord = np.clip(x_coord, 0, self.width)
                    y_coord = np.clip(y_coord, 0, self.height)
                    points.append((x_coord, y_coord))

                points_flat = [coord for point in points for coord in point]  # Flatten points
                if points_flat:
                    Line(points=points_flat, width=1.5)

            except ValueError as e:
                logging.error(f"Error during spline math: {e}")

            for point in self.points:
                Ellipse(pos=(point.x*self.width - point.get_width()/2, point.y*self.height - point.get_height()/2), size=(point.get_width(), point.get_height()))
            
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

    def set_point_list(self, point_list):
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
        self.__update_curve()

class ToneCurveApp(App):

    def build(self):
        root = BoxLayout()
        label = Label()
        label.text = "Tone Curve"
        root.add_widget(label)
        tone_curve_widget = CurveWidget(size_hint=(1, 1))
        root.add_widget(tone_curve_widget)
        return root


if __name__ == '__main__':
    ToneCurveApp().run()