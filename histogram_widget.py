
import numpy as np
import cv2
from kivy.app import App as KVApp
from kivy.uix.image import Image as KVImage
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.graphics import Color as KVColor, Rectangle as KVRectangle, Line as KVLine, PushMatrix as KVPushMatrix, PopMatrix as KVPopMatrix
from kivy.metrics import dp

import core

class HistogramWidget(KVImage):
    
    def _load_image(self, image_path):
        # 画像を読み込み、ヒストグラムを計算
        pixels = cv2.imread(image_path)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        pixels = pixels.astype(np.float32)/256.0
        self.draw_histogram(pixels)

    def draw_histogram(self, pixels, blue_count, black_count):

        # 手動対数変換関数
        def manual_scale(data):
            """
            データをスケール変換
            """
            return np.sqrt(data)
        
        # RGB各チャンネルのヒストグラムを計算
        r_hist, g_hist, b_hist = [cv2.calcHist([pixels], [i], None, [256], [0, 2.0]) for i in range(3)]
        r_hist = r_hist.squeeze(axis=-1)
        r_hist[0] = max(0, r_hist[0] - black_count)
        r_hist = manual_scale(r_hist)
        g_hist = g_hist.squeeze(axis=-1)
        g_hist[0] = max(0, g_hist[0] - black_count)
        g_hist = manual_scale(g_hist)
        b_hist = b_hist.squeeze(axis=-1)
        b_hist[0] = max(0, b_hist[0] - black_count)
        b_hist[255] = max(0, b_hist[255] - blue_count)
        b_hist = manual_scale(b_hist)
        
        # 輝度の計算
        luminance = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        l_hist = cv2.calcHist([luminance], [0], None, [256], [0, 2.0])
        l_hist = l_hist.squeeze(axis=-1)
        l_hist[0] = max(0, l_hist[0] - black_count)
        l_hist = manual_scale(l_hist)
        #l_hist = np.clip(l_hist, 0, 255)

        # ヒストグラムの最大値を取得
        max_value = max(np.max(r_hist), np.max(g_hist), np.max(b_hist), np.max(l_hist))

        # ヒストグラムを描画
        self.canvas.clear()
        with self.canvas:
            KVColor((0.8, 0.8, 0.8, 1))
            KVLine(rectangle=(self.pos[0], self.pos[1], 256, 128+64), width=1)
            KVLine(rectangle=(self.pos[0]+256, self.pos[1], 256, 128+64), width=1)
        self.__draw_histogram_bars(r_hist, max_value, (1, 0, 0, 0.8))
        self.__draw_histogram_bars(g_hist, max_value, (0, 1, 0, 0.8))
        self.__draw_histogram_bars(b_hist, max_value, (0, 0, 1, 0.8))
        self.__draw_histogram_bars(l_hist, max_value, (0.8, 0.8, 0.8, 1))  # 輝度ヒストグラムをグレーで表示

    def __draw_histogram_bars(self, histogram, max_value, color, offset_x=0, offset_y=0):
        bar_width = 2
        offset_x += self.pos[0]
        offset_y += self.pos[1]
        
        with self.canvas:
            KVPushMatrix()
            KVColor(*color)
            for x, value in enumerate(histogram):
                height = min(127+64, (value / max_value) * (127+64))  # ヒストグラムの高さを設定
                KVRectangle(pos=(x * bar_width + offset_x, offset_y), size=(bar_width, height))
            KVPopMatrix()

class Histogram_WidgetApp(KVApp):
    def build(self):
        histogram = HistogramWidget()
        histogram._load_image("/Users/uniuyuni/PythonProjects/escargot/picture/DSCF0007.JPG")
        return histogram

if __name__ == '__main__':
    Histogram_WidgetApp().run()
