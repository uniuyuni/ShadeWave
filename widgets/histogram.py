
import numpy as np
import cv2
from kivy.app import App as KVApp
from kivy.uix.image import Image as KVImage
from kivy.graphics import Color as KVColor, Rectangle as KVRectangle, Line as KVLine, PushMatrix as KVPushMatrix, PopMatrix as KVPopMatrix, Scale as KVScale, Translate as KVTranslate

import macos as device

class HistogramWidget(KVImage):
    
    def _load_image(self, image_path):
        # 画像を読み込み、ヒストグラムを計算
        pixels = cv2.imread(image_path)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        pixels = pixels.astype(np.float32)/256.0
        self.draw_histogram(pixels)
        
    def set_histogram_data(self, pixels, blue_count=0, black_count=0):
        if pixels is None:
            self.canvas.clear()
            self.last_hist_data = None
            return
        self.draw_histogram(pixels, blue_count, black_count)

    def on_size(self, *args):
        if hasattr(self, 'last_hist_data') and self.last_hist_data:
            self.draw_histogram_from_data(self.last_hist_data)

    def on_pos(self, *args):
        if hasattr(self, 'last_hist_data') and self.last_hist_data:
            self.draw_histogram_from_data(self.last_hist_data)

    @staticmethod
    def calculate_histogram_data(pixels, blue_count, black_count):
        # 手動対数変換関数
        def manual_scale(data):
            """
            データをスケール変換
            """
            return np.sqrt(data)
        
        # RGB各チャンネルのヒストグラムを計算
        r_hist, g_hist, b_hist = [cv2.calcHist([pixels], [i], None, [256+256], [0, 1.5]) for i in range(3)]
        # calcHist は通常 (bins, 1) を返すが、環境によって既に (bins,) の場合があるため
        # squeeze(axis=-1) だと後者で ValueError になる。reshape(-1) ならどちらでも安全。
        r_hist = r_hist.reshape(-1)
        r_hist[0] = max(0, r_hist[0] - black_count)
        r_hist = manual_scale(r_hist)
        g_hist = g_hist.reshape(-1)
        g_hist[0] = max(0, g_hist[0] - black_count)
        g_hist = manual_scale(g_hist)
        b_hist = b_hist.reshape(-1)
        b_hist[0] = max(0, b_hist[0] - black_count)
        b_hist[255] = max(0, b_hist[255] - blue_count)
        b_hist = manual_scale(b_hist)

        # 輝度の計算
        luminance = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        l_hist = cv2.calcHist([luminance], [0], None, [256+256], [0, 1.5])
        l_hist = l_hist.reshape(-1)
        l_hist[0] = max(0, l_hist[0] - black_count)
        l_hist = manual_scale(l_hist)
        #l_hist = np.clip(l_hist, 0, 255)

        # ヒストグラムの表示スケールを取得。
        # 1つのbinに極端な値があると全体が潰れるため、最大値ではなく上位percentileを使う。
        hist_values = np.concatenate((r_hist, g_hist, b_hist, l_hist))
        max_value = max(1e-6, np.percentile(hist_values, 99.5))

        return (r_hist, g_hist, b_hist, l_hist, max_value)

    def draw_histogram_from_data(self, hist_data):
        self.last_hist_data = hist_data

        r_hist, g_hist, b_hist, l_hist, max_value = hist_data
        
        # ヒストグラムを描画
        self.canvas.clear()
        
        # Original fixed dimensions
        fixed_w = 256 + 64 + 32 + 128 + 32  # 512
        fixed_h = 128 + 64                  # 192
        
        # Calculate scaling factors
        scale_x = device.dpi_scale() * 0.5 * (self.width / (device.dpi_scale() * 256))
        scale_y = device.dpi_scale() * 0.5

        with self.canvas:
            KVPushMatrix()
            KVTranslate(self.x, self.y)
            KVScale(scale_x, scale_y, 1)
                    
            self.__draw_histogram_bars(r_hist, max_value, (1, 0, 0, 0.6))
            self.__draw_histogram_bars(g_hist, max_value, (0, 1, 0, 0.6))
            self.__draw_histogram_bars(b_hist, max_value, (0, 0, 1, 0.6))
            self.__draw_histogram_bars(l_hist, max_value, (0.8, 0.8, 0.8, 1))#, offset_x_ref=256+64+32)  # Ref offset for luminance
            
            KVColor((0.8, 0.8, 0.8, 1))
            KVLine(rectangle=(0, 0, 256+64+32, 128+64), width=1)
            KVLine(rectangle=(0+256+64+32, 0, 128+32, 128+64), width=1)
            KVPopMatrix()

    def draw_histogram(self, pixels, blue_count, black_count):
        hist_data = self.calculate_histogram_data(pixels, blue_count, black_count)
        self.draw_histogram_from_data(hist_data)

    def __draw_histogram_bars(self, histogram, max_value, color, offset_x_ref=0, offset_y_ref=0):
        bar_width = 1
        # Drawing in local coordinates (0,0 based), so just use refs
        offset_x = offset_x_ref
        offset_y = offset_y_ref
        
        with self.canvas:
            KVColor(*color)
            for x, value in enumerate(histogram):
                height = min(127+64, (value / max_value) * (127+64))  # ヒストグラムの高さを設定
                KVRectangle(pos=(x * bar_width + offset_x, offset_y), size=(bar_width, height))

class HistogramApp(KVApp):
    def build(self):
        histogram = HistogramWidget()
        # 開発用のローカルパスに依存しないようにする
        # histogram._load_image(<path>)
        return histogram

if __name__ == '__main__':
    HistogramApp().run()
