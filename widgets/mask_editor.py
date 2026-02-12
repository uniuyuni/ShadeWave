
import cv2
import numpy as np

from kivy.app import App as KVApp
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.uix.widget import Widget as KVWidget
from kivy.uix.image import Image as KVImage
from kivy.graphics import (
    Color as KVColor, Rectangle as KVRectangle, PushMatrix as KVPushMatrix, 
    PopMatrix as KVPopMatrix, Line as KVLine, Translate as KVTranslate,
    ScissorPush as KVScissorPush, ScissorPop as KVScissorPop
)
from kivy.properties import NumericProperty as KVNumericProperty
from kivy.graphics.texture import Texture as KVTexture
from kivy.clock import Clock as KVClock
from kivy.core.window import Window as KVWindow

import config
import params
import cores.core as core
import utils.kvutils as kvutils
import macos as device

class MaskEditor(KVFloatLayout):
    brush_size = KVNumericProperty(300)
    
    def __init__(self, param, **kwargs):
        self.effect_ctrl_param = kwargs.pop('effect_ctrl_param', None)
        self.touch_up_callback = kwargs.pop('touch_up_callback', None)
        super(MaskEditor, self).__init__(**kwargs)
        
        # 初期化
        self.pos_hint = {'x':0, 'top': 1}

        self.param = param
        self.canvas_width = param.get('img_size', (512, 512))[0]
        self.canvas_height = param.get('img_size', (512, 512))[1]
        self.tcg_info = params.param_to_tcg_info(param)
        self.drawing = False
        self.erasing = False
        
        # 表示用テクスチャの作成
        self.texture_size = (config.get_config('preview_width'), config.get_config('preview_height'))
        
        # マスクデータの初期化
        self.clear_mask()
        # self.canvas_texture は create_ui または update_canvas で生成/更新されるが、
        # ここでは描画用のFBO的な役割を果たすテクスチャを準備するわけではなく、
        # マスクを可視化するためのテクスチャを管理する。
        # 実際には draw_mask_texture を使用して描画する。
        
        self.callback_param = param
        
        self.last_touch_pos = None
        self.cursor_instruction = None

        # ブラシカーソル初期化
        self.init_cursor()

        # ウィンドウのマウスイベントをバインド（カーソル追従用）
        KVWindow.bind(mouse_pos=self.on_mouse_pos)

        # 描画更新
        self.bind(size=self.delay_update_canvas, pos=self.delay_update_canvas)
        self.delay_update_canvas()

    def init_cursor(self):
        with self.canvas.after:
            KVPushMatrix()
            self.cursor_color = KVColor(1, 1, 1, 1) # 白色のカーソル
            self.cursor_line = KVLine(circle=(0, 0, 10), width=1.5)
            self.cursor_translate = KVTranslate(0, 0)
            KVPopMatrix()

    def on_mouse_pos(self, window, pos):
        # 画面内にあるときだけカーソルを更新
        if self.collide_point(*pos):
             self.update_cursor(pos[0], pos[1])
        else:
            # 画面外ならカーソルを隠す（または何もしない）
            self.cursor_line.circle = (0,0,0) # 半径0で見えなくする簡易手法

    def update_cursor(self, x, y):
        # brush_size は画像ピクセル単位
        # 画面上のサイズ（Points）に変換する
        
        # params.disp_info[4] はズーム倍率
        scale = params.get_disp_info(self.tcg_info)[4] * device.dpi_scale()
        
        # brush_size * scale = 表示上の画像サイズ(Points)
        visual_radius = (self.brush_size / 2) * scale
        
        self.cursor_line.circle = (x, y, visual_radius)

    def clear_mask(self):
        self.mask = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)

    def get_mask(self):
        # 正方形パディングされている場合はクロップして返す？ 
        # 元の実装を見ると `self.mask = np.zeros((imax, imax)...)` となっていたが、
        # 今回は `(canvas_height, canvas_width)` で作成した。
        # 必要ならパディングする。
        return self.mask

    def delay_update_canvas(self, *args):
        # Paramの内容が変わっている可能性があるため、tcg_infoを更新
        self.tcg_info = params.param_to_tcg_info(self.param)
        KVClock.schedule_once(self.update_canvas, 0)

    def update_canvas(self, *args):
        self.canvas.before.clear()
        
        self.tcg_info = params.param_to_tcg_info(self.param) # double check
        
        h, w = self.mask.shape[:2]
        if h == 0 or w == 0: return

        # テクスチャ更新
        # RGBAを使用する（デバッグのため、より確実な方法を選択）
        la_img = np.empty((h, w, 4), dtype=np.uint8)
        la_img[:] = 255 # White
        la_img[..., 3] = self.mask # Alpha channel
        
        texture_a = KVTexture.create(size=(w, h), colorfmt='rgba', bufferfmt='ubyte')
        texture_a.blit_buffer(la_img.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        # texture_a.flip_vertical() # params.tcg_to_windowで返される座標系によっては反転不要かも。要確認。
        # MaskEditor2では `texture.flip_vertical()` している。
        texture_a.flip_vertical()

        # 描画位置の計算
        
        # 左下と右上の座標を取得
        x0, y0 = params.tcg_to_window(-0.5, 0.5, self, self.texture_size, self.tcg_info, normalize=True)
        x1, y1 = params.tcg_to_window(0.5, -0.5, self, self.texture_size, self.tcg_info, normalize=True)
        
        rect_x = min(x0, x1)
        rect_y = min(y0, y1)
        rect_w = abs(x1 - x0)
        rect_h = abs(y1 - y0)
        
        with self.canvas.before:
            KVColor(1, 0, 0, 0.5)
            KVRectangle(texture=texture_a, pos=(rect_x, rect_y), size=(rect_w, rect_h))

    def _window_to_mask_coords(self, wx, wy):
        tx, ty = params.window_to_tcg(wx, wy, self.parent, self.texture_size, self.tcg_info, normalize=True)
        ix, iy = params.tcg_to_ref_image(tx, ty, self.mask, self.tcg_info, apply_disp_info=True)
        ix, iy = ix / self.tcg_info['disp_info'][4], iy / self.tcg_info['disp_info'][4]
        m_max = max(self.mask.shape[:2])
        m_h, m_w = self.mask.shape[:2]
        ix, iy = ix - (m_max - m_w) // 2, iy - (m_max - m_h) // 2
        return int(ix), int(iy)

    def on_touch_down(self, touch):
        # Paramの内容が変わっている可能性があるため、tcg_infoを更新
        self.tcg_info = params.param_to_tcg_info(self.param)
        
        if self.collide_point(*touch.pos):
            # スクロールによるブラシサイズ変更
            if touch.is_mouse_scrolling:
                if touch.button == 'scrollup':
                    self.brush_size = max(1, self.brush_size * 0.9)
                elif touch.button == 'scrolldown':
                    self.brush_size *= 1.1
                self.update_cursor(touch.x, touch.y)
                return True

            # 左クリックで描画、右クリックで消去
            if touch.button == 'left':
                self.drawing = True
                self.erasing = False
            elif touch.button == 'right':
                self.drawing = False
                self.erasing = True
            else:
                return super(MaskEditor, self).on_touch_down(touch)

            self.last_touch_pos = self._window_to_mask_coords(touch.x, touch.y)
            
            # 点を描画
            self.paint_point(self.last_touch_pos)
            
            # カーソル更新
            self.update_cursor(touch.x, touch.y)
            return True
        return super(MaskEditor, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        # Paramの内容が変わっている可能性があるため、tcg_infoを更新
        self.tcg_info = params.param_to_tcg_info(self.param)

        if self.drawing or self.erasing:
            current_pos = self._window_to_mask_coords(touch.x, touch.y)
            
            # 線を描画（補間）
            self.paint_line(self.last_touch_pos, current_pos)
            
            self.last_touch_pos = current_pos
            self.update_cursor(touch.x, touch.y)
            return True
        return super(MaskEditor, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.drawing or self.erasing:
            self.drawing = False
            self.erasing = False
            
            # コールバック処理
            if self.touch_up_callback:
                if self.effect_ctrl_param:
                    kvutils.get_root_widget(self).begin_history_effect_ctrl(*self.effect_ctrl_param)
                self.touch_up_callback(self.callback_param, self.get_mask())
                if self.effect_ctrl_param:
                    kvutils.get_root_widget(self).end_history_effect_ctrl(*self.effect_ctrl_param)
            
            return True
        return super(MaskEditor, self).on_touch_up(touch)
        
    def paint_point(self, pos):
        val = 0 if self.erasing else 255
        cv2.circle(self.mask, pos, int(self.brush_size // 2), val, -1, lineType=cv2.LINE_AA)
        self.delay_update_canvas()

    def paint_line(self, pt1, pt2):
        val = 0 if self.erasing else 255
        cv2.line(self.mask, pt1, pt2, val, thickness=int(self.brush_size), lineType=cv2.LINE_AA)
        cv2.circle(self.mask, pt2, int(self.brush_size // 2), val, -1, lineType=cv2.LINE_AA)
        self.delay_update_canvas()

class MyApp(KVApp):
    def build(self):
        layout = KVBoxLayout(orientation='vertical')
        # デバッグ用パラメータ
        param = {'img_size': (512, 512), 'disp_info': (0,0,512,512,1.0)}
        mask_widget = MaskEditor(param=param, size_hint=(1, 1))
        layout.add_widget(mask_widget)
        return layout

if __name__ == '__main__':
    MyApp().run()
