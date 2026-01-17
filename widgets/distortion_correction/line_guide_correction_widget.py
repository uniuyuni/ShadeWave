"""
ラインガイド補正Widget

KivyMDベースのGUIウィジェット
"""

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.properties import ListProperty, StringProperty, NumericProperty
from kivy.graphics import Color, Line
from kivy.clock import mainthread
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image as KivyImage
import numpy as np
import cv2

from cores.distortion_correction.warp_correction import correct_with_lines
import params


class LineGuideCorrectionWidget(FloatLayout):
    """ラインガイド補正Widget"""
    
    lines_tcg = ListProperty([])  # [((x1,y1), (x2,y2)), ...]
    interpolation = StringProperty('bicubic')
    preview_scale = NumericProperty(0.25)
    
    def __init__(self, texture_size, param, **kwargs):
        super().__init__(**kwargs)
        
        self.texture_size = texture_size
        self.tcg_info = params.param_to_tcg_info(param)

        self.on_callback = None
        
        # 線描画用のオーバーレイWidget
        self.draw_overlay = Widget()
        self.add_widget(self.draw_overlay)
        
        # 状態変数
        self.selected_line_index = -1
        self.selected_point_index = -1 # 0: start, 1: end
        self.dragging = False
        self.start_new_line_point = None
        
        # キーボードイベント
        from kivy.core.window import Window
        Window.bind(on_key_down=self._on_key_down)

        # タッチイベントをバインド
        self.draw_overlay.bind(
            on_touch_down=self._on_drawing_touch_down,
            on_touch_move=self._on_drawing_touch_move,
            on_touch_up=self._on_drawing_touch_up
        )

        # UIコントロール
        button_layout = MDBoxLayout(
            orientation='horizontal',
            size_hint=(None, None),
            size=(100, 48),
            pos_hint={'center_x': 0.5, 'y': 0.02},
            adaptive_size=True,
            spacing=20,
            padding=10
        )
        
        self.clear_btn = MDRaisedButton(text="Reset")
        self.clear_btn.bind(on_press=self.clear_lines)
        button_layout.add_widget(self.clear_btn)

        self.apply_btn = MDRaisedButton(text="Apply")
        self.apply_btn.bind(on_press=lambda x: self._apply())
        button_layout.add_widget(self.apply_btn)

        self.revert_btn = MDRaisedButton(text="Revert")
        self.revert_btn.bind(on_press=self.revert_lines)
        button_layout.add_widget(self.revert_btn)
        
        self.add_widget(button_layout)
        
        # プロパティの変更を監視
        self.bind(lines_tcg=self._redraw_lines) # lines_tcg変更時に再描画
        self.bind(size=self._redraw_lines)
        self.bind(pos=self._redraw_lines)

        self._redraw_lines()

    def set_callback(self, callback):
        self.on_callback = callback

    def on_edit_start(self):
        """編集開始イベント（ヒストリー管理用）"""
        if self.on_callback:
            self.on_callback('start', self)
    
    def on_edit_end(self):
        """編集終了イベント（ヒストリー管理用）"""
        if self.on_callback:
            self.on_callback('end', self)

    def set_lines(self, lines):
        """外部からラインを設定"""
        self.lines_tcg = lines
        #self._redraw_lines()
        
    def _apply(self):
        """現在のライン設定を適用"""
        if self.on_callback:
            # 外部(Effect)に渡すときは、単純なリストとして渡す
            # warp_correction.correct_with_lines は lines だけ受け取るようになった
            self.on_callback('apply', self)

    def _on_key_down(self, window, key, scancode, codepoint, modifiers):
        """キー入力処理"""
        # Delete or Backspace
        if key in [8, 127]: # 8: Backspace, 127: Delete
            if self.selected_line_index >= 0:
                self.lines_tcg.pop(self.selected_line_index)
                self.selected_line_index = -1
                self.selected_point_index = -1
                self._redraw_lines()
                return True
        return False

    def clear_lines(self, instance=None):
        """全ての線を削除"""
        self.on_edit_start()
        self.lines_tcg = []
        self.selected_line_index = -1
        self.selected_point_index = -1
        self.on_edit_end()
        self._apply()
        #self._redraw_lines()

    def revert_lines(self, instance=None):
        """オリジナルの状態（補正なし）をプレビューしつつ、ガイドラインは維持する"""
        backup = list(self.lines_tcg)
        self.lines_tcg = []
        # Update UI to clear lines (implicitly via bind)
        self._apply()
        # Restore lines so they are visible on top of uncorrected image
        self.lines_tcg = backup
        #self._redraw_lines()

    def _get_tcg_pos(self, touch_pos):
        """タッチ位置(Widget座標)をTCG座標に変換"""
        tx, ty = params.window_to_tcg(touch_pos[0], touch_pos[1], self, self.texture_size, self.tcg_info)
        return tx, ty

    def _get_window_pos(self, tx, ty):
        """TCG座標をウィンドウ座標に変換"""
        wx, wy = params.tcg_to_window(tx, ty, self, self.texture_size, self.tcg_info)
        return wx, wy

    def _hit_test(self, tcg_x, tcg_y, touch_tcg_x, touch_tcg_y):
        """ヒットテスト（TCG空間での距離）"""
        dist = np.sqrt((tcg_x - touch_tcg_x)**2 + (tcg_y - touch_tcg_y)**2)
        # 閾値は適当（画面サイズに依存すべきだが）
        return dist < 0.05 

    def _clamp_tcg(self, x, y):
        """TCG座標を画像範囲内に制限"""
        # TCG座標はアスペクト比依存で範囲が変わる (-0.5 ~ 0.5付近)
        if x < -0.5:
            x = -0.5
        elif x > 0.5:
            x = 0.5
        if y < -0.5:
            y = -0.5
        elif y > 0.5:
            y = 0.5
        return x, y

    def _on_drawing_touch_down(self, instance, touch):
        """タッチ開始"""
        if not self.collide_point(*touch.pos):
            return False
            
        touch.grab(self)
        
        tx, ty = self._get_tcg_pos(touch.pos)
        
        # 既存ラインの端点ヒットテスト
        hit_line = -1
        hit_point = -1
        min_dist = 1000
        
        for i, line in enumerate(self.lines_tcg):
            p1, p2 = line
            # Start
            if self._hit_test(p1[0], p1[1], tx, ty):
                dist = np.sqrt((p1[0]-tx)**2 + (p1[1]-ty)**2)
                if dist < min_dist:
                    min_dist = dist
                    hit_line = i
                    hit_point = 0
            # End
            if self._hit_test(p2[0], p2[1], tx, ty):
                dist = np.sqrt((p2[0]-tx)**2 + (p2[1]-ty)**2)
                if dist < min_dist:
                    min_dist = dist
                    hit_line = i
                    hit_point = 1
        
        if hit_line >= 0:
            # 既存ポイントのドラッグ開始
            self.selected_line_index = hit_line
            self.selected_point_index = hit_point
            self.dragging = True
            self._redraw_lines()
        else:
            # 新規ライン作成開始
            self.start_new_line_point = (tx, ty)
            self.selected_line_index = -1
            self.selected_point_index = -1
            self.dragging = True
            
        return True

    def _on_drawing_touch_move(self, instance, touch):
        """ドラッグ中"""
        if touch.grab_current is not self:
            return False
            
        tx, ty = self._get_tcg_pos(touch.pos)
        tx, ty = self._clamp_tcg(tx, ty)
        
        if self.start_new_line_point:
            # 新規ラインを描画中に表示（一時的）
            self._redraw_lines(temp_line=(self.start_new_line_point, (tx, ty)))
            
        elif self.selected_line_index >= 0:
            # 既存点の移動
            line = list(self.lines_tcg[self.selected_line_index])
            line[self.selected_point_index] = (tx, ty)
            self.lines_tcg[self.selected_line_index] = tuple(line)
            #self._redraw_lines()
            
        return True

    def _on_drawing_touch_up(self, instance, touch):
        """タッチ終了"""
        if touch.grab_current is not self:
            return False
        
        touch.ungrab(self)
        
        tx, ty = self._get_tcg_pos(touch.pos)
        tx, ty = self._clamp_tcg(tx, ty)
        
        if self.start_new_line_point:
            self.on_edit_start()
            # 新規ライン確定
            # 一定以上長さがある場合のみ
            dist = np.sqrt((self.start_new_line_point[0]-tx)**2 + (self.start_new_line_point[1]-ty)**2)
            if dist > 0.01:
                self.lines_tcg.append((self.start_new_line_point, (tx, ty)))
                self.selected_line_index = len(self.lines_tcg) - 1 # 選択状態にする
                self.selected_point_index = -1 # ポイント選択解除
            
            self.start_new_line_point = None
            #self._redraw_lines()
            self.on_edit_end()
            
        elif self.dragging:
            # 移動終了、確定
            self.dragging = False


        return True

    def _redraw_lines(self, instance=None, value=None, temp_line=None):
        """すべての線を再描画"""
        self.draw_overlay.canvas.clear()
        
        LINE_WIDTH = 1.5
        POINT_SIZE = 5
        COLOR_NORMAL = (0.9, 0.9, 0.9, 0.8)
        COLOR_SELECTED = (1.0, 0.8, 0.2, 1.0)
        
        with self.draw_overlay.canvas:
            # 確定済みライン
            for i, line in enumerate(self.lines_tcg):
                p1 = line[0]
                p2 = line[1]
                
                wx1, wy1 = self._get_window_pos(p1[0], p1[1])
                wx2, wy2 = self._get_window_pos(p2[0], p2[1])
                
                if i == self.selected_line_index:
                    Color(*COLOR_SELECTED)
                else:
                    Color(*COLOR_NORMAL)
                    
                Line(points=[wx1, wy1, wx2, wy2], width=LINE_WIDTH)
                
                # 端点（コントロールポイント）
                Color(1,1,1,1) if (i==self.selected_line_index and self.selected_point_index==0) else Color(*COLOR_NORMAL)
                Line(circle=(wx1, wy1, POINT_SIZE), width=1.2)
                
                Color(1,1,1,1) if (i==self.selected_line_index and self.selected_point_index==1) else Color(*COLOR_NORMAL)
                Line(circle=(wx2, wy2, POINT_SIZE), width=1.2)

            # 作成中のライン
            if temp_line:
                p1 = temp_line[0]
                p2 = temp_line[1]
                # temp_lineはTCG座標で来る想定
                wx1, wy1 = self._get_window_pos(p1[0], p1[1])
                wx2, wy2 = self._get_window_pos(p2[0], p2[1])
                
                Color(*COLOR_SELECTED)
                Line(points=[wx1, wy1, wx2, wy2], width=LINE_WIDTH, dash_length=5, dash_offset=5)
                 
    @mainthread
    def update_preview(self, *args):
        pass
    
    def get_correction_params(self) -> dict:
        """現在のパラメータを取得"""
        return {
            "reference_lines": list(self.lines_tcg),
        }
    
    def set_correction_params(self, params: dict):
        """パラメータを設定"""
        self.lines_tcg = params.get('reference_lines', [])
        #self._redraw_lines()

    def on_lines_change(self):
        pass

