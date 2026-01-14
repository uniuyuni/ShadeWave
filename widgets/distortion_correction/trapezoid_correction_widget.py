"""
台形補正Widget（水平・垂直）

KivyMDベースのGUIウィジェット
"""

from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (
    BooleanProperty
)
from kivy.graphics import Color, Line
from kivy.clock import mainthread

import cores.core as core


class TrapezoidCorrectionWidget(FloatLayout):
    """台形補正Widget（水平・垂直）"""
    
    # プロパティ
    show_guides = BooleanProperty(True)
    
    def __init__(self, texture_size, param, **kwargs):
        super().__init__(**kwargs)
        
        self.texture_size = texture_size
        self.tcg_info = core.param_to_tcg_info(param)

        self.on_callback = None
        
        self.bind(show_guides=self._on_show_guides_change)

        self.update_preview()

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
    
    @mainthread
    def update_preview(self, *args):
        # ガイド線を描画
        if self.show_guides:
            self._draw_guides()
    
    def get_correction_params(self) -> dict:
        """
        現在のパラメータを取得
        
        Returns:
            dict: {"horizontal": float, "vertical": float, "rotation": float, 
                   "focal_length": float, "offset_x": float, "offset_y": float}
        """
        return {
        }
    
    def set_correction_params(self, params: dict):
        """
        パラメータを設定
        
        Args:
            params: dict
        """
    
    def _on_show_guides_change(self, instance, value):
        """show_guides変更時"""
        if value:
            self._draw_guides()
        else:
            self._clear_guides()
    
    def _draw_guides(self):
        """ガイド線を描画"""
        self.canvas.after.clear()
        with self.canvas.after:
            Color(1, 1, 1, 0.6)  # 白色半透明
            
            # 水平ガイド線（3本: 上、中、下）
            for i in [1, 2, 3]:
                norm_y = i / 4.0
                tcg_y = -(norm_y - 0.5) # Core Y is down?, TCG standard. core TCG: -0.5=Top?
                # core.py TCG: Center 0,0. Range -0.5 ~ 0.5.
                # In core.tcg_to_window: cx, cy are passed.
                # If I want 25% from top: Top is -0.5? Bottom is 0.5?
                # core.py: 
                # x_img = (x_tcg + 0.5) * (width - 1)
                # y_img = (y_tcg + 0.5) * (height - 1)
                # So -0.5 is 0 (Top/Left). 0.5 is 1 (Bottom/Right).
                
                start_tcg_y = (norm_y - 0.5)
                
                # Start (-0.5, y), End (0.5, y)
                kx1, ky1 = core.tcg_to_window(-0.5, start_tcg_y, self, self.texture_size, self.tcg_info)
                kx2, ky2 = core.tcg_to_window(0.5, start_tcg_y, self, self.texture_size, self.tcg_info)
                
                Line(points=[kx1, ky1, kx2, ky2], width=1)
            
            # 垂直ガイド線（3本: 左、中、右）
            for i in [1, 2, 3]:
                norm_x = i / 4.0
                start_tcg_x = norm_x - 0.5
                
                # Start (x, -0.5), End (x, 0.5)
                kx1, ky1 = core.tcg_to_window(start_tcg_x, -0.5, self, self.texture_size, self.tcg_info)
                kx2, ky2 = core.tcg_to_window(start_tcg_x, 0.5, self, self.texture_size, self.tcg_info)
                
                Line(points=[kx1, ky1, kx2, ky2], width=1)
    
    def _clear_guides(self):
        """ガイド線をクリア"""
        self.canvas.after.clear()

