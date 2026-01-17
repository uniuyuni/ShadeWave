"""
レンズ歪み補正Widget

KivyMDベースのGUIウィジェット
"""

from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (
    ObjectProperty, NumericProperty, BooleanProperty, StringProperty
)
from kivy.graphics import Color, Line
from kivy.clock import mainthread
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image as KivyImage
import numpy as np
import cv2

from cores.distortion_correction.lens_distortion import correct_lens_distortion, detect_lens_distortion
import params


class LensDistortionWidget(FloatLayout):
    """レンズ歪み補正Widget"""
    
    # プロパティ
    source_image = ObjectProperty(None)
    strength = NumericProperty(0)
    scale = NumericProperty(1.0)  # スケールパラメータ
    show_grid = BooleanProperty(True)
    
    def __init__(self, texture_size, param, **kwargs):
        super().__init__(**kwargs)
        
        self.texture_size = texture_size
        self.tcg_info = params.param_to_tcg_info(param)

        self.on_callback = None
        
        # 自動検出ボタン
        self.auto_detect_btn = MDRaisedButton(
            text="Auto",
            size_hint=(None, None),
            size=(120, 48),
            pos_hint={'center_x': 0.5, 'y': 0.02},
        )
        self.auto_detect_btn.bind(on_press=self.on_auto_detect)
        self.add_widget(self.auto_detect_btn)
        
        # プロパティの変更を監視
        self.bind(show_grid=self._on_show_grid_change)

        self.update_preview()

    def set_image(self, image):
        self.source_image = image

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
        """プレビューを更新（リアルタイム）"""
        # グリッドを描画
        if self.show_grid:
            self._draw_grid()
    
    def on_auto_detect(self, instance):
        """自動検出ボタンが押された"""
        if self.source_image is None:
            return
        
        try:            
            detected_strength = detect_lens_distortion(self.source_image)
            self.strength = detected_strength
            
        except Exception as e:
            print(f"Error in auto detection: {e}")
    
    def get_correction_params(self) -> dict:
        """
        現在のパラメータを取得
        
        Returns:
            dict: {"lens_distortion_strength": float, "lens_distortion_scale": float}
        """
        return {
            "lens_distortion_strength": float(self.strength),
            "lens_distortion_scale": float(self.scale)
        }
    
    def set_correction_params(self, params: dict):
        """
        パラメータを設定
        
        Args:
            params: dict、{"lens_distortion_strength": float, "lens_distortion_scale": float}
        """
        self.strength = params.get('lens_distortion_strength', 0.0)
        self.scale = params.get('lens_distortion_scale', 1.0)

    def _on_show_grid_change(self, instance, value):
        """show_grid変更時"""
        if value:
            self._draw_grid()
        else:
            self._clear_grid()
    
    def _draw_grid(self):
        """グリッドを描画（正方形セル）"""
        self.canvas.after.clear()
    
        w, h = self.tcg_info['original_img_size']
        if h == 0 or w == 0:
            return
            
        aspect = w / h
        base_grid = 8  # 短辺の分割数
        
        if w >= h:
            ny = base_grid
            nx = int(round(base_grid * aspect))
        else:
            nx = base_grid
            ny = int(round(base_grid / aspect))
        
        with self.canvas.after:
            Color(1, 1, 1, 0.5)  # 白色半透明
            
            # 縦線
            for i in range(1, nx):
                t = i / nx
                tcg_x = t - 0.5
                
                kx1, ky1 = params.tcg_to_window(tcg_x, -0.5, self, self.texture_size, self.tcg_info)
                kx2, ky2 = params.tcg_to_window(tcg_x, 0.5, self, self.texture_size, self.tcg_info)
                Line(points=[kx1, ky1, kx2, ky2], width=1)
            
            # 横線
            for i in range(1, ny):
                t = i / ny
                tcg_y = t - 0.5
                
                kx1, ky1 = params.tcg_to_window(-0.5, tcg_y, self, self.texture_size, self.tcg_info)
                kx2, ky2 = params.tcg_to_window(0.5, tcg_y, self, self.texture_size, self.tcg_info)
                Line(points=[kx1, ky1, kx2, ky2], width=1)
    
    def _clear_grid(self):
        """グリッドをクリア"""
        self.canvas.after.clear()
