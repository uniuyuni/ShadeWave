"""
ポイントワープWidget

KivyMDベースのGUIウィジェット
"""

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter
from kivy.properties import (
    ObjectProperty, ListProperty, NumericProperty, StringProperty
)
from kivy.graphics import Color, Line
from kivy.graphics.texture import Texture
from kivy.clock import mainthread
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image as KivyImage
import numpy as np
import cv2
import math
from cores.distortion_correction.warp_correction import warp_points
import cores.core as core


class PointWarpWidget(FloatLayout):
    """ポイントワープWidget"""
    
    # プロパティ
    source_image = ObjectProperty(None)
    src_points_tcg = ListProperty([])  # [(x, y), ...]
    dst_points_tcg = ListProperty([])  # [(x, y), ...]
    influence = NumericProperty(100)
    interpolation = StringProperty('bicubic')
    preview_scale = NumericProperty(0.25)
    
    def __init__(self, texture_size, param, **kwargs):
        super().__init__(**kwargs)
        
        self.texture_size = texture_size
        self.param = param
        self.tcg_info = core.param_to_tcg_info(param)
        
        # 画像表示Widget
        self.image_widget = KivyImage(
            size_hint=(1, 1),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            allow_stretch=True,
            keep_ratio=True,
            color=[1, 1, 1, 0]  # 透明（テクスチャなし時）
        )
        self.add_widget(self.image_widget)
        
        # ハンドルリスト
        self.src_handles = []
        self.dst_handles = []
        
        # コントロールボタン群
        button_layout = MDBoxLayout(
            orientation='horizontal',
            size_hint=(None, None),
            size=(600, 48),
            pos_hint={'center_x': 0.5, 'y': 0.02},
            spacing=10
        )
        
        self.add_point_btn = MDRaisedButton(
            text="ポイント追加", 
            size_hint_x=None, 
            width=130
        )
        self.add_point_btn.bind(on_press=self.add_point_pair)
        button_layout.add_widget(self.add_point_btn)
        
        self.remove_btn = MDFlatButton(
            text="最後のポイント削除",
            size_hint_x=None,
            width=170
        )
        self.remove_btn.bind(on_press=self.remove_last_pair)
        button_layout.add_widget(self.remove_btn)
        
        self.clear_btn = MDFlatButton(
            text="すべてクリア",
            size_hint_x=None,
            width=120
        )
        self.clear_btn.bind(on_press=self.clear_all_points)
        button_layout.add_widget(self.clear_btn)
        
        self.swap_btn = MDFlatButton(
            text="srcとdstを入れ替え",
            size_hint_x=None,
            width=170
        )
        self.swap_btn.bind(on_press=self.swap_src_dst)
        button_layout.add_widget(self.swap_btn)
        
        self.add_widget(button_layout)
        
        # ポイント追加モード
        self.adding_point = False
        self.temp_src_pos = None
        
        # プロパティの変更を監視
        self.bind(src_points_tcg=self._on_points_change)
        self.bind(dst_points_tcg=self._on_points_change)
        self.bind(influence=self._on_influence_change)

    def set_edit_start_callback(self, callback):
        self.on_edit_start_callback = callback

    def set_edit_end_callback(self, callback):
        self.on_edit_end_callback = callback

    def on_edit_start(self):
        """編集開始イベント（ヒストリー管理用）"""
        if self.on_edit_start_callback:
            self.on_edit_start_callback(self)
    
    def on_edit_end(self):
        """編集終了イベント（ヒストリー管理用）"""
        if self.on_edit_end_callback:
            self.on_edit_end_callback(self)
        
    def set_image(self, image: np.ndarray):
        """
        画像をセット
        
        Args:
            image: numpy.ndarray、dtype=float32、shape=(H, W, 3)
        """
        self.source_image = image
        if image is not None:
             h, w = image.shape[:2]
             self.texture_size = (w, h)
        self._display_image(image)
    
    def add_point_pair(self, instance):
        """ポイントペアを追加（ユーザーに2回タップさせる）"""
        # 簡易実装: 画像中心にデフォルトペアを追加
        if self.source_image is None:
            return
        
        h, w = self.source_image.shape[:2]
        
        # ランダムな位置に追加（正規化座標）
        import random
        offset_x = random.uniform(-0.25, 0.25)
        offset_y = random.uniform(-0.25, 0.25)
        
        src_tcg = (offset_x, offset_y)
        dst_tcg = (offset_x, offset_y)  # 初期位置は同じ
        
        new_src = list(self.src_points_tcg)
        new_src.append(src_tcg)
        self.src_points_tcg = new_src
        
        new_dst = list(self.dst_points_tcg)
        new_dst.append(dst_tcg)
        self.dst_points_tcg = new_dst
        
        self._rebuild_handles()
        self.update_preview()
    
    def remove_last_pair(self, instance):
        """最後のポイントペアを削除"""
        if len(self.src_points_tcg) > 0:
            self.src_points_tcg = self.src_points_tcg[:-1]
            self.dst_points_tcg = self.dst_points_tcg[:-1]
            self._rebuild_handles()
            self.update_preview()
    
    def clear_all_points(self, instance):
        """すべてのポイントをクリア"""
        self.src_points_tcg = []
        self.dst_points_tcg = []
        self._rebuild_handles()
        self.update_preview()
    
    def swap_src_dst(self, instance):
        """srcとdstを入れ替え"""
        temp = list(self.src_points_tcg)
        self.src_points_tcg = list(self.dst_points_tcg)
        self.dst_points_tcg = temp
        self._rebuild_handles()
        self.update_preview()
    
    @mainthread
    def update_preview(self, *args):
        """プレビューを更新"""
        if self.source_image is None:
            return
        
        if len(self.src_points_tcg) == 0:
            self._display_image(self.source_image)
            return
        
        # 低解像度化
        h, w = self.source_image.shape[:2]
        preview_h = int(h * self.preview_scale)
        preview_w = int(w * self.preview_scale)
        preview_image = cv2.resize(self.source_image, (preview_w, preview_h))
        
        # スケール調整したポイント
        scaled_src = [
            (x * self.preview_scale, y * self.preview_scale)
            for x, y in self.src_points_tcg
        ]
        
        scaled_dst = [
            (x * self.preview_scale, y * self.preview_scale)
            for x, y in self.dst_points_tcg
        ]
        
        # 補正実行
        try:
            corrected = warp_points(
                preview_image,
                scaled_src,
                scaled_dst,
                self.influence,
                interpolation='bilinear'
            )
            self._display_image(corrected)
        except Exception as e:
            print(f"Error in correction: {e}")
            self._display_image(preview_image)
    
    def get_correction_params(self) -> dict:
        """
        現在のパラメータを取得（TCG座標系）
        
        Returns:
            dict
        """
        return {
            "src_points": list(self.src_points_tcg),
            "dst_points": list(self.dst_points_tcg),
            "influence": float(self.influence)
        }
    
    def set_correction_params(self, params: dict):
        """
        パラメータを設定（TCG座標系）
        
        Args:
            params: dict
        """
        self.src_points_tcg = params.get('src_points', [])
        self.dst_points_tcg = params.get('dst_points', [])
        self.influence = params.get('influence', 100.0)
        self._rebuild_handles()
        self.update_preview()
    
    def get_corrected_image(self) -> np.ndarray:
        """
        フル解像度で補正した画像を取得
        
        Returns:
            補正後画像
        """
        if self.source_image is None:
            return None
        
        if len(self.src_points_tcg) == 0:
            return self.source_image.copy()
        
        return warp_points(
            self.source_image,
            self.src_points_tcg,
            self.dst_points_tcg,
            self.influence,
            interpolation=self.interpolation
        )
    
    def _rebuild_handles(self):
        """ハンドルを再構築"""
        if self.source_image is None:
            return
        
        # 既存のハンドルを削除
        for handle in self.src_handles + self.dst_handles:
            self.remove_widget(handle)
        self.src_handles = []
        self.dst_handles = []
        
        # h, w = self.source_image.shape[:2]
        # image_shape = (h, w)
        # widget_size = (self.image_widget.width, self.image_widget.height)
        
        # if widget_size[0] == 0 or widget_size[1] == 0:
        #    return
        
        # srcハンドル作成
        for i, tcg_pos in enumerate(self.src_points_tcg):
            # kivy_pos = tcg_to_kivy_coords(tcg_pos, widget_size, image_shape)
            kx, ky = core.tcg_to_window(tcg_pos[0], tcg_pos[1], self, self.texture_size, self.tcg_info)
            handle = self._create_src_handle(i, (kx, ky))
            self.src_handles.append(handle)
            self.add_widget(handle)
        
        # dstハンドル作成
        for i, tcg_pos in enumerate(self.dst_points_tcg):
            # kivy_pos = tcg_to_kivy_coords(tcg_pos, widget_size, image_shape)
            kx, ky = core.tcg_to_window(tcg_pos[0], tcg_pos[1], self, self.texture_size, self.tcg_info)
            handle = self._create_dst_handle(i, (kx, ky))
            self.dst_handles.append(handle)
            self.add_widget(handle)
        
        # 矢印を描画
        self._update_arrows()
    
    def _create_src_handle(self, index: int, pos: tuple):
        """srcポイントハンドルを作成"""
        handle = Scatter(
            size=(35, 35),
            do_rotation=False,
            do_scale=False,
            auto_bring_to_front=True
        )
        
        # 円形を描画（赤色）
        with handle.canvas:
            Color(1, 0.2, 0.2, 0.8)
            from kivy.graphics import Ellipse
            Ellipse(pos=(0, 0), size=(35, 35))
        
        # 位置を設定
        handle.center = pos
        
        # ドラッグイベントをバインド
        handle.bind(pos=lambda inst, val: self._on_src_point_move(index, inst))
        
        return handle
    
    def _create_dst_handle(self, index: int, pos: tuple):
        """dstポイントハンドルを作成"""
        handle = Scatter(
            size=(35, 35),
            do_rotation=False,
            do_scale=False,
            auto_bring_to_front=True
        )
        
        # 円形を描画（緑色）
        with handle.canvas:
            Color(0.2, 1, 0.2, 0.8)
            from kivy.graphics import Ellipse
            Ellipse(pos=(0, 0), size=(35, 35))
        
        # 位置を設定
        handle.center = pos
        
        # ドラッグイベントをバインド
        handle.bind(pos=lambda inst, val: self._on_dst_point_move(index, inst))
        
        return handle
    
    def _on_src_point_move(self, index: int, handle):
        """srcポイントがドラッグされた"""
        if self.source_image is None:
            return
        
        # h, w = self.source_image.shape[:2]
        # image_shape = (h, w)
        # widget_size = (self.image_widget.width, self.image_widget.height)
        
        # Kivy座標→TCG座標
        kx, ky = handle.center
        # tcg_pos = kivy_to_tcg_coords(kivy_pos, widget_size, image_shape)
        tcg_pos = core.window_to_tcg(kx, ky, self, self.texture_size, self.tcg_info)
        
        # 更新
        new_src = list(self.src_points_tcg)
        new_src[index] = tcg_pos
        self.src_points_tcg = new_src
        
        self._update_arrows()
        self.update_preview()
    
    def _on_dst_point_move(self, index: int, handle):
        """dstポイントがドラッグされた"""
        if self.source_image is None:
            return
        
        # h, w = self.source_image.shape[:2]
        # image_shape = (h, w)
        # widget_size = (self.image_widget.width, self.image_widget.height)
        
        # Kivy座標→TCG座標
        kx, ky = handle.center
        # tcg_pos = kivy_to_tcg_coords(kivy_pos, widget_size, image_shape)
        tcg_pos = core.window_to_tcg(kx, ky, self, self.texture_size, self.tcg_info)
        
        # 更新
        new_dst = list(self.dst_points_tcg)
        new_dst[index] = tcg_pos
        self.dst_points_tcg = new_dst
        
        self._update_arrows()
        self.update_preview()
    
    def _on_points_change(self, instance, value):
        """ポイント変更時"""
        self.dispatch('on_points_change')
    
    def _on_influence_change(self, instance, value):
        """influence変更時"""
        self.dispatch("on_edit_start")
        self.update_preview()
        self.dispatch("on_edit_end")
        self.dispatch('on_influence_change')
    
    def _update_arrows(self):
        """矢印（接続線）を再描画"""
        self.canvas.after.clear()
        
        if len(self.src_handles) != len(self.dst_handles):
            return
        
        with self.canvas.after:
            Color(1, 1, 1, 0.6)  # 白色半透明
            
            for src_h, dst_h in zip(self.src_handles, self.dst_handles):
                # 線を描画
                Line(
                    points=[
                        src_h.center_x, src_h.center_y,
                        dst_h.center_x, dst_h.center_y
                    ],
                    width=2
                )
                
                # 矢印の頭を描画（簡易版）
                dx = dst_h.center_x - src_h.center_x
                dy = dst_h.center_y - src_h.center_y
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 10:
                    # 正規化
                    dx /= length
                    dy /= length
                    
                    # 矢印の頭の位置
                    arrow_size = 10
                    arrow_x = dst_h.center_x
                    arrow_y = dst_h.center_y
                    
                    # 矢印の2つの端点
                    angle = math.atan2(dy, dx)
                    left_angle = angle + math.pi * 3 / 4
                    right_angle = angle - math.pi * 3 / 4
                    
                    left_x = arrow_x + arrow_size * math.cos(left_angle)
                    left_y = arrow_y + arrow_size * math.sin(left_angle)
                    right_x = arrow_x + arrow_size * math.cos(right_angle)
                    right_y = arrow_y + arrow_size * math.sin(right_angle)
                    
                    Line(points=[left_x, left_y, arrow_x, arrow_y, right_x, right_y], width=2)
    
    def _display_image(self, image: np.ndarray):
        """画像を表示用に変換してWidgetに設定"""
        # float32 [0, 1] RGB画像をそのまま使用
        h, w = image.shape[:2]
        
        # Kivy Textureに変換（RGB, float32）
        texture = Texture.create(size=(w, h), colorfmt='rgb', bufferfmt='float')
        texture.blit_buffer(image.tobytes(), colorfmt='rgb', bufferfmt='float')
        texture.flip_vertical()
        
        self.image_widget.texture = texture
    
    def on_points_change(self):
        """ポイント変更イベント"""
        pass
    
    def on_influence_change(self):
        """影響度変更イベント"""
        pass
    
    def _update_arrows(self):
        """矢印（接続線）を再描画"""
        self.canvas.after.clear()
        
        if len(self.src_handles) != len(self.dst_handles):
            return
        
        with self.canvas.after:
            Color(1, 1, 1, 0.6)  # 白色半透明
            
            for src_h, dst_h in zip(self.src_handles, self.dst_handles):
                # 線を描画
                Line(
                    points=[
                        src_h.center_x, src_h.center_y,
                        dst_h.center_x, dst_h.center_y
                    ],
                    width=2
                )
                
                # 矢印の頭を描画（簡易版）
                dx = dst_h.center_x - src_h.center_x
                dy = dst_h.center_y - src_h.center_y
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 10:
                    # 正規化
                    dx /= length
                    dy /= length
                    
                    # 矢印の頭の位置
                    arrow_size = 10
                    arrow_x = dst_h.center_x
                    arrow_y = dst_h.center_y
                    
                    # 矢印の2つの端点
                    angle = math.atan2(dy, dx)
                    left_angle = angle + math.pi * 3 / 4
                    right_angle = angle - math.pi * 3 / 4
                    
                    left_x = arrow_x + arrow_size * math.cos(left_angle)
                    left_y = arrow_y + arrow_size * math.sin(left_angle)
                    right_x = arrow_x + arrow_size * math.cos(right_angle)
                    right_y = arrow_y + arrow_size * math.sin(right_angle)
                    
                    Line(points=[left_x, left_y, arrow_x, arrow_y, right_x, right_y], width=2)
    
    def _display_image(self, image: np.ndarray):
        """画像を表示用に変換してWidgetに設定"""
        # float32 [0, 1] RGB画像をそのまま使用
        h, w = image.shape[:2]
        
        # Kivy Textureに変換（RGB, float32）
        texture = Texture.create(size=(w, h), colorfmt='rgb', bufferfmt='float')
        texture.blit_buffer(image.tobytes(), colorfmt='rgb', bufferfmt='float')
        texture.flip_vertical()
        
        self.image_widget.texture = texture
    
    def on_points_change(self):
        """ポイント変更イベント"""
        pass
    
    def on_influence_change(self):
        """影響度変更イベント"""
        pass

