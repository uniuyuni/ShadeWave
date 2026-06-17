"""
4点自由補正Widget
"""

from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.uix.scatter import Scatter as KVScatter
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.properties import ListProperty as KVListProperty, StringProperty as KVStringProperty
from kivy.graphics import Color as KVColor, Line as KVLine, PushMatrix as KVPushMatrix, PopMatrix as KVPopMatrix, Translate as KVTranslate, Ellipse as KVEllipse
from kivy.uix.image import Image as KVImage
from kivy.clock import mainthread as kvmainthread
import numpy as np
import cv2

from cores.distortion_correction.four_point_correction import correct_four_points, detect_rectangle
import params
from utils import kvutils
from widgets.scaled_button import ScaledButton

class FourPointCorrectionWidget(KVFloatLayout):
    """4点自由補正Widget"""
    
    # プロパティ
    corner_positions_tcg = KVListProperty([])  # TCG座標系
    interpolation = KVStringProperty('bicubic')
    
    def __init__(self, texture_size, param, **kwargs):
        super().__init__(**kwargs)

        self.texture_size = texture_size
        self.tcg_info = params.param_to_tcg_info(param)
        self.image_shape = (param['original_img_size'][1], param['original_img_size'][0])

        self.on_callback = None

        # ハンドルリスト
        self.handles = []

        button_layout = KVBoxLayout(
            orientation='horizontal',
            size_hint=(None, None),
            pos_hint={'center_x': 0.5, 'y': 0.035},
        )
        button_layout.ref_width = 180
        button_layout.ref_height = 22
        button_layout.ref_layout_spacing = 10
        button_layout.ref_layout_padding = 5
        button_layout.bind(minimum_height=button_layout.setter('height'))
        kvutils.traverse_widget(button_layout)

        # リセットボタン
        self.reset_btn = ScaledButton(text="Reset")
        self.reset_btn.set_ref_metrics()
        self.reset_btn.bind(on_press=self.reset_corners)
        button_layout.add_widget(self.reset_btn)

        # 適用ボタン
        self.apply_btn = ScaledButton(text="Apply")
        self.apply_btn.set_ref_metrics()
        self.apply_btn.bind(on_press=self._apply_corners)
        button_layout.add_widget(self.apply_btn)

        # 戻すボタン
        self.revert_btn = ScaledButton(text="Revert")
        self.revert_btn.set_ref_metrics()
        self.revert_btn.bind(on_press=self._revert_corners)
        button_layout.add_widget(self.revert_btn)

        self.add_widget(button_layout)

        # プロパティの変更を監視
        self.updating_handles = False
        self.is_dragging = False
        self.grab_current = -1
        self._using_default_corners = True
        #self.bind(corner_positions_tcg=self._on_corners_change)
        self.bind(size=self._sync_tcg_to_kivy, pos=self._sync_tcg_to_kivy)

        self._reset_corners()
        kvutils.install_ref_scaling(self)

    @staticmethod
    def _default_corners():
        return [
            (-0.5, -0.5),
            (0.5, -0.5),
            (0.5, 0.5),
            (-0.5, 0.5),
        ]

    def set_callback(self, callback):
        self.on_callback = callback

    def set_texture_size(self, texture_size):
        self.texture_size = texture_size
        self._sync_tcg_to_kivy()

    def on_edit_start(self):
        """編集開始イベント（ヒストリー管理用）"""
        if self.on_callback:
            self.on_callback('start', self)
    
    def on_edit_end(self):
        """編集終了イベント（ヒストリー管理用）"""
        if self.on_callback:
            self.on_callback('end', self)

    def on_apply_corners(self):
        """コーナー適用イベント"""
        if self.on_callback:
            self.on_callback('apply', self)

    def reset_corners(self, *args):
        self.on_edit_start()
        self._reset_corners()
        self.on_edit_end()  

    def _reset_corners(self):        
        # TCG座標系で四隅を設定（正規化座標）
        # 画像の四隅に対応
        self._using_default_corners = True
        self.corner_positions_tcg = self._default_corners()
        self._sync_tcg_to_kivy()
        self._apply_corners()

    def _apply_corners(self, *args):
        self.on_apply_corners()
    
    def _revert_corners(self, *args):
        backup = self.corner_positions_tcg.copy()
        self._reset_corners() # Reset UI and Params to Identity
        self._apply_corners() # Apply restored state to Params
        self.corner_positions_tcg = backup # Restore backup to internal state
        self._sync_tcg_to_kivy() # Restore UI handles
    
    def get_correction_params(self) -> dict:
        """
        現在のパラメータを取得（TCG座標系）
        
        Returns:
            dict: {"four_points": [(x,y), ...]}
        """
        if self._using_default_corners:
            return {"four_points": []}
        return {"four_points": list(self.corner_positions_tcg)}
    
    def set_correction_params(self, param: dict):
        """
        パラメータを設定（TCG座標系）
        
        Args:
            param: dict、{"four_points": [(x,y), ...]}
        """
        four_points = param.get('four_points', [])
        self._using_default_corners = four_points == []
        # 未設定の Four Points は param としては [] のまま扱う。
        # 表示時だけ、四隅に置いたマーカー中心が画面外へ出た場合に端へ戻す。
        self.tcg_info = params.param_to_tcg_info(param)
        self.corner_positions_tcg = list(four_points) if four_points != [] else self._default_corners()
        self._sync_tcg_to_kivy()
        self.update_preview()
    
    @kvmainthread
    def update_preview(self, *args):
        pass

    @kvmainthread
    def _sync_tcg_to_kivy(self, *args):
        """TCG座標をKivy座標に同期してハンドルを更新"""
        self.updating_handles = True
        try:
            # 既存ハンドル数チェック
            if len(self.handles) != 4:
                for handle in self.handles:
                    self.remove_widget(handle)
                self.handles = []
                for i in range(4):
                    handle = self._create_handle(i, (0,0))
                    self.handles.append(handle)
                    self.add_widget(handle)
    
            for i, (tx, ty) in enumerate(self.corner_positions_tcg):
                kx, ky = params.tcg_to_window(tx, ty, self, self.texture_size, self.tcg_info)
                if self._using_default_corners:
                    kx, ky = self._clamp_handle_center_to_widget(kx, ky, self.handles[i])
                self.handles[i].center = (kx, ky)
            
            # 接続線を描画
            self._update_lines()
        finally:
            self.updating_handles = False
    
    def _sync_kivy_to_tcg(self, moved_index=None, preserve_default_corners=False):
        """Kivy座標をTCG座標に同期"""

        self.corner_positions_tcg = []
        default_corners = self._default_corners()
        for i, handle in enumerate(self.handles):
            if preserve_default_corners and i != moved_index:
                self.corner_positions_tcg.append(default_corners[i])
                continue
            kx, ky = handle.center
            tx, ty = params.window_to_tcg(kx, ky, self, self.texture_size, self.tcg_info)
            self.corner_positions_tcg.append((tx, ty))

    def _clamp_handle_center_to_widget(self, x, y, _handle):
        wx, wy = self.to_window(*self.pos)
        min_x = wx
        min_y = wy
        max_x = wx + self.width
        max_y = wy + self.height
        if max_x < min_x:
            min_x = max_x = wx + self.width / 2
        if max_y < min_y:
            min_y = max_y = wy + self.height / 2
        return (
            min(max(x, min_x), max_x),
            min(max(y, min_y), max_y),
        )
            
    def _create_handle(self, index: int, pos: tuple):
        """ハンドルを作成"""
        handle = KVScatter(
            size=(40, 40),
            size_hint=(None, None), # 重要: タッチ判定を制限する
            do_rotation=False,
            do_scale=False,
            auto_bring_to_front=True
        )
        
        # 円形を描画
        with handle.canvas:
            #KVPushMatrix()
            KVColor(0.2, 0.6, 1.0, 0.8)  # 青色
            KVEllipse(pos=(0, 0), size=(40, 40))
            #KVPopMatrix()
        
        # 位置を設定
        handle.center = pos
        
        # ドラッグイベントをバインド
        handle.bind(pos=lambda inst, touch: self._on_handle_move(index))
        handle.bind(on_touch_up=lambda inst, touch: self._on_handle_release(index, touch))
        
        return handle
    
    def _on_handle_move(self, index: int):
        """ハンドルがドラッグされた"""
        if self.updating_handles:
            return
        
        self.is_dragging = True
        preserve_default_corners = self._using_default_corners
        self.on_edit_start()

        self.grab_current = index
        
        self._sync_kivy_to_tcg(
            moved_index=index,
            preserve_default_corners=preserve_default_corners,
        )
        self._using_default_corners = False
        # 範囲制限
        for i, pos in enumerate(self.corner_positions_tcg):
            if pos[0] < -0.5:
                self.corner_positions_tcg[i] = (-0.5, pos[1])
            elif pos[0] > 0.5:
                self.corner_positions_tcg[i] = (0.5, pos[1])
            if pos[1] < -0.5:
                self.corner_positions_tcg[i] = (pos[0], -0.5)
            elif pos[1] > 0.5:
                self.corner_positions_tcg[i] = (pos[0], 0.5)
        self._sync_tcg_to_kivy()
        self._update_lines()
        
    def _on_handle_release(self, index, touch):
        """ハンドルが離された"""
        if self.grab_current == index:
            self.is_dragging = False
            self.on_edit_end()
    
    def _update_lines(self):
        """接続線を再描画"""
        self.canvas.after.clear()
        
        if len(self.handles) != 4:
            return
        
        with self.canvas.after:
            KVColor(1, 1, 1, 1)  # 白色
            
            # 4点を結ぶ線
            points = []
            for handle in self.handles:
                points.extend([handle.center_x, handle.center_y])
            
            # 最初の点に戻る
            points.extend([self.handles[0].center_x, self.handles[0].center_y])
            
            KVLine(points=points, width=2)
