"""
メッシュワープWidget

KivyMDベースのGUIウィジェット
"""

from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.uix.widget import Widget as KVWidget
from kivy.uix.button import Button as KVButton
from kivy.properties import ListProperty as KVListProperty, DictProperty as KVDictProperty
from kivy.graphics import Color, Line
from kivy.clock import mainthread as kvmainthread
from kivymd.uix.button import MDRaisedButton, MDIconButton, MDTextButton
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
import numpy as np
import cv2

from cores.distortion_correction.warp_correction import get_mesh_coordinates
import params

class MeshWarpWidget(KVFloatLayout):
    """メッシュワープWidget"""
    
    # プロパティ
    mesh_size = KVListProperty([4, 4]) # [rows, cols]
    control_offsets_tcg = KVDictProperty({}) # {(row, col): (off_x, off_y)}
    
    def __init__(self, texture_size, param, **kwargs):
        super().__init__(**kwargs)
        
        self.texture_size = texture_size
        self.tcg_info = params.param_to_tcg_info(param)

        self.on_callback = None
        
        # 描画用オーバーレイ
        self.draw_overlay = KVWidget()
        self.add_widget(self.draw_overlay)
        
        # 状態変数
        self.selected_point = None # (row, col)
        self.dragging = False
        
        # UIコントロール（リセットボタンなど）
        button_layout = MDBoxLayout(
            orientation='horizontal',
            pos_hint={'center_x': 0.5, 'y': 0.02},
            adaptive_size=True,
            spacing=10,
            padding=10
        )

        self.reset_btn = MDRaisedButton(text="Reset")
        self.reset_btn.bind(on_press=self.reset_mesh)
        button_layout.add_widget(self.reset_btn)

        # Rows Control
        button_layout.add_widget(MDLabel(text="Y:", size_hint_x=None, width=40, halign='right', theme_text_color="Custom", text_color=(1,1,1,1)))
        btn_row_minus = MDTextButton(text="-", width=20, height=20, on_release=lambda x: self.change_mesh_size(-2, 0))
        btn_row_minus.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        btn_row_minus.size_hint = (None, None)
        btn_row_minus.ref_width = 20
        btn_row_minus.ref_height = 20
        btn_row_minus.font_size = "24sp"
        button_layout.add_widget(btn_row_minus)
        
        self.label_rows = MDLabel(text=str(self.mesh_size[0]), size_hint_x=None, width=40, halign='center', theme_text_color="Custom", text_color=(1,1,1,1))
        self.label_rows.size_hint_x = None
        self.label_rows.ref_width = 40
        button_layout.add_widget(self.label_rows)
        
        btn_row_plus = MDTextButton(text="+", width=20, height=20, on_release=lambda x: self.change_mesh_size(2, 0))
        btn_row_plus.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        btn_row_plus.size_hint = (None, None)
        btn_row_plus.ref_width = 20
        btn_row_plus.ref_height = 20
        button_layout.add_widget(btn_row_plus)

        # Cols Control
        button_layout.add_widget(MDLabel(text="X:", size_hint_x=None, width=40, halign='right', theme_text_color="Custom", text_color=(1,1,1,1)))
        btn_col_minus = MDTextButton(text="-", width=20, height=20, on_release=lambda x: self.change_mesh_size(0, -2))
        btn_col_minus.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        btn_col_minus.size_hint = (None, None)
        btn_col_minus.ref_width = 20
        btn_col_minus.ref_height = 20
        btn_col_minus.font_size = "24sp"
        button_layout.add_widget(btn_col_minus)
        
        self.label_cols = MDLabel(text=str(self.mesh_size[1]), size_hint_x=None, width=40, halign='center', theme_text_color="Custom", text_color=(1,1,1,1))
        self.label_cols.size_hint_x = None
        self.label_cols.ref_width = 40
        button_layout.add_widget(self.label_cols)
        
        btn_col_plus = MDTextButton(text="+", width=20, height=20, on_release=lambda x: self.change_mesh_size(0, 2))
        btn_col_plus.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        btn_col_plus.size_hint = (None, None)
        btn_col_plus.ref_width = 20
        btn_col_plus.ref_height = 20
        button_layout.add_widget(btn_col_plus)
        
        self.apply_btn = MDRaisedButton(text="Apply")
        self.apply_btn.bind(on_press=lambda x: self._apply())
        button_layout.add_widget(self.apply_btn)

        self.add_widget(button_layout)
        
        # Update labels on property change
        self.bind(mesh_size=self._update_labels)

        # タッチイベント
        self.draw_overlay.bind(
            on_touch_down=self._on_touch_down,
            on_touch_move=self._on_touch_move,
            on_touch_up=self._on_touch_up
        )
        
        # プロパティ監視
        self.bind(mesh_size=self._redraw_mesh)
        self.bind(control_offsets_tcg=self._redraw_mesh)
        self.bind(size=self._redraw_mesh)
        self.bind(pos=self._redraw_mesh)

    def _update_labels(self, instance, value):
        self.label_rows.text = str(value[0])
        self.label_cols.text = str(value[1])

    def change_mesh_size(self, d_row, d_col):
        """メッシュサイズを変更し、オフセットを補間する"""
        rows, cols = self.mesh_size
        new_rows = rows + d_row
        new_cols = cols + d_col
        
        # 制限 (偶数のみ、最小4、最大20程度)
        if new_rows < 4: new_rows = 4
        if new_cols < 4: new_cols = 4
        if new_rows > 16: new_rows = 16
        if new_cols > 16: new_cols = 16
        
        if new_rows == rows and new_cols == cols:
            return

        # 現在のオフセットを密な配列に変換 (rows+1, cols+1, 2)
        current_field = np.zeros((rows + 1, cols + 1, 2), dtype=np.float32)
        for (r, c), offset in self.control_offsets_tcg.items():
            current_field[r, c] = offset
            
        # リサイズ (補間)
        # cv2.resize takes (width, height) -> (cols, rows)
        new_field = cv2.resize(current_field, (new_cols + 1, new_rows + 1), interpolation=cv2.INTER_LINEAR)
        
        # 配列を新しいDictに戻す
        new_offsets = {}
        for r in range(new_rows + 1):
            for c in range(new_cols + 1):
                off = new_field[r, c]
                if abs(off[0]) > 1e-6 or abs(off[1]) > 1e-6:
                    new_offsets[(r, c)] = (float(off[0]), float(off[1]))
        
        # 一括更新
        self.control_offsets_tcg = new_offsets
        self.on_edit_start()
        self.mesh_size = [new_rows, new_cols]
        self.on_edit_end()

    def set_callback(self, callback):
        self.on_callback = callback

    def on_edit_start(self):
        if self.on_callback:
            self.on_callback('start', self)

    def on_edit_end(self):
        if self.on_callback:
            self.on_callback('end', self)

    def reset_mesh(self, instance=None):
        """メッシュをリセット"""
        self.on_edit_start()
        self.control_offsets_tcg = {}
        self.on_edit_end()
        self._redraw_mesh()
        self._apply()

    def get_correction_params(self) -> dict:
        """現在のパラメータを取得"""
        # JSON互換のためキーを文字列に変換
        cp_str_keys = {f"{k[0]},{k[1]}": v for k, v in self.control_offsets_tcg.items()}
        return {
            "mesh_size": list(self.mesh_size),
            "control_points": cp_str_keys,
        }
    
    def set_correction_params(self, params: dict):
        """パラメータを設定"""
        self.mesh_size = params.get('mesh_size', [4, 4])
        # Dict key must be tuple, json might give list/string
        raw_offsets = params.get('control_points', {})
        # Ensure keys are tuples
        cleaned_offsets = {}
        for k, v in raw_offsets.items():
            if isinstance(k, str):
                try:
                    # eval is dangerous but simple context depends on usage
                    # Better parse "1,2" -> (1,2)
                    parts = k.strip('()').split(',')
                    key = (int(parts[0]), int(parts[1]))
                except:
                    continue
            else:
                key = tuple(k)
            cleaned_offsets[key] = tuple(v)
            
        self.control_offsets_tcg = cleaned_offsets
        self._redraw_mesh()

    def _apply(self):
        """適用"""
        if self.on_callback:
            self.on_callback('apply', self)

    def _get_tcg_pos(self, touch_pos):
        tx, ty = params.window_to_tcg(touch_pos[0], touch_pos[1], self, self.texture_size, self.tcg_info)
        return tx, ty

    def _get_window_pos(self, tx, ty):
        wx, wy = params.tcg_to_window(tx, ty, self, self.texture_size, self.tcg_info)
        return wx, wy
        
    def _hit_test(self, tcg_x, tcg_y, touch_tcg_x, touch_tcg_y):
        # TCG空間での距離判定 
        # メッシュの密度によるが、適当な閾値
        dist = np.sqrt((tcg_x - touch_tcg_x)**2 + (tcg_y - touch_tcg_y)**2)
        return dist < 0.04 

    def _on_touch_down(self, instance, touch):
        if not self.collide_point(*touch.pos):
            return False
            
        touch.grab(self)
        
        tx, ty = self._get_tcg_pos(touch.pos)
        
        # ヒットテスト
        rows, cols = self.mesh_size
        img_shape = (self.texture_size[1], self.texture_size[0]) # H, W
        base_coords = get_mesh_coordinates(img_shape, (rows, cols)) # Shape: (rows+1, cols+1, 2)
        
        min_dist = 1000
        hit_pt = None
        
        for r in range(rows + 1):
            for c in range(cols + 1):
                bx, by = base_coords[r, c]
                off_x, off_y = self.control_offsets_tcg.get((r, c), (0, 0))
                cx = bx + off_x
                cy = by + off_y
                
                dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                if dist < 0.05: # Hit radius
                    if dist < min_dist:
                        min_dist = dist
                        hit_pt = (r, c)
        
        if hit_pt:
            # ダブルクリック判定
            if touch.is_double_tap:
                # リセット
                if hit_pt in self.control_offsets_tcg:
                    self.on_edit_start()
                    del self.control_offsets_tcg[hit_pt]
                    self.on_edit_end()
                    self._redraw_mesh()
                return True

            self.selected_point = hit_pt
            self.dragging = True
            self.on_edit_start()
            self._redraw_mesh()
            return True
            
        return True # Consume touch to prevent propagation

    def _on_touch_move(self, instance, touch):
        if touch.grab_current is not self:
            return False
            
        if self.dragging and self.selected_point:
            tx, ty = self._get_tcg_pos(touch.pos)
            
            # 元の座標を取得してオフセットを計算
            rows, cols = self.mesh_size
            img_shape = (self.texture_size[1], self.texture_size[0])
            base_coords = get_mesh_coordinates(img_shape, (rows, cols))
            r, c = self.selected_point
            bx, by = base_coords[r, c]
            
            off_x = tx - bx
            off_y = ty - by
            
            # 更新 (DictProperty update hack)
            new_offsets = dict(self.control_offsets_tcg)
            new_offsets[self.selected_point] = (off_x, off_y)
            self.control_offsets_tcg = new_offsets
            
            # リアルタイム反映する場合はここで _apply だが重いので描画のみ
            self._redraw_mesh()
            
        return True

    def _on_touch_up(self, instance, touch):
        if touch.grab_current is not self:
            return False
        
        touch.ungrab(self)
        
        if self.dragging:
            self.dragging = False
            self.selected_point = None
            self.on_edit_end()
            self._redraw_mesh()
            
        return True

    def _redraw_mesh(self, *args):
        self.draw_overlay.canvas.clear()
        
        rows, cols = self.mesh_size
        if self.texture_size[0] == 0: return

        img_shape = (self.texture_size[1], self.texture_size[0])
        base_coords = get_mesh_coordinates(img_shape, (rows, cols))
        
        # 現在の変形後座標を計算
        current_points = np.zeros_like(base_coords)
        for r in range(rows + 1):
            for c in range(cols + 1):
                bx, by = base_coords[r, c]
                off_x, off_y = self.control_offsets_tcg.get((r, c), (0, 0))
                current_points[r, c] = [bx + off_x, by + off_y]

        # 描画
        with self.draw_overlay.canvas:
            Color(0.0, 1.0, 1.0, 0.6) # Cyan lines
            
            # Horizontal lines
            for r in range(rows + 1):
                pts = []
                for c in range(cols + 1):
                    ctx, cty = current_points[r, c]
                    wx, wy = self._get_window_pos(ctx, cty)
                    pts.extend([wx, wy])
                Line(points=pts, width=1.2)
                
            # Vertical lines
            for c in range(cols + 1):
                pts = []
                for r in range(rows + 1):
                    ctx, cty = current_points[r, c]
                    wx, wy = self._get_window_pos(ctx, cty)
                    pts.extend([wx, wy])
                Line(points=pts, width=1.2)
            
            # Control points
            for r in range(rows + 1):
                for c in range(cols + 1):
                    if self.selected_point == (r, c):
                         Color(1.0, 0.8, 0.2, 1.0)
                         size = 8
                    else:
                         Color(1.0, 1.0, 1.0, 0.8)
                         size = 5
                         
                    ctx, cty = current_points[r, c]
                    wx, wy = self._get_window_pos(ctx, cty)
                    Line(circle=(wx, wy, size), width=1.5)
