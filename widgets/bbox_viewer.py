
import logging

from kivy.app import App as KVApp
from kivy.uix.widget import Widget as KVWidget
from kivy.graphics import Color, Line
from kivy.core.window import Window as KVWindow

import config
import params
import utils.kvutils as kvutils

class BoundingBoxViewer(KVWidget):
    def __init__(self, param=None, size=(800, 600), initial_view=(0, 0, 1000, 800, 1.0), on_delete=None, **kwargs):
        super().__init__(**kwargs)

        self.pos_hint = {'x': 0, 'top': 1}

        # 座標変換の基準となる param（primary_param）。disp_info(ズーム/パン/クロップ)を
        # 毎回ここから読み直すことで、MaskEditor(mask1) と同じく拡大表示に追従する。
        self.param = param

        # 最大表示サイズ（ビュー表示用）
        self.max_display_width, self.max_display_height = size

        # 表示範囲とスケール（param が無い場合のフォールバック / デモ用）
        self.view_x, self.view_y, self.view_w, self.view_h, self.scale = initial_view

        # バウンディングボックス（マスク座標系 = original_img_size 基準の (x, y, w, h)）
        self.boxes = []  # [(x, y, w, h), ...]
        self.selected_index = None
        self.overlapping_indices = []  # 重複しているボックスのインデックス
        self.overlap_cycle_index = 0  # 重複時の選択サイクル

        # コールバック
        self.on_delete_callback = on_delete

        # 描画スタイル設定（カスタマイズ可能）
        self.normal_color = (1, 0, 0, 1)  # 赤
        self.selected_color = (0, 1, 0, 1)  # 緑
        self.line_width = 2
        self.selected_line_width = 3

        # イベントバインド
        self.bind(size=self.on_size_change)
        self.bind(pos=self.on_pos_change)

        # キーボードイベントをバインド
        KVWindow.bind(on_key_down=self.on_key_down)

    def __del__(self):
        KVWindow.unbind(on_key_down=self.on_key_down)

    def set_param(self, param):
        """座標変換の基準 param を更新する"""
        self.param = param
        self.redraw()

    def set_boxes(self, boxes):
        """バウンディングボックスのリストを設定"""
        self.boxes = list(boxes)
        self.selected_index = None
        self.overlapping_indices = []
        self.overlap_cycle_index = 0
        self.redraw()

    def set_view(self, x, y, w, h, scale):
        """表示範囲とスケールを設定（param 未指定のフォールバック用）"""
        self.view_x, self.view_y, self.view_w, self.view_h, self.scale = x, y, w, h, scale
        self.redraw()

    def set_display_size(self, size):
        """表示用の最大サイズを更新"""
        self.max_display_width, self.max_display_height = size
        self.redraw()

    def set_style(self, normal_color=None, selected_color=None, line_width=None, selected_line_width=None):
        """描画スタイルを設定"""
        if normal_color:
            self.normal_color = normal_color
        if selected_color:
            self.selected_color = selected_color
        if line_width:
            self.line_width = line_width
        if selected_line_width:
            self.selected_line_width = selected_line_width
        self.redraw()

    # -------------------------------------------------------------------------
    # 座標変換（MaskEditor と同じ TCG 座標系を使い、拡大表示/パン/回転に追従する）
    # -------------------------------------------------------------------------
    def _root_widget(self):
        try:
            return kvutils.get_root_widget(self)
        except Exception:
            return None

    def _is_space_panning(self):
        root = self._root_widget()
        return bool(getattr(root, 'is_press_space', False))

    def _mask_shape(self):
        """マスク座標系の (高さ, 幅) を返す。mask1 の MaskEditor と同じ original_img_size 基準。"""
        orig = self.param.get('original_img_size') if self.param else None
        if not orig:
            return None
        w, h = int(orig[0]), int(orig[1])
        return h, w

    def _mask_to_window(self, mx, my, tcg_info, texture_size):
        """マスク座標をウィンドウ座標に変換（MaskEditor._window_to_mask_coords の逆変換）"""
        m_h, m_w = self._mask_shape()
        m_max = max(m_h, m_w)
        px = mx + (m_max - m_w) // 2
        py = my + (m_max - m_h) // 2
        tx, ty = params.ref_image_to_tcg(px, py, None, tcg_info, apply_disp_info=False)
        return params.tcg_to_window(tx, ty, self, texture_size, tcg_info, normalize=True)

    def _window_to_mask(self, wx, wy, tcg_info, texture_size):
        """ウィンドウ座標をマスク座標に変換（MaskEditor._window_to_mask_coords 相当）"""
        m_h, m_w = self._mask_shape()
        m_max = max(m_h, m_w)
        tx, ty = params.window_to_tcg(wx, wy, self, texture_size, tcg_info, normalize=True)
        ix, iy = params.tcg_to_ref_image(tx, ty, None, tcg_info, apply_disp_info=False)
        ix, iy = ix - (m_max - m_w) // 2, iy - (m_max - m_h) // 2
        return ix, iy

    def _point_in_box(self, point_x, point_y, box):
        """点がボックス内にあるかチェック（マスク座標系）"""
        box_x, box_y, box_w, box_h = box
        return (box_x <= point_x <= box_x + box_w and
                box_y <= point_y <= box_y + box_h)

    def _find_overlapping_boxes(self, mask_x, mask_y):
        """指定した点に重なるボックスのインデックスを取得（後方から前方の順）"""
        overlapping = []
        for i in range(len(self.boxes) - 1, -1, -1):  # 後方から検索（上位優先）
            if self._point_in_box(mask_x, mask_y, self.boxes[i]):
                overlapping.append(i)
        return overlapping

    def redraw(self):
        """画面を再描画（現在の disp_info に基づき、拡大表示に追従する）"""
        self.canvas.clear()

        # param が無い / original_img_size が未定義なら描画できない
        if self.param is None or self._mask_shape() is None:
            return

        tcg_info = params.param_to_tcg_info(self.param)
        texture_size = config.get_preview_texture_size()

        with self.canvas:
            for i, box in enumerate(self.boxes):
                box_x, box_y, box_w, box_h = box

                # ボックスの四隅をマスク座標→ウィンドウ座標に変換
                # （回転/反転が掛かっていても正しい四辺形になるよう4隅を個別変換）
                corners = (
                    (box_x, box_y),
                    (box_x + box_w, box_y),
                    (box_x + box_w, box_y + box_h),
                    (box_x, box_y + box_h),
                )
                points = []
                for cx, cy in corners:
                    wx, wy = self._mask_to_window(cx, cy, tcg_info, texture_size)
                    points.extend((wx, wy))

                # 選択状態に応じて色と線幅を設定
                if i == self.selected_index:
                    Color(*self.selected_color)
                    width = self.selected_line_width
                else:
                    Color(*self.normal_color)
                    width = self.line_width

                # 矩形の枠を描画（閉じたポリライン）
                Line(points=points, width=width, close=True)

    def on_touch_down(self, touch):
        """マウスクリック処理"""
        # ズーム(スクロール)やスペースパンは preview 側に委ねてズーム操作を妨げない
        if self.param is None or self._mask_shape() is None:
            return super().on_touch_down(touch)
        if touch.is_mouse_scrolling or self._is_space_panning():
            return super().on_touch_down(touch)

        if self.collide_point(*touch.pos):
            tcg_info = params.param_to_tcg_info(self.param)
            texture_size = config.get_preview_texture_size()

            # ウィンドウ座標をマスク座標に変換
            mask_x, mask_y = self._window_to_mask(touch.x, touch.y, tcg_info, texture_size)

            logging.debug("Touch: (%s, %s)", touch.x, touch.y)
            logging.debug("Mask: (%s, %s)", mask_x, mask_y)

            # 重なるボックスを検索
            overlapping = self._find_overlapping_boxes(mask_x, mask_y)

            if overlapping:
                if overlapping != self.overlapping_indices:
                    # 新しい場所をクリック
                    self.overlapping_indices = overlapping
                    self.overlap_cycle_index = 0
                    self.selected_index = overlapping[0]
                else:
                    # 同じ場所を再クリック - 次のボックスに切り替え
                    self.overlap_cycle_index = (self.overlap_cycle_index + 1) % len(overlapping)
                    self.selected_index = overlapping[self.overlap_cycle_index]
                self.redraw()
                return True

            # 何もない場所をクリック - 選択解除して preview 側にタッチを渡す
            self.selected_index = None
            self.overlapping_indices = []
            self.overlap_cycle_index = 0
            self.redraw()

        return super().on_touch_down(touch)

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        """キー入力処理"""
        if key == 8:  # Backspace
            if self.selected_index is not None and self.selected_index < len(self.boxes):
                deleted_box = self.boxes[self.selected_index]
                deleted_index = self.selected_index

                # ボックスを削除
                del self.boxes[self.selected_index]

                # 選択状態をリセット
                self.selected_index = None
                self.overlapping_indices = []
                self.overlap_cycle_index = 0

                # コールバック呼び出し
                if self.on_delete_callback:
                    self.on_delete_callback(deleted_index, deleted_box)

                self.redraw()

    def on_size_change(self, instance, value):
        """ウィンドウサイズ変更時の処理"""
        self.redraw()

    def on_pos_change(self, instance, value):
        """位置変更時の処理"""
        self.redraw()


class BoundingBoxApp(KVApp):
    def build(self):
        from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
        from kivy.uix.button import Button as KVButton

        def on_delete(index, box):
            logging.info("Deleted box %s: %s", index, box)

        # テスト用の param（400x300 の画像、等倍表示）
        orig = (400, 300)
        maxsize = max(orig)
        param = {
            'original_img_size': orig,
            'disp_info': (0.0, 0.0, orig[0] / maxsize, orig[1] / maxsize, 1.0),
        }
        config.set_preview_texture_size(*orig)

        # テスト用のバウンディングボックス（マスク座標系）
        test_boxes = [
            (50, 50, 100, 80),
            (120, 70, 90, 60),
            (80, 80, 70, 50),  # 重複するボックス
            (200, 150, 120, 90),
            (180, 200, 80, 70),
        ]

        # FloatLayoutを作成してテスト
        layout = KVFloatLayout()

        viewer = BoundingBoxViewer(
            param=param,
            size=orig,  # ビューの最大表示サイズ
            on_delete=on_delete
        )
        # pos_hintとsize_hintを使用して位置とサイズを指定
        viewer.pos_hint = {'x': 0.1, 'y': 0.2}  # 画面の10%,20%の位置
        viewer.size_hint = (0.8, 0.7)  # 画面の80%,70%のサイズ

        viewer.set_boxes(test_boxes)

        # テスト用ボタンを追加（拡大表示の確認）
        btn = KVButton(
            text='Zoom x1.5',
            size_hint=(None, None),
            size=(100, 50),
            pos=(10, 10)
        )

        def change_view_button(instance):
            param['disp_info'] = (0.0, 0.0, orig[0] / maxsize, orig[1] / maxsize, 1.5)
            viewer.redraw()
            logging.info("Zoom changed to 1.5")

        btn.bind(on_press=change_view_button)

        layout.add_widget(viewer)
        layout.add_widget(btn)

        return layout


if __name__ == '__main__':
    BoundingBoxApp().run()
