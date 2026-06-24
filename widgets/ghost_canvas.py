"""LensGhostCanvas — preview_widget に重ねる光源CP配置オーバーレイ。

DistortionCanvas(widgets/distortion_painter.py) を範に、Liquify と同じ流儀で
preview に mount される。光源CP(コントロールポイント)を左クリックで追加/選択、
ドラッグで移動、右クリックで削除する。CP は正規化TCG座標で保持し、回転/反転/
クロップに追従する(params.window_to_tcg / tcg_to_window)。

実際のゴースト描画は LensGhostEffect.make_diff(create_ghost)が行う。本キャンバスは
配置UIと callback('focus'/'start'/'apply'/'end') の通知のみを担う。
"""

import time

from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.graphics import Color, Line, Ellipse, PushMatrix, PopMatrix
from kivy.graphics.scissor_instructions import ScissorPush, ScissorPop

import params
import config


class LensGhostCanvas(KVFloatLayout):
    GRAB_RADIUS = 22   # window px。CP掴み判定
    CP_RADIUS = 7      # マーカー半径(window px)

    def __init__(self, image_widget=None, coords=None, callback=None, **kwargs):
        super().__init__(**kwargs)
        self.image_widget = image_widget
        self.coords = [tuple(c) for c in coords] if coords else []  # 正規化TCG (cx, cy)
        self.callback = callback
        self.selected = -1
        self._dragging = False
        self._last_apply = 0.0   # ドラッグ中の再描画スロットル用(秒)
        self.tcg_info = params.param_to_tcg_info({})
        # マーカー再描画はメインスレッドで(set_primary_param は描画スレッドから呼ばれ得る)。
        self._marker_trigger = Clock.create_trigger(self._refresh_markers, 0)
        self.bind(parent=self.on_parent_changed)

    # ---------- preview への追従 ----------
    def on_parent_changed(self, instance, parent):
        if parent:
            if self.image_widget is None:
                self.image_widget = parent
            self._sync_bounds()
            parent.bind(pos=self._sync_bounds, size=self._sync_bounds)

    def _sync_bounds(self, *args):
        if self.image_widget is not None:
            self.pos = self.image_widget.pos
            self.size = self.image_widget.size
        self._refresh_markers()

    # ---------- effect 連携 ----------
    def set_primary_param(self, primary_param):
        self.tcg_info = params.param_to_tcg_info(primary_param)
        # 描画スレッドから呼ばれ得るのでマーカー更新はメインスレッドへ委譲(trigger は合体される)。
        self._marker_trigger()

    def set_coords(self, coords):
        self.coords = [tuple(c) for c in coords] if coords else []
        if self.selected >= len(self.coords):
            self.selected = -1
        self._refresh_markers()

    def get_coords(self):
        return [tuple(c) for c in self.coords]

    # ---------- 座標変換 (DistortionCanvas と同方式) ----------
    def _window_to_tcg(self, cx, cy):
        return params.window_to_tcg(cx, cy, self, config.get_preview_texture_size(), self.tcg_info)

    def _tcg_to_window(self, cx, cy):
        return params.tcg_to_window(cx, cy, self, config.get_preview_texture_size(), self.tcg_info)

    # ---------- マウス ----------
    def on_touch_down(self, touch):
        if self.image_widget is None or not self.image_widget.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if self.callback is not None:
            self.callback('focus', self)
        if getattr(touch, 'is_mouse_scrolling', False):
            return super().on_touch_down(touch)

        right = getattr(touch, 'button', 'left') == 'right'
        idx = self._nearest(touch.x, touch.y)
        if right:
            if idx is not None:
                self._emit('start')
                del self.coords[idx]
                self.selected = -1
                self._refresh_markers()
                self._emit('apply')
                self._emit('end')
        elif idx is not None:
            self.selected = idx
            self._dragging = True
            self._refresh_markers()
            self._emit('start')
        else:
            self._emit('start')
            self.coords.append(self._window_to_tcg(touch.x, touch.y))
            self.selected = len(self.coords) - 1
            self._dragging = True
            self._refresh_markers()
            self._emit('apply')
        return True

    def on_touch_move(self, touch):
        if self._dragging and 0 <= self.selected < len(self.coords):
            self.coords[self.selected] = self._window_to_tcg(touch.x, touch.y)
            # マーカー(CP位置)は毎フレーム即追従。重いゴースト再描画(apply)は
            # スロットルして溜め込まない。最終位置は on_touch_up の 'end' で必ず反映。
            self._refresh_markers()
            now = time.monotonic()
            if now - self._last_apply >= 0.05:
                self._last_apply = now
                self._emit('apply')
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self._dragging:
            self._dragging = False
            self._emit('end')
            return True
        return super().on_touch_up(touch)

    def _nearest(self, wx, wy):
        best, best_d = None, self.GRAB_RADIUS ** 2
        for i, (cx, cy) in enumerate(self.coords):
            x, y = self._tcg_to_window(cx, cy)
            d = (x - wx) ** 2 + (y - wy) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def _emit(self, proc):
        if self.callback is not None:
            self.callback(proc, self)

    # ---------- マーカー描画 ----------
    def _refresh_markers(self, *args):
        self.canvas.after.clear()
        if not self.coords:
            return
        with self.canvas.after:
            PushMatrix()
            ScissorPush(x=int(self.pos[0]), y=int(self.pos[1]),
                        width=int(self.size[0]), height=int(self.size[1]))
            for i, (cx, cy) in enumerate(self.coords):
                x, y = self._tcg_to_window(cx, cy)
                r = self.CP_RADIUS
                if i == self.selected:
                    # アクティブ: 紺の塗り + 白縁 (はっきり見えるように塗りつぶし)
                    Color(0.08, 0.13, 0.55, 1.0)
                    Ellipse(pos=(x - r, y - r), size=(2 * r, 2 * r))
                    Color(1.0, 1.0, 1.0, 1.0)
                    Line(circle=(x, y, r), width=1.6)
                else:
                    # 非アクティブ: 白の塗り + 暗縁
                    Color(1.0, 1.0, 1.0, 1.0)
                    Ellipse(pos=(x - r, y - r), size=(2 * r, 2 * r))
                    Color(0.1, 0.1, 0.1, 0.85)
                    Line(circle=(x, y, r), width=1.2)
            ScissorPop()
            PopMatrix()
