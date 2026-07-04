"""
ラインガイド補正Widget
"""

from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.uix.widget import Widget as KVWidget
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.properties import ListProperty as KVListProperty, StringProperty as KVStringProperty, NumericProperty as KVNumericProperty
from kivy.graphics import Color as KVColor, Line as KVLine
from kivy.clock import mainthread as kvmainthread
from kivy.uix.image import Image as KVImage
import numpy as np
import cv2

from cores.distortion_correction.warp_correction import correct_with_lines, calculate_lines_homography
import params
from utils import kvutils
from widgets.scaled_button import ScaledButton

# 破綻(地平線越え/向き反転/退化/非凸)しそうな配置では、Lines は捨てずに安全な最大限まで
# 減衰(クランプ)して適用する。そのため純粋な破綻(赤)ではなく「頭打ち」を示すアンバーにする。
_LINE_COLOR_UNSAFE = (1.0, 0.65, 0.1, 0.95)

# 端点(CP)の表示半径と当たり半径(いずれもウィンドウ px)。当たり判定は表示円より
# ほんの少し大きい程度にして、以前のように大幅に広くならないようにする。
_POINT_SIZE = 5
_POINT_HIT_RADIUS_PX = 12

# calculate_lines_homography は5本目以降を無視して最後の4本だけを使う仕様なので、
# UI 側でも4本を上限にして「作れるのに使われない」線が生まれないようにする。
_MAX_LINES = 4


def _geom_signed_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _geom_quad_convex(pts):
    n = len(pts)
    sign = 0
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        c = pts[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _homography_is_safe(H, size):
    """ホモグラフィ H を [0,size] 画像四隅に適用し、表示破綻
    (透視分母 w の符号反転=地平線越え / 向き反転 / 退化 / 非凸) しないか判定する。"""
    if H is None:
        return True
    M = np.asarray(H, dtype=np.float64)
    if M.shape != (3, 3) or not np.all(np.isfinite(M)):
        return False
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64)
    out = np.concatenate([corners, np.ones((4, 1), dtype=np.float64)], axis=1) @ M.T
    w = out[:, 2]
    if not (np.all(w > 0.0) or np.all(w < 0.0)):
        return False
    aw = np.abs(w)
    if np.max(aw) <= 0.0 or np.min(aw) / np.max(aw) < 0.02:
        return False
    pts = out[:, :2] / w[:, None]
    if not np.all(np.isfinite(pts)):
        return False
    sa = _geom_signed_area(corners)
    da = _geom_signed_area(pts)
    if da == 0.0 or (sa > 0.0) != (da > 0.0):
        return False
    ratio = abs(da) / abs(sa)
    if ratio < 1e-3 or ratio > 1e3:
        return False
    return _geom_quad_convex(pts)


class LineGuideCorrectionWidget(KVFloatLayout):
    """ラインガイド補正Widget"""
    
    lines_tcg = KVListProperty([])  # [((x1,y1), (x2,y2)), ...]
    interpolation = KVStringProperty('bicubic')
    preview_scale = KVNumericProperty(0.25)
    
    def __init__(self, texture_size, param, **kwargs):
        super().__init__(**kwargs)
        
        self.texture_size = texture_size
        self.tcg_info = params.param_to_tcg_info(param)

        self.on_callback = None
        
        # 線描画用のオーバーレイWidget
        self.draw_overlay = KVWidget()
        self.add_widget(self.draw_overlay)
        
        # 状態変数
        self.selected_line_index = -1
        self.selected_point_index = -1 # 0: start, 1: end
        self.dragging = False
        self.start_new_line_point = None

        # タッチイベントをバインド
        self.draw_overlay.bind(
            on_touch_down=self._on_drawing_touch_down,
            on_touch_move=self._on_drawing_touch_move,
            on_touch_up=self._on_drawing_touch_up
        )

        # UIコントロール
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
        
        self.clear_btn = ScaledButton(text="Reset")
        self.clear_btn.set_ref_metrics()
        self.clear_btn.bind(on_press=self.clear_lines)
        button_layout.add_widget(self.clear_btn)

        self.apply_btn = ScaledButton(text="Apply")
        self.apply_btn.set_ref_metrics()
        self.apply_btn.bind(on_press=lambda x: self.apply_lines())
        button_layout.add_widget(self.apply_btn)

        self.revert_btn = ScaledButton(text="Revert")
        self.revert_btn.set_ref_metrics()
        self.revert_btn.bind(on_press=self.revert_lines)
        button_layout.add_widget(self.revert_btn)
        
        self.add_widget(button_layout)
        
        # プロパティの変更を監視
        self.bind(lines_tcg=self._redraw_lines) # lines_tcg変更時に再描画
        self.bind(size=self._redraw_lines)
        self.bind(pos=self._redraw_lines)

        self._redraw_lines()
        kvutils.install_ref_scaling(self)

    def set_callback(self, callback):
        self.on_callback = callback

    def set_texture_size(self, texture_size):
        self.texture_size = texture_size
        self._redraw_lines()

    def set_view_param(self, param):
        """表示中 preview と同じ座標系(tcg_info)へ再同期する (resize / cmd+F 後の表示リセット)。
        lines_tcg は編集状態なので保持し、view 変換だけ差し替える。"""
        self.tcg_info = params.param_to_tcg_info(param)
        self._redraw_lines()

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
        
    def apply_lines(self):
        """ラインを確定して適用する (画像への適用はここでのみ行う)。
        描画/削除/移動では適用せず、Apply ボタン経由のこの経路でのみ適用する。"""
        self.on_edit_start()
        self.on_edit_end()
        self._redraw_lines()

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

    def _hit_test_window(self, tcg_x, tcg_y, touch_pos):
        """ヒットテスト（ウィンドウ座標=表示円と同じ空間で判定）。
        端点の表示位置と touch 位置の距離を px で比較し、表示円に近い当たり判定にする。"""
        wx, wy = self._get_window_pos(tcg_x, tcg_y)
        return np.hypot(wx - touch_pos[0], wy - touch_pos[1])

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
        
        # 既存ラインの端点ヒットテスト (ウィンドウ座標=表示円と同じ空間で px 判定)
        hit_line = -1
        hit_point = -1
        min_dist = _POINT_HIT_RADIUS_PX

        for i, line in enumerate(self.lines_tcg):
            for pt_idx, p in ((0, line[0]), (1, line[1])):
                d = self._hit_test_window(p[0], p[1], touch.pos)
                if d < min_dist:
                    min_dist = d
                    hit_line = i
                    hit_point = pt_idx
        
        is_right = getattr(touch, 'button', 'left') == 'right'

        if hit_line >= 0:
            if is_right:
                # 右クリックでラインを削除 (適用はしない。Apply で適用)
                self.lines_tcg.pop(hit_line)
                self.selected_line_index = -1
                self.selected_point_index = -1
                self._redraw_lines()
                return True
            # 既存ポイントのドラッグ開始
            self.selected_line_index = hit_line
            self.selected_point_index = hit_point
            self.dragging = True
            self._redraw_lines()
        elif is_right:
            # 右クリックは何もしない（新規ライン作成はしない）
            return True
        elif len(self.lines_tcg) >= _MAX_LINES:
            # 上限 (4本) に達している場合は新規ライン作成を開始しない。
            # (5本以上作れても calculate_lines_homography が最後の4本しか使わず、
            # 見えている線が実際には無視されてしまうため)
            self.start_new_line_point = None
            self.selected_line_index = -1
            self.selected_point_index = -1
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
            # 新規ライン確定 (適用はしない。ガイド線を追加するだけで、
            # 実際の画像への適用は Apply ボタンでのみ行う)
            # 一定以上長さがある場合のみ
            dist = np.sqrt((self.start_new_line_point[0]-tx)**2 + (self.start_new_line_point[1]-ty)**2)
            if dist > 0.01 and len(self.lines_tcg) < _MAX_LINES:
                self.lines_tcg.append((self.start_new_line_point, (tx, ty)))
                self.selected_line_index = len(self.lines_tcg) - 1 # 選択状態にする
                self.selected_point_index = -1 # ポイント選択解除

            self.start_new_line_point = None
            self._redraw_lines()

        elif self.dragging:
            # 移動終了、確定 (適用はしない。Apply で適用)
            self.dragging = False


        return True

    def _lines_are_safe(self):
        """現在の参照線から実際にホモグラフィを計算し、表示破綻しない配置かを判定する。
        2本未満や有効な H が得られない場合は「破綻ではない」(=赤くしない) 扱い。"""
        lines = list(self.lines_tcg)
        if len(lines) < 2:
            return True
        try:
            info = dict(self.tcg_info) if isinstance(self.tcg_info, dict) else {}
            # ライン用 tcg_info は回転/反転を除いて評価する (effects._line_homography_tcg_info と同じ)
            info['rotation'] = 0.0
            info['rotation2'] = 0.0
            info['flip_mode'] = 0
            orig = info.get('original_img_size')
            size = int(max(orig)) if orig else 1000
            H = calculate_lines_homography(lines, size, size, tcg_info=info)
        except Exception:
            return True  # 計算失敗時は過剰警告を避けて赤くしない
        if H is None:
            return True  # 有効な補正が出ない (本数/対応点不足)。破綻ではない
        return _homography_is_safe(H, size)

    def _redraw_lines(self, instance=None, value=None, temp_line=None):
        """すべての線を再描画"""
        self.draw_overlay.canvas.clear()

        LINE_WIDTH = 1.5
        POINT_SIZE = _POINT_SIZE
        COLOR_NORMAL = (0.9, 0.9, 0.9, 0.8)
        COLOR_SELECTED = (0.2, 0.6, 1.0, 1.0)  # 青 (アンバー警告色と紛れないように)

        # 破綻しそうな配置なら全ラインを赤で描く (リアルタイム警告)
        safe = self._lines_are_safe()

        with self.draw_overlay.canvas:
            # 確定済みライン
            for i, line in enumerate(self.lines_tcg):
                p1 = line[0]
                p2 = line[1]

                wx1, wy1 = self._get_window_pos(p1[0], p1[1])
                wx2, wy2 = self._get_window_pos(p2[0], p2[1])

                if not safe:
                    KVColor(*_LINE_COLOR_UNSAFE)
                elif i == self.selected_line_index:
                    KVColor(*COLOR_SELECTED)
                else:
                    KVColor(*COLOR_NORMAL)

                KVLine(points=[wx1, wy1, wx2, wy2], width=LINE_WIDTH)

                # 端点（コントロールポイント）
                if not safe:
                    KVColor(*_LINE_COLOR_UNSAFE)
                elif i == self.selected_line_index and self.selected_point_index == 0:
                    KVColor(1, 1, 1, 1)
                else:
                    KVColor(*COLOR_NORMAL)
                KVLine(circle=(wx1, wy1, POINT_SIZE), width=1.2)

                if not safe:
                    KVColor(*_LINE_COLOR_UNSAFE)
                elif i == self.selected_line_index and self.selected_point_index == 1:
                    KVColor(1, 1, 1, 1)
                else:
                    KVColor(*COLOR_NORMAL)
                KVLine(circle=(wx2, wy2, POINT_SIZE), width=1.2)

            # 作成中のライン
            if temp_line:
                p1 = temp_line[0]
                p2 = temp_line[1]
                # temp_lineはTCG座標で来る想定
                wx1, wy1 = self._get_window_pos(p1[0], p1[1])
                wx2, wy2 = self._get_window_pos(p2[0], p2[1])
                
                KVColor(*COLOR_SELECTED)
                KVLine(points=[wx1, wy1, wx2, wy2], width=LINE_WIDTH, dash_length=5, dash_offset=5)
                 
    @kvmainthread
    def update_preview(self, *args):
        pass
    
    def get_correction_params(self) -> dict:
        """現在のパラメータを取得"""
        return {
            "reference_lines": list(self.lines_tcg),
        }
    
    def set_correction_params(self, param: dict):
        """パラメータを設定"""
        self.tcg_info = params.param_to_tcg_info(param)
        self.lines_tcg = param.get('reference_lines', [])
        #self._redraw_lines()

    def on_lines_change(self):
        pass
