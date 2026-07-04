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
import os
import logging

from cores.distortion_correction.four_point_correction import correct_four_points, detect_rectangle, calculate_four_point_homography
import params
from utils import kvutils
from widgets.scaled_button import ScaledButton

_DEBUG_4PT = os.getenv("PLATYPUS_DEBUG_4PT", "0").strip().lower() in {"1", "true", "yes", "on"}

# CP / 接続線の色 (3状態):
#   通常(青)    : 問題なし
#   クランプ(アンバー): 四角形は健全だが透視が強く、適用時に安全上限まで減衰される (=それっぽく効く)
#   破綻(赤)    : 四角形自体が壊れている (非凸/自己交差/向き反転/退化)。減衰しても意味をなさない
_CP_COLOR_NORMAL = (0.2, 0.6, 1.0, 0.8)    # 青
_CP_COLOR_CLAMPED = (1.0, 0.65, 0.1, 0.9)  # アンバー
_CP_COLOR_UNSAFE = (1.0, 0.2, 0.2, 0.9)    # 赤
_LINE_COLOR_NORMAL = (1.0, 1.0, 1.0, 1.0)
_LINE_COLOR_CLAMPED = (1.0, 0.65, 0.1, 1.0)
_LINE_COLOR_UNSAFE = (1.0, 0.25, 0.25, 1.0)

def _dbg4pt(msg, *args):
    if _DEBUG_4PT:
        logging.warning("[4PT] " + msg, *args)

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
        self._last_state = 'ok'  # 直近の安全判定 ('ok'/'clamped'/'broken', 接続線の色に使用)
        #self.bind(corner_positions_tcg=self._on_corners_change)
        self.bind(size=self._sync_tcg_to_kivy, pos=self._sync_tcg_to_kivy)

        self._reset_corners()
        kvutils.install_ref_scaling(self)

    def on_touch_down(self, touch):
        """CP(ハンドル)/ボタン以外への touch は、下層の CropEditor 等へ伝播させず
        ここで消費する。伝播を許すと Ge タブでこのエディタが開いている間、CP 以外の
        場所をドラッグすると裏の CropEditor が動いてしまう (クロップ枠のドラッグ)。"""
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos):
            return True
        return False

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

    def set_view_param(self, param):
        """表示中 preview と同じ座標系(tcg_info)へ再同期する (resize / cmd+F 後の表示リセット)。
        corner_positions_tcg は編集状態なので保持し、view 変換だけ差し替える。"""
        self.tcg_info = params.param_to_tcg_info(param)
        self._sync_tcg_to_kivy()

    def on_edit_start(self):
        """編集開始イベント（ヒストリー管理用）"""
        if self.on_callback:
            _dbg4pt("-> callback('start')")
            self.on_callback('start', self)

    def on_edit_end(self):
        """編集終了イベント（ヒストリー管理用）"""
        if self.on_callback:
            _dbg4pt("-> callback('end')")
            self.on_callback('end', self)

    def on_apply_corners(self):
        """コーナー適用イベント"""
        if self.on_callback:
            _dbg4pt("-> callback('apply')")
            self.on_callback('apply', self)

    def reset_corners(self, *args):
        _dbg4pt("reset_corners (Reset button) pressed")
        self._reset_corners()
        self._commit_and_apply()

    def _reset_corners(self):
        # 隅を初期(画像四隅)に戻す。適用はしない (ハンドル/オーバーレイのみ更新)。
        self._using_default_corners = True
        self.corner_positions_tcg = self._default_corners()
        self._sync_tcg_to_kivy()

    def _commit_and_apply(self):
        """現在の隅を param へ確定し、画像へ適用する (履歴付き)。
        ドラッグ中は適用せず、Apply / Reset / Revert ボタン経由でのみここを通す。"""
        _dbg4pt("_commit_and_apply corners=%s flag=%s",
                list(self.corner_positions_tcg), self._using_default_corners)
        self.on_edit_start()
        self.on_edit_end()

    def _apply_corners(self, *args):
        # Apply ボタン: 現在のハンドル位置を確定して適用する。
        self._commit_and_apply()

    def _revert_corners(self, *args):
        _dbg4pt("_revert_corners (Revert button) pressed corners=%s", list(self.corner_positions_tcg))
        # 補正なし(未適用)状態をプレビューしつつ、ハンドル位置は維持する。
        # (Apply を押せば維持した位置で再適用できる)
        backup = list(self.corner_positions_tcg)
        backup_flag = self._using_default_corners
        self._reset_corners()       # 隅をデフォルトに (適用なし)
        self._commit_and_apply()    # identity を適用 (未補正プレビュー)
        # 内部状態(ハンドル位置と flag)を復元。画像/param は未補正のまま。
        # flag を復元しないと、次回ドラッグ時に他の隅がデフォルトへ飛ぶ等の不整合になる。
        self.corner_positions_tcg = backup
        self._using_default_corners = backup_flag
        self._sync_tcg_to_kivy()

    def get_correction_params(self) -> dict:
        """
        現在のパラメータを取得（TCG座標系）

        Returns:
            dict: {"four_points": [(x,y), ...]}
        """
        # 隅が画像四隅 (デフォルト) のときのみ「未設定」として [] を返す。
        # _using_default_corners フラグではなく実際の隅座標で判定することで、
        # ドラッグ後に再描画 (set_correction_params) 経由でフラグが同期ずれしても、
        # Apply が [] を返して補正がリセットされてしまう不具合を防ぐ。
        is_default = list(self.corner_positions_tcg) == self._default_corners()
        result = {"four_points": []} if is_default else {"four_points": list(self.corner_positions_tcg)}
        _dbg4pt("get_correction_params -> %s (corners=%s flag=%s is_default=%s)",
                result['four_points'], list(self.corner_positions_tcg),
                self._using_default_corners, is_default)
        return result

    def set_correction_params(self, param: dict):
        """
        パラメータを設定（TCG座標系）

        Args:
            param: dict、{"four_points": [(x,y), ...]}
        """
        four_points = param.get('four_points', [])
        _dbg4pt("set_correction_params incoming four_points=%s", four_points)
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

            # 破綻判定に応じて CP 色を更新 (青/赤) してから接続線を描画
            self._apply_safety_colors()
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
        
        # 円形を描画 (色は破綻判定でリアルタイムに変える)
        with handle.canvas:
            #KVPushMatrix()
            color = KVColor(*_CP_COLOR_NORMAL)  # 青色
            KVEllipse(pos=(0, 0), size=(40, 40))
            #KVPopMatrix()
        handle._cp_color = color

        # 位置を設定
        handle.center = pos

        # ドラッグイベントをバインド
        handle.bind(pos=lambda inst, touch: self._on_handle_move(index))
        handle.bind(on_touch_down=lambda inst, touch: self._on_handle_touch_down(index, inst, touch))
        handle.bind(on_touch_up=lambda inst, touch: self._on_handle_release(index, touch))

        return handle
    
    def _on_handle_move(self, index: int):
        """ハンドルがドラッグされた。
        ここでは画像へ適用せず、内部座標とオーバーレイ(ハンドル/接続線)のみ更新する。
        実際の画像への適用は Apply ボタン (_commit_and_apply) でのみ行う。"""
        if self.updating_handles:
            return

        self.is_dragging = True
        preserve_default_corners = self._using_default_corners

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
        """ハンドルが離された。ドラッグでは適用しないので、状態を戻すだけ。
        grab_current は必ず -1 に戻す (この bind はボタン等の無関係な touch_up でも
        発火するため、残っていると誤動作の原因になる)。"""
        if self.grab_current == index:
            self.is_dragging = False
            self.grab_current = -1

    def _on_handle_touch_down(self, index, instance, touch):
        """ハンドル上の touch_down。右クリックならその CP だけデフォルト(画像四隅)へ
        リセットし、Scatter のドラッグを止める (True を返して既定処理を抑止)。"""
        if getattr(touch, 'button', 'left') == 'right' and instance.collide_point(*touch.pos):
            self._reset_single_corner(index)
            return True
        return False

    def _reset_single_corner(self, index):
        """CP を1つだけデフォルト位置(画像四隅)へ戻す。適用はしない (Apply で適用)。"""
        default = self._default_corners()
        corners = list(self.corner_positions_tcg)
        if index < 0 or index >= len(default) or index >= len(corners):
            return
        corners[index] = default[index]
        self.corner_positions_tcg = corners
        # 全 CP がデフォルトに戻ったら未設定扱いに戻す
        if list(corners) == default:
            self._using_default_corners = True
        _dbg4pt("_reset_single_corner index=%s corners=%s", index, corners)
        self._sync_tcg_to_kivy()

    # --- 破綻(非凸/自己交差/向き反転)の局所判定とリアルタイム着色 ---
    @staticmethod
    def _signed_area(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    @staticmethod
    def _quad_is_convex(pts):
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

    @classmethod
    def _homography_is_safe(cls, H, size):
        """ホモグラフィ H を [0,size] 画像四隅に適用し、表示破綻 (透視分母 w の符号反転
        =地平線越え / 向き反転 / 退化 / 非凸) しないか判定する。effects 側の Apply ガードと
        同じ基準 (凸な四角形でも極端な透視だと画像四隅が破綻するのを検出する)。"""
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
        sa = cls._signed_area(corners)
        da = cls._signed_area(pts)
        if da == 0.0 or (sa > 0.0) != (da > 0.0):
            return False
        ratio = abs(da) / abs(sa)
        if ratio < 1e-3 or ratio > 1e3:
            return False
        return cls._quad_is_convex(pts)

    def _fourpoint_homography_safe(self, corners):
        """現在の CP から実際の4点ホモグラフィを計算し、画像四隅を破綻させないか判定する。
        Apply 側 (_update_matrix_param / make_diff) と同じ計算・基準なので、
        「赤くならないのに Apply で変化しない (=ガードで捨てられる)」ズレを無くす。"""
        reset = self._default_corners()
        if list(corners) == reset:
            return True  # 恒等 (補正なし)
        info = self.tcg_info if isinstance(self.tcg_info, dict) else None
        if not info:
            return True
        orig = info.get('original_img_size')
        size = int(max(orig)) if orig else 1000

        class _Dummy:
            def __init__(self, n):
                self.shape = (n, n, 3)

        dummy = _Dummy(size)
        try:
            src = [params.tcg_to_ref_image(cx, cy, dummy, info) for cx, cy in corners]
            dst = [params.tcg_to_ref_image(cx, cy, dummy, info) for cx, cy in reset]
            H_inv = calculate_four_point_homography(src, dst)
            H = np.linalg.inv(H_inv)
        except Exception:
            return False  # 計算不能 = Apply でも捨てられる → 破綻扱い(赤)
        return self._homography_is_safe(H, size)

    def _quad_geometry_ok(self, pts):
        """四角形そのものが健全 (凸 & 向き保持 & 非退化) かを判定。"""
        if not np.all(np.isfinite(pts)):
            return False
        area = self._signed_area(pts)
        if abs(area) < 1e-9:
            return False  # 退化 (潰れ)
        ref = self._signed_area(np.asarray(self._default_corners(), dtype=np.float64))
        if (area > 0.0) != (ref > 0.0):
            return False  # 向き反転 (裏表逆)
        return self._quad_is_convex(pts)  # 自己交差(bowtie)でない

    def _safety_state(self, corners):
        """CP 配置を 'ok' / 'clamped' / 'broken' の3状態で返す。
          - broken : 四角形自体が壊れている (非凸/自己交差/向き反転/退化)
          - clamped: 四角形は健全だが透視が強く、適用時に安全上限まで減衰される
          - ok     : 問題なし"""
        if corners is None or len(corners) != 4:
            return 'ok'
        pts = np.asarray(corners, dtype=np.float64)
        if not self._quad_geometry_ok(pts):
            return 'broken'
        if not self._fourpoint_homography_safe(corners):
            return 'clamped'
        return 'ok'

    def _apply_safety_colors(self):
        """現在の CP 配置の安全性 (3状態) を判定し、ハンドル色 (と接続線色) を更新する。"""
        state = self._safety_state(list(self.corner_positions_tcg))
        self._last_state = state
        color = {
            'ok': _CP_COLOR_NORMAL,
            'clamped': _CP_COLOR_CLAMPED,
            'broken': _CP_COLOR_UNSAFE,
        }[state]
        for handle in self.handles:
            col = getattr(handle, '_cp_color', None)
            if col is not None:
                col.rgba = color

    def _update_lines(self):
        """接続線を再描画"""
        self.canvas.after.clear()
        
        if len(self.handles) != 4:
            return
        
        with self.canvas.after:
            # 状態に応じて接続線の色も変える (通常=白 / クランプ=アンバー / 破綻=赤)
            _line_color = {
                'ok': _LINE_COLOR_NORMAL,
                'clamped': _LINE_COLOR_CLAMPED,
                'broken': _LINE_COLOR_UNSAFE,
            }.get(getattr(self, '_last_state', 'ok'), _LINE_COLOR_NORMAL)
            KVColor(*_line_color)

            # 4点を結ぶ線
            points = []
            for handle in self.handles:
                points.extend([handle.center_x, handle.center_y])
            
            # 最初の点に戻る
            points.extend([self.handles[0].center_x, self.handles[0].center_y])
            
            KVLine(points=points, width=2)
