
import numpy as np
import cv2

import params
from kivy.app import App as KVApp
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.clock import Clock as KVClock
from kivy.graphics.texture import Texture as KVTexture
from kivy.properties import ObjectProperty as KVObjectProperty, ListProperty as KVListProperty

from cores.lens_ghost import create_ghost, GHOST_PRESETS
# kv 内で ParamSlider / HoverSpinner を使うために import(Factory 登録)。
from widgets.param_slider import ParamSlider  # noqa: F401
from widgets.hover_spinner import HoverSpinner  # noqa: F401

IMG_W, IMG_H = 700, 500

# create_ghost に整数で渡す必要があるパラメータ(range() / default_rng() が float を拒否)。
INT_PARAMS = {'random_seed', 'base_radius', 'num_components', 'spike_density'}


class GhostEditor(KVBoxLayout):
    """ゴーストエディタ本体。パラメータUI(スライダー群)は ghosteditor.kv 側に集約し、
    本クラスは「マウス操作(光源の追加/選択/移動/削除)」と「レンダリング」に専念する。
    スライダーは kv で id='slider_<param名>' として宣言し、ここで収集して値を読み書きする。
    """

    image_widget = KVObjectProperty(None)
    param_container = KVObjectProperty(None)
    preset_names = KVListProperty([])

    GRAB_RADIUS = 22   # 画像px。CP掴み判定
    CP_RADIUS = 7      # CPマーカー半径

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preset_names = list(GHOST_PRESETS.keys())
        # 光源CPは正規化TCG座標で保持する(liquify と同方式)。回転/反転/クロップに追従する。
        # standalone は幾何変換なし(identity)。platypus 組み込み時は実 param(rotation/crop等)を入れる。
        self.param = {
            'original_img_size': (IMG_W, IMG_H),
            'rotation': 0.0, 'rotation2': 0.0, 'flip_mode': 0, 'matrix': np.eye(3),
        }
        self._refimg = np.empty((IMG_H, IMG_W, 3), dtype=np.float32)  # 形状参照用(中身は未使用)
        self.lights = [{'pos': self._px_to_tcg(460, 160), 'params': {}}]
        self.selected = 0
        self.sliders = {}          # param_name -> ParamSlider (kv の id から収集)
        self.update_event = None
        self._syncing = False
        self._dragging = False
        self._ready = False
        KVClock.schedule_once(self._init_from_kv, 0)

    # ---------- TCG 座標変換 ----------
    # 光源CPは正規化TCGで保持し、描画/判定の直前に作業画像のピクセルへ変換する。
    # 組み込み時はクリック→TCG を params.window_to_tcg に、tcg_info を実 param 由来に差し替える。
    def _tcg_info(self):
        return params.param_to_tcg_info(self.param)

    def _px_to_tcg(self, px, py):
        return params.ref_image_to_tcg(px, py, self._refimg, self._tcg_info())

    def _tcg_to_px(self, cx, cy):
        x, y = params.tcg_to_ref_image(cx, cy, self._refimg, self._tcg_info())
        return (int(round(x)), int(round(y)))

    # ---------- kv のスライダー収集 ----------
    def _init_from_kv(self, dt):
        for wid, widget in self.ids.items():
            if wid.startswith('slider_'):
                name = wid[len('slider_'):]
                widget.param_name = name
                self.sliders[name] = widget
        # 選択中の光源の params をスライダー既定値から初期化。
        self.lights[self.selected]['params'] = {n: s.value for n, s in self.sliders.items()}
        self._ready = True
        self.update_image()

    # ---------- 選択光源 ⇔ スライダー ----------
    def _selected_light(self):
        return self.lights[self.selected] if self.lights else None

    def _sync_sliders_from_light(self):
        light = self._selected_light()
        if light is None:
            return
        self._syncing = True
        for name, slider in self.sliders.items():
            if name in light['params']:
                slider.set_slider_value(light['params'][name])
        self._syncing = False

    def on_param_change(self, slider):
        if self._syncing or not self._ready:
            return
        name = getattr(slider, 'param_name', None)
        light = self._selected_light()
        if name is not None and light is not None:
            light['params'][name] = slider.value
        self._schedule_update()

    def on_preset_selected(self, preset_name):
        preset = GHOST_PRESETS.get(preset_name)
        light = self._selected_light()
        if not preset or light is None:
            return
        # プリセットは「選択中の光源」の params にのみ適用(座標等メタキーは無視)。
        for name in self.sliders:
            if name in preset:
                light['params'][name] = preset[name]
        self._sync_sliders_from_light()
        self.update_image()

    def on_randomize_seed(self):
        light = self._selected_light()
        if light is None or 'random_seed' not in self.sliders:
            return
        light['params']['random_seed'] = int(np.random.randint(0, 100000))
        self._sync_sliders_from_light()
        self.update_image()

    # ---------- 光源の追加/選択/移動/削除(マウス) ----------
    def on_touch_down(self, touch):
        iw = self.image_widget
        if iw is not None and iw.collide_point(*touch.pos):
            coords = self._touch_to_image_coords(touch)
            if coords is not None:
                right = getattr(touch, 'button', 'left') == 'right'
                idx = self._nearest_light(coords)
                if right:
                    if idx is not None:
                        self._delete_light(idx)
                elif idx is not None:
                    self._select_light(idx)
                    self._dragging = True
                else:
                    self._add_light(coords)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._dragging and self.image_widget is not None and self.lights:
            coords = self._touch_to_image_coords(touch)
            if coords is not None:
                self.lights[self.selected]['pos'] = self._px_to_tcg(*coords)
                self._schedule_update(0.03)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self._dragging:
            self._dragging = False
            return True
        return super().on_touch_up(touch)

    def _nearest_light(self, coords):
        # coords は画像ピクセル。各光源の TCG をピクセルへ戻して距離比較。
        cx, cy = coords
        best, best_d = None, self.GRAB_RADIUS ** 2
        for i, lt in enumerate(self.lights):
            x, y = self._tcg_to_px(*lt['pos'])
            d = (x - cx) ** 2 + (y - cy) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def _select_light(self, idx):
        if idx != self.selected:
            self.selected = idx
            self._sync_sliders_from_light()
        self.update_image()

    def _add_light(self, coords):
        base = self._selected_light()
        p = dict(base['params']) if base else {n: s.value for n, s in self.sliders.items()}
        self.lights.append({'pos': self._px_to_tcg(*coords), 'params': p})
        self.selected = len(self.lights) - 1
        self._sync_sliders_from_light()
        self.update_image()

    def _delete_light(self, idx):
        if len(self.lights) <= 1:
            return
        del self.lights[idx]
        self.selected = min(self.selected, len(self.lights) - 1)
        self._sync_sliders_from_light()
        self.update_image()

    def _touch_to_image_coords(self, touch):
        iw = self.image_widget
        tex_w, tex_h = iw.norm_image_size
        if tex_w <= 0 or tex_h <= 0:
            return None
        off_x = iw.x + (iw.width - tex_w) / 2.0
        off_y = iw.y + (iw.height - tex_h) / 2.0
        lx = touch.x - off_x
        ly = touch.y - off_y
        if lx < 0 or lx > tex_w or ly < 0 or ly > tex_h:
            return None
        ix = int(lx / tex_w * IMG_W)
        iy = int((1.0 - ly / tex_h) * IMG_H)  # Kivy y上向き＋表示反転済み → numpy 行
        return (max(0, min(IMG_W - 1, ix)), max(0, min(IMG_H - 1, iy)))

    # ---------- 描画 ----------
    def _schedule_update(self, delay=0.1):
        if self.update_event:
            self.update_event.cancel()
        self.update_event = KVClock.schedule_once(lambda dt: self.update_image(), delay)

    def _coerce(self, params):
        p = dict(params)
        for k in INT_PARAMS:
            if k in p:
                p[k] = int(round(p[k]))
        return p

    def update_image(self):
        if self.image_widget is None:
            return
        acc = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
        px_list = [self._tcg_to_px(*lt['pos']) for lt in self.lights]  # TCG → 作業画像ピクセル
        for lt, px in zip(self.lights, px_list):
            if lt['params']:
                acc += create_ghost(np.zeros((IMG_H, IMG_W, 3), np.float32), [px], **self._coerce(lt['params']))
        acc = np.clip(acc, 0.0, 1.0)

        buf = (acc * 255).astype(np.uint8)
        for i, px in enumerate(px_list):
            color = (80, 160, 255) if i == self.selected else (255, 255, 255)
            cv2.circle(buf, (int(px[0]), int(px[1])), self.CP_RADIUS, color, -1)
            if i == self.selected:
                cv2.circle(buf, (int(px[0]), int(px[1])), self.CP_RADIUS + 3, (80, 160, 255), 1)

        buf = cv2.flip(buf, 0)
        texture = KVTexture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image_widget.texture = texture


class GhostEditorApp(KVApp):
    def build(self):
        return GhostEditor()


if __name__ == '__main__':
    GhostEditorApp().run()
