
#from splashscreen import display_splash_screen, close_splash_screen
#display_splash_screen("assets/platypus.png")

if __name__ == '__main__':
    import sys as _sys_early
    import os as _os_early
    import multiprocessing as _mp_early
    # PyInstaller: kv/json 等は sys._MEIPASS 配下に同梱される
    if getattr(_sys_early, "frozen", False) and hasattr(_sys_early, "_MEIPASS"):
        _os_early.chdir(_sys_early._MEIPASS)
    # OpenMP ランタイムの情報メッセージを抑える（llvm-openmp / 混在時）
    _os_early.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
    _os_early.environ.setdefault("KMP_WARNINGS", "0")
    _os_early.environ.setdefault("LIBOMP_VERBOSE", "0")
    _os_early.environ.setdefault("KIVY_NO_ARGS", "1")  # 子プロセスでの Kivy 引数誤解釈を防ぐ
    # Finder から起動した .app は PATH が /usr/bin:/bin:/usr/sbin:/sbin に絞られ、
    # /usr/local/bin (公式 ExifTool) や /opt/homebrew/bin (Homebrew) が含まれない
    if _sys_early.platform == "darwin":
        _extra_paths = [p for p in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin")
                        if _os_early.path.isdir(p)]
        _path_parts = _os_early.environ.get("PATH", "").split(_os_early.pathsep)
        _missing_paths = [p for p in _extra_paths if p not in _path_parts]
        if _missing_paths:
            _os_early.environ["PATH"] = _os_early.pathsep.join(_missing_paths + _path_parts)
    _mp_early.freeze_support()  # frozen 実行時の multiprocessing 子プロセス分岐を早期処理

    try:
        #import matplotlib
        #matplotlib.use('Agg', force=True)  # matplotlib読み込み前にAgg固定
        #matplotlib.interactive(False)
        pass
    except ImportError:
        pass

    # PyInstaller + Kivy 既定フックでは tkinter が除外される。バンドル実行時はスキップ。
    if not getattr(_sys_early, "frozen", False):
        import tkinter as tk

        # tk.Tk()で落ちるのを回避するためのパッチ
        tk = tk.Tk()
        tk.withdraw()
        tk.destroy()

    from kivy.config import Config
    Config.set('input', 'mouse', 'mouse,disable_multitouch')  # 右クリック赤丸消去
    Config.set('kivy', 'exit_on_escape', '0')  # kivy ESC無効
    Config.set('kivy', 'kivy_clock', 'interrupt')

    from kivymd.app import MDApp
    from kivymd.uix.boxlayout import MDBoxLayout
    from kivy.core.window import Window as KVWindow
    from kivy.graphics.texture import Texture as KVTexture
    from kivy.properties import (
        BooleanProperty as KVBooleanProperty,
        ListProperty as KVListProperty,
        NumericProperty as KVNumericProperty,
    )
    from kivy.clock import Clock as KVClock, mainthread as kvmainthread
    from kivy.graphics.transformation import Matrix as KVMatrix
    from kivy.uix.label import Label as KVLabel
    from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
    from kivy.uix.button import Button as KVButton
    from kivy.uix.popup import Popup as KVPopup
    from kivy.uix.textinput import TextInput as KVTextInput

    import threading
    import threads
    import os

    import cores.colour_functions as colour_functions
    import re
    import time
    import multiprocessing
    import math
    import logging
    # ログレベルの設定
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pyvips").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    #logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)

    import define
    import cores.core as core
    import params
    from enums import ImageFidelity, LoadStage, coerce_load_stage
    from image_fidelity import pipeline_loading_flag
    import effects
    import pipeline
    import utils.utils as utils
    import utils.kvutils as kvutils
    import utils.dialogutils as dialogutils
    from utils import perf_trace
    from utils import preset_utils
    from utils import rating_utils
    from utils import rating_io
    import macos as device

    import cores.film_emulator as film_emulator
    from cores.coating_simulator import CoatingSimulator
    import config
    import export
    import processing_dialog
    from processing_dialog import create_processing_dialog
    from async_worker import AsyncWorker
    import waitinfo
    import history

    import widgets.metainfo
    import widgets.float_input
    import widgets.param_slider
    import widgets.color_picker
    import widgets.hover_spinner
    import widgets.histogram
    import widgets.viewer
    import widgets.curve
    import widgets.bbox_viewer
    import widgets.mask_editor2
    import widgets.history_content as history_content
    import widgets.mask2_content as mask2_content
    import widgets.preset_content as preset_content
    from widgets.effect_selector import EffectSelector
    from widgets.export_dialog import ExportDialog, ExportConfirmDialog
    import widgets.collapsible_box
    import widgets.compact_switch
    import widgets.modern_checkbox
    from widgets.switch_reset_map import build_switch_reset_targets

if __name__ != '__main__':
    class ImportBlocker:
        """特定のモジュールのインポートをブロックする"""
        def __init__(self, blocked_modules):
            self.blocked_modules = set(blocked_modules)
        
        def find_module(self, fullname, path=None):
            # ブロック対象のモジュールかチェック
            for blocked in self.blocked_modules:
                if fullname == blocked or fullname.startswith(blocked + '.'):
                    return self
            return None
        
        def load_module(self, fullname):
            raise ImportError(f"Module '{fullname}' is blocked in child process")

    def init_worker():
        """子プロセス初期化時に実行される"""
        import sys
        import multiprocessing
        # インポートフックを設定
        blocker = ImportBlocker(['kivy', 'kivymd', 'matplotlib', 'tkinter'])
        sys.meta_path.insert(0, blocker)
        print(f"子プロセス {multiprocessing.current_process().name}: インポートブロッカー設定完了")
    
    #init_worker()

import os
import numpy as np
import cv2

import file_cache_system

# OpenCVの設定
cv2.ocl.setUseOpenCL(True)
cv2.setUseOptimized(True)

if __name__ == '__main__':

    def pillow_init():
        import PIL.Image as PILImage
        import PIL.Jpeg2KImagePlugin
        import PIL.JpegImagePlugin
        import PIL.PngImagePlugin
        import PIL.TiffImagePlugin
        import PIL.GifImagePlugin
        import PIL.BmpImagePlugin
        PILImage._initialized = 2
        PILImage.init()

    def _load_stage_allows_ui(stage, imgset):
        """
        プレビュー（RAW 埋め込み）または単発 RGB が表示可能になった段階でのみ True。
        RAW フルデコード完了は fidelity が FULL のときのみ True（それ以外はローディング継続）。
        """
        if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE):
            return True
        if stage == LoadStage.FULL_DECODE and getattr(imgset, 'fidelity', None) == ImageFidelity.FULL:
            return True
        return False

    def _load_stage_ends_file_loading_indicator(stage, imgset):
        """
        ファイル読み込み用インジケータ（_actively_loading / is_processing）を消す段階。
        プレビュー到達でパラメータを触れる loading 解除とは分離し、RAW はフルデコード完了まで
        スピナーを表示し続ける。
        """
        if stage == LoadStage.FULL_DECODE:
            return True
        if stage == LoadStage.RGB_DONE:
            return True
        if stage == LoadStage.FIRST_PAINTABLE and getattr(imgset, 'fidelity', None) == ImageFidelity.FULL:
            return True
        return False

    class GeModePanel(MDBoxLayout):
        visible = KVBooleanProperty(True)

        def on_touch_down(self, touch):
            if not self.visible:
                return False
            return super().on_touch_down(touch)

        def on_touch_move(self, touch):
            if not self.visible:
                return False
            return super().on_touch_move(touch)

        def on_touch_up(self, touch):
            if not self.visible:
                return False
            return super().on_touch_up(touch)

    class MainWidget(MDBoxLayout):
        # === 読み込み状態フラグの役割分担 ===
        # loading:              on_select 直後 〜 最初のピクセル表示可能 (FIRST_PAINTABLE/RGB_DONE)
        #                       Select/Export 等の「ロード中は触らせたくない」UI を無効化する短期フラグ。
        # image_loaded:         imgset が確定して編集可能な状態か (起動時/フォルダ切替時=False)。
        #                       全パラメータ UI の gate。loading とは独立で、loading=False でも
        #                       image_loaded=False の局面が存在する（起動直後・フォルダ切替直後）。
        # mask2_wait_full_load: RAW プレビュー段階では True、フル復号完了で False。Mask2 専用 gate。
        # _actively_loading:    ファイル選択開始 〜 フル復号完了。スピナーアニメ用 is_processing 算出の入力。
        # is_processing:        _actively_loading or 非同期タスク有り（スピナー表示）。
        # === 内部状態（UI 非連動） ===
        # _last_image_fidelity: 直前 fid。PREVIEW→FULL 遷移時の pmck heavy merge 判定。
        # _expected_file_path:  期待しているファイル。遅延 FCS コールバックを破棄するため。
        loading = KVBooleanProperty(False)
        mask2_wait_full_load = KVBooleanProperty(True)
        preview_size = KVListProperty([100, 100])
        is_processing = KVBooleanProperty(False)
        export_in_progress = KVBooleanProperty(False)
        export_done = KVNumericProperty(0)
        export_total = KVNumericProperty(0)
        image_loaded = KVBooleanProperty(False)

        def __init__(self, cache_system, **kwargs):
            super(MainWidget, self).__init__(**kwargs)

            self.texture = None
            self.imgset = None
            self.click_x = 0
            self.click_y = 0        
            self.crop_image = None
            self.crop_image_view_key = None
            self.is_zoomed = False
            self.drag_center_start = None
            self.is_press_space = False

            self.primary_param = {}
            self.primary_effects = effects.create_effects(
                lens_modifier_callback=self.lens_modifier_callback,
                distortion_callback=self.distortion_callback,
                geometry_callback=self.geometry_callback,
                crop_callback=self.crop_callback)
            #self.primary_effects[0]['crop'].set_editing_callback(self.crop_editing)
            self.inpaint_edit = None
            self.patchmatch_inpaint_edit = None
            self.cache_system = cache_system
            self.ids['viewer'].set_cache_system(self.cache_system)
            self._rgb_xmp_rating_had = {}

            self.async_worker = AsyncWorker()
            # self.async_worker.start() # Start explicitly after config init
            self.processor = pipeline.AsyncPipelineManager(self.async_worker)
            KVClock.schedule_interval(self.update_async_results, 0.1)
            self.pipeline_version = 0
            
            self.apply_draw_image_center = None
            self.apply_draw_fast_display = False
            self.apply_draw_skip_histogram = False
            self._fast_display_transform_cache = {}
            self.draw_event = threading.Event()
            self.apply_thread = threading.Thread(target=self.draw_image, daemon=False)
            self.apply_thread.start()
            self.enabledelay = None
            self._actively_loading = False  # ファイル選択によるロード中フラグ（起動時のloading: Trueとは別管理）

            self.history = history.History()
            self.current_op = None
            self._copied_effect_param = None

            self.run_set2widget_all = False

            self.is_press_space = False
            # on_select で選んだパス。FCS の遅延コールバックが別ファイル向けなら無視する（primary_param と imgset の不整合防止）
            self._expected_file_path = None

            self._export_cancel_event = threading.Event()
            self._export_thread = None
            self._clamping_preview_window = False
            self._preview_min_w = 0
            self._preview_min_h = 0
            self._debug_resize_label = None
            preset_utils.ensure_preset_dir()

            KVWindow.bind(on_key_down=self.on_key_down)
            KVWindow.bind(on_key_up=self.on_key_up)
            KVClock.schedule_once(lambda _dt: self.update_mask2_options_enabled(), 0)

        def on_start(self, *args, **kwargs):
            #self.ids['preview_widget'].ref_size_hint_min = (config.get_config("preview_width"), config.get_config("preview_height"))
            #self.ids['preview_widget'].ref_size_hint_max = (config.get_config("preview_width") * 1.1, config.get_config("preview_height") * 1.1)
            pass

        def update_async_results(self, dt):
            if self.async_worker:
                results = self.async_worker.poll_results()
                dirty = False
                for task_id, result_image, error_msg in results:
                    if error_msg:
                        logging.error(f"Async Task {task_id} failed: {error_msg}")
                    elif result_image is not None:
                        # Update cache in manager
                        key = self.processor.update_result(task_id, result_image)
                        if key:
                            dirty = True
                            logging.info(f"Async Task {task_id} ({key}) completed.")
                
                if dirty:
                    # Trigger redraw
                    # We need to make sure we don't spam redraws?
                    self.start_draw_image()

            if self.async_worker:
                for msg in self.async_worker.poll_messages():
                    if msg['type'] == 'waitinfo':
                        waitinfo.set_text(msg['tag'], msg['text'], self)
            
            # 処理状態の更新
            if self.async_worker:
                has_tasks = self.async_worker.has_pending_tasks()
                queue_empty = self.async_worker.input_queue.empty()
                active_count = len(self.async_worker.active_shms)
                should_processing = has_tasks or self._actively_loading
                if self.is_processing != should_processing:
                    logging.info(f"is_processing changed: {self.is_processing} -> {should_processing} (queue_empty={queue_empty}, active_shms={active_count}, actively_loading={self._actively_loading})")
                    self.is_processing = should_processing

        def get_preview_window_minimum_size(self):
            """
            返す min_w, min_h は Kivy 窓（通常は macOS 論理 pt 系）向け。

            m は ref*dpi（バッキング幅）のまま。窓の最小は min(w, h)/dpi_scale と同じ m_log = m/dpi
            を make し、0.55 列・上段+bar+viewer(ref) の論理高で揃える。ceil(m/0.55) を
            バッキングのまま使うと min・cap が画面いっぱいに張り付きがち（Retina）。
            """
            m = int(kvutils.preview_min_edge_for_window(config.get_preview_min_size()))
            if m < 1:
                return 0, 0, m
            dps = float(device.dpi_scale())
            if dps < 0.01:
                dps = 1.0
            m_log = m / dps
            col_frac = 0.55
            min_w = int(math.ceil(m_log / col_frac))
            # main.kv: プレビュー下 bar ref 30, 下段 viewer ref 160（論理 ref）
            min_h = int(math.ceil(m_log + 30.0 + 160.0))
            return min_w, min_h, m

        def _clamp_window_to_preview_minimum(self):
            """SDL の minimum_* が効かない環境向け: 手動で窓を既定最小以上に戻す。"""
            if self._clamping_preview_window:
                return
            min_w, min_h = self._preview_min_w, self._preview_min_h
            if min_w < 1 or min_h < 1:
                return
            w, h = KVWindow.size
            if w + 0.5 >= min_w and h + 0.5 >= min_h:
                return
            self._clamping_preview_window = True
            try:
                KVWindow.size = (max(int(w), min_w), max(int(h), min_h))
            finally:
                self._clamping_preview_window = False

        def sync_preview_widget_min_size(self):
            """
            m0=ref*dpi: kvutils.preview_min_edge_for_window（枠は NSScreen ポイント * dpi して m0 と同系で比較）。

            minimum_* / cap の sw, sh は get_window_screen_size() のポイント。min_w, min_h は
            m/dpi（論理）基準（get_preview_window_minimum_size）で、Retina ではバッキングのまま
            ceil(m/0.55) を使うと cap が画面いっぱいに張り付きがちになるのを避ける。
            """
            min_w, min_h, m = self.get_preview_window_minimum_size()
            self._preview_min_w, self._preview_min_h = min_w, min_h
            pw = self.ids.get("preview_widget")
            if pw is not None:
                pw.size_hint_min = (m, m)
            if min_w > 0 and min_h > 0:
                sw, sh = kvutils.get_window_screen_size()
                KVWindow.minimum_width = min(min_w, max(1, int(sw * 0.99)))
                KVWindow.minimum_height = min(min_h, max(1, int(sh * 0.99)))

        def _update_resize_debug_display(self):
            if not getattr(define, "RESIZE_DEBUG", False):
                return
            try:
                ww, wh = int(KVWindow.size[0]), int(KVWindow.size[1])
            except Exception:
                ww, wh = 0, 0
            try:
                wkmin = int(getattr(KVWindow, "minimum_width", 0) or 0)
                wkhmin = int(getattr(KVWindow, "minimum_height", 0) or 0)
            except Exception:
                wkmin, wkhmin = 0, 0
            try:
                sys_w, sys_h = int(KVWindow.system_size[0]), int(KVWindow.system_size[1])
            except Exception:
                sys_w, sys_h = 0, 0
            pw = self.ids.get("preview_widget")
            pww, pwh = (int(pw.size[0]), int(pw.size[1])) if pw is not None else (0, 0)
            tgeo = config.get_preview_texture_size()
            tw, th = int(tgeo[0]), int(tgeo[1])
            # get_config("preview_width/height") も同じ（process_pipeline に渡る値）
            tcfg_w = int(config.get_config("preview_width"))
            tcfg_h = int(config.get_config("preview_height"))
            sw, sh = kvutils.get_window_screen_size()
            pminw, pminh = int(self._preview_min_w), int(self._preview_min_h)
            mon = None
            if hasattr(device, "get_app_window_screen_size_points"):
                try:
                    mon = device.get_app_window_screen_size_points()
                except Exception:
                    mon = None
            mon_s = f"{int(mon[0])}x{int(mon[1])}" if mon and len(mon) >= 2 else "—"
            try:
                dps = float(device.dpi_scale())
            except Exception:
                dps = 0.0
            monb = None
            if hasattr(device, "get_app_window_screen_backing_pixel_size"):
                try:
                    monb = device.get_app_window_screen_backing_pixel_size()
                except Exception:
                    monb = None
            monb_s = f"{int(monb[0])}x{int(monb[1])}" if monb and len(monb) >= 2 else "—"
            line1 = (
                f"win {ww}x{wh}  |  kivy_sys {sys_w}x{sys_h}  scn_pt {sw}x{sh}  scn_bak {monb_s}  nss {mon_s}  dpi {dps:.2f}"
            )
            if tcfg_w == tw and tcfg_h == th:
                texpart = f"tex&pipeline {tw}x{th}"
            else:
                texpart = f"tex_cfg {tcfg_w}x{tcfg_h}  tex_store {tw}x{th}  (!)"
            line2 = (
                f"Kmin {wkmin}x{wkhmin}  pmin {pminw}x{pminh}  |  previewWgt {pww}x{pwh}  |  {texpart}"
            )
            print(f"[RESIZE-DEBUG] {line1}", flush=True)
            print(f"[RESIZE-DEBUG] {line2}", flush=True)
            logging.info("RESIZE-DEBUG %s  %s", line1, line2)
            if self._debug_resize_label is not None and pw is not None:
                self._debug_resize_label.text = f"{line1}\n{line2}"
                tw0 = max(200.0, float(pw.size[0]) - 8.0)
                self._debug_resize_label.text_size = (tw0, None)
                self._debug_resize_label.width = tw0
                self._debug_resize_label.height = self._debug_resize_label.texture_size[1] + 6.0
                self._debug_resize_label.pos = (4, 4)

        def on_kv_post(self, *args, **kwargs):
            super(MainWidget, self).on_kv_post(*args, **kwargs)

            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self._set_film_presets()
            self._set_lens_presets()

            KVClock.schedule_once(lambda dt: self.sync_distortion_mode_sliders(), 0)
            KVClock.schedule_once(lambda dt: self.sync_preview_widget_min_size(), 0)

            self.mask2_panel = mask2_content.create_mask2_content_panel(self.ids['mask_editor2'])
            self.ids['masks_box'].add_widget(self.mask2_panel)
            #self.ids['masks_box'].ids['content'].add_widget(self.mask2_panel)

            self.preset_panel = preset_content.create_preset_content_panel()
            self.ids['presets_box'].add_widget(self.preset_panel)

            self.history_panel = history_content.create_history_content_panel(self._on_history_selected)
            self.ids['history_box'].add_widget(self.history_panel)
            #self.ids['history_box'].ids['content'].add_widget(self.history_panel)
            self.update_load_dependent_panels_enabled()

            if getattr(define, "RESIZE_DEBUG", False):
                self._debug_resize_label = KVLabel(
                    text="",
                    size_hint=(None, None),
                    halign="left",
                    valign="bottom",
                    color=(1, 0.92, 0.15, 0.95),
                    font_size=10,
                )
                self.ids["preview_widget"].add_widget(self._debug_resize_label)
                KVClock.schedule_once(lambda _dt: self._update_resize_debug_display(), 0.05)

        def get_preview_texture_size(self):
            preview_widget = self.ids.get('preview_widget')
            min_side = config.get_config('preview_size')
            if preview_widget is None:
                return (min_side, min_side)

            widget_side = min(preview_widget.width, preview_widget.height) / device.dpi_scale()
            side = max(min_side, int(round(widget_side)))
            return (side, side)

        def update_preview_texture_size(self, force=False):
            size = self.get_preview_texture_size()
            changed = force or size != config.get_preview_texture_size()
            if changed:
                config.set_preview_texture_size(*size)
            return changed

        def sync_distortion_mode_sliders(self):
            """
            group 'distortion' 内のトグル: Lens 時はレンズ2本、Trapezoid 時は H/V/焦点、
            それ以外（Lines / Mesh / Four 等）では5本とも無効。
            """
            if "btn_lens" not in self.ids or "btn_trapezoid" not in self.ids:
                return

            lens_on = self.ids["btn_lens"].state == "down"
            trap_on = self.ids["btn_trapezoid"].state == "down"
            self.ids["slider_lens_distortion_strength"].disabled = not lens_on
            self.ids["slider_lens_distortion_scale"].disabled = not lens_on
            self.ids["slider_correct_trapezoid_h"].disabled = not trap_on
            self.ids["slider_correct_trapezoid_v"].disabled = not trap_on
            self.ids["slider_focal_length"].disabled = not trap_on

        @kvmainthread
        def refresh_preview_overlays(self, dt=0):
            # apply_thread 上の draw_image_core から呼ばれるため、ここをメインスレッド化する
            # （Canvas / グラフィックス操作は Kivy ではメインスレッド専用）
            texture_size = config.get_preview_texture_size()
            mask_editor = self.ids['mask_editor2']
            mask_editor.set_texture_size(*texture_size)
            if self._is_mask2_enabled():
                mask_editor.reposition_mask_image()

            if self.inpaint_edit is not None:
                self.inpaint_edit.set_display_size(texture_size)
            if self.patchmatch_inpaint_edit is not None:
                self.patchmatch_inpaint_edit.set_display_size(texture_size)

            geometry_effect = self.primary_effects[0].get('geometry')
            if geometry_effect is not None:
                geometry_effect.update_geometry_editor_texture_size()

            crop_effect = self.primary_effects[0].get('crop')
            if crop_effect is not None and params.has_original_img_size(self.primary_param):
                crop_effect.update_crop_editor_preview_size(self.primary_param)

        @kvmainthread
        def refresh_mask2_overlay(self, dt=0):
            mask_editor = self.ids.get('mask_editor2')
            if mask_editor is not None and self._is_mask2_enabled():
                mask_editor.refresh_active_mask_overlay()

        def empty_image(self):
            with threads.primary_param_lock:
                # 画像が無い状態。編集系 UI は全て無効化する。
                self.image_loaded = False
                # mask2/preset/history パネルも未選択状態に同期させる。
                # update_mask2_options_enabled 内で update_load_dependent_panels_enabled も
                # 走るので、両方の連動 UI が一度で正しく無効化される。
                self.update_mask2_options_enabled()
                self.update_preview_texture_size()
                self.texture = KVTexture.create(size=(config.get_config('preview_width'), config.get_config('preview_height')), colorfmt='rgb', bufferfmt='float')
                self.texture.flip_vertical()
                self.ids["preview"].texture = None

                self.imgset = None
                self.click_x = 0
                self.click_y = 0
                self.is_zoomed = False
                self.crop_image = None
                self._last_image_fidelity = None

                #core.clean_lensfun()

                self.primary_effects = effects.create_effects(
                    lens_modifier_callback=self.lens_modifier_callback,
                    distortion_callback=self.distortion_callback,
                    geometry_callback=self.geometry_callback,
                    crop_callback=self.crop_callback)
                self.reset_param(self.primary_param)
                self.ids['mask_editor2'].clear_mask()
        
        def start_draw_image_and_crop(self, imgset, center_pos=None, fast_display=False, skip_histogram=False):
            if self.imgset is imgset:
                self.start_draw_image(
                    center_pos,
                    invalidate_crop=True,
                    fast_display=fast_display,
                    skip_histogram=skip_histogram,
                )

        def sync_draw_image_and_crop(self, imgset):
            if self.imgset is imgset:
                self.sync_draw_image(invalidate_crop=True)

        def _debug_mask_geom_image_stats(self, image):
            try:
                return {
                    "shape": tuple(int(v) for v in image.shape),
                    "hash": hash(np.ascontiguousarray(image).tobytes()),
                    "min": float(np.nanmin(image)),
                    "max": float(np.nanmax(image)),
                    "sum": float(np.nansum(image)),
                    "nonzero": int(np.count_nonzero(image)),
                }
            except Exception:
                return {"shape": getattr(image, "shape", None)}

        @kvmainthread
        def blit_image(self, img, frame_version=None, allow_stale=False, dt=0):
            if frame_version is not None and frame_version < self.pipeline_version and not allow_stale:
                if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                    logging.warning(
                        "[MASK_GEOM] blit_image skipped stale frame_version=%s current_version=%s",
                        frame_version,
                        self.pipeline_version,
                    )
                return
            if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                logging.warning(
                    "[MASK_GEOM] blit_image frame_version=%s current_version=%s img_draw=%s",
                    frame_version,
                    self.pipeline_version,
                    self._debug_mask_geom_image_stats(img),
                )
            logging.debug("[PERF] blit_image: Start. Time: %s", time.time())
            perf_trace.event("blit_image.enter", shape=list(img.shape))
            # Texture Resizing logic
            is_dither = config.get_config('display_output_dither')
            is_downscale = config.get_config('display_output_downscale')
            target_fmt = 'ubyte' if (is_dither or is_downscale) else 'float'

            if self.texture is None or self.texture.size != (img.shape[1], img.shape[0]):
                self.texture = KVTexture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb', bufferfmt=target_fmt)
                self.texture.flip_vertical()

            if is_dither:
                img = core.jjn_dither_uint8(img)
                self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            elif is_downscale:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            else:
                self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')

            # Update Preview Widget Size
            self.ids["preview"].texture = None
            self.ids["preview"].texture = self.texture
            try:
                self.ids["preview"].canvas.ask_update()
                self.ids["transform_wrapper"].canvas.ask_update()
                self.ids["preview_widget"].canvas.ask_update()
            except Exception:
                logging.exception("preview canvas ask_update failed")

            self.resize()
            self.refresh_mask2_overlay()

            #Singnalを送る
            import signals
            signals.blit_image.emit()

            # 1 トレース = 1 画像表示。ここで JSONL に書き出す。
            perf_trace.event("blit_image.done")
            perf_trace.flush(reason="blit_done")

        @kvmainthread
        def draw_histogram_view(self, hist_data):
            #logging.debug(f"draw_histogram_view")
            self.ids["histogram"].draw_histogram_from_data(hist_data)

        def _display_output_encoding(self, colourspace):
            cs_lower = str(colourspace).lower()
            if 'srgb' in cs_lower or 'rec.709' in cs_lower or 'rec709' in cs_lower:
                return 'srgb'
            if 'display p3' in cs_lower or 'p3-d65' in cs_lower:
                return 'srgb'
            if 'adobe' in cs_lower:
                return 'gamma-2.2'
            if 'prophoto' in cs_lower or 'romm' in cs_lower:
                return 'gamma-1.8'
            return 'linear'

        def _get_fast_display_basis(self, src_space, dst_space, cat):
            key = (src_space, dst_space, cat)
            basis = self._fast_display_transform_cache.get(key)
            if basis is None:
                basis = colour_functions.RGB_to_RGB(
                    np.eye(3, dtype=np.float32),
                    src_space,
                    dst_space,
                    cat,
                    apply_cctf_decoding=False,
                    apply_cctf_encoding=False,
                    apply_gamut_mapping=False,
                ).astype(np.float32)
                self._fast_display_transform_cache[key] = basis
            return basis

        def _fast_display_color_transform(self, img, src_space, dst_space, cat):
            basis = self._get_fast_display_basis(src_space, dst_space, cat)
            src = np.asarray(img, dtype=np.float32)
            out = (src.reshape(-1, 3) @ basis).reshape(src.shape)
            np.maximum(out, 0.0, out=out)

            encoding = self._display_output_encoding(dst_space)
            if encoding == 'srgb':
                encoded = np.empty_like(out, dtype=np.float32)
                low = out <= 0.0031308
                encoded[low] = out[low] * 12.92
                encoded[~low] = 1.055 * np.power(out[~low], 1.0 / 2.4) - 0.055
                return encoded
            if encoding == 'gamma-2.2':
                return np.power(out, 1.0 / 2.2).astype(np.float32, copy=False)
            if encoding == 'gamma-1.8':
                return np.power(out, 1.0 / 1.8).astype(np.float32, copy=False)
            return out

        def draw_image_core(self, center_pos=None, fast_display=False, skip_histogram=False):
            with threads.primary_param_lock:
                if (self.imgset is not None) and (self.imgset.img is not None):
                    if not params.has_original_img_size(self.primary_param):
                        logging.warning("draw_image_core: original_img_size 未定義のため描画しません")
                        return

                    self.update_preview_texture_size()
                    self.refresh_preview_overlays()

                    frame_version = self.pipeline_version
                    current_tab = self.ids["effects"].current_tab.text
                    mask2_on = self._is_mask2_on()
                    crop_image_view_key = "full" if current_tab == "Ge" else "crop"
                    if self.crop_image_view_key != crop_image_view_key:
                        self.crop_image = None
                        self.crop_image_view_key = crop_image_view_key
                    if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                        logging.warning(
                            "[MASK_GEOM] draw_image_core start frame_version=%s current_tab=%s center_pos=%s fast_display=%s skip_histogram=%s",
                            frame_version,
                            current_tab,
                            center_pos,
                            fast_display,
                            skip_histogram,
                        )
                    img, self.crop_image = pipeline.process_pipeline(self.imgset.img, self.crop_image, self.is_zoomed, config.get_config('preview_width'), config.get_config('preview_height'), self.click_x, self.click_y, self.primary_effects, self.primary_param, self.ids['mask_editor2'], self.processor, frame_version, current_tab=current_tab, loading_flag=pipeline_loading_flag(self.imgset), is_drag=self.is_press_space, center_pos=center_pos, mask2_active=mask2_on)
                    logging.debug("[PERF] draw_image_core: process_pipeline finished. Time: %s", time.time())
                    perf_trace.event("draw_image_core.pipeline_done")
                    if img is None:
                        return
                    if frame_version < self.pipeline_version and not fast_display:
                        if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                            logging.warning(
                                "[MASK_GEOM] draw_image_core skipped stale frame_version=%s current_version=%s",
                                frame_version,
                                self.pipeline_version,
                            )
                        return
                    elif frame_version < self.pipeline_version:
                        if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                            logging.warning(
                                "[MASK_GEOM] draw_image_core allowing stale fast frame frame_version=%s current_version=%s",
                                frame_version,
                                self.pipeline_version,
                            )

                    debug_mask_geom = os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled()
                    post_t0 = time.perf_counter() if debug_mask_geom else None
                    img = np.array(img)
                    utils.print_nan_inf(img, "output")

                    src_space = getattr(self.imgset, 'color_space', 'ProPhoto RGB')
                    dst_space = config.get_config('display_color_gamut')
                    cat = config.get_config('cat')
                    color_t0 = time.perf_counter() if debug_mask_geom else None
                    if fast_display:
                        img = self._fast_display_color_transform(img, src_space, dst_space, cat)
                    else:
                        img = colour_functions.RGB_to_RGB(img, src_space, dst_space, cat,
                                                apply_cctf_decoding=False, apply_cctf_encoding=True, apply_gamut_mapping=True).astype(np.float32)
                    color_ms = (time.perf_counter() - color_t0) * 1000.0 if debug_mask_geom else 0.0

                    # Ge タブでは Mask2 モードでも full-preview なので zero-wrap しない。
                    # CropEditor の起動可否は CropEffect 側で Mask2 ON/OFF を見て制御する。
                    crop_editing = current_tab == "Ge"

                    # プレビュー表示
                    preview_t0 = time.perf_counter() if debug_mask_geom else None
                    img_draw = core.apply_out_of_range_exposure(img, self.ids['toggle_overexposure'].state == 'down', self.ids['toggle_underexposure'].state == 'down')
                    img_draw, _ = core.apply_zero_wrap(img_draw, self.primary_param, crop_editing=crop_editing)
                    img_draw = np.clip(img_draw, 0, 1)
                    preview_ms = (time.perf_counter() - preview_t0) * 1000.0 if debug_mask_geom else 0.0
                    if debug_mask_geom:
                        logging.warning(
                            "[MASK_GEOM] draw_image_core ready_to_blit frame_version=%s current_version=%s img=%s img_draw=%s",
                            frame_version,
                            self.pipeline_version,
                            self._debug_mask_geom_image_stats(img),
                            self._debug_mask_geom_image_stats(img_draw),
                        )

                    #描画をスケジューリング
                    self.blit_image(img_draw, frame_version, allow_stale=fast_display)

                    # ヒストグラムは表示画像を投げてから計算する。Mask2 操作中の体感遅延を
                    # 減らすため、プレビュー反映をヒストグラム更新で待たせない。
                    hist_ms = 0.0
                    if not skip_histogram:
                        hist_t0 = time.perf_counter() if debug_mask_geom else None
                        img_hist, exclude_count = core.apply_zero_wrap(img, self.primary_param, crop_editing=crop_editing)
                        hist_data = widgets.histogram.HistogramWidget.calculate_histogram_data(img_hist, 0, exclude_count)
                        hist_ms = (time.perf_counter() - hist_t0) * 1000.0 if debug_mask_geom else 0.0
                        if frame_version == self.pipeline_version:
                            self.draw_histogram_view(hist_data)
                    if debug_mask_geom:
                        logging.warning(
                            "[MASK_GEOM] draw_image_core post timings frame_version=%s fast_display=%s skip_histogram=%s color_ms=%.1f preview_ms=%.1f hist_ms=%.1f total_ms=%.1f",
                            frame_version,
                            fast_display,
                            skip_histogram,
                            color_ms,
                            preview_ms,
                            hist_ms,
                            (time.perf_counter() - post_t0) * 1000.0,
                        )
                    """
                    try:
                        if self.enabledelay is not None:
                            self.enabledelay.cancel()  # 既にスケジュール済みならキャンセル
                    except:
                        pass  # 未スケジュール時は無視
                    self.enabledelay = KVClock.schedule_once(partial(self.blit_image, img_draw), -1)
                    """

        def draw_image(self):
            last_processed_version = -1
            while self.apply_thread is not None:

                if last_processed_version >= self.pipeline_version:
                    self.draw_event.wait(timeout=0.1)
                    self.draw_event.clear()

                current_version = self.pipeline_version
                if last_processed_version < current_version:
                    center_pos = self.apply_draw_image_center
                    fast_display = self.apply_draw_fast_display
                    skip_histogram = self.apply_draw_skip_histogram
                    self.draw_image_core(center_pos, fast_display=fast_display, skip_histogram=skip_histogram)
                    last_processed_version = current_version
            
        def start_draw_image(self, center_pos=None, invalidate_crop=False, fast_display=False, skip_histogram=False):
            if invalidate_crop:
                self.crop_image = None
            self.pipeline_version += 1
            self.apply_draw_image_center = center_pos
            self.apply_draw_fast_display = fast_display
            self.apply_draw_skip_histogram = skip_histogram
            self.processor.set_pipeline_version(self.pipeline_version)
            if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                logging.warning(
                    "[MASK_GEOM] start_draw_image version=%s center_pos=%s invalidate_crop=%s fast_display=%s skip_histogram=%s active=%s",
                    self.pipeline_version,
                    center_pos,
                    invalidate_crop,
                    fast_display,
                    skip_histogram,
                    getattr(self.ids['mask_editor2'].get_active_mask(), 'mask_id', None),
                )
            self.draw_event.set()

        def sync_draw_image(self, invalidate_crop=False):
            if invalidate_crop:
                self.crop_image = None
            self.pipeline_version += 1
            self.draw_image_core()
                
        def crop_editing(self):
            self.apply_effects_lv(4, 'vignette')

        def lens_modifier_callback(self):
            self.run_set2widget_all = True
            try:
                self.primary_effects[0]['lens_modifier'].set2widget(self, self.primary_param)
            finally:
                self.run_set2widget_all = False

        def distortion_callback(self, proc, widget):
            match proc:
                case 'start':
                    self.begin_history_effect_ctrl(1, 'distortion')
                case 'update' | 'apply':
                    self.apply_effects_lv(1, 'distortion')
                case 'end':
                    self.primary_param.update(widget.get_distortion_params())
                    self.end_history_effect_ctrl(1, 'distortion')

        def geometry_callback(self, proc, widget):
            match proc:
                case 'start':
                    self.begin_history_effect_ctrl(0, 'geometry')
                case 'update' | 'apply':
                    self.apply_effects_lv(0, 'geometry', sync=True)
                    # Update widget with new params (especially matrix for correct display)
                    #widget.set_correction_params(self.primary_param)
                case 'end':
                    self.primary_param.update(widget.get_correction_params())
                    self.end_history_effect_ctrl(0, 'geometry')

        def crop_callback(self, proc, widget):
            match proc:
                case 'start':
                    self.begin_history_effect_ctrl(0, 'crop')
                case 'update' | 'apply':
                    params.set_crop_rect(self.primary_param, widget.get_crop_rect())
                case 'end':
                    self.end_history_effect_ctrl(0, 'crop')

        def _get_active_effects(self, mask_id=None, lv=None, subname=None):
            if mask_id is None:
                mask = self.ids['mask_editor2'].get_active_mask()
            else:
                mask = self.ids['mask_editor2'].find_mask(mask_id)

            if mask is None:
                return (self.primary_effects, self.primary_param, None)

            # マスクパラメータの振り分け
            if lv is not None:
                composit_mask = self.ids['mask_editor2'].find_composit_mask(mask)
                if lv == 3:
                    if subname == 'mask2_draw_effects' and not mask.is_composit():
                        mask = composit_mask
                    elif subname == 'mask_geometry' and not mask.is_composit():
                        # マスク自身の Geometry 変形は Composit 直下に保存・適用する
                        mask = composit_mask
                else:
                    # それ以外は親のCompositMaskへ（自分がCompositMaskなら自分へ）
                    if not mask.is_composit():
                        composit_mask = self.ids['mask_editor2'].find_composit_mask(mask)
                        if composit_mask is not None:
                            mask = composit_mask

            return (composit_mask.effects, mask.effects_param, mask.mask_id)
        
        def apply_effects_lv(self, lv, effect, sync=False, subname=None, defer_draw=False):
            if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and (
                lv == 3 or subname == 'mask_geometry' or effect == 'mask_geometry'
            ):
                logging.warning(
                    "[MASK_GEOM] apply_effects_lv enter lv=%s effect=%s subname=%s sync=%s defer_draw=%s run_set2widget_all=%s",
                    lv,
                    effect,
                    subname,
                    sync,
                    defer_draw,
                    self.run_set2widget_all,
                )
            if self.run_set2widget_all == True:
                if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and (
                    lv == 3 or subname == 'mask_geometry' or effect == 'mask_geometry'
                ):
                    logging.warning(
                        "[MASK_GEOM] apply_effects_lv skipped run_set2widget_all lv=%s effect=%s subname=%s",
                        lv,
                        effect,
                        subname,
                    )
                return

            current_effects, current_param, mask_id = self._get_active_effects(lv=lv, subname=subname or effect)
            if effect is None:
                effects.set2param_all(current_effects, current_param, self)
            else:
                effect = effect if isinstance(effect, list) else [effect]
                for e in effect:
                    current_effects[lv][e].set2param(current_param, self)
            if lv == 0:
                self.sync_distortion_mode_sliders()

            mask_geometry_update = False
            # Mask Geometry: slider 変更後 active Composit の mask Geom matrix を再構築。
            # set_draw_mask (= mask.update_mask 同期呼出) より前に行うことで、
            # update_mask 内の direction-preserving 等の matrix 依存計算が新 matrix
            # を使うようになる (古い matrix で 1 frame 描画 → schedule 経由で新 matrix
            # で再描画、という見た目のジャンプを回避)。
            if lv == 3:
                _eff_list = effect if isinstance(effect, list) else [effect] if effect is not None else []
                mask_geometry_update = subname == 'mask_geometry' or 'mask_geometry' in _eff_list
                if mask_geometry_update:
                    if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        _keys = (
                            "switch_mask_geometry",
                            "mask_rotation",
                            "mask_translation_x",
                            "mask_translation_y",
                            "mask_scale_x",
                            "mask_scale_y",
                            "mask_flip_mode",
                        )
                        logging.warning(
                            "[MASK_GEOM] apply_effects_lv lv=%s effect=%s subname=%s target_mask=%s params=%s",
                            lv,
                            effect,
                            subname,
                            mask_id,
                            {key: current_param.get(key) for key in _keys if key in current_param},
                        )
                    self.ids['mask_editor2']._set_active_composit_matrix(redraw_mask=True)
                    self.ids['mask_editor2'].skip_next_mask_overlay_refresh(clear=False)

            self.ids['mask_editor2'].set_draw_mask(
                self._should_draw_mask_overlay(lv, subname),
                refresh=not mask_geometry_update,
            )
            #self.apply_rotation_flip_for_wrapper()

            # defer_draw=True のとき呼び出し側は後でまとめて start/sync を発火する想定。
            # pipeline_version の無駄な多段進行と apply_thread の捨て描画を避ける。
            if defer_draw:
                return
            if sync == False:
                self.start_draw_image(
                    fast_display=mask_geometry_update,
                    skip_histogram=mask_geometry_update,
                )
            else:
                self.sync_draw_image()

        def set_effect_param(self, lv, effect, arg):
            if self.run_set2widget_all == True:
                return

            current_effects, current_param, _ = self._get_active_effects(lv=lv)
            current_effects[lv][effect].set2param2(current_param, arg)
            self.ids['mask_editor2'].set_draw_mask(self._should_draw_mask_overlay(lv, None))
            #self.apply_rotation_flip_for_wrapper()
            self.start_draw_image()

        def _should_draw_mask_overlay(self, lv, subname=None):
            return lv == 3 and subname != 'mask2_draw_effects'

        def apply_rotation_flip_for_wrapper(self):
            # Calculate Rotation/Flip for Hardware
            #if self.ids["effects"].current_tab.text == "Ge":
            if False:
                rotation_effect = self.primary_effects[0]['geometry']
                angle = rotation_effect._get_param(self.primary_param,'rotation') + rotation_effect._get_param(self.primary_param,'rotation2')
                flip = rotation_effect._get_param(self.primary_param,'flip_mode')
            else:
                angle = 0
                flip = 0

            # Apply Hardware Rotation/Flip to Wrapper
            wrapper = self.ids['transform_wrapper']
            if wrapper:
                wrapper.size = self.texture.size
                
                # Scale for Flip
                sx = -1 if (flip & 1) else 1
                sy = -1 if (flip & 2) else 1

                # Reset transform
                wrapper.transform = KVMatrix()
                
                # Apply Rotation & Flip via Matrix
                mat = KVMatrix()
                mat.scale(sx, sy, 1)
                mat.rotate(math.radians(angle), 0, 0, 1)
                wrapper.transform = mat
                
                # Center wrapper (KV binding should handle this, but re-triggering might be needed if transform affected pos)
                # Setting transform updates pos, so KV binding might fight. 
                # Ideally KV binding `center: self.parent.center` wins on next layout cycle.
                # To be safe, we can manually set center after transform if KV doesn't catch it immediately.
                if self.ids['preview_widget']:
                    wrapper.center = self.ids['preview_widget'].center

        def begin_history_layer_ctrl(self, layer_ctrl, op, index, op_type):
            self.current_op = history.Operation(type="Layer")
            self.current_op.set_backup_layer(layer_ctrl, op, index, op_type)

        def end_history_layer_ctrl(self, layer_ctrl, op, index):
            if self.current_op is None:
                logging.warning("MainWidget.end_history_layer_ctrl None.")
                return

            # current_op が Layer 型でない場合 (effect 系の begin が混入したケース等) は安全に抜ける
            if getattr(self.current_op, 'type', None) != 'Layer' or not hasattr(self.current_op, 'layer_ctrl'):
                logging.warning(f"MainWidget.end_history_layer_ctrl Type Mismatch (got type={getattr(self.current_op, 'type', None)}).")
                self.current_op = None
                return

            if self.current_op.layer_ctrl is not layer_ctrl:
                logging.warning("MainWidget.end_history_layer_ctrl Unmatching.")
                return

            if self.current_op.set_update_layer(layer_ctrl, op, index) is not None:
                self.history.append(self.current_op)
                self.history_panel.set_history(self.history)
                self.current_op = None

        def begin_history_reset_all(self):
            self.current_op = history.Operation(type="All")
            self.current_op.set_backup_all(self.primary_param, self.ids['mask_editor2'])

        def end_history_reset_all(self):
            if self.current_op is None:
                return

            if self.current_op.type != "All":
                logging.warning(f"MainWidget.end_history_reset_all Type Unmatching. {self.current_op.type}")
                return

            if self.current_op.check_backup_all(self.primary_param, self.ids['mask_editor2']) == False:
                self.history.append(self.current_op)
                self.history_panel.set_history(self.history)
                self.current_op = None

        def reset_all(self):
            
            # セーブしないパラメータ（メタデータ等）は維持する。
            # 全体Resetでは crop_rect などの編集内容は維持しない。
            temp_param = {}
            params.copy_special_param(temp_param, self.primary_param)
            
            self.primary_param.clear()
            self.primary_param.update(temp_param)
            
            # 初期化パラメータ設定
            params.set_image_param(self.primary_param, self.imgset.img)

            # マスク関連全消去
            self.ids['mask2'].state = 'normal' # マスクモードを抜けないとおかしくなる
            self._disable_mask2()
            self.ids['mask_editor2'].clear_mask()
            
            # UIを先に初期値へ同期しないと、apply_effects_lv が古いウィジェット値をparamへ戻してしまう。
            self.set2widget_all(self.primary_effects, self.primary_param)

            # クロップエディタ起動時はそれの初期化も行う
            self.primary_effects[0]['crop'].reset2_crop_editor(self.primary_param)
            self.primary_effects[0]['crop'].reset_crop_editor()
            self.apply_effects_lv(0, 'crop') # 描画を走らせる

            # これでファイルが消えるはず
            self.save_current_sidecar()
           
            # UIと表示の更新
            self.set2widget_all(self.primary_effects, self.primary_param)

        def begin_history_effect_ctrl(self, lv, effect, subname=None):
            if self.run_set2widget_all == True:
                return False
            current_effects, current_param, mask_id = self._get_active_effects(lv=lv, subname=subname or effect)
            effect_list = effect if isinstance(effect, list) else [effect]
            self.current_op = history.Operation(lv, effect_list, subname, mask_id)
            self.current_op.set_backup(current_effects, current_param, subname)
            return True
        
        def end_history_effect_ctrl(self, lv, effect, subname=None):            
            effect_list = effect if isinstance(effect, list) else [effect]
            redraw_full_after_edit = lv == 3 and (
                subname == 'mask_geometry' or 'mask_geometry' in effect_list
            )
            
            if self.current_op is None:
                logging.warning(f"MainWidget.end_history_effect_ctrl None. {effect_list}")
                return
            
            if self.current_op.subname != subname:
                logging.warning(f"MainWidget.end_history_effect_ctrl Subname Unmatching. {effect_list}")
                return

            if self.current_op.lv != lv or self.current_op.effect_list != effect_list:
                logging.warning(f"MainWidget.end_history_effect_ctrl LV or Effect Unmatching. {effect_list}")

            current_effects, current_param, mask_id = self._get_active_effects(
                self.current_op.mask_id, lv=lv, subname=subname or effect
            )
            if self.current_op.set_update(current_effects, current_param, subname) is not None:
                self.history.append(self.current_op)
                self.history_panel.set_history(self.history)
            self.current_op = None
            if redraw_full_after_edit:
                self.start_draw_image()

        def apply_crop_button_action(self, action):
            if not self.begin_history_effect_ctrl(0, 'crop'):
                return
            self.primary_effects[0]['crop'].apply_crop_button_action(self.primary_param, self, action)
            self.ids['mask_editor2'].set_draw_mask(self._should_draw_mask_overlay(0, None))
            self.start_draw_image_and_crop(self.imgset)
            self.end_history_effect_ctrl(0, 'crop')
            self.save_current_sidecar()

        def begin_rotation_crop_preview(self):
            self.primary_effects[0]['crop'].begin_rotation_preview(self.primary_param)

        def end_rotation_crop_preview(self):
            self.primary_effects[0]['crop'].end_rotation_preview(self.primary_param)

        def reset_switch_defaults_for_label(self, head_label):
            switch_id = None
            # self.ids の値はしばしば WeakProxy なので `is head_label` では一致しない。
            # 実体との比較は `==` に任せつつ、既に無効になったプロキシは ReferenceError とする。
            for key in self.ids:
                try:
                    value = self.ids[key]
                except ReferenceError:
                    continue
                if value is head_label:
                    switch_id = key
                    break
                try:
                    if value == head_label:
                        switch_id = key
                        break
                except ReferenceError:
                    continue
            if switch_id is None:
                return False

            target = self._switch_reset_targets().get(switch_id)
            if target is None:
                return False

            lv, effect, subname = target
            self._reset_effect_defaults(lv, effect, subname)
            return True

        def _switch_reset_targets(self):
            return build_switch_reset_targets()

        def _reset_effect_defaults(self, lv, effect, subname=None):
            effect_list = effect if isinstance(effect, list) else [effect]
            if not self.begin_history_effect_ctrl(lv, effect, subname):
                return

            current_effects, current_param, _ = self._get_active_effects(lv=lv, subname=subname or effect)
            if subname is not None:
                default_params = current_effects[lv][effect_list[0]].get_param_dict(current_param, subname)
            else:
                default_params = {}
                for effect_name in effect_list:
                    default_params.update(current_effects[lv][effect_name].get_param_dict(current_param))

            for key, value in default_params.items():
                current_param[key] = value.copy() if isinstance(value, list) else value

            for effect_name in effect_list:
                current_effects[lv][effect_name].set2widget(self, current_param)

            self.apply_effects_lv(lv, effect, subname=subname)
            self.end_history_effect_ctrl(lv, effect, subname)

        def _undo(self):        
            if self.history.can_undo():
                if self.history.undo(self):
                    self.history_panel.set_history(self.history)
                    #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                    self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                    self._set_diff_list_to_inpaint_edit()
                    self._sync_editor_modes_after_history()
                    self.start_draw_image_and_crop(self.imgset)

        def _redo(self):        
            if self.history.can_redo():
                if self.history.redo(self):
                    self.history_panel.set_history(self.history)
                    #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                    self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                    self._set_diff_list_to_inpaint_edit()
                    self._sync_editor_modes_after_history()
                    self.start_draw_image_and_crop(self.imgset)

        def _on_history_selected(self, index):
            if self.mask2_wait_full_load:
                return

            if index < self.history.current_index:
                n = self.history.current_index - index
                for _ in range(n):
                    self.history.undo(self)
                self.history_panel.set_history(self.history)
                #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                self._set_diff_list_to_inpaint_edit()
                self._sync_editor_modes_after_history()
                self.start_draw_image_and_crop(self.imgset)

            elif index >= self.history.current_index:
                n = index - self.history.current_index
                for _ in range(n):
                    self.history.redo(self)
                self.history_panel.set_history(self.history)
                #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                self._set_diff_list_to_inpaint_edit()
                self._sync_editor_modes_after_history()
                self.start_draw_image_and_crop(self.imgset)

        def _sync_editor_modes_after_history(self):
            self.primary_effects[0]['crop'].sync_crop_editor_mode_from_widget(self, self.primary_param)

        def reset_param(self, param):
            param.clear()

        def set2widget_all(self, _effects, param, reset_effects=True):
            if _effects is None:
                _effects = self.primary_effects
                param = self.primary_param

            self.run_set2widget_all = True
            try:
                effects.set2widget_all(self, _effects, param, reset_effects=reset_effects)
            finally:
                self.run_set2widget_all = False

        def _viewer_snapshot_rating(self, file_path: str) -> int:
            if not file_path:
                return 0
            v = self.ids.get("viewer")
            if v:
                for d in v.data:
                    if d.get("file_path") == file_path:
                        return int(d.get("rating", 0) or 0)
            return 0

        def save_current_sidecar(self):
            if self.imgset is not None:
                param2 = effects.delete_default_param_all(self.primary_effects, self.primary_param) # プライマリのデフォルト値は消す
                param2['image_fidelity'] = getattr(self.imgset, 'fidelity', ImageFidelity.FULL).value
                param2.pop("rating", None)
                raw_r = 0
                if rating_utils.is_raw_path(self.imgset.file_path):
                    raw_r = self._viewer_snapshot_rating(self.imgset.file_path)
                result = params.save_json(
                    self.imgset.file_path, param2, self.ids['mask_editor2'], raw_sidecar_rating=raw_r
                )
                if result == False:
                    # 失敗時はファイルを削除
                    params.delete_empty_param_json(self.imgset.file_path)
                viewer = self.ids.get("viewer")
                if viewer:
                    viewer.set_pmck_indicator_for_path(self.imgset.file_path)

        def _snapshot_current_param(self):
            snap = params.serialize(self.primary_param, self.ids['mask_editor2']) or {"primary_param": {}}
            params.copy_special_param(snap["primary_param"], self.primary_param)
            params.copy_remain_param(snap["primary_param"], self.primary_param)
            return snap

        def _open_effect_selector(self, on_decide):
            selected = preset_utils.get_saved_selector_switch_keys(config)
            dialog = EffectSelector(selected_switch_keys=selected)

            def _decide(inst, _selection):
                switch_keys = inst.get_selected_switch_keys()
                preset_utils.save_selector_switch_keys(config, switch_keys)
                on_decide(switch_keys)

            dialog.bind(on_decide=_decide)
            dialog.open()

        def _can_use_effect_settings_transfer(self):
            return (
                self.imgset is not None
                and self.loading is False
                and self.mask2_wait_full_load is False
                and self.primary_param.get("image_fidelity") == ImageFidelity.FULL.value
            )

        def _warn_effect_settings_transfer_not_ready(self):
            self.show_warning_dialog("Please wait until the image finishes loading.")

        def copy_effect_settings(self):
            if not self._can_use_effect_settings_transfer() or not self.primary_param:
                self._warn_effect_settings_transfer_not_ready()
                return

            def _copy(switch_keys):
                partial = preset_utils.collect_selected_primary_param(
                    self.primary_effects, self.primary_param, switch_keys
                )
                self._copied_effect_param = {
                    "switch_keys": list(switch_keys),
                    "primary_param": partial,
                }

            self._open_effect_selector(_copy)

        def _apply_partial_to_current(self, partial_param, history_name="Paste Settings"):
            if self.imgset is None:
                return False
            op = history.Operation(type="All")
            op.set_backup_all(self.primary_param, self.ids['mask_editor2'])
            preset_utils.apply_partial_primary_param(self.primary_param, partial_param)
            self.set2widget_all(self.primary_effects, self.primary_param)
            self.start_draw_image()
            if op.set_update_all(self.primary_param, self.ids['mask_editor2'], history_name) is not None:
                self.history.append(op)
                self.history_panel.set_history(self.history)
            self.save_current_sidecar()
            return True

        def paste_effect_settings(self):
            if not self._can_use_effect_settings_transfer():
                self._warn_effect_settings_transfer_not_ready()
                return
            copied = self._copied_effect_param
            if not copied or not copied.get("primary_param"):
                self.show_warning_dialog("No copied settings.")
                return
            cards = self.ids['viewer'].get_selected_cards()
            if not cards:
                if self.imgset is None:
                    return
                self._apply_partial_to_current(copied["primary_param"])
                return
            if len(cards) == 1:
                card = cards[0]
                if self.imgset is not None and card.file_path == self.imgset.file_path:
                    self._apply_partial_to_current(copied["primary_param"])
                    return
            self._paste_effect_settings_to_cards(cards, copied["primary_param"])

        def _paste_effect_settings_to_cards(self, cards, partial_param):
            current_path = self.imgset.file_path if self.imgset is not None else None
            current_backup = self._snapshot_current_param() if current_path and any(c.file_path == current_path for c in cards) else None

            def _job():
                items = []
                for card in cards:
                    item = preset_utils.backup_pmck_for_batch(card.file_path)
                    preset_utils.apply_partial_to_pmck_file(card.file_path, partial_param)
                    items.append(item)
                return items

            items = processing_dialog.wait_prosessing(_job)
            if current_backup is not None:
                preset_utils.apply_partial_primary_param(self.primary_param, partial_param)
                self.set2widget_all(self.primary_effects, self.primary_param)
                self.start_draw_image()
            current_update = self._snapshot_current_param() if current_backup is not None else None
            op = history.Operation(type="BatchPaste")
            op.set_batch_paste(items, current_backup=current_backup, current_update=current_update)
            self.history.append(op)
            self.history_panel.set_history(self.history)
            for item in items:
                self._refresh_pmck_indicator_for_image_path(item.get("image_path"))

        def _refresh_pmck_indicator_for_image_path(self, image_path):
            viewer = self.ids.get("viewer")
            if viewer and image_path:
                viewer.set_pmck_indicator_for_path(
                    image_path, exists=os.path.exists(image_path + ".pmck")
                )

        def show_warning_dialog(self, message):
            layout = KVBoxLayout(orientation="vertical")
            layout.ref_padding = 10
            layout.ref_spacing = 10
            layout.add_widget(KVLabel(text=message))
            btn = KVButton(text="OK", size_hint_y=None)
            btn.ref_height = 40
            layout.add_widget(btn)
            popup = KVPopup(title="Warning", content=layout, size_hint=(None, None), auto_dismiss=True)
            popup.ref_width = 420
            popup.ref_height = 160
            dialogutils.install_ref_scaling(popup)
            btn.bind(on_release=popup.dismiss)
            popup.open()

        def start_add_preset(self):
            if self.mask2_wait_full_load:
                return
            if not self._can_use_effect_settings_transfer() or not self.primary_param:
                self._warn_effect_settings_transfer_not_ready()
                return

            def _selected(switch_keys):
                partial = preset_utils.collect_selected_primary_param(
                    self.primary_effects, self.primary_param, switch_keys
                )
                self._open_preset_name_dialog(partial)

            self._open_effect_selector(_selected)

        def _open_preset_name_dialog(self, partial_param):
            layout = KVBoxLayout(orientation="vertical")
            layout.ref_padding = 10
            layout.ref_spacing = 10
            text_input = KVTextInput(multiline=False, hint_text="Preset name", size_hint_y=None)
            text_input.ref_height = 32
            buttons = KVBoxLayout(orientation="horizontal", size_hint_y=None)
            buttons.ref_height = 40
            buttons.ref_spacing = 8
            cancel_btn = KVButton(text="Cancel")
            ok_btn = KVButton(text="OK")
            buttons.add_widget(cancel_btn)
            buttons.add_widget(ok_btn)
            layout.add_widget(text_input)
            layout.add_widget(buttons)
            popup = KVPopup(title="Save Preset", content=layout, size_hint=(None, None), auto_dismiss=False)
            popup.ref_width = 440
            popup.ref_height = 180
            dialogutils.install_ref_scaling(popup)

            def _save(*_args):
                try:
                    path = preset_utils.preset_path_for_name(text_input.text)
                    preset_utils.save_preset_json(path, preset_utils.build_preset_dict(partial_param))
                except Exception as e:
                    self.show_warning_dialog(str(e))
                    return
                popup.dismiss()
                self.refresh_preset_panel()

            ok_btn.bind(on_release=_save)
            cancel_btn.bind(on_release=popup.dismiss)
            text_input.bind(on_text_validate=_save)
            popup.open()

        def refresh_preset_panel(self):
            panel = getattr(self, "preset_panel", None)
            if panel is not None:
                panel.refresh_list()

        def apply_preset_path(self, preset_path):
            if self.mask2_wait_full_load:
                return
            if self.imgset is None:
                return
            try:
                partial = preset_utils.load_preset_json(preset_path)
            except Exception as e:
                self.show_warning_dialog(str(e))
                return
            self._apply_partial_to_current(partial, history_name="Apply Preset")

        def confirm_delete_preset(self, preset_name, preset_path):
            if self.mask2_wait_full_load:
                return
            layout = KVBoxLayout(orientation="vertical")
            layout.ref_padding = 10
            layout.ref_spacing = 10
            layout.add_widget(KVLabel(text=f'Delete preset "{preset_name}"?'))
            buttons = KVBoxLayout(orientation="horizontal", size_hint_y=None)
            buttons.ref_height = 40
            buttons.ref_spacing = 8
            cancel_btn = KVButton(text="Cancel")
            delete_btn = KVButton(text="Delete")
            buttons.add_widget(cancel_btn)
            buttons.add_widget(delete_btn)
            layout.add_widget(buttons)
            popup = KVPopup(title="Delete Preset", content=layout, size_hint=(None, None), auto_dismiss=False)
            popup.ref_width = 440
            popup.ref_height = 180
            dialogutils.install_ref_scaling(popup)

            def _delete(*_args):
                try:
                    os.remove(preset_path)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    self.show_warning_dialog(str(e))
                    return
                popup.dismiss()
                self.refresh_preset_panel()

            cancel_btn.bind(on_release=popup.dismiss)
            delete_btn.bind(on_release=_delete)
            popup.open()
        
        @kvmainthread
        def on_select(self, card):
            logging.debug("[PERF] on_select: Start. Time: %s", time.time())
            perf_trace.select_start(card.file_path if card is not None else None)
            perf_trace.event("on_select.enter")
            # ロード開始
            self.loading = True
            self.mask2_wait_full_load = True
            if 'mask2' in self.ids:
                # ファイル切替時: 状態を戻してから無効化
                self.ids['mask2'].state = 'normal'
            self.update_mask2_options_enabled()
            self._actively_loading = True  # アニメーション表示開始
            with threads.primary_param_lock:
                # 前の設定を保存
                self.save_current_sidecar()
                # 前のエフェクトを終了
                effects.finalize_all(self.primary_effects, self.primary_param, self)
                # 空のイメージをセット
                self.empty_image()

            if card is not None:
                self._expected_file_path = card.file_path
                self._clear_exif_data()
                # 別ファイルへ切り替わるので前回の pmck キャッシュを破棄
                self._last_pmck_dict = None
                self.cache_system.register_for_preload(card.file_path, card.exif_data, None, True)
                self.cache_system.get_file(card.file_path, lambda f1, f2, f3, f4, f5, f6: file_cache_system.run_method(self, "on_fcs_get_file", config._config, f1, f2, f3, f4, f5, f6))
            else:
                self._expected_file_path = None
                self._clear_exif_data()
                # カードなし（フォルダ空など）— get_file が呼ばれず loading が解除されないのを防ぐ
                self.loading = False
                self._actively_loading = False
                self.update_mask2_options_enabled()
        
        @kvmainthread
        def on_fcs_get_file(self, file_path, imgset, exif_data, param, history_obj, stage):
            stage = coerce_load_stage(stage)
            if file_path != getattr(self, '_expected_file_path', None):
                logging.info(
                    "FCS: 現在の選択と異なるファイルのコールバックを無視します: %r (expected %r)",
                    file_path,
                    getattr(self, '_expected_file_path', None),
                )
                return
            logging.debug("[PERF] on_fcs_get_file: Called. stage: %s, Time: %s", stage, time.time())
            perf_trace.event("on_fcs_get_file.enter", stage=str(stage))
            _img = getattr(imgset, "img", None) if imgset is not None else None
            shape_str = getattr(_img, "shape", None) if _img is not None else None
            print(f"Load image SHAPE: {shape_str} fidelity: {getattr(imgset, 'fidelity', None)}, Proc: {stage}")

            if imgset is None or getattr(imgset, "img", None) is None:
                logging.error(f"画像データがありません: {file_path} (stage={stage})")
                self.empty_image()
                if self.processor:
                    self.processor.cancel_all()
                self.loading = False
                self._actively_loading = False
                # 失敗パスはここで終わり blit_image に到達しない。トレースを取りこぼさないよう
                # ここで明示的に flush する。
                perf_trace.event("on_fcs_get_file.load_failed", stage=str(stage))
                perf_trace.flush(reason="load_failed")
                return

            if _load_stage_allows_ui(stage, imgset):
                self.loading = False
                # ここで初めて param と imgset が編集可能な対になる。UI 全体を解禁。
                self.image_loaded = True
            if _load_stage_ends_file_loading_indicator(stage, imgset):
                self._actively_loading = False
            # Mask2 はフル読み込み完了までは無効化
            if stage == LoadStage.RGB_DONE or (
                stage == LoadStage.FULL_DECODE and getattr(imgset, 'fidelity', None) == ImageFidelity.FULL
            ):
                self.mask2_wait_full_load = False
            self.update_load_dependent_panels_enabled()
            self.update_mask2_options_enabled()

            if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE):
                card = self.ids['viewer'].get_card(file_path)
                if card is not None:
                    # 一度も描画してないので値が設定されてない。暫定処置
                    self.update_preview_texture_size()
                    self.ids['mask_editor2'].set_texture_size(config.get_config('preview_width'), config.get_config('preview_height'))
                    self.ids['mask_editor2'].set_primary_param(param, params.get_disp_info(param))

                    # フル解像のときだけ pmck から重い結果を復元（RAW プレビュー段階ではスキップ）
                    param['image_fidelity'] = getattr(imgset, 'fidelity', ImageFidelity.FULL).value
                    load_heavy = param['image_fidelity'] == ImageFidelity.FULL.value
                    pmck_dict = params.load_json(file_path, param, self.ids['mask_editor2'], load_heavy=load_heavy)
                    # RAW: プレビュー段階で読んだ dict を保持しておき、FULL_DECODE 遷移時の
                    # merge_heavy_from_pmck で再パースを避ける。
                    self._last_pmck_dict = (file_path, pmck_dict)

            # Cancel previous background tasks
            if self.processor:
                self.processor.cancel_all()
                self.ids['histogram'].set_histogram_data(None)  # Reset histogram?

            with threads.primary_param_lock:
                if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE):
                    # １回目の時だけパラメータを反映して、編集できる様にする
                    self.primary_param.clear()
                    self.primary_param.update(param)
                    if file_path and rating_utils.is_rgb_path(file_path):
                        # サムネ用 get_metadata 由来は param / exif_data 引数（同一参照想定）に。空なら exiftool 追読だけが頼り
                        pex = self.primary_param.get("exif_data")
                        if not isinstance(pex, dict):
                            pex = {}
                        if not pex and isinstance(exif_data, dict) and exif_data:
                            pex = exif_data
                        self.primary_param["exif_data"] = {**pex}
                        ex = self.primary_param["exif_data"]
                        rating_io.merge_xmp_star_tags_into_exif(file_path, ex)
                        try:
                            r = int(rating_utils.parse_exif_rating_value(ex) or 0)
                            self.ids["viewer"].set_rating_for_path(file_path, r)
                        except Exception:
                            pass
                    logging.debug("[PERF] on_fcs_get_file: Merged Params. Time: %s", time.time())
                    perf_trace.event("on_fcs_get_file.params_merged")
                    self.set2widget_all(self.primary_effects, self.primary_param)
                    self.update_mask2_options_enabled()

                    # 特別あつかいでエディタを起動できるなら起動する。
                    # ここでは描画キックを抑止し、後段の start_draw_image_and_crop に集約する。
                    self.apply_effects_lv(1, 'distortion', defer_draw=True)
                    self.apply_effects_lv(0, 'crop', defer_draw=True)

                    # ヒストリーの設定
                    if history_obj is None:
                        self.history = history.History()
                        self.cache_system.set_history(file_path, self.history)
                    else:
                        self.history = history_obj

                    self.history_panel.set_history(self.history)

                # RAW フルデコード完了時にレンズ補正まわりを param から反映
                if stage == LoadStage.FULL_DECODE and param.get('rgb_or_raw') == 'raw':
                    self.primary_param['lens_modifier'] = param['lens_modifier']
                    if param['lens_modifier'] == True:
                        self.primary_param['exif_data'] = param['exif_data']
                    self.primary_param['rgb_or_raw'] = param['rgb_or_raw']
                    self.primary_param['auto_exposure'] = param['auto_exposure']
                    # プレビュー段階で入った original_img_size / img_size のままだと
                    # フル解像バッファとずれて crop_image が空領域になり黒画面になる。
                    # FULL_DECODEは履歴外で来るため、ユーザー編集値 crop_rect 自体は上書きしない。
                    for _k in ('original_img_size', 'img_size'):
                        if _k in param:
                            self.primary_param[_k] = param[_k]
                    if params.get_crop_rect(self.primary_param) is not None:
                        disp_info = core.convert_rect_to_info(
                            params.get_crop_rect(self.primary_param),
                            config.get_preview_texture_side()/max(self.primary_param['original_img_size'])
                        )
                        params.set_disp_info(self.primary_param, disp_info)
                    self.ids['mask_editor2'].set_primary_param(
                        self.primary_param, params.get_disp_info(self.primary_param)
                    )
                    self.primary_effects[0]['crop'].sync_crop_editor_mode_from_widget(self, self.primary_param)

                self.imgset = imgset

                fid = getattr(imgset, 'fidelity', ImageFidelity.FULL)
                self.primary_param['image_fidelity'] = fid.value
                prev_fid = getattr(self, '_last_image_fidelity', None)
                if fid == ImageFidelity.FULL and prev_fid == ImageFidelity.PREVIEW:
                    cached_pmck = None
                    last = getattr(self, '_last_pmck_dict', None)
                    if last is not None and last[0] == file_path:
                        cached_pmck = last[1]
                    params.merge_heavy_from_pmck(
                        file_path, self.primary_param, self.ids['mask_editor2'], cached_dict=cached_pmck
                    )
                self._last_image_fidelity = fid

                params.apply_original_geometry_if_missing(self.primary_param, imgset.img)
                if not params.has_original_img_size(self.primary_param):
                    logging.error("on_fcs_get_file: デコード画像からも original_img_size を確定できません")
                    self.loading = False
                    self._actively_loading = False
                    perf_trace.event("on_fcs_get_file.no_geometry", stage=str(stage))
                    perf_trace.flush(reason="no_geometry")
                    return

                effects.reeffect_all(self.primary_effects)
                self.start_draw_image_and_crop(imgset)
            if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE) or (
                stage == LoadStage.FULL_DECODE and param.get('rgb_or_raw') == 'raw'
            ):
                display_exif = self.primary_param.get("exif_data", exif_data)
                if isinstance(display_exif, dict) and display_exif:
                    self._set_exif_data(display_exif, file_path=file_path)
            self._sync_exif_rating_row()

        def _image_interaction_ready(self):
            """ズーム／ドラッグ等のプレビュー上の操作を受け付けて良い状態か。

            画像が確定していない間（起動直後・フォルダ切替直後・ロード途中）に操作させると、
            is_zoomed 等の内部状態だけが変化して描画は走らず、見た目の整合性が崩れる。
            """
            return (
                self.image_loaded
                and self.imgset is not None
                and getattr(self.imgset, 'img', None) is not None
            )

        def _is_mask2_on(self):
            """Mask2 トグルが ON 状態か。マスク編集モードの判定軸。
            個別マスクが Active かどうかではなく、Mask2 パネル全体が有効か否かで判定する
            (ON 中ならマスク未選択でも『マスク Geometry モード』として扱う)。"""
            try:
                return self.ids['mask2'].state == 'down'
            except Exception:
                return False

        def _is_image_geometry_mode(self):
            """Geometry タブを開いていて、かつ Mask2 が OFF の状態。
            画像本体の Geometry を編集している状態として、ズームは禁止する。
            Mask2 ON 中 (= マスク Geometry モード) はズームを許可する。"""
            try:
                if self.ids["effects"].current_tab.text != "Ge":
                    return False
                return not self._is_mask2_on()
            except Exception:
                return False

        def _active_mask_consumes_double_tap(self, touch):
            """アクティブマスクがダブルタップを自分の動作で消費するかどうか。
            (例: PolylineMask の開放確定。これが True なら preview の zoom 切替を抑制する)"""
            try:
                mask = self.ids["mask_editor2"].get_active_mask()
            except Exception:
                return False
            if mask is None:
                return False
            consumer = getattr(mask, 'consumes_double_tap', None)
            if not callable(consumer):
                return False
            try:
                return bool(consumer(touch))
            except Exception:
                return False

        def on_image_touch_down(self, touch):
            if self.ids['preview_widget'].collide_point(*touch.pos):
                # 画像未確定の間は preview 上のジェスチャを受け付けない
                if not self._image_interaction_ready():
                    return False
                # ズーム操作: 画像 Geometry モード時のみ抑制 (マスク Geometry モード時は許可)
                # ただしアクティブマスクがダブルタップを消費 (= PolylineMask の確定) する場合も抑制
                if (touch.is_double_tap == True
                        and not self._is_image_geometry_mode()
                        and not self._active_mask_consumes_double_tap(touch)):
                    self.is_zoomed = not self.is_zoomed
                    if self.is_zoomed == False:
                        self.click_x, self.click_y = 0, 0
                        self.drag_center_start = None
                        disp_info = core.convert_rect_to_info(params.get_crop_rect(self.primary_param), config.get_preview_texture_side()/max(self.primary_param['original_img_size']))
                        params.set_disp_info(self.primary_param, disp_info)

                    else:
                        # ウィンドウ座標からローカルイメージ座標に変換
                        self.click_x, self.click_y = utils.to_texture(touch.pos, self.ids['preview'])

                    effects.reeffect_all(self.primary_effects, 1)
                    self.start_draw_image_and_crop(self.imgset)

                # ドラッグ操作
                elif self.is_zoomed == True:
                    # ドラッグ開始時の中心位置を計算して保存
                    disp_info = params.get_disp_info(self.primary_param)
                    if disp_info is not None:
                        dx, dy, dw, dh, scale = disp_info
                        self.drag_center_start = (dx + dw/2, dy + dh/2)

            return False

        def on_image_touch_move(self, touch):
            if self.collide_point(*touch.pos):
                if self.is_zoomed == True:
                    if self.is_press_space == True and self.drag_center_start is not None:
                        # 表示倍率を取得
                        disp_info = params.get_disp_info(self.primary_param)
                        scale = disp_info[4] * device.dpi_scale()
                        
                        # 画面上の移動量
                        diff_screen_x = touch.pos[0] - touch.opos[0]
                        diff_screen_y = touch.pos[1] - touch.opos[1]

                        # 画像上の移動量に変換
                        offset_x = -diff_screen_x / scale
                        offset_y = diff_screen_y / scale # KivyはY軸上向き、画像座標系の移動としては...

                        # 新しい中心位置の計算
                        new_cx = self.drag_center_start[0] + offset_x
                        new_cy = self.drag_center_start[1] + offset_y 

                        effects.reeffect_all(self.primary_effects, 1)
                        self.start_draw_image_and_crop(
                            self.imgset,
                            center_pos=(new_cx, new_cy),
                            fast_display=True,
                            skip_histogram=True,
                        )

            return False
                    
        def on_image_touch_up(self, touch):
            if self.is_zoomed == True:
                if self.drag_center_start is not None:
                    self.drag_center_start = None

            return False

        def on_select_press(self):
            self.save_current_sidecar()
            device.FileChooser(title="Select Folder", mode="dir", filters=[("Jpeg Files", "*.jpg")], on_selection=self.handle_for_dir_selection).run()

        def on_export_press(self):
            self.save_current_sidecar()

            dialog = ExportDialog(callback=self.handle_export_dialog)
            dialog.open()

        def handle_export_dialog(self, preset):
            if self.export_in_progress:
                return
            # 保存先ファイルの存在チェック
            cards = self.ids['viewer'].get_selected_cards()
            isfile = False
            for x in cards:
                ex_path = self._make_export_path(x.file_path, preset)
                if os.path.isfile(ex_path):
                    isfile = True
                    break

            if isfile == True:
                dialog = ExportConfirmDialog(preset=preset, callback=self.handle_confirm_dialog)
                dialog.open()

            elif len(cards) > 0:
                self.handle_confirm_dialog('Overwrite', preset)            

        def handle_confirm_dialog(self, select, preset):
            if select in ['Overwrite', 'Rename']:
                cards = self.ids['viewer'].get_selected_cards()
                if not cards or self.export_in_progress:
                    return

                self._export_cancel_event.clear()
                n = len(cards)
                self.export_total = n
                self.export_done = 0
                self.export_in_progress = True

                rating_by_path = {
                    c.file_path: self._export_snapshot_rating(c.file_path) for c in cards
                }

                def _export_job():
                    exported_ok = []
                    try:
                        for i, x in enumerate(cards):
                            if self._export_cancel_event.is_set():
                                break
                            ok = False
                            try:
                                ex_path = self._make_export_path(x.file_path, preset)
                                if select == 'Rename':
                                    ex_path = self._find_not_duplicate_filename(ex_path)
                                if select == 'Overwrite':
                                    if os.path.isfile(ex_path): # あっても無くても'Overwrite'
                                        self.cache_system.delete_file(ex_path)
                                        os.remove(ex_path) # ほんとは消さなくても良さそうだけど、通知がおかしいので

                                resize_str = ""
                                if preset['size_mode'] == "Long Edge":
                                    _, _, width, height = core.get_exif_image_size(x.exif_data)
                                    if width >= height:
                                        resize_str = preset['size_value'] + "x"
                                    else:
                                        resize_str = "x" + preset['size_value']
                                if preset['size_mode'] == "Pixels": resize_str = preset['size_value']
                                if preset['size_mode'] == "Percentage": resize_str = preset['size_value'] + "%"

                                ex_r = rating_by_path.get(x.file_path, 0)
                                exfile = export.ExportFile(x.file_path, x.exif_data, export_rating=ex_r)
                                ok = exfile.write_to_file(
                                    ex_path,
                                    preset['quality'],
                                    resize_str,
                                    preset['sharpen']/100,
                                    preset['icc_profile'],
                                    preset['metadata'],
                                    preset['gps'],
                                    preset['dithering'],
                                    cancel_event=self._export_cancel_event,
                                )
                            except Exception:
                                logging.exception("export failed for %s", x.file_path)
                                ok = False
                            if ok:
                                exported_ok.append(ex_path)
                            done = i + 1
                            KVClock.schedule_once(
                                lambda dt, d=done: setattr(self, "export_done", d),
                                0,
                            )
                            if not ok:
                                break
                    finally:
                        done_paths = list(exported_ok)
                        KVClock.schedule_once(
                            lambda dt, ep=done_paths: self._export_finish_ui(dt, ep), 0
                        )

                self._export_thread = threading.Thread(target=_export_job, daemon=True)
                self._export_thread.start()

        def _make_export_path(self, path, preset):
            dirname, basename = os.path.split(path)
            basename_with_out_ext, ext = os.path.splitext(basename)
            if len(preset['output_path']) > 0:
                if preset['output_path'] != '/':
                    ex_path = os.path.join(dirname, preset['output_path'], basename_with_out_ext) + preset['format']
                else:
                    ex_path = os.path.join(preset['output_path'], basename_with_out_ext) + preset['format']
            else:
                ex_path = os.path.join(dirname, basename_with_out_ext) + preset['format']
            return ex_path

        def _find_not_duplicate_filename(self, path):
            addnum = -1
            while os.path.isfile(path) == True:
                path_with_out_ext, ext = os.path.splitext(path)
                path_with_out_ext = re.split('-[0-9]+$', path_with_out_ext)
                path = path_with_out_ext[0] + str(addnum) + ext
                addnum -= 1

            return path

        #--------------------------------
        # パラメータの全消去

        def on_reset_press(self):
            # パラメータバックアップ
            temp = self.primary_param['color_temperature_reset']
            tint = self.primary_param['color_tint_reset']
            Y = self.primary_param['color_Y']

            # 全消去
            self.primary_param.clear()

            # 初期化パラメータ設定
            params.set_image_param(self.primary_param, self.imgset.img)
            params.set_temperature_to_param(self.primary_param, temp, tint, Y)

            # マスク関連全消去
            is_mask2_enabled = self._is_mask2_enabled()
            self._disable_mask2()
            self.ids['mask_editor2'].clear_mask()
            if is_mask2_enabled:
                self._enable_mask2()
            
            # クロップエディタ起動時はそれの初期化も行う
            self.primary_effects[0]['crop'].reset2_crop_editor(self.primary_param)
            self.primary_effects[0]['crop'].reset_crop_editor()
            self.apply_effects_lv(0, 'crop') # 描画を走らせる

            # これでファイルが消えるはず
            self.save_current_sidecar()

            # widget更新
            self.set2widget_all(self.primary_effects, self.primary_param)

        #--------------------------------
        # Mask2関連

        def _is_mask2_enabled(self):
            return self.ids['mask_editor2'].opacity == 1 and self.ids['mask_editor2'].disabled == False

        def _set_disabled_for_ids(self, id_names, disabled):
            for id_name in id_names:
                widget = self.ids.get(id_name)
                if widget is not None:
                    widget.disabled = disabled

        def update_mask2_options_enabled(self):
            self.update_load_dependent_panels_enabled()
            editor = self.ids.get('mask_editor2')
            mask2_button = self.ids.get('mask2')
            mask2_panel = self.ids.get('mask2_panel')
            if editor is None or mask2_button is None or mask2_panel is None:
                return
            mask2_enabled = (
                mask2_button.state == 'down'
                and self.mask2_wait_full_load == False
                and editor.disabled == False
            )
            active_mask = editor.get_active_mask() if mask2_enabled else None
            created_mask = editor.get_created_mask() if mask2_enabled else None
            current_mask = created_mask if created_mask is not None else active_mask
            has_mask_context = mask2_enabled and current_mask is not None
            is_composit = bool(has_mask_context and current_mask.is_composit())
            class_name = current_mask.__class__.__name__ if current_mask is not None else ''
            is_freedraw = class_name == 'FreeDrawMask'
            # brush hardness は FreeDraw と Polyline で共有 (PolylineMask は線幅エッジの soft 制御に流用)
            has_brush_hardness = class_name in ('FreeDrawMask', 'PolylineMask')
            is_polyline = class_name == 'PolylineMask'
            is_face = class_name == 'FaceMask'
            mask_specific_enabled = has_mask_context and not is_composit

            mask2_panel.disabled = not has_mask_context
            self._set_disabled_for_ids(
                (
                    'switch_lens_modifier',
                    'checkbox_color_modification',
                    'checkbox_subpixel_distortion',
                    'checkbox_geometry_distortion',
                ),
                mask2_button.state == 'down',
            )
            self._set_disabled_for_ids(
                (
                    'switch_fringe_removal',
                    ''
                    'slider_rca_purple_amount',
                    'slider_rca_green_amount',
                    'slider_rca_fringe_width',
                    'slider_rca_edge_threshold',
                ),
                mask2_button.state == 'down',
            )

            self._set_disabled_for_ids(
                (
                    'switch_mask2_draw_effects',
                    'switch_rca',
                    'slider_mask2_color_dodge',
                    'slider_mask2_color_burn',
                    'slider_mask2_mix_black',
                    'slider_mask2_mix_white',
                    'slider_mask2_skin_smooth_amount',
                    'slider_mask2_skin_smooth_radius_bias',
                ),
                not has_mask_context,
            )
            self._set_disabled_for_ids(
                (
                    'switch_mask2_settings',
                    'checkbox_mask2_invert',
                    'switch_mask2_depth',
                    'slider_mask2_depth_min',
                    'slider_mask2_depth_max',
                    'switch_mask2_hue',
                    'slider_mask2_hue_distance',
                    'slider_mask2_hue_min',
                    'slider_mask2_hue_max',
                    'switch_mask2_lum',
                    'slider_mask2_lum_distance',
                    'slider_mask2_lum_min',
                    'slider_mask2_lum_max',
                    'switch_mask2_sat',
                    'slider_mask2_sat_distance',
                    'slider_mask2_sat_min',
                    'slider_mask2_sat_max',
                    'switch_mask2_options',
                    'slider_mask2_blur',
                    'slider_mask2_open_space',
                    'slider_mask2_close_space',
                ),
                not mask_specific_enabled,
            )
            # Mask Geometry: Composit を含む has_mask_context のとき有効。
            # CompositMask は集約後のマスク画像に matrix を適用する。
            self._set_disabled_for_ids(
                (
                    'switch_mask_geometry',
                    'slider_mask_rotation',
                    'checkbox_mask_flip_h',
                    'checkbox_mask_flip_v',
                    'slider_mask_translation_x',
                    'slider_mask_translation_y',
                    'slider_mask_scale_x',
                    'slider_mask_scale_y',
                ),
                not has_mask_context,
            )
            self._set_disabled_for_ids(
                (
                    'slider_mask2_freedraw_brush_hardness',
                ),
                not (mask_specific_enabled and has_brush_hardness),
            )
            self._set_disabled_for_ids(
                (
                    'checkbox_mask2_polyline_fill',
                ),
                not (mask_specific_enabled and is_polyline),
            )
            self._set_disabled_for_ids(
                (
                    'switch_mask2_face',
                    'grid_mask2_face',
                    'checkbox_mask2_face_face',
                    'checkbox_mask2_face_brows',
                    'checkbox_mask2_face_eyes',
                    'checkbox_mask2_face_nose',
                    'checkbox_mask2_face_mouth',
                    'checkbox_mask2_face_lips',
                ),
                not (mask_specific_enabled and is_face),
            )
            self._set_disabled_for_ids(
                (
                    'checkbox_mask2_allow_over_one',
                    'checkbox_mask2_allow_under_zero',
                ),
                True,
            )

        def update_load_dependent_panels_enabled(self):
            # ファイル未選択時はプリセット／ヒストリ系パネルも無効化する。
            disabled = bool(self.mask2_wait_full_load) or not bool(self.image_loaded)
            for panel_name in ("preset_panel", "history_panel"):
                panel = getattr(self, panel_name, None)
                if panel is not None:
                    panel.disabled = disabled

        def _enable_mask2(self):
            self.ids['mask_editor2'].opacity = 1
            self.ids['mask_editor2'].disabled = False
            self.update_preview_texture_size()
            self.ids['mask_editor2'].set_texture_size(config.get_config('preview_width'), config.get_config('preview_height'))
            self.ids['mask_editor2'].set_primary_param(self.primary_param, params.get_disp_info(self.primary_param))
            self.ids['mask_editor2'].restore_last_active_mask()
            self.ids['mask_editor2'].update()
            self.update_mask2_options_enabled()

        def _disable_mask2(self):
            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self.ids['mask_editor2'].set_active_mask(None)
            self.ids['mask_editor2'].end()
            self.update_mask2_options_enabled()

        def on_mask2_press(self, value):
            if self.mask2_wait_full_load:
                self.ids['mask2'].state = 'normal'
                kvutils.find_widget(self, 'mask2_content_panel').disabled = True
                self._disable_mask2()
                return
            if value == "down":
                self._enable_mask2()
                kvutils.find_widget(self, 'mask2_content_panel').disabled = False
                # マスク Geometry モードへ遷移: Ge タブ上のクロップ枠/lens エディタを閉じる
                try:
                    if self.ids["effects"].current_tab.text == "Ge":
                        self.primary_effects[0]['geometry'].close_geometry_editor(self)
                        self.primary_effects[0]['crop'].sync_crop_editor_mode_from_widget(self, self.primary_param)
                except Exception:
                    pass
            else:
                kvutils.find_widget(self, 'mask2_content_panel').disabled = True
                self._disable_mask2()
                # マスク Geometry モードから画像 Geometry モードへ抜けるとき、
                # Ge タブ上で拡大表示中ならリセットする (画像 Geometry はズーム禁止のため)。
                try:
                    if self.is_zoomed and self.ids["effects"].current_tab.text == "Ge":
                        self.is_zoomed = False
                        self.click_x, self.click_y = 0, 0
                        self.drag_center_start = None
                        disp_info = core.convert_rect_to_info(
                            params.get_crop_rect(self.primary_param),
                            config.get_preview_texture_side() / max(self.primary_param['original_img_size']),
                        )
                        params.set_disp_info(self.primary_param, disp_info)
                        effects.reeffect_all(self.primary_effects, 1)
                        self.start_draw_image_and_crop(self.imgset)
                except Exception:
                    pass
                # 画像 Geometry モードへ復帰: Ge タブ上ならクロップ枠を再展開する
                try:
                    if self.ids["effects"].current_tab.text == "Ge":
                        self.apply_effects_lv(0, "crop")
                except Exception:
                    pass

        def set_mask2_hue_range(self, color_str):
            # イベント発火させる代入
            self.ids['slider_mask2_hue_min'].ids['slider'].value = core.HLS_COLOR_SETTING[color_str]['center'] - core.HLS_COLOR_SETTING[color_str]['width'][0] - core.HLS_COLOR_SETTING[color_str]['fade_width'][0]
            self.ids['slider_mask2_hue_max'].ids['slider'].value = core.HLS_COLOR_SETTING[color_str]['center'] + core.HLS_COLOR_SETTING[color_str]['width'][1] + core.HLS_COLOR_SETTING[color_str]['fade_width'][1]

        #--------------------------------

        def _enable_inpaint_edit(self):
            if self.inpaint_edit is None:
                self.update_preview_texture_size()
                self.inpaint_edit = widgets.bbox_viewer.BoundingBoxViewer(size=config.get_preview_texture_size(),
                                    initial_view=params.get_disp_info(self.primary_param),
                                    on_delete=self._on_inpaint_edit)
                self._set_diff_list_to_inpaint_edit()
                self.ids['preview_widget'].add_widget(self.inpaint_edit)
                #print(f"Inpaint x:{self.inpaint_edit.x}, y:{self.inpaint_edit.y}")
                #print(f"Preview x:{self.ids['preview'].x}, y:{self.ids['preview'].y}")
                #print(f"Mask2 x:{self.ids['mask_editor2'].x}, y:{self.ids['mask_editor2'].y}")

        def _disable_inpaint_edit(self):
            if self.inpaint_edit is not None:
                self.ids['preview_widget'].remove_widget(self.inpaint_edit)
                del self.inpaint_edit
                self.inpaint_edit = None

        def _set_diff_list_to_inpaint_edit(self):
            if self.inpaint_edit is not None:
                boxes = []
                for inpaint_diff in self.primary_param.get('inpaint_diff_list', []):
                    boxes.append(inpaint_diff.disp_info)
                self.inpaint_edit.set_boxes(boxes)

        def _on_inpaint_edit(self, deleted_index, deleted_box):
            self.begin_history_effect_ctrl(0, 'inpaint')
            self.primary_param['inpaint_diff_list'].pop(deleted_index)
            self.end_history_effect_ctrl(0, 'inpaint')
            self.apply_effects_lv(0, 'inpaint')

        def on_inpaint_edit_press(self, value):
            if value == "down":
                self._enable_inpaint_edit()
            else:
                self._disable_inpaint_edit()

        #--------------------------------

        def _enable_patchmatch_inpaint_edit(self):
            if self.patchmatch_inpaint_edit is None:
                self.update_preview_texture_size()
                self.patchmatch_inpaint_edit = widgets.bbox_viewer.BoundingBoxViewer(size=config.get_preview_texture_size(),
                                    initial_view=params.get_disp_info(self.primary_param),
                                    on_delete=self._on_patchmatch_inpaint_edit)
                self._set_diff_list_to_patchmatch_inpaint_edit()
                self.ids['preview_widget'].add_widget(self.patchmatch_inpaint_edit)
                #print(f"Inpaint x:{self.inpaint_edit.x}, y:{self.inpaint_edit.y}")
                #print(f"Preview x:{self.ids['preview'].x}, y:{self.ids['preview'].y}")
                #print(f"Mask2 x:{self.ids['mask_editor2'].x}, y:{self.ids['mask_editor2'].y}")

        def _disable_patchmatch_inpaint_edit(self):
            if self.patchmatch_inpaint_edit is not None:
                self.ids['preview_widget'].remove_widget(self.patchmatch_inpaint_edit)
                del self.patchmatch_inpaint_edit
                self.patchmatch_inpaint_edit = None

        def _set_diff_list_to_patchmatch_inpaint_edit(self):
            if self.patchmatch_inpaint_edit is not None:
                boxes = []
                for inpaint_diff in self.primary_param.get('patchmatch_inpaint_diff_list', []):
                    boxes.append(inpaint_diff.disp_info)
                self.patchmatch_inpaint_edit.set_boxes(boxes)

        def _on_patchmatch_inpaint_edit(self, deleted_index, deleted_box):
            self.begin_history_effect_ctrl(0, 'patchmatch_inpaint')
            self.primary_param['patchmatch_inpaint_diff_list'].pop(deleted_index)
            self.end_history_effect_ctrl(0, 'patchmatch_inpaint')
            self.apply_effects_lv(0, 'patchmatch_inpaint')

        def on_patchmatch_inpaint_edit_press(self, value):
            if value == "down":
                self._enable_patchmatch_inpaint_edit()
            else:
                self._disable_patchmatch_inpaint_edit()

        #--------------------------------

        def handle_for_dir_selection(self, selection):
            if selection is not None:
                # フォルダ切替で viewer.data は刷新されるが、MainWidget は直前の imgset を
                # 保持し続けてしまう。新フォルダから新たに選択されるまで編集 UI を無効化する。
                with threads.primary_param_lock:
                    self.save_current_sidecar()
                    effects.finalize_all(self.primary_effects, self.primary_param, self)
                    self.empty_image()  # image_loaded=False / imgset=None
                self._expected_file_path = None
                self._last_pmck_dict = None
                self._clear_exif_data()
                # フォルダ切替時は「何もロードしていない」状態に戻す。
                # 前回 on_select 直後にフォルダ切替が走った場合に loading=True が残るのを防ぐ。
                self.loading = False
                self._actively_loading = False
                self.update_mask2_options_enabled()
                config.set_config('import_path', selection[0].decode())

        #--------------------------------

        def on_lut_select_folder(self):
            device.FileChooser(title="Select LUT Folder", mode="dir", filters=[("CUBE Files", "*.cube")], on_selection=self.handle_for_lut).run()

        def handle_for_lut(self, selection):
            if selection is not None:
                path = selection[0].decode()
                config.set_config('lut_path', path)

        #--------------------------------

        def on_color_match_select_source(self):
            device.FileChooser(
                title="Select Color Match Source Image",
                mode="open",
                filters=[("Image Files", "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.webp")],
                on_selection=self.handle_for_color_match_source,
            ).run()

        def handle_for_color_match_source(self, selection):
            if selection is None or len(selection) == 0:
                return
            path = selection[0]
            if isinstance(path, bytes):
                path = path.decode()
            try:
                import io as _io
                import pyvips
                from PIL import ImageCms
                import cores.colour_functions as colour_functions
                with pyvips.Image.new_from_file(path) as vips_image:
                    long_side = max(vips_image.width, vips_image.height)
                    if long_side > 1024:
                        vips_image = vips_image.resize(1024.0 / long_side)
                    # imageset._load_rgb と同じ手順で ICC プロファイル名を取得
                    try:
                        icc_data = vips_image.get("icc-profile-data")
                        if icc_data is None:
                            src_icc_profile_name = "sRGB"
                        else:
                            profile = ImageCms.ImageCmsProfile(_io.BytesIO(icc_data))
                            src_icc_profile_name = ImageCms.getProfileDescription(profile).strip()
                    except Exception:
                        src_icc_profile_name = "sRGB"
                    arr = np.array(vips_image)
                if arr.ndim == 3 and arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                arr = core.convert_to_float32(arr)
                if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                src_space = core.ICC_PROFILE_TO_COLOR_SPACE.get(src_icc_profile_name, 'sRGB')
                arr = colour_functions.RGB_to_RGB(
                    arr, src_space, 'ProPhoto RGB', config.get_config('cat'),
                    apply_cctf_decoding=True, apply_gamut_mapping=True,
                ).astype(np.float32)
                # MKL を知覚均等空間で走らせるため、保存時点で sRGB ガンマエンコード済みにしておく
                import cores.color as color
                arr = color.srgb_gamma_encode(arr).astype(np.float32)
                arr = np.ascontiguousarray(arr)
            except Exception as e:
                logging.warning(f"on_color_match_select_source failed to load {path}: {e}")
                return

            if self.imgset is None:
                return

            self.begin_history_effect_ctrl(0, 'color_match')
            self.primary_param['color_match_source_image'] = arr
            self.apply_effects_lv(0, 'color_match')
            self.end_history_effect_ctrl(0, 'color_match')
            self.save_current_sidecar()

        #--------------------------------

        def on_current_tab(self, current):
            # 描画中の操作 (Polyline の途中など) はタブ切替で確定させる。
            try:
                self.ids['mask_editor2'].commit_in_progress()
            except Exception:
                pass

            if current.text == "Ge":
                # 画像 Geometry モード (Mask2 OFF) のときのみ拡大表示をリセットする。
                # マスク Geometry モード (Mask2 ON) のときは拡大表示を維持する。
                if not self._is_mask2_on():
                    self.is_zoomed = False
                # マスク Geometry モードに入った時は lens/four-points 系エディタも閉じる
                if self._is_mask2_on():
                    self.primary_effects[0]['geometry'].close_geometry_editor(self)
            else:
                self.primary_effects[0]['geometry'].close_geometry_editor(self)

            if self.imgset is not None:
                # apply_effects_lv(0, 'crop') 内の sync_crop_editor_mode_from_widget は
                # Mask2 ON 時はクロップエディタを閉じるよう CropEffect 側で対応済み。
                self.apply_effects_lv(0, "geometry")
                self.apply_effects_lv(0, "crop")
                self.apply_effects_lv(1, "distortion")



        def set_lut_path(self, path):
            lut_values = ['None']
            effects.LUTEffect.file_pathes = { 'None': None, }

            file_list = os.listdir(path)
            file_list.sort()
            for file_name in file_list:
                file_path = os.path.join(path, file_name)
                if file_name.lower().endswith(('.cube')):
                    lut_values.append(file_name)
                    effects.LUTEffect.file_pathes[file_name] = file_path
            self.ids['lut_spinner'].values = lut_values

        def _set_film_presets(self):
            presets = ['None']

            film_presets = film_emulator.emulator.get_presets()
            for preset in film_presets:
                presets.append(preset)

            self.ids['spinner_film_preset'].values = presets

        def _set_lens_presets(self):
            presets = ['None']
            for _key, data in CoatingSimulator().presets.items():
                presets.append(data['name'])
            self.ids['spinner_coating_preset'].values = presets

        def _clear_exif_data(self):
            for exif_id in (
                'exif_file_name',
                'exif_file_size',
                'exif_create_date',
                'exif_image_size',
                'exif_iso_speed',
                'exif_aperture',
                'exif_shutter_speed',
                'exif_exposure_compensation',
                'exif_flash',
                'exif_white_balance',
                'exif_focal_length',
                'exif_exposure_program',
                'exif_make',
                'exif_model',
                'exif_lens_model',
                'exif_software',
            ):
                self.ids[exif_id].value = '-'

            row = self.ids.get("exif_rating_row", None)
            if row is not None:
                row.rating = 0

        def _set_exif_data(self, exif_data, file_path=None):
            fp = file_path or (self.imgset.file_path if self.imgset is not None else None)
            if exif_data is not None and fp and rating_utils.is_rgb_path(fp):
                rating_io.merge_xmp_star_tags_into_exif(fp, exif_data)
            self.ids['exif_file_name'].value = exif_data.get("FileName", "-")
            self.ids['exif_file_size'].value = exif_data.get("FileSize", "-")
            self.ids['exif_create_date'].value = exif_data.get("CreateDate", "-")
            _, _, width, height = core.get_exif_image_size_with_orientation(exif_data)
            self.ids['exif_image_size'].value = str(width) + "x" + str(height)
            self.ids['exif_iso_speed'].value = str(exif_data.get("ISO", "-"))
            self.ids['exif_aperture'].value = str(exif_data.get("ApertureValue", exif_data.get("Aperture", "-")))
            self.ids['exif_shutter_speed'].value = str(exif_data.get("ShutterSpeedValue", "-"))
            self.ids['exif_exposure_compensation'].value = str(exif_data.get("ExposureCompensation", "-"))
            self.ids['exif_flash'].value = exif_data.get("Flash", "-")
            self.ids['exif_white_balance'].value = exif_data.get("WhiteBalance", "-")
            self.ids['exif_focal_length'].value = exif_data.get("FocalLength", "-")
            self.ids['exif_exposure_program'].value = exif_data.get("PictureMode", "-")
            self.ids['exif_make'].value = exif_data.get("Make", "-")
            self.ids['exif_model'].value = exif_data.get("Model", "-")
            self.ids['exif_lens_model'].value = exif_data.get("LensModel", "-")
            self.ids['exif_software'].value = exif_data.get("Software", "-")
            #self.ids['exif_'].value = exif_data.get("", "-")
            if self.imgset and self.imgset.file_path and rating_utils.is_rgb_path(self.imgset.file_path):
                self._rgb_xmp_rating_had[self.imgset.file_path] = rating_utils.exif_had_xmp_rating_tag(exif_data)
            self._sync_exif_rating_row()

        def _sync_exif_rating_row(self):
            row = self.ids.get("exif_rating_row", None)
            if row is None or self.imgset is None or not self.imgset.file_path:
                if row is not None:
                    row.rating = 0
                return
            fp = self.imgset.file_path
            r = 0
            for d in self.ids["viewer"].data:
                if d.get("file_path") == fp:
                    r = int(d.get("rating", 0) or 0)
                    break
            else:
                r = int(
                    rating_utils.effective_rating_display(
                        fp, self.primary_param.get("exif_data", {}) or {}, self.primary_param
                    )
                )
            row.rating = int(r)

        def apply_exif_pane_rating_slot(self, slot: int):
            if not self.imgset or not self.imgset.file_path:
                return
            cur = int(self.ids["exif_rating_row"].rating or 0)
            new_r = rating_utils.new_rating_on_slot_click(cur, int(slot))
            self.apply_paths_rating([self.imgset.file_path], new_r)

        def apply_paths_rating(self, file_paths, new_r: int):
            new_r = int(new_r) if new_r is not None else 0
            for fp in file_paths:
                if not fp:
                    continue
                if rating_utils.is_raw_path(fp):
                    # RAW: 星は viewer の data のみ即時更新。pmck へは他パラメータと同様、
                    # ファイル切替・エクスポート前・手動/終了保存など save_current_sidecar 経由のみ。
                    self.ids["viewer"].set_rating_for_path(fp, new_r)
                    if self.imgset and self.imgset.file_path == fp:
                        with threads.primary_param_lock:
                            self.primary_param.pop("rating", None)
                else:
                    had = self._rgb_xmp_rating_had.get(fp, rating_utils.exif_had_xmp_rating_tag(
                        (self.primary_param.get("exif_data") or {}) if (self.imgset and self.imgset.file_path == fp) else (self._viewer_exif_for_path(fp) or {})))
                    try:
                        rating_io.write_rgb_file_xmp_rating(fp, new_r, had)
                    except Exception as e:
                        logging.exception("RGB rating")
                        rating_io.notify_write_error(str(e))
                        continue
                    exif = None
                    if fp in self.cache_system.cache:
                        exif = self.cache_system.cache[fp][1]
                        rating_io.update_exif_dict_after_rgb_write(exif, new_r, had)
                    with threads.primary_param_lock:
                        if self.imgset and self.imgset.file_path == fp and self.primary_param.get("exif_data"):
                            rating_io.update_exif_dict_after_rgb_write(self.primary_param["exif_data"], new_r, had)
                    if new_r > 0:
                        self._rgb_xmp_rating_had[fp] = True
                    elif new_r == 0 and had:
                        self._rgb_xmp_rating_had[fp] = True
                    else:
                        self._rgb_xmp_rating_had[fp] = False
                    for d in self.ids["viewer"].data:
                        if d.get("file_path") == fp and exif is not None:
                            d["exif_data"] = exif
                            break
                    self.ids["viewer"].set_rating_for_path(fp, new_r)
            self._sync_exif_rating_row()

        def _viewer_exif_for_path(self, file_path: str):
            for d in self.ids["viewer"].data:
                if d.get("file_path") == file_path:
                    return d.get("exif_data")
            return None

        def _export_snapshot_rating(self, file_path: str) -> int:
            """Export 用の 0～5。選択カードは常に viewer にある。"""
            return self._viewer_snapshot_rating(file_path)

        def request_export_cancel(self):
            self._export_cancel_event.set()

        def _export_finish_ui(self, dt=None, exported_ok=None):
            self.export_in_progress = False
            self.export_done = 0
            self.export_total = 0
            self._export_thread = None
            viewer = self.ids['viewer']
            # エクスポート直後: 先に vips 作成→のち exiftool で星、watch より前にサムネ載せが走ると 0 星のまま
            for p in exported_ok or []:
                if p:
                    viewer.refresh_exif_for_exported_path(p)
            if exported_ok:
                retry_paths = list(exported_ok)
                KVClock.schedule_once(
                    lambda _dt, pl=retry_paths: self._export_retry_viewer_exif(list(pl)), 0.35
                )
            fp = self.imgset.file_path if self.imgset is not None else None
            if fp:
                viewer.set_selection_silent(fp)
            else:
                viewer.clear_selection()

        def _export_retry_viewer_exif(self, paths):
            v = self.ids["viewer"]
            for p in paths or []:
                if p:
                    v.refresh_exif_for_exported_path(p)

        def on_export_bar_press(self):
            if self.export_in_progress:
                self.request_export_cancel()
            else:
                self.on_export_press()

        def shutdown(self):
            #self.processor.stop()
            viewer = self.ids.get("viewer")
            if viewer and getattr(viewer, "watch_directory", None):
                preset_utils.cleanup_pmck_backup_files(viewer.watch_directory)
            for op in getattr(self.history, "operations", []):
                if getattr(op, "type", None) == "BatchPaste":
                    for item in getattr(op, "batch_items", []):
                        preset_utils.finalize_batch_item(item)
            if self.async_worker:
                self.async_worker.stop()
            
            if self.apply_thread is not None:
                t = self.apply_thread
                self.apply_thread = None
                self.draw_event.set()  # 待機中のスレッドを起こす
                t.join()

        def resize(self):
            self.sync_preview_widget_min_size()
            self._clamp_window_to_preview_minimum()
            preview_changed = self.update_preview_texture_size()
            if getattr(define, "RESIZE_DEBUG", False):
                KVClock.schedule_once(lambda _dt: self._update_resize_debug_display(), 0.05)
            if self.imgset is not None and self.imgset.img is not None:
                h, w = self.imgset.img.shape[:2]
                self.preview_size = [kvutils.dpi_scale_width(w), kvutils.dpi_scale_height(h)]
            self.ids["transform_wrapper"].scale = device.dpi_scale()
            self.ids["transform_wrapper"].center = self.ids['preview_widget'].center
            self.refresh_preview_overlays()
            if preview_changed and self.imgset is not None and self.imgset.img is not None:
                self.start_draw_image_and_crop(self.imgset)
                
        def on_key_down(self, window, key, scancode, codepoint, modifier):
            #print(f"key:{key}, scancode:{scancode}, codepoint:{codepoint}, modifier:{modifier}")

            if key == 32:
                if self.is_press_space == False:
                    self.sync_draw_image_and_crop(self.imgset)
                self.is_press_space = True
                return True

            if (key == 115 and ('ctrl' in modifier or 'meta' in modifier)):  # Sキー
                self.save_current_sidecar()
                return True

            if (key == 99 and ('ctrl' in modifier or 'meta' in modifier)):  # Cキー
                self.copy_effect_settings()
                return True

            if (key == 118 and ('ctrl' in modifier or 'meta' in modifier)):  # Vキー
                self.paste_effect_settings()
                return True
                                
            if (key == 122 and ('shift' not in modifier) and ('ctrl' in modifier or 'meta' in modifier)):  # Zキー
                self._undo()
                return True
                    
            if (key == 122 and ('shift' in modifier) and ('ctrl' in modifier or 'meta' in modifier)):  # shift-Zキー
                self._redo()
                return True

        def on_key_up(self, window, key, *args):
            if key == 32:
                self.is_press_space = False
                self.sync_draw_image_and_crop(self.imgset)
                return True
        
    class MainApp(MDApp):
        def __init__(self, cache_system, **kwargs):
            super(MainApp, self).__init__(**kwargs)
            
            self.title = define.APPNAME
            self.theme_cls.theme_style = 'Dark'
            self.theme_cls.primary_palette = 'Blue'
            
            self.cache_system = cache_system
            self._setup_window_handle = None

        def build(self):
            self.main_widget = MainWidget(self.cache_system)

            config.init_config(self.main_widget)
            config.load_config()
            
            # Start worker after config is loaded
            self.main_widget.async_worker.start()

            # window setup
            self._setup_window_handle = KVClock.schedule_interval(self._setup_window, 0.02)
            
            return self.main_widget

        def _setup_window(self, dt):
            if device.set_window_autosave(define.APPNAME, "PlatypusMainWindow"):
                KVClock.unschedule(self._setup_window_handle)
                self.on_window_resize(KVWindow, KVWindow.width, KVWindow.height)
        
        def on_start(self):
            self.main_widget.on_start()

            KVWindow.bind(on_resize=self.on_window_resize)
            """
            display = device.get_current_display()
            KVWindow.size = (display["width"] * 0.9, display["height"] * 0.9)
            KVWindow.left = (display["width"] - display["width"] * 0.9) // 2
            KVWindow.top = (display["height"] - display["height"] * 0.9) // 2
            """
            #close_splash_screen()
            return super().on_start()

        def on_stop(self):
            self.main_widget.request_export_cancel()
            t = self.main_widget._export_thread
            if t is not None and t.is_alive():
                t.join(timeout=3.0)
            self.main_widget._export_finish_ui()
            self.main_widget.save_current_sidecar()
            self.main_widget.shutdown()

        def on_window_resize(self, window, width, height):
            kvutils.traverse_widget(self.root)
            self.main_widget.resize()

        def _widget_pos(self, root, pos):
            kvutils.traverse_widget(root)
            self.main_widget.resize()

if __name__ == '__main__':
    from kivy.factory import Factory
    from widgets.rating_row import RatingRow
    Factory.register("RatingRow", cls=RatingRow)
    # 処理中ダイアログ作成
    create_processing_dialog()

    # PILイメージプラグイン抑制
    pillow_init()
    
    # メインプロセスでマネージャーを作成
    cache_system = file_cache_system.FileCacheSystem(max_cache_size=100, max_concurrent_loads=20)
        
    # ここでシステムを使用...
    MainApp(cache_system).run()
        
    # 終了時にクリーンアップ
    cache_system.shutdown()
