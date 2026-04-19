
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

    import threading
    import threads

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
    import macos as device

    import cores.film_emulator as film_emulator
    from cores.coating_simulator import CoatingSimulator
    import config
    import export
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
    from widgets.export_dialog import ExportDialog, ExportConfirmDialog
    import widgets.collapsible_box
    import widgets.compact_switch

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

    class MainWidget(MDBoxLayout):
        loading = KVBooleanProperty(False)
        preview_size = KVListProperty([100, 100])
        is_processing = KVBooleanProperty(False)
        export_in_progress = KVBooleanProperty(False)
        export_done = KVNumericProperty(0)
        export_total = KVNumericProperty(0)

        def __init__(self, cache_system, **kwargs):
            super(MainWidget, self).__init__(**kwargs)

            self.texture = None
            self.imgset = None
            self.click_x = 0
            self.click_y = 0        
            self.crop_image = None
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

            self.async_worker = AsyncWorker()
            # self.async_worker.start() # Start explicitly after config init
            self.processor = pipeline.AsyncPipelineManager(self.async_worker)
            KVClock.schedule_interval(self.update_async_results, 0.1)
            self.pipeline_version = 0
            
            self.apply_draw_image_center = None
            self.draw_event = threading.Event()
            self.apply_thread = threading.Thread(target=self.draw_image, daemon=False)
            self.apply_thread.start()
            self.enabledelay = None
            self._actively_loading = False  # ファイル選択によるロード中フラグ（起動時のloading: Trueとは別管理）

            self.history = history.History()
            self.current_op = None

            self.run_set2widget_all = False

            self.is_press_space = False
            # on_select で選んだパス。FCS の遅延コールバックが別ファイル向けなら無視する（primary_param と imgset の不整合防止）
            self._expected_file_path = None

            self._export_cancel_event = threading.Event()
            self._export_thread = None

            KVWindow.bind(on_key_down=self.on_key_down)
            KVWindow.bind(on_key_up=self.on_key_up)

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

        def on_kv_post(self, *args, **kwargs):
            super(MainWidget, self).on_kv_post(*args, **kwargs)

            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self._set_film_presets()
            self._set_lens_presets()

            self.mask2_panel = mask2_content.create_mask2_content_panel(self.ids['mask_editor2'])
            self.ids['masks_box'].add_widget(self.mask2_panel)
            #self.ids['masks_box'].ids['content'].add_widget(self.mask2_panel)

            self.history_panel = history_content.create_history_content_panel(self._on_history_selected)
            self.ids['history_box'].add_widget(self.history_panel)
            #self.ids['history_box'].ids['content'].add_widget(self.history_panel)

        def empty_image(self):
            with threads.primary_param_lock:
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
        
        def start_draw_image_and_crop(self, imgset, center_pos=None):
            if self.imgset is imgset:
                self.crop_image = None
                self.start_draw_image(center_pos)

        def sync_draw_image_and_crop(self, imgset):
            if self.imgset is imgset:
                self.crop_image = None
                self.pipeline_version += 1
                self.draw_image_core()

        @kvmainthread
        def blit_image(self, img, dt=0):
            logging.debug("[PERF] blit_image: Start. Time: %s", time.time())
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

            self.resize()

            #Singnalを送る
            import signals
            signals.blit_image.emit()

        @kvmainthread
        def draw_histogram_view(self, hist_data):
            #logging.debug(f"draw_histogram_view")
            self.ids["histogram"].draw_histogram_from_data(hist_data)

        def draw_image_core(self, center_pos=None):
            with threads.primary_param_lock:
                if (self.imgset is not None) and (self.imgset.img is not None):
                    if not params.has_original_img_size(self.primary_param):
                        logging.warning("draw_image_core: original_img_size 未定義のため描画しません")
                        return

                    img, self.crop_image = pipeline.process_pipeline(self.imgset.img, self.crop_image, self.is_zoomed, config.get_config('preview_width'), config.get_config('preview_height'), self.click_x, self.click_y, self.primary_effects, self.primary_param, self.ids['mask_editor2'], self.processor, self.pipeline_version, current_tab=self.ids["effects"].current_tab.text, loading_flag=pipeline_loading_flag(self.imgset), is_drag=self.is_press_space, center_pos=center_pos)
                    logging.debug("[PERF] draw_image_core: process_pipeline finished. Time: %s", time.time())
                    if img is None:
                        return

                    img = np.array(img)
                    utils.print_nan_inf(img, "output")

                    src_space = getattr(self.imgset, 'color_space', 'ProPhoto RGB')
                    img = colour_functions.RGB_to_RGB(img, src_space, config.get_config('display_color_gamut'), config.get_config('cat'),
                                            apply_cctf_decoding=False, apply_cctf_encoding=True, apply_gamut_mapping=True).astype(np.float32)

                    # ヒストグラム表示
                    img_hist, exclude_count = core.apply_zero_wrap(img, self.primary_param)
                    hist_data = widgets.histogram.HistogramWidget.calculate_histogram_data(img_hist, 0, exclude_count)
                    self.draw_histogram_view(hist_data)

                    # プレビュー表示
                    img_draw = core.apply_out_of_range_exposure(img, self.ids['toggle_overexposure'].state == 'down', self.ids['toggle_underexposure'].state == 'down')
                    img_draw, _ = core.apply_zero_wrap(img_draw, self.primary_param)
                    img_draw = np.clip(img_draw, 0, 1)

                    #描画をスケジューリング
                    self.blit_image(img_draw)
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
                    self.draw_image_core(center_pos)
                    last_processed_version = current_version
            
        def start_draw_image(self, center_pos=None):
            self.pipeline_version += 1
            self.apply_draw_image_center = center_pos
            self.processor.set_pipeline_version(self.pipeline_version)
            self.draw_event.set()

        def sync_draw_image(self):
            self.pipeline_version += 1
            self.draw_image_core()
                
        def crop_editing(self):
            self.apply_effects_lv(4, 'vignette')

        def lens_modifier_callback(self):
            self.primary_effects[0]['lens_modifier'].set2widget(self, self.primary_param)

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

        def _get_active_effects(self, mask_id=None, lv=None):
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
                    # Mask2パラメータは常に自分自身
                    pass
                else: 
                    # それ以外は親のCompositMaskへ（自分がCompositMaskなら自分へ）
                    if not mask.is_composit():
                        composit_mask = self.ids['mask_editor2'].find_composit_mask(mask)
                        if composit_mask is not None:
                            mask = composit_mask

            return (composit_mask.effects, mask.effects_param, mask.mask_id)
        
        def apply_effects_lv(self, lv, effect, sync=False):
            if self.run_set2widget_all == True:
                return
                
            current_effects, current_param, mask_id = self._get_active_effects(lv=lv)
            if effect is None:
                effects.set2param_all(current_effects, current_param, self)
            else:
                effect = effect if isinstance(effect, list) else [effect]
                for e in effect:
                    current_effects[lv][e].set2param(current_param, self)
            self.ids['mask_editor2'].set_draw_mask(lv == 3)
            #self.apply_rotation_flip_for_wrapper()
            if sync == False:
                self.start_draw_image()
            else:
                self.sync_draw_image()

        def set_effect_param(self, lv, effect, arg):
            if self.run_set2widget_all == True:
                return

            current_effects, current_param, _ = self._get_active_effects(lv=lv)
            current_effects[lv][effect].set2param2(current_param, arg)
            self.ids['mask_editor2'].set_draw_mask(lv == 3)
            #self.apply_rotation_flip_for_wrapper()
            self.start_draw_image()

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
            
            # セーブしないパラメータ（メタデータ等）と、維持するパラメータ（クロップ、色収差等）は維持する
            temp_param = {}
            params.copy_special_param(temp_param, self.primary_param)
            params.copy_remain_param(temp_param, self.primary_param)
            
            self.primary_param.clear()
            self.primary_param.update(temp_param)
            
            # 初期化パラメータ設定
            params.set_image_param(self.primary_param, self.imgset.img)

            # マスク関連全消去
            self.ids['mask2'].state = 'normal' # マスクモードを抜けないとおかしくなる
            self._disable_mask2()
            self.ids['mask_editor2'].clear_mask()
            
            # クロップエディタ起動時はそれの初期化も行う
            self.primary_effects[0]['crop'].reset2_crop_editor(self.primary_param)
            self.primary_effects[0]['crop'].reset_crop_editor()
            self.apply_effects_lv(0, 'crop') # 描画を走らせる

            # これでファイルが消えるはず
            self.save_current_sidecar()
           
            # UIと表示の更新
            effects.set2widget_all(self, self.primary_effects, self.primary_param)

        def begin_history_effect_ctrl(self, lv, effect, subname=None):
            current_effects, current_param, mask_id = self._get_active_effects(lv=lv)
            effect_list = effect if isinstance(effect, list) else [effect]
            self.current_op = history.Operation(lv, effect_list, subname, mask_id)
            self.current_op.set_backup(current_effects, current_param, subname)
            return True
        
        def end_history_effect_ctrl(self, lv, effect, subname=None):            
            effect_list = effect if isinstance(effect, list) else [effect]
            
            if self.current_op is None:
                logging.warning(f"MainWidget.end_history_effect_ctrl None. {effect_list}")
                return
            
            if self.current_op.subname != subname:
                logging.warning(f"MainWidget.end_history_effect_ctrl Subname Unmatching. {effect_list}")
                return

            if self.current_op.lv != lv or self.current_op.effect_list != effect_list:
                logging.warning(f"MainWidget.end_history_effect_ctrl LV or Effect Unmatching. {effect_list}")

            current_effects, current_param, mask_id = self._get_active_effects(self.current_op.mask_id, lv=lv)
            if self.current_op.set_update(current_effects, current_param, subname) is not None:
                self.history.append(self.current_op)
                self.history_panel.set_history(self.history)
                self.current_op = None

        def _undo(self):        
            if self.history.can_undo():
                if self.history.undo(self):
                    self.history_panel.set_history(self.history)
                    #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                    self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                    self._set_diff_list_to_inpaint_edit()
                    self.start_draw_image()

        def _redo(self):        
            if self.history.can_redo():
                if self.history.redo(self):
                    self.history_panel.set_history(self.history)
                    #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                    self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                    self._set_diff_list_to_inpaint_edit()
                    self.start_draw_image()

        def _on_history_selected(self, index):
            if index < self.history.current_index:
                n = self.history.current_index - index
                for _ in range(n):
                    self.history.undo(self)
                self.history_panel.set_history(self.history)
                #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                self._set_diff_list_to_inpaint_edit()
                self.start_draw_image()

            elif index >= self.history.current_index:
                n = index - self.history.current_index
                for _ in range(n):
                    self.history.redo(self)
                self.history_panel.set_history(self.history)
                #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                self._set_diff_list_to_inpaint_edit()
                self.start_draw_image()

        def reset_param(self, param):
            param.clear()

        def set2widget_all(self, _effects, param):
            if _effects is None:
                _effects = self.primary_effects
                param = self.primary_param

            self.run_set2widget_all = True
            effects.set2widget_all(self, _effects, param)
            self.run_set2widget_all = False

        def save_current_sidecar(self):
            if self.imgset is not None:
                param2 = effects.delete_default_param_all(self.primary_effects, self.primary_param) # プライマリのデフォルト値は消す
                param2['image_fidelity'] = getattr(self.imgset, 'fidelity', ImageFidelity.FULL).value
                result = params.save_json(self.imgset.file_path, param2, self.ids['mask_editor2'])
                if result == False:
                    # 失敗時はファイルを削除
                    params.delete_empty_param_json(self.imgset.file_path)
        
        @kvmainthread
        def on_select(self, card):
            logging.debug("[PERF] on_select: Start. Time: %s", time.time())
            # ロード開始
            self.loading = True
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
                self.cache_system.register_for_preload(card.file_path, card.exif_data, None, True)
                exif_data, _ = self.cache_system.get_file(card.file_path, lambda f1, f2, f3, f4, f5, f6: file_cache_system.run_method(self, "on_fcs_get_file", config._config, f1, f2, f3, f4, f5, f6))

                # とりあえずEXIF表示
                self._set_exif_data(exif_data)
            else:
                self._expected_file_path = None
                # カードなし（フォルダ空など）— get_file が呼ばれず loading が解除されないのを防ぐ
                self.loading = False
                self._actively_loading = False
        
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
                return

            if _load_stage_allows_ui(stage, imgset):
                self.loading = False
                self._actively_loading = False

            if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE):
                card = self.ids['viewer'].get_card(file_path)
                if card is not None:
                    # 一度も描画してないので値が設定されてない。暫定処置
                    self.ids['mask_editor2'].set_texture_size(config.get_config('preview_width'), config.get_config('preview_height'))
                    self.ids['mask_editor2'].set_primary_param(param, params.get_disp_info(param))

                    # フル解像のときだけ pmck から重い結果を復元（RAW プレビュー段階ではスキップ）
                    param['image_fidelity'] = getattr(imgset, 'fidelity', ImageFidelity.FULL).value
                    load_heavy = param['image_fidelity'] == ImageFidelity.FULL.value
                    params.load_json(file_path, param, self.ids['mask_editor2'], load_heavy=load_heavy)

            # Cancel previous background tasks
            if self.processor:
                self.processor.cancel_all()
                self.ids['histogram'].set_histogram_data(None)  # Reset histogram?

            with threads.primary_param_lock:
                if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE):
                    # １回目の時だけパラメータを反映して、編集できる様にする
                    self.primary_param.clear()
                    self.primary_param.update(param)
                    logging.debug("[PERF] on_fcs_get_file: Merged Params. Time: %s", time.time())
                    self.set2widget_all(self.primary_effects, self.primary_param)

                    # 特別あつかいでエディタを起動できるなら起動する
                    self.apply_effects_lv(1, 'distortion')
                    self.apply_effects_lv(0, 'crop')

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

                self.imgset = imgset

                fid = getattr(imgset, 'fidelity', ImageFidelity.FULL)
                self.primary_param['image_fidelity'] = fid.value
                prev_fid = getattr(self, '_last_image_fidelity', None)
                if fid == ImageFidelity.FULL and prev_fid == ImageFidelity.PREVIEW:
                    params.merge_heavy_from_pmck(file_path, self.primary_param, self.ids['mask_editor2'])
                self._last_image_fidelity = fid

                params.apply_original_geometry_if_missing(self.primary_param, imgset.img)
                if not params.has_original_img_size(self.primary_param):
                    logging.error("on_fcs_get_file: デコード画像からも original_img_size を確定できません")
                    self.loading = False
                    self._actively_loading = False
                    return

                effects.reeffect_all(self.primary_effects)
                self.start_draw_image_and_crop(imgset)

        def on_image_touch_down(self, touch):
            if self.ids['preview_widget'].collide_point(*touch.pos):
                # ズーム操作
                if touch.is_double_tap == True and self.ids["effects"].current_tab.text != "Ge":
                    self.is_zoomed = not self.is_zoomed
                    if self.is_zoomed == False:
                        self.click_x, self.click_y = 0, 0
                        self.drag_center_start = None
                        disp_info = core.convert_rect_to_info(params.get_crop_rect(self.primary_param), config.get_config('preview_size')/max(self.primary_param['original_img_size']))
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
                        self.start_draw_image_and_crop(self.imgset, center_pos=(new_cx, new_cy))

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
            dialog.bind(pos=MDApp.get_running_app()._widget_pos)
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

                def _export_job():
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

                                exfile = export.ExportFile(x.file_path, x.exif_data)
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
                            done = i + 1
                            KVClock.schedule_once(
                                lambda dt, d=done: setattr(self, "export_done", d),
                                0,
                            )
                            if not ok:
                                break
                    finally:
                        KVClock.schedule_once(self._export_finish_ui, 0)

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

        def _enable_mask2(self):
            self.ids['mask_editor2'].opacity = 1
            self.ids['mask_editor2'].disabled = False
            self.ids['mask_editor2'].set_texture_size(config.get_config('preview_width'), config.get_config('preview_height'))
            self.ids['mask_editor2'].set_primary_param(self.primary_param, params.get_disp_info(self.primary_param))
            self.ids['mask_editor2'].update()

        def _disable_mask2(self):
            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self.ids['mask_editor2'].set_active_mask(None)
            self.ids['mask_editor2'].end()

        def on_mask2_press(self, value):
            if value == "down":
                self._enable_mask2()
                kvutils.find_widget(self, 'mask2_content_panel').disabled = False
            else:
                kvutils.find_widget(self, 'mask2_content_panel').disabled = True
                self._disable_mask2()

        def set_mask2_hue_range(self, color_str):
            # イベント発火させる代入
            self.ids['slider_mask2_hue_min'].ids['slider'].value = core.HLS_COLOR_SETTING[color_str]['center'] - core.HLS_COLOR_SETTING[color_str]['width'][0] - core.HLS_COLOR_SETTING[color_str]['fade_width'][0]
            self.ids['slider_mask2_hue_max'].ids['slider'].value = core.HLS_COLOR_SETTING[color_str]['center'] + core.HLS_COLOR_SETTING[color_str]['width'][1] + core.HLS_COLOR_SETTING[color_str]['fade_width'][1]

        #--------------------------------

        def _enable_inpaint_edit(self):
            if self.inpaint_edit is None:
                self.inpaint_edit = widgets.bbox_viewer.BoundingBoxViewer(size=(config.get_config('preview_width'), config.get_config('preview_height')),
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
                self.patchmatch_inpaint_edit = widgets.bbox_viewer.BoundingBoxViewer(size=(config.get_config('preview_width'), config.get_config('preview_height')),
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
                config.set_config('import_path', selection[0].decode())

        #--------------------------------

        def on_lut_select_folder(self):
            device.FileChooser(title="Select LUT Folder", mode="dir", filters=[("CUBE Files", "*.cube")], on_selection=self.handle_for_lut).run()

        def handle_for_lut(self, selection):
            if selection is not None:
                path = selection[0].decode()
                config.set_config('lut_path', path)

        #--------------------------------

        def on_current_tab(self, current):
            if current.text == "Ge":
                self.is_zoomed = False

            else:
                self.primary_effects[0]['geometry'].close_geometry_editor(self)

            if self.imgset is not None:
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

        def _set_exif_data(self, exif_data):
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
        
        def request_export_cancel(self):
            self._export_cancel_event.set()

        def _export_finish_ui(self, *args):
            self.export_in_progress = False
            self.export_done = 0
            self.export_total = 0
            self._export_thread = None
            self.ids['viewer'].clear_selection()

        def on_export_bar_press(self):
            if self.export_in_progress:
                self.request_export_cancel()
            else:
                self.on_export_press()

        def shutdown(self):
            #self.processor.stop()
            if self.async_worker:
                self.async_worker.stop()
            
            if self.apply_thread is not None:
                t = self.apply_thread
                self.apply_thread = None
                self.draw_event.set()  # 待機中のスレッドを起こす
                t.join()

        def resize(self):
            # Update Property for KV Stencil
            if self.imgset is not None and self.imgset.img is not None:
                h, w = self.imgset.img.shape[:2]
                self.preview_size = [kvutils.dpi_scale_width(w), kvutils.dpi_scale_height(h)]
                self.ids["transform_wrapper"].scale = device.dpi_scale()
                self.ids["transform_wrapper"].center = self.ids['preview_widget'].center
                
        def on_key_down(self, window, key, scancode, codepoint, modifier):
            print(f"key:{key}, scancode:{scancode}, codepoint:{codepoint}, modifier:{modifier}")

            if key == 32:
                if self.is_press_space == False:
                    self.sync_draw_image_and_crop(self.imgset)
                self.is_press_space = True
                return True

            if (key == 115 and ('ctrl' in modifier or 'meta' in modifier)):  # Sキー
                self.save_current_sidecar()
                return True
                                
            if (key == 122 and ('shift' not in modifier) and ('ctrl' in modifier or 'meta' in modifier)):  # Zキー
                self._undo()
                    
            if (key == 122 and ('shift' in modifier) and ('ctrl' in modifier or 'meta' in modifier)):  # shift-Zキー
                self._redo()

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


