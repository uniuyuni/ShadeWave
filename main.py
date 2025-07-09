
#from splashscreen import display_splash_screen, close_splash_screen
#display_splash_screen("platypus.png")

if __name__ == '__main__':
    import matplotlib
    import tkinter as tk
    
    # tk.Tk()で落ちるのを回避するためのパッチ
    matplotlib.use('tkagg')
    tk = tk.Tk()
    tk.withdraw()
    tk.destroy()

    from kivymd.app import MDApp
    from kivymd.uix.boxlayout import MDBoxLayout
    from kivy.core.window import Window as KVWindow
    from kivy.graphics.texture import Texture as KVTexture
    from kivy.clock import Clock, mainthread
    from kivy.properties import BooleanProperty as KVBooleanProperty
    from functools import partial
    import numpy as np
    import re
    import multiprocessing
    import colour
    import logging

    import core
    import curve
    import effects
    import pipeline
    import param_slider
    import histogram_widget
    import metainfo
    import utils
    import kvutils
    import mask_editor2
    import color_picker
    import macos
    import film_emulator
    import lens_simulator
    import config
    import export
    from export_dialog import ExportDialog, ExportConfirmDialog
    import color
    import file_cache_system
    import hover_spinner
    import float_input
    import bbox_viewer
    import params
    from processing_dialog import create_processing_dialog
    import viewer_widget
    import imageset
    import define

import os
import jax
import cv2

import file_cache_system

os.environ['JAX_LOG_VERBOSITY'] = '0'
jax.config.update("jax_platform_name", "METAL")
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

    # プリコンパイル
    def precompile():
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        msk = np.ones((32, 32), dtype=np.float32)

        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
        hls = core.adjust_hls_color_one(hls, 'red', 0, 18/100, 0)

        core.fast_median_filter(rgb[..., 0])
        core.apply_mask(rgb, msk, rgb)


    class MainWidget(MDBoxLayout):
        loading: KVBooleanProperty(False)

        def __init__(self, cache_system, **kwargs):
            super(MainWidget, self).__init__(**kwargs)

            self.texture_height = config.get_config('preview_height')
            self.texture_width = config.get_config('preview_width')
            self.texture = None
            self.imgset = None
            self.click_x = 0
            self.click_y = 0        
            self.crop_image = None
            self.is_zoomed = False
            self.drag_start_point = None
            self.primary_param = {}
            self.primary_effects = effects.create_effects()
            #self.primary_effects[0]['crop'].set_editing_callback(self.crop_editing)
            self.apply_thread = None
            self.is_draw_image = False
            self.inpaint_edit = None
            self.cache_system = cache_system
            self.ids['viewer'].set_cache_system(self.cache_system)

            KVWindow.bind(on_key_down=self.on_key_down)

        def on_kv_post(self, *args, **kwargs):
            super(MainWidget, self).on_kv_post(*args, **kwargs)

            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self._set_film_presets()
            self._set_lens_presets()
    
        def empty_image(self):
            self.texture = KVTexture.create(size=(self.texture_width, self.texture_height), colorfmt='rgb', bufferfmt='float')
            self.texture.flip_vertical()
            self.ids["preview"].texture = None

            self.imgset = None
            self.click_x = 0
            self.click_y = 0
            self.is_zoomed = False
            self.crop_image = None

            self.reset_param(self.primary_param)
            self.ids['mask_editor2'].clear_mask()
        
        def start_draw_image_and_crop(self, imgset, offset=(0, 0)):
            if self.is_draw_image == False:
                if self.imgset == imgset:
                    self.crop_image = None
                    #effects.reeffect_all(self.primary_effects)
                    self.start_draw_image(offset)

        # @mainthread
        def blit_image(self, img):
            if config.get_config('display_output_dither'):
                img = core.jjn_dither_uint8(img)
                self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            else:
                self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')
            self.ids["preview"].texture = None # 更新のために必要
            self.ids["preview"].texture = self.texture

        def draw_histogram(self, img, blue_count=0, black_count=0):
            logging.debug(f"draw_histogram blue_count={blue_count}, black_count={black_count}")
            self.ids["histogram"].draw_histogram(img, blue_count, black_count)

        def draw_image(self, offset, dt):
            if (self.imgset is not None) and (self.imgset.img is not None):
                img, self.crop_image = pipeline.process_pipeline(self.imgset.img, offset, self.crop_image, self.is_zoomed, self.texture_width, self.texture_height, self.click_x, self.click_y, self.primary_effects, self.primary_param, self.ids['mask_editor2'])
                img = np.array(img)
                utils.print_nan_inf("output", img)

                # ヒストグラム表示
                img_hist, exclude_count = core.apply_zero_wrap(img, self.primary_param)
                img_hist = colour.RGB_to_RGB(img_hist, 'ProPhoto RGB', config.get_config('display_color_gamut'), config.get_config('cat'),
                                        apply_cctf_encoding=True, apply_gamut_mapping=False).astype(np.float32)
                self.draw_histogram(img_hist, 0, exclude_count)

                # プレビュー表示
                img_draw = core.apply_out_of_range_exposure(img, self.ids['toggle_overexposure'].state == 'down', self.ids['toggle_underexposure'].state == 'down')
                img_draw, _ = core.apply_zero_wrap(img_draw, self.primary_param)
                img_draw = np.clip(img_draw, 0, 1)
                img_draw = colour.RGB_to_RGB(img_draw, 'ProPhoto RGB', config.get_config('display_color_gamut'), config.get_config('cat'),
                                        apply_cctf_encoding=True, apply_gamut_mapping=True).astype(np.float32)
                self.blit_image(img_draw)

            self.is_draw_image = False

        def start_draw_image(self, offset=(0, 0)):
            if self.is_draw_image == False: #２重コール防止
                self.is_draw_image = True
                Clock.schedule_once(partial(self.draw_image, offset), -1)
            #self.apply_thread = threading.Thread(target=self.draw_image, daemon=True)
            #self.apply_thread.start()
        
        def crop_editing(self):
            self.apply_effects_lv(4, 'vignette')

        def apply_effects_lv(self, lv, effect):
            mask = self.ids['mask_editor2'].get_active_mask()
            if mask is None:
                self.primary_effects[lv][effect].set2param(self.primary_param, self)
            else:
                mask.effects[lv][effect].set2param(mask.effects_param, self)

            self.ids['mask_editor2'].set_draw_mask(lv == 3)
            self.start_draw_image()

        def set_effect_param(self, lv, effect, arg):
            mask = self.ids['mask_editor2'].get_active_mask()
            if mask is None:
                self.primary_effects[lv][effect].set2param2(self.primary_param, arg)
            else:
                mask.effects[lv][effect].set2param2(mask.effects_param, arg)

            self.ids['mask_editor2'].set_draw_mask(lv == 3)
            self.start_draw_image()
        
        def reset_param(self, param):
            param.clear()

        def set2widget_all(self, _effects, param):
            if _effects is None:
                _effects = self.primary_effects
                param = self.primary_param

            effects.set2widget_all(self, _effects, param)

        def save_current_sidecar(self):
            if self.imgset is not None:
                param2 = effects.delete_default_param_all(self.primary_effects, self.primary_param) # プライマリのデフォルト値は消す
                result = params.save_json(self.imgset.file_path, param2, self.ids['mask_editor2'])
                if result == False:
                    # 失敗時はファイルを削除
                    params.delete_empty_param_json(self.imgset.file_path)
        
        @mainthread
        def on_select(self, card):
            # ロード開始
            self.loading = True
            # 前の設定を保存
            self.save_current_sidecar()
            # 前のエフェクトを終了
            effects.finalize_all(self.primary_effects, self.primary_param, self)
            # 空のイメージをセット
            self.empty_image()

            if card is not None:
                self.cache_system.register_for_preload(card.file_path, card.exif_data, None, True)
                exif_data, _ = self.cache_system.get_file(card.file_path, lambda f1, f2, f3, f4, f5: file_cache_system.run_method(self, "on_fcs_get_file", f1, f2, f3, f4, f5))

                # とりあえずEXIF表示
                self._set_exif_data(exif_data)
        
        @mainthread
        def on_fcs_get_file(self, file_path, imgset, exif_data, param, flag):
            print(f"Load image SHAPE: {imgset.img.shape} FLAG: {imgset.flag}, Proc: {flag}")

            if flag == 0:
                # 最終的なパラメータを合成
                card = self.ids['viewer'].get_card(file_path)
                if card is not None:
                    # 一度も描画してないので値が設定されてない。暫定処置
                    self.ids['mask_editor2'].set_texture_size(config.get_config('preview_width'), config.get_config('preview_height'))
                    self.ids['mask_editor2'].set_primary_param(param, param['disp_info'])

                    # パラメータを読み込んで追加設定
                    params.load_json(file_path, param, self.ids['mask_editor2'])

                # １回目の時だけパラメータを反映して、編集できる様にする
                self.primary_param = param
                self.set2widget_all(self.primary_effects, param)
                self.apply_effects_lv(0, 'crop') # 特別あつかい

                # ロード終了
                self.loading = False
                
            self.imgset = imgset
            effects.reeffect_all(self.primary_effects)
            self.start_draw_image_and_crop(imgset)

        def on_image_touch_down(self, touch):
            if self.collide_point(*touch.pos):
                # ズーム操作
                if touch.is_double_tap == True and self.ids["effects"].current_tab.text != "Geometry":
                    self.is_zoomed = not self.is_zoomed
                    if self.is_zoomed == False:
                        self.click_x, self.click_y = 0, 0
                        self.primary_param['disp_info'] = None
                    else:
                        # ウィンドウ座標からローカルイメージ座標に変換
                        self.click_x, self.click_y = utils.to_texture(touch.pos, self.ids['preview'])

                    effects.reeffect_all(self.primary_effects, 1)
                    self.start_draw_image_and_crop(self.imgset)

                # ドラッグ操作
                elif self.is_zoomed == True:
                    self.drag_start_point = touch.pos

        def on_image_touch_move(self, touch):
            if self.collide_point(*touch.pos):
                if self.is_zoomed == True:
                    if self.drag_start_point != None:
                        offset_x = touch.pos[0] - self.drag_start_point[0]
                        offset_y = touch.pos[1] - self.drag_start_point[1]
                        offset_x = -offset_x
                        effects.reeffect_all(self.primary_effects, 1)
                        self.start_draw_image_and_crop(self.imgset, (offset_x, offset_y))

                        self.drag_start_point = touch.pos
                    
        def on_image_touch_up(self, touch):
            if self.is_zoomed == True:
                if self.drag_start_point != None:
                    self.drag_start_point = None

        def on_select_press(self):
            self.save_current_sidecar()
            macos.FileChooser(title="Select Folder", mode="dir", filters=[("Jpeg Files", "*.jpg")], on_selection=self.handle_for_dir_selection).run()

        def on_export_press(self):
            self.save_current_sidecar()

            dialog = ExportDialog(callback=self.handle_export_dialog)
            dialog.bind(pos=MDApp.get_running_app().on_widget_pos)
            dialog.open()

        def handle_export_dialog(self, preset):
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
                for x in cards:
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
                    exfile.write_to_file(ex_path, preset['quality'], resize_str, preset['sharpen']/100, preset['icc_profile'], preset['metadata'], preset['dithering'])

        def _make_export_path(seslf, path, preset):
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

        def on_reset_press(self):
            # パラメータバックアップ
            temp = self.primary_param['color_temperature_reset']
            tint = self.primary_param['color_tint_reset']
            Y = self.primary_param['color_Y']

            # 全消去
            self.primary_param = {}

            # 初期化パラメータ設定
            params.set_image_param(self.primary_param, self.imgset.img)
            params.set_temperature_to_param(self.primary_param, temp, tint, Y)

            # マスク関連全消去
            self._disable_mask2()
            self.ids['mask_editor2'].clear_mask()
            
            # クロップエディタ起動時はそれの初期化も行う
            self.primary_effects[0]['crop'].reset2_crop_editor(self.primary_param)
            self.primary_effects[0]['crop'].reset_crop_editor()
            self.apply_effects_lv(0, 'crop') # 描画を走らせる

            # これでファイルが消えるはず
            self.save_current_sidecar()

        #--------------------------------

        def _set_image_for_mask2(self, param):
            #self.ids['mask_editor2'].set_orientation(param.get('rotation', 0), param.get('rotation2', 0), param.get('flip_mode', 0))
            self.ids['mask_editor2'].set_texture_size(self.texture_width, self.texture_height)
            self.ids['mask_editor2'].set_primary_param(param, params.get_disp_info(param))
            self.ids['mask_editor2'].update()

        def _enable_mask2(self):
            self.ids['mask_editor2'].opacity = 1
            self.ids['mask_editor2'].disabled = False
            self._set_image_for_mask2(self.primary_param)
            #Clock.schedule_once(self._delay_set_image, -1)   # editor2のサイズが未決定なので遅らせる

        def _disable_mask2(self):
            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self.ids['mask_editor2'].set_active_mask(None)
            self.ids['mask_editor2'].end()

        def on_mask2_press(self, value):
            if value == "down":
                self._enable_mask2()
            else:
                self._disable_mask2()

        #--------------------------------

        def _enable_inpaint_edit(self):
            if self.inpaint_edit is None:
                self.inpaint_edit = bbox_viewer.BoundingBoxViewer(size=(config.get_config('preview_width'), config.get_config('preview_height')),
                                    initial_view=params.get_disp_info(self.primary_param),
                                    on_delete=self._on_inpaint_edit)
                boxes = []
                for inpaint_diff in self.primary_param.get('inpaint_diff_list', []):
                    boxes.append(inpaint_diff.disp_info)
                self.inpaint_edit.set_boxes(boxes)
                self.ids['preview_widget'].add_widget(self.inpaint_edit)
                #print(f"Inpaint x:{self.inpaint_edit.x}, y:{self.inpaint_edit.y}")
                #print(f"Preview x:{self.ids['preview'].x}, y:{self.ids['preview'].y}")
                #print(f"Mask2 x:{self.ids['mask_editor2'].x}, y:{self.ids['mask_editor2'].y}")

        def _disable_inpaint_edit(self):
            if self.inpaint_edit is not None:
                self.ids['preview_widget'].remove_widget(self.inpaint_edit)
                del self.inpaint_edit
                self.inpaint_edit = None

        def _on_inpaint_edit(self, deleted_index, deleted_box):
            self.primary_param['inpaint_diff_list'].pop(deleted_index)
            self.apply_effects_lv(0, 'inpaint')

        def on_inpaint_edit_press(self, value):
            if value == "down":
                self._enable_inpaint_edit()
            else:
                self._disable_inpaint_edit()

        #--------------------------------

        def handle_for_dir_selection(self, selection):
            if selection is not None:
                config.set_config('import_path', selection[0].decode())

        #--------------------------------

        def on_lut_select_folder(self):
            macos.FileChooser(title="Select LUT Folder", mode="dir", filters=[("CUBE Files", "*.cube")], on_selection=self.handle_for_lut).run()

        def handle_for_lut(self, selection):
            if selection is not None:
                path = selection[0].decode()
                config.set_config('lut_path', path)

        #--------------------------------

        def on_current_tab(self, current):
            if current.text == "Geometry":
                self.is_zoomed = False

            if self.imgset is not None:
                self.apply_effects_lv(0, "crop")


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

            for preset in lens_simulator.LensSimulator.LENS_PRESETS:
                presets.append(preset)

            self.ids['spinner_lens_preset'].values = presets

        def _set_exif_data(self, exif_data):
            self.ids['exif_file_name'].value = exif_data.get("FileName", "-")
            self.ids['exif_file_size'].value = exif_data.get("FileSize", "-")
            self.ids['exif_create_date'].value = exif_data.get("CreateDate", "-")
            _, _, width, height = core.get_exif_image_size_with_orientation(exif_data)
            self.ids['exif_image_size'].value = str(width) + "x" + str(height)
            self.ids['exif_iso_speed'].value = str(exif_data.get("ISO", "-"))
            self.ids['exif_aperture'].value = str(exif_data.get("ApertureValue", "-"))
            self.ids['exif_shutter_speed'].value = exif_data.get("ShutterSpeedValue", "-")
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
        
        def shutdown(self):
            pass
            
        def on_key_down(self, window, key, scancode, codepoint, modifier):
            print(f"key:{key}, scancode:{scancode}, codepoint:{codepoint}, modifier:{modifier}")
            if (key == 115 and ('ctrl' in modifier or 'meta' in modifier)):  # Sキー
                self.save_current_sidecar()
                return True
        
    class MainApp(MDApp):
        def __init__(self, cache_system, **kwargs):
            super(MainApp, self).__init__(**kwargs)
            
            self.title = define.APPNAME
            self.theme_cls.theme_style = 'Dark'
            self.theme_cls.primary_palette = 'Blue'
            
            self.cache_system = cache_system

        def build(self): 
            self.main_widget = MainWidget(self.cache_system)

            display = kvutils.get_current_dispay()
            KVWindow.size = (display["width"] * 0.9, display["height"] * 0.9)
            KVWindow.left = (display["width"] - display["width"] * 0.9) // 2
            KVWindow.top = (display["height"] - display["height"] * 0.9) // 2

            # testcode
            #self.main_widget.ids['viewer'].set_path(os.getcwd() + "/picture")

            config.set_main_widget(self.main_widget)
            config.load_config()

            return self.main_widget
        
        def on_start(self):
            KVWindow.bind(on_resize=self.on_window_resize)

            #close_splash_screen()
            return super().on_start()

        def on_stop(self):
            self.main_widget.save_current_sidecar()
            self.main_widget.shutdown()

        def on_window_resize(self, window, width, height):
            kvutils.traverse_widget(self.root)

        def on_widget_pos(self, root, pos):
            kvutils.traverse_widget(root)


if __name__ == '__main__':
    # 処理中ダイアログ作成
    create_processing_dialog()

    # マルチプロセスのサポートを有効にする
    multiprocessing.freeze_support()

    # PILイメージプラグイン抑制
    pillow_init()

    # プリコンパイル
    precompile()
    
    # メインプロセスでマネージャーを作成
    cache_system = file_cache_system.FileCacheSystem(max_cache_size=100, max_concurrent_loads=20)
        
    # ここでシステムを使用...
    MainApp(cache_system).run()
        
    # 終了時にクリーンアップ
    cache_system.shutdown()


