import os
import threading
import base64
import rawpy
import exiftool
import numpy as np
import cv2
from watchfiles import watch
import time
import pyvips

from kivymd.app import MDApp
from kivy.core.window import Window as KVWindow
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image as KVImage
from kivy.uix.label import Label as KVLabel
from kivymd.uix.card import MDCard
from kivy.graphics.texture import Texture as KVTexture
from kivy.properties import Property as KVProperty, StringProperty as KVStringProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty, BooleanProperty as KVBooleanProperty
from kivy.clock import mainthread as kvmainthread
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior

import define
import cores.core as core
import utils.kvutils as kvutils
from widgets.draggable_widget import DraggableWidget

class ThumbnailCard(RecycleDataViewBehavior, MDCard):
    file_path = KVStringProperty()
    thumb_source = KVObjectProperty(None, allownone=True, force_dispatch=True)
    rating = KVNumericProperty(0)
    selected = KVBooleanProperty(False)
    ctx = KVObjectProperty(None)
    index = KVNumericProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.exif_data = None
        self.orientation = 'vertical'
        self.size_hint = (None, 1)
        self.md_bg_color = [0.1, 0.1, 0.1, 1]
        self.radius = [5, 5, 5, 5]
        self.elevation = 2

        vbox = MDBoxLayout(orientation='vertical')
        vbox.ref_padding = 8

        # サムネイル表示
        self.image = KVImage(source='assets/spinner.gif', size_hint_y=0.7, anim_delay=0.02)
        vbox.add_widget(self.image)

        # ファイル名ラベル
        self.label = KVLabel(text="", bold=True, font_size='9pt', size_hint_y=0.3)
        vbox.add_widget(self.label)

        self.add_widget(vbox)

        self.bind(file_path=self.update_filename)

    def on_parent(self, instance, value):
        self._set_width()
    
    def on_size(self, instance, value):
        self._set_width()
    
    def _set_width(self):
        if self.parent:
            self.width = self.parent.height * 0.7

    def update_filename(self, instance, value):
        if value:
            self.label.text = os.path.basename(value)

    def refresh_view_attrs(self, rv, index, data):
        """ Catch and handle the view changes """
        self.index = index
        self._set_width()
        return super(ThumbnailCard, self).refresh_view_attrs(rv, index, data)

    def on_selected(self, instance, value):
        self.md_bg_color = [0.8, 0.8, 0.8, 1] if value else [0.1, 0.1, 0.1, 1]

    def on_thumb_source(self, instance, thumb):
        if thumb is None:
            self.image.source = 'assets/spinner.gif'
            self.image.texture = None
            return

        self.texture = KVTexture.create(size=(thumb.shape[1], thumb.shape[0]), colorfmt='rgb', bufferfmt='ushort')
        self.texture.flip_vertical()
        self.texture.blit_buffer(thumb.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.image.source = ''
        #self.image.size = (thumb.shape[1], thumb.shape[0])
        self.image.texture = self.texture

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.ctx:
                self.ctx.handle_selection(self.index, touch)
                return True
        return super().on_touch_down(touch)

class ViewerWidget(RecycleView, DraggableWidget):
    last_selected_index = KVNumericProperty(None, allownone=True)
    cols = KVNumericProperty(4)
    thumb_width = KVNumericProperty(120*2)
    
    # Selection state
    selected_indices = set()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []
        self.watch_directory = None

        threading.Thread(target=self._watchfiles_thread, daemon=True).start()
        KVWindow.bind(on_key_down=self.on_key_down)

    def _watchfiles_thread(self):
        action_type_map = {
            1: self._added_file,
            2: self._modified_file,
            3: self._deleted_file,
        }

        while True:
            watch_directory = self.watch_directory
            if watch_directory is not None:
                try:
                    for changes in watch(watch_directory):
                        for action, path in changes:
                            if action in action_type_map:
                                action_type_map[action](path)
                except Exception:
                    pass
            time.sleep(1)

    @kvmainthread
    def _added_file(self, file_path):
        if self.is_supported_image(file_path):
            file_list = [d['file_path'] for d in self.data]
            if file_path in file_list:
                return
            
            new_item = {
                'file_path': file_path,
                'thumb_source': None,
                'exif_data': None,
                'selected': False,
                'ctx': self,
            }
            
            idx = 0
            for i, d in enumerate(self.data):
                if d['file_path'] > file_path:
                    idx = i
                    break
                idx = i + 1
            
            self.data.insert(idx, new_item)
            self.cols = max(1, len(self.data))
            self.load_images({file_path: idx})

    @kvmainthread
    def _deleted_file(self, file_path):
        for i, d in enumerate(self.data):
            if d['file_path'] == file_path:
                self.data.pop(i)
                break
        self.cols = max(1, len(self.data))

    def _modified_file(self, file_path):
        pass

    def set_path(self, directory):
        self.data = []
        self.selected_indices.clear()
        self.last_selected_index = None

        file_list = os.listdir(directory)
        file_list.sort()
        
        new_data = []
        file_path_dict = {} # path -> index mapping for loader
        
        for i, file_name in enumerate(file_list):
            if self.is_supported_image(file_name):
                file_path = os.path.join(directory, file_name)
                new_data.append({
                    'file_path': file_path,
                    'thumb_source': None,
                    'exif_data': None,
                    'selected': False,
                    'ctx': self,
                })
                file_path_dict[file_path] = len(new_data) - 1

        self.data = new_data
        self.cols = max(1, len(self.data)) # Not used for logic, but might be used by UI binding?
        
        self.load_images(file_path_dict)
        self.watch_directory = directory

    def load_images(self, file_path_dict):
        if len(file_path_dict) > 0:
            threading.Thread(target=self.load_images_thread, args=(file_path_dict, 16), daemon=True).start()

    def load_images_thread(self, file_path_dict, chunk_size):
        file_path_list = list(file_path_dict.keys())
        
        for i in range(0, len(file_path_list), chunk_size):
            chunk = file_path_list[i:i + chunk_size]
            
            with exiftool.ExifToolHelper(common_args=['-b', '-s']) as et:
                exif_data_list = et.get_metadata(chunk)
        
            thumb_data_list = self.process_exif_data(chunk, exif_data_list)
            
            updates = {}
            for k in range(len(chunk)):
                file_path = chunk[k]
                if file_path not in file_path_dict: continue
                idx = file_path_dict[file_path]
                
                if idx < len(self.data) and self.data[idx]['file_path'] == file_path:
                    item = self.data[idx]
                    item['thumb_source'] = thumb_data_list[k]
                    item['exif_data'] = exif_data_list[k]
                    updates[idx] = item
            
            self._apply_updates(updates)

    @kvmainthread
    def _apply_updates(self, updates):
        for idx, item in updates.items():
            self.data[idx] = item
        self.refresh_from_data()

    def is_supported_image(self, file_name):
        return file_name.lower().endswith(define.SUPPORTED_FORMATS_RGB) or file_name.lower().endswith(define.SUPPORTED_FORMATS_RAW)

    def process_exif_data(self, file_path_list, exif_data_list):
        thumb_data_list = []
        try:
            for i in range(len(file_path_list)):
                exif_data = exif_data_list[i]
                file_path = file_path_list[i]

                thumb_base64 = exif_data.get('ThumbnailImage', None)
                if thumb_base64 is not None:
                    image = np.frombuffer(base64.b64decode(thumb_base64[7:]), dtype=np.uint8)
                    thumb = cv2.imdecode(image, 1)
                    thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                else:
                    if file_path.lower().endswith(define.SUPPORTED_FORMATS_RAW):
                        with rawpy.imread(file_path) as raw:
                            thumb = raw.postprocess()
                    else:
                        with pyvips.Image.new_from_file(file_path) as vips_image:
                            thumb = np.array(vips_image)
                thumb = core.convert_to_float32(thumb)

                thumb_size = self._calc_resize_image((thumb.shape[1], thumb.shape[0]), self.thumb_width)
                thumb = cv2.resize(thumb, thumb_size)

                # Orientation
                orientation = exif_data.get('Orientation')
                if orientation is not None:
                    if orientation == 'Rotate 180':
                        thumb = cv2.rotate(thumb, cv2.ROTATE_180)
                    elif orientation == 'Rotate 270 CW':
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif orientation == 'Rotate 90 CW':
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                    elif orientation == 'Mirror horizontal':
                        thumb = cv2.flip(thumb, 1)
                    elif orientation == 'Mirror vertical':
                        thumb = cv2.flip(thumb, 0)
                    elif orientation == 'Mirror horizontal and rotate 270 CW':
                        thumb = cv2.flip(thumb, 1)
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif orientation == 'Mirror horizontal and rotate 90 CW':
                        thumb = cv2.flip(thumb, 1)
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                
                thumb_data_list.append(thumb)

            return thumb_data_list

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return [None]*len(file_path_list)

    def handle_selection(self, index, touch):
        # We also need to notify MainWidget about selection change
        should_notify = False
        
        if not touch.is_mouse_scrolling and touch.button == 'left':
            if 'shift' in KVWindow.modifiers and self.last_selected_index is not None:
                if not( 'ctrl' in KVWindow.modifiers or 'meta' in KVWindow.modifiers ):
                    self.clear_selection()
                
                start = min(self.last_selected_index, index)
                end = max(self.last_selected_index, index)
                
                for i in range(start, end + 1):
                    self.select_at(i)
                should_notify = True
                    
            else:
                if 'ctrl' in KVWindow.modifiers or 'meta' in KVWindow.modifiers:
                    self.toggle_at(index)
                    should_notify = True # Toggle always changes selection
                else:
                    self.clear_selection()
                    self.select_at(index)
                    should_notify = True
                
                self.last_selected_index = index
            
            if should_notify:
                 self.notify_selection_change(index)

    def notify_selection_change(self, index):
        app = MDApp.get_running_app()
        if app and hasattr(app, 'main_widget'):
             # Create a mock card object for the newly selected item
             # If multiple items selected, MainWidget usually takes the last one or iterates.
             # MainWidget.on_select takes a single 'card' argument.
             selected_data = self.data[index]
             class MockCard:
                 def __init__(self, d):
                     self.file_path = d['file_path']
                     self.exif_data = d['exif_data']
             
             app.main_widget.on_select(MockCard(selected_data))

    def select_at(self, index):
        if 0 <= index < len(self.data):
            self.data[index]['selected'] = True
            self.selected_indices.add(index)
            self.refresh_from_data()

    def toggle_at(self, index):
        if 0 <= index < len(self.data):
            val = not self.data[index]['selected']
            self.data[index]['selected'] = val
            if val: self.selected_indices.add(index)
            else: self.selected_indices.discard(index)
            self.refresh_from_data()

    def clear_selection(self):
        for idx in self.selected_indices:
            if idx < len(self.data):
                self.data[idx]['selected'] = False
        self.selected_indices.clear()
        self.refresh_from_data()

    def get_selected_cards(self):
        res = []
        class MockCard:
             def __init__(self, d):
                 self.file_path = d['file_path']
                 self.exif_data = d['exif_data']
                 self.thumb_source = d['thumb_source']
        
        for idx in self.selected_indices:
            if idx < len(self.data):
                res.append(MockCard(self.data[idx]))
        return res

    def get_card(self, file_path):
        for d in self.data:
            if d['file_path'] == file_path:
                class MockCard:
                     def __init__(self, d):
                         self.file_path = d['file_path']
                         self.exif_data = d['exif_data']
                return MockCard(d)
        return None

    def set_cache_system(self, cache_system):
        self.cache_system = cache_system
        self.bind(scroll_x=self._request_current_view_cards)

    @kvmainthread
    def _request_current_view_cards(self, instance, value):
        pass

    def get_drag_files(self):
        file_paths = []
        for card in self.get_selected_cards():
            if card.thumb_source is not None:
                file_paths.append((card.file_path, (card.thumb_source * 255).astype(np.uint8)))
        return file_paths

    def _calc_resize_image(self, original_size, max_length):
        width, height = original_size
        if width > height:
            scale_factor = max_length / width
        else:
            scale_factor = max_length / height
        return (int(width * scale_factor), int(height * scale_factor))

    def on_scroll_start(self, touch, check_children=True):
        # マウスホイールの縦スクロールを横スクロールに変換する
        # touch.buttonを一時的に書き換え、super()呼び出し後に元に戻す
        # （touchオブジェクトは共有のため、他のウィジェットへの副作用を防ぐ）
        if touch.is_mouse_scrolling:
            original_button = touch.button
            if touch.button == 'scrolldown':
                touch.button = 'scrollright'
            elif touch.button == 'scrollup':
                touch.button = 'scrollleft'
            result = super().on_scroll_start(touch, check_children)
            touch.button = original_button  # 他のウィジェットのために元の値に戻す
            return result
        return super().on_scroll_start(touch, check_children)

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        if (key == 97 and ('ctrl' in modifier or 'meta' in modifier)):  # A
            self.clear_selection()
            for i in range(len(self.data)):
                self.select_at(i)
            # Notify MainWidget via last item?
            if self.data:
                self.notify_selection_change(len(self.data)-1)
            return True
