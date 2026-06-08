import os
import threading
import base64
import rawpy
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
from kivy.uix.floatlayout import FloatLayout
from kivymd.uix.card import MDCard
from kivy.graphics.texture import Texture as KVTexture
from kivy.properties import Property as KVProperty, StringProperty as KVStringProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty, BooleanProperty as KVBooleanProperty
from kivy.clock import Clock as KVClock
from kivy.clock import mainthread as kvmainthread
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior

import logging

import define
import cores.core as core
import utils.kvutils as kvutils
from utils import rating_utils
from utils import rating_io
from utils.exiftool_safe import safe_get_metadata
from widgets.draggable_widget import DraggableWidget
from widgets.rating_row import RatingRow
from utils.paths import rel
from utils import preset_utils


_PMCK_ICON_REF_SIZE = 12
_PMCK_ICON_MARGIN_REF = 2
_THUMBNAIL_CARD_WIDTH_RATIO = 0.7

class ThumbnailCard(RecycleDataViewBehavior, MDCard):
    file_path = KVStringProperty()
    thumb_source = KVObjectProperty(None, allownone=True, force_dispatch=True)
    rating = KVNumericProperty(0)
    pmck_exists = KVBooleanProperty(False)
    load_pending = KVBooleanProperty(True)
    selected = KVBooleanProperty(False)
    ctx = KVObjectProperty(None)
    index = KVNumericProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._bound_layout_parent = None
        self.exif_data = None
        self.orientation = 'vertical'
        self.size_hint = (None, 1)
        self.md_bg_color = [0.1, 0.1, 0.1, 1]
        self.radius = [5, 5, 5, 5]
        self.elevation = 2

        vbox = MDBoxLayout(orientation='vertical')
        vbox.ref_padding = 8

        # サムネイル表示
        self.image_box = FloatLayout(size_hint_y=0.62)
        self.image = KVImage(
            source=rel("assets", "spinner.gif"),
            anim_delay=0.02,
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
        )
        self.image_box.add_widget(self.image)
        self.pmck_icon = KVImage(
            source=rel("assets", "pmck_indicator.png"),
            size_hint=(None, None),
            size=(
                kvutils.dpi_scale_width(_PMCK_ICON_REF_SIZE),
                kvutils.dpi_scale_height(_PMCK_ICON_REF_SIZE),
            ),
            allow_stretch=True,
            keep_ratio=True,
            mipmap=True,
            opacity=0,
        )
        self.pmck_icon.ref_width = _PMCK_ICON_REF_SIZE
        self.pmck_icon.ref_height = _PMCK_ICON_REF_SIZE
        self.image_box.add_widget(self.pmck_icon)
        self.image_box.bind(pos=self._update_pmck_icon_layout, size=self._update_pmck_icon_layout)
        self.image.bind(
            pos=self._update_pmck_icon_layout,
            size=self._update_pmck_icon_layout,
            norm_image_size=self._update_pmck_icon_layout,
        )
        vbox.add_widget(self.image_box)

        # ファイル名ラベル
        self.label = KVLabel(
            text="",
            bold=True,
            font_size='9pt',
            size_hint_y=0.28,
            shorten=True,
            shorten_from="center",
            max_lines=1,
            halign="center",
            valign="middle",
        )
        self.label.bind(size=self.label.setter("text_size"))
        vbox.add_widget(self.label)

        self.rating_row = RatingRow(size_hint_y=0.1)
        vbox.add_widget(self.rating_row)

        self.add_widget(vbox)

        self.bind(file_path=self.update_filename)

    def on_parent(self, instance, value):
        if self._bound_layout_parent is not None:
            self._bound_layout_parent.unbind(height=self._set_width)
        self._bound_layout_parent = value
        if value is not None:
            value.bind(height=self._set_width)
        self._set_width()
        KVClock.schedule_once(lambda _dt: self._set_width(), 0)
    
    def on_size(self, instance, value):
        self._set_width()
    
    def _set_width(self, *_args):
        layout_height = self.parent.height if self.parent else self.height
        if layout_height <= 0:
            return
        width = layout_height * _THUMBNAIL_CARD_WIDTH_RATIO
        if abs(self.width - width) > 0.5:
            self.width = width

    def _update_pmck_icon_layout(self, *_args):
        if not hasattr(self, "pmck_icon"):
            return
        margin = kvutils.dpi_scale_width(_PMCK_ICON_MARGIN_REF)
        try:
            image_w, image_h = self.image.norm_image_size
        except (TypeError, ValueError):
            image_w, image_h = self.image_box.size
        if image_w <= 0 or image_h <= 0:
            image_w, image_h = self.image_box.size
        image_x = self.image_box.x + max(0, (self.image_box.width - image_w) / 2)
        image_y = self.image_box.y + max(0, (self.image_box.height - image_h) / 2)
        self.pmck_icon.pos = (
            image_x + image_w - self.pmck_icon.width - margin,
            image_y + margin,
        )

    def update_filename(self, instance, value):
        if value:
            self.label.text = os.path.basename(value)

    def refresh_view_attrs(self, rv, index, data):
        """ Catch and handle the view changes """
        self.index = index
        self._set_width()
        r = super(ThumbnailCard, self).refresh_view_attrs(rv, index, data)
        self.rating_row.rating = int(data.get("rating", 0) or 0)
        self.rating_row.card_index = index
        self.rating_row.ctx = data.get("ctx")
        self.rating_row.exif_pane = False
        self.pmck_exists = bool(data.get("pmck_exists", False))
        self.load_pending = bool(data.get("load_pending", False))
        self.pmck_icon.opacity = 1.0 if self.pmck_exists else 0.0
        self._update_pmck_icon_layout()
        return r

    def refresh_view_layout(self, rv, index, layout, viewport):
        r = super().refresh_view_layout(rv, index, layout, viewport)
        self._set_width()
        return r

    def on_selected(self, instance, value):
        self.md_bg_color = [0.8, 0.8, 0.8, 1] if value else [0.1, 0.1, 0.1, 1]

    def on_thumb_source(self, instance, thumb):
        if thumb is None:
            self.image.source = rel("assets", "spinner.gif")
            self.image.texture = None
            return

        self.texture = KVTexture.create(size=(thumb.shape[1], thumb.shape[0]), colorfmt='rgb', bufferfmt='ushort')
        self.texture.flip_vertical()
        self.texture.blit_buffer(thumb.tobytes(), colorfmt='rgb', bufferfmt='float')
        self.image.source = ''
        #self.image.size = (thumb.shape[1], thumb.shape[0])
        self.image.texture = self.texture
        self._update_pmck_icon_layout()

    def on_touch_down(self, touch):
        # 子（星スロット）へ先に伝播。ここで丸呑みするとタッチが RatingRow に届かない。
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if self.load_pending and not touch.is_mouse_scrolling and touch.button == 'left':
            return True
        for child in reversed(self.children):
            if child.dispatch("on_touch_down", touch):
                return True
        if self.ctx:
            self.ctx.handle_selection(self.index, touch)
            return True
        return super().on_touch_down(touch)

class ViewerWidget(RecycleView, DraggableWidget):
    last_selected_index = KVNumericProperty(None, allownone=True)
    cols = KVNumericProperty(4)
    card_width = KVNumericProperty(112)
    thumb_width = KVNumericProperty(120*2)
    
    # Selection state
    selected_indices = set()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []
        self.watch_directory = None
        self._card_width_layout_event = None

        threading.Thread(target=self._watchfiles_thread, daemon=True).start()
        KVWindow.bind(on_key_down=self.on_key_down)
        self.bind(height=self._schedule_card_width_sync)
        KVClock.schedule_once(lambda _dt: self._sync_card_width(), 0)

    def on_kv_post(self, base_widget):
        self._sync_card_width()

    def _schedule_card_width_sync(self, *_args):
        if self._card_width_layout_event is None:
            self._card_width_layout_event = KVClock.schedule_once(
                lambda _dt: self._sync_card_width(), 0
            )

    def _sync_card_width(self):
        self._card_width_layout_event = None
        if self.height <= 0:
            return
        width = max(1, self.height * _THUMBNAIL_CARD_WIDTH_RATIO)
        if abs(self.card_width - width) > 0.5:
            self.card_width = width
        layout = getattr(self, "layout_manager", None)
        if layout is None and self.children:
            layout = self.children[0]
        if layout is not None and hasattr(layout, "default_size"):
            layout.default_size = (width, None)
        self.refresh_from_layout()

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
        pmck_image_path = self._image_path_for_pmck_sidecar(file_path)
        if pmck_image_path is not None:
            self.set_pmck_indicator_for_path(pmck_image_path, True)
            return
        if self.is_supported_image(file_path):
            file_list = [d['file_path'] for d in self.data]
            if file_path in file_list:
                return
            
            new_item = {
                'file_path': file_path,
                'thumb_source': None,
                'exif_data': None,
                'load_pending': True,
                'selected': False,
                'ctx': self,
                'rating': 0,
                'pmck_exists': os.path.exists(file_path + ".pmck"),
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
        pmck_image_path = self._image_path_for_pmck_sidecar(file_path)
        if pmck_image_path is not None:
            self.set_pmck_indicator_for_path(pmck_image_path, False)
            return
        for i, d in enumerate(self.data):
            if d['file_path'] == file_path:
                self.data.pop(i)
                break
        self.cols = max(1, len(self.data))

    @kvmainthread
    def _modified_file(self, file_path):
        """
        エクスポート等で「先にファイル作成 → 後から exiftool で星」となると、
        追加 (watch) 時点では星が無い。追記後の modify でメタ＆星表示を再取得する。
        """
        pmck_image_path = self._image_path_for_pmck_sidecar(file_path)
        if pmck_image_path is not None:
            self.set_pmck_indicator_for_path(pmck_image_path, os.path.exists(file_path))
            return
        if not self.is_supported_image(file_path):
            return
        want = self._norm_path_key(file_path)
        for i, d in enumerate(self.data):
            if self._norm_path_key(d.get("file_path") or "") != want:
                continue
            fp = d["file_path"]
            self.load_images({fp: i})
            break

    def set_path(self, directory):
        preset_utils.cleanup_pmck_backup_files(directory)
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
                    'load_pending': True,
                    'selected': False,
                    'ctx': self,
                    'rating': 0,
                    'pmck_exists': os.path.exists(file_path + ".pmck"),
                })
                file_path_dict[file_path] = len(new_data) - 1

        self.data = new_data
        self.cols = max(1, len(self.data)) # Not used for logic, but might be used by UI binding?
        
        self.load_images(file_path_dict)
        self.watch_directory = directory

    def load_images(self, file_path_dict):
        if len(file_path_dict) > 0:
            self._set_load_pending(file_path_dict, True)
            threading.Thread(target=self.load_images_thread, args=(file_path_dict, 16), daemon=True).start()

    @kvmainthread
    def _set_load_pending(self, file_path_dict, pending):
        changed = False
        for file_path, idx in file_path_dict.items():
            if idx < len(self.data) and self.data[idx].get('file_path') == file_path:
                self.data[idx]['load_pending'] = bool(pending)
                changed = True
        if changed:
            self.refresh_from_data()

    @staticmethod
    def _norm_path_key(p: str) -> str:
        try:
            return os.path.normcase(os.path.abspath(p))
        except OSError:
            return os.path.normcase(p or "")

    def load_images_thread(self, file_path_dict, chunk_size):
        file_path_list = list(file_path_dict.keys())

        for i in range(0, len(file_path_list), chunk_size):
            chunk = file_path_list[i:i + chunk_size]
            try:
                # -a -G1 keeps duplicate Rating tags as group-qualified keys.
                # safe_get_metadata also adds short-name aliases for existing UI code.
                exif_data_list = safe_get_metadata(
                    chunk,
                    common_args=["-b", "-s", "-a", "-G1", "-x", "IFD1:PreviewTIFF", "-x", "SubIFD1:PreviewTIFF"],
                )

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
                        item['load_pending'] = False
                        ex0 = exif_data_list[k] or {}
                        if rating_utils.is_raw_path(file_path):
                            item['rating'] = rating_io.read_raw_pmck_rating_value(file_path)
                        else:
                            item['rating'] = rating_utils.parse_exif_rating_value(ex0)
                        item['pmck_exists'] = os.path.exists(file_path + ".pmck")
                        updates[idx] = item

                self._apply_updates(updates)
            except Exception:
                logging.exception("load_images_thread: chunk 処理失敗。スキップして続行 (chunk size=%d)", len(chunk))
                self._finish_failed_chunk(chunk, file_path_dict)

    @kvmainthread
    def _apply_updates(self, updates):
        for idx, item in updates.items():
            self.data[idx] = item
        self.refresh_from_data()

    @kvmainthread
    def _finish_failed_chunk(self, chunk, file_path_dict):
        for file_path in chunk:
            idx = file_path_dict.get(file_path)
            if idx is None:
                continue
            if idx < len(self.data) and self.data[idx].get('file_path') == file_path:
                self.data[idx]['load_pending'] = False
                if self.data[idx].get('exif_data') is None:
                    self.data[idx]['exif_data'] = {}
        self.refresh_from_data()

    def is_supported_image(self, file_name):
        return file_name.lower().endswith(define.SUPPORTED_FORMATS_RGB) or file_name.lower().endswith(define.SUPPORTED_FORMATS_RAW)

    def _image_path_for_pmck_sidecar(self, file_path):
        if not file_path:
            return None
        s = str(file_path)
        if not s.lower().endswith(".pmck"):
            return None
        image_path = s[:-5]
        return image_path if self.is_supported_image(image_path) else None

    def set_pmck_indicator_for_path(self, file_path, exists=None):
        if not file_path:
            return False
        want = self._norm_path_key(file_path)
        found = False
        pmck_exists = os.path.exists(file_path + ".pmck") if exists is None else bool(exists)
        for d in self.data:
            if self._norm_path_key(d.get("file_path") or "") != want:
                continue
            d["pmck_exists"] = pmck_exists
            if rating_utils.is_raw_path(file_path):
                d["rating"] = rating_io.read_raw_pmck_rating_value(file_path)
            found = True
            break
        if found:
            self.refresh_from_data()
            app = MDApp.get_running_app()
            main_widget = getattr(app, "main_widget", None) if app else None
            imgset = getattr(main_widget, "imgset", None) if main_widget else None
            if imgset and self._norm_path_key(getattr(imgset, "file_path", "") or "") == want:
                sync = getattr(main_widget, "_sync_exif_rating_row", None)
                if sync:
                    sync()
        return found

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
                    if thumb.ndim == 2:
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
                    elif thumb.shape[2] == 4:
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGRA2RGB)
                    elif thumb.shape[2] > 4:
                        thumb = cv2.cvtColor(thumb[:, :, :3], cv2.COLOR_BGR2RGB)
                    else:
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                else:
                    if file_path.lower().endswith(define.SUPPORTED_FORMATS_RAW):
                        with rawpy.imread(file_path) as raw:
                            thumb = raw.postprocess()
                    else:
                        with pyvips.Image.new_from_file(file_path) as vips_image:
                            thumb = np.array(vips_image)
                            if thumb.ndim == 3 and thumb.shape[2] > 3:
                                thumb = thumb[:, :, :3]
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
        if not self._is_item_ready(index):
            return
        should_notify = False
        
        if not touch.is_mouse_scrolling and touch.button == 'left':
            if (
                'shift' in KVWindow.modifiers
                and self.last_selected_index is not None
                and index is not None
            ):
                anchor = self.last_selected_index
                if not( 'ctrl' in KVWindow.modifiers or 'meta' in KVWindow.modifiers ):
                    self.clear_selection()
                
                start = min(anchor, index)
                end = max(anchor, index)
                
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
        if not self._is_item_ready(index):
            return
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
        if self._is_item_ready(index):
            self.data[index]['selected'] = True
            self.selected_indices.add(index)
            self.refresh_from_data()

    def toggle_at(self, index):
        if self._is_item_ready(index):
            val = not self.data[index]['selected']
            self.data[index]['selected'] = val
            if val: self.selected_indices.add(index)
            else: self.selected_indices.discard(index)
            self.refresh_from_data()

    def clear_selection(self):
        for d in self.data:
            d['selected'] = False
        self.selected_indices.clear()
        self.last_selected_index = None
        self.refresh_from_data()

    def refresh_exif_for_exported_path(self, file_path: str) -> bool:
        """
        エクスポート直後: ファイルには exiftool で星が入ったあと。watch より前に
        サムネ取得が走ると 0 星のままになるため、該当行のメタ＆星を取り直す。
        """
        if not file_path:
            return False
        want = self._norm_path_key(file_path)
        for i, d in enumerate(self.data):
            if self._norm_path_key(d.get("file_path") or "") != want:
                continue
            self.load_images({d["file_path"]: i})
            return True
        return False

    def _is_item_ready(self, index):
        return (
            index is not None
            and 0 <= index < len(self.data)
            and not bool(self.data[index].get("load_pending", False))
            and self.data[index].get("exif_data") is not None
        )

    def set_selection_silent(self, file_path):
        """サムネの選択表示だけを合わせる。on_select（画像の再ロード）は呼ばない。"""
        if not self.data or file_path is None:
            self.clear_selection()
            return
        idx = next((i for i, d in enumerate(self.data) if d['file_path'] == file_path), None)
        for i, d in enumerate(self.data):
            d['selected'] = idx is not None and i == idx
        if idx is not None:
            self.selected_indices = {idx}
            self.last_selected_index = idx
        else:
            self.selected_indices.clear()
            self.last_selected_index = None
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

    def set_rating_for_path(self, file_path, rating_value: int):
        for i, d in enumerate(self.data):
            if d.get("file_path") == file_path:
                d["rating"] = int(rating_value)
                break
        self.refresh_from_data()

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

    def on_rating_slot(self, index, slot: int):
        if not self._is_item_ready(index):
            return
        cur = int(self.data[index].get("rating", 0) or 0)
        new_r = rating_utils.new_rating_on_slot_click(cur, slot)
        if len(self.selected_indices) > 1 and index in self.selected_indices:
            target_paths = [self.data[i]["file_path"] for i in sorted(self.selected_indices)]
        else:
            target_paths = [self.data[index]["file_path"]]
        app = MDApp.get_running_app()
        if app and hasattr(app, "main_widget"):
            app.main_widget.apply_paths_rating(target_paths, new_r)
