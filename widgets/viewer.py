import os
import threading
import base64
import io
import numpy as np
import cv2
from watchfiles import watch
import time
import pyvips
from PIL import Image as PILImage, ImageOps as PILImageOps

from kivymd.app import MDApp
from kivy.core.window import Window as KVWindow
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image as KVImage
from kivy.uix.label import Label as KVLabel
from kivy.uix.floatlayout import FloatLayout
from kivymd.uix.card import MDCard
from kivy.graphics.texture import Texture as KVTexture
from kivy.graphics import Color as KVColor, Rectangle as KVRectangle
from kivy.metrics import dp as kvdp
from kivy.properties import Property as KVProperty, StringProperty as KVStringProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty, BooleanProperty as KVBooleanProperty
from kivy.clock import Clock as KVClock
from kivy.clock import mainthread as kvmainthread
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior

import logging

import libraw_enhanced as lre
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
_HOVER_HINT_DELAY = 0.7
_EMBEDDED_PREVIEW_KEYS = ("PreviewImage", "JpgFromRaw", "PreviewTIFF", "OtherImage")
_EMBEDDED_THUMBNAIL_KEYS = ("ThumbnailImage", "ThumbnailTIFF")


def _first_value(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _format_file_size(file_path):
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return None
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _format_image_size(exif_data):
    try:
        _, _, width, height = core.get_exif_image_size_with_orientation(exif_data or {})
    except Exception:
        width = _first_value(exif_data, "ImageWidth", "ExifImageWidth")
        height = _first_value(exif_data, "ImageHeight", "ExifImageHeight")
    try:
        width = int(str(width).split()[0])
        height = int(str(height).split()[0])
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    mp = width * height / 1_000_000.0
    return f"{width} x {height} · {mp:.1f} MP"


def _format_aperture(value):
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.lower().startswith("f/"):
        return text
    try:
        return f"f/{float(text):g}"
    except ValueError:
        return text


def _build_file_hint_text(file_path, exif_data):
    exif_data = exif_data or {}
    lines = [os.path.basename(file_path or "")]
    directory = os.path.dirname(file_path or "")
    if directory:
        lines.append(directory)

    date = _first_value(exif_data, "CreateDate", "DateCreated", "FileModifyDate", "ModifyDate")
    size_text = _format_image_size(exif_data)
    file_size = _format_file_size(file_path)
    if date:
        lines.extend(["", str(date)])
    metrics = " · ".join(part for part in (size_text, file_size) if part)
    if metrics:
        lines.append(metrics)

    make = _first_value(exif_data, "Make")
    model = _first_value(exif_data, "Model")
    camera = " ".join(str(part).strip() for part in (make, model) if part)
    lens = _first_value(exif_data, "LensModel", "Lens", "LensInfo")
    if camera or lens:
        lines.append("")
    if camera:
        lines.append(camera)
    if lens:
        lines.append(str(lens))

    exposure_parts = [
        f"ISO {exif_data.get('ISO')}" if exif_data.get("ISO") not in (None, "") else None,
        _first_value(exif_data, "ExposureTime", "ShutterSpeedValue"),
        _format_aperture(_first_value(exif_data, "Aperture", "FNumber", "ApertureValue")),
        _first_value(exif_data, "FocalLength"),
    ]
    exposure = " · ".join(str(part) for part in exposure_parts if part)
    if exposure:
        lines.append(exposure)
    return "\n".join(line for line in lines if line is not None)


class FileHint(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.opacity = 0
        self.label = KVLabel(
            text="",
            color=(0.96, 0.96, 0.96, 1),
            font_size="10pt",
            halign="left",
            valign="middle",
            size_hint=(None, None),
        )
        self.add_widget(self.label)
        with self.canvas.before:
            KVColor(0.04, 0.04, 0.04, 0.94)
            self._bg = KVRectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *_args):
        self._bg.pos = self.pos
        self._bg.size = self.size
        pad = kvdp(8)
        self.label.pos = (self.x + pad, self.y + pad)
        self.label.size = (max(1, self.width - pad * 2), max(1, self.height - pad * 2))
        self.label.text_size = self.label.size

    def show(self, text, mouse_pos):
        if not text:
            self.hide()
            return
        pad = kvdp(8)
        self.label.text = text
        self.label.text_size = (kvdp(360), None)
        self.label.texture_update()
        width = min(max(kvdp(220), self.label.texture_size[0] + pad * 2), kvdp(420))
        self.label.text_size = (width - pad * 2, None)
        self.label.texture_update()
        height = self.label.texture_size[1] + pad * 2
        self.size = (width, height)
        x = mouse_pos[0] + kvdp(14)
        y = mouse_pos[1] - height - kvdp(14)
        x = min(max(kvdp(4), x), max(kvdp(4), KVWindow.width - width - kvdp(4)))
        y = min(max(kvdp(4), y), max(kvdp(4), KVWindow.height - height - kvdp(4)))
        self.pos = (x, y)
        self.opacity = 1

    def hide(self):
        self.opacity = 0

class ThumbnailCard(RecycleDataViewBehavior, MDCard):
    file_path = KVStringProperty()
    thumb_source = KVObjectProperty(None, allownone=True, force_dispatch=True)
    rating = KVNumericProperty(0)
    pmck_exists = KVBooleanProperty(False)
    ai_job_state = KVStringProperty("")
    ai_job_progress = KVStringProperty("")
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
        vbox.ref_layout_padding = 8

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
        self.ai_job_icon = KVImage(
            source=rel("assets", "spinner.gif"),
            anim_delay=0.03,
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
            opacity=0,
        )
        self.image_box.add_widget(self.ai_job_icon)
        self.ai_job_progress_label = KVLabel(
            text="",
            bold=True,
            font_size='12sp',
            size_hint=(None, None),
            size=(kvdp(54), kvdp(22)),
            halign="right",
            valign="middle",
            color=(1, 1, 1, 0.95),
            outline_width=1,
            outline_color=(0, 0, 0, 0.9),
            opacity=0,
        )
        self.ai_job_progress_label.bind(size=self.ai_job_progress_label.setter("text_size"))
        self.image_box.add_widget(self.ai_job_progress_label)
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
        if hasattr(self, "ai_job_progress_label"):
            self.ai_job_progress_label.pos = (
                image_x + image_w - self.ai_job_progress_label.width - margin,
                image_y + margin + self.pmck_icon.height + kvutils.dpi_scale_height(2),
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
        self.exif_data = data.get("exif_data")
        self.pmck_exists = bool(data.get("pmck_exists", False))
        self.ai_job_state = str(data.get("ai_job_state") or "")
        self.ai_job_progress = str(data.get("ai_job_progress") or "")
        self.load_pending = bool(data.get("load_pending", False))
        self.pmck_icon.opacity = 1.0 if self.pmck_exists else 0.0
        self.ai_job_icon.opacity = 1.0 if self.ai_job_state in {"queued", "running"} else (0.65 if self.ai_job_state == "error" else 0.0)
        self.ai_job_progress_label.text = self.ai_job_progress
        self.ai_job_progress_label.opacity = 1.0 if self.ai_job_state == "running" and self.ai_job_progress else 0.0
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
        self._watch_directory_lock = threading.Lock()
        self._watch_stop_event = None
        self._card_width_layout_event = None
        self._hover_recheck_event = None
        self._hover_hint_event = None
        self._hover_index = None
        self._file_hint = None
        self._file_hint_path = None

        threading.Thread(target=self._watchfiles_thread, daemon=True).start()
        KVWindow.bind(on_key_down=self.on_key_down)
        KVWindow.bind(mouse_pos=self._on_window_mouse_pos)
        self.bind(height=self._schedule_card_width_sync)
        self.bind(scroll_x=self._on_viewer_scroll_position)
        KVClock.schedule_once(lambda _dt: self._sync_card_width(), 0)

    def on_kv_post(self, base_widget):
        self._sync_card_width()

    def show_file_hint(self, file_path, exif_data, mouse_pos):
        if self._file_hint is None:
            self._file_hint = FileHint()
        if self._file_hint.parent is None:
            KVWindow.add_widget(self._file_hint)
        self._file_hint.show(_build_file_hint_text(file_path, exif_data), mouse_pos)
        self._file_hint_path = file_path

    def move_file_hint(self, mouse_pos):
        if self._file_hint is not None and self._file_hint.opacity > 0:
            self._file_hint.show(self._file_hint.label.text, mouse_pos)

    def hide_file_hint(self, file_path=None):
        if file_path is not None and self._file_hint_path != file_path:
            return
        if self._file_hint is not None:
            self._file_hint.hide()
        self._file_hint_path = None

    def _visible_thumbnail_cards(self):
        stack = list(self.children)
        while stack:
            widget = stack.pop()
            if isinstance(widget, ThumbnailCard):
                yield widget
            try:
                stack.extend(widget.children)
            except Exception:
                pass

    def hover_index_at_window_pos(self, pos):
        card = self.hover_card_at_window_pos(pos)
        if card is None:
            return None
        try:
            index = int(getattr(card, "index", -1))
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(self.data):
            return index
        return None

    def hover_card_at_window_pos(self, pos):
        if not self.collide_point(*pos):
            return None

        local_pos = self.to_local(pos[0], pos[1])
        for card in self._visible_thumbnail_cards():
            if card.collide_point(*local_pos):
                return card
        return None

    def _cancel_hover_hint(self):
        if self._hover_hint_event is not None:
            self._hover_hint_event.cancel()
            self._hover_hint_event = None

    def _schedule_hover_hint(self, index, mouse_pos):
        self._cancel_hover_hint()
        if not (0 <= index < len(self.data)):
            return
        item = self.data[index]
        if item.get("load_pending") or not item.get("file_path"):
            return
        expected_path = item.get("file_path")
        self._hover_hint_event = KVClock.schedule_once(
            lambda _dt: self._show_hover_hint(index, expected_path, mouse_pos),
            _HOVER_HINT_DELAY,
        )

    def _show_hover_hint(self, index, expected_path, mouse_pos):
        self._hover_hint_event = None
        if self._hover_index != index or self.hover_index_at_window_pos(KVWindow.mouse_pos) != index:
            return
        if not (0 <= index < len(self.data)):
            return
        item = self.data[index]
        if item.get("file_path") != expected_path or item.get("load_pending"):
            return
        self.show_file_hint(item.get("file_path"), item.get("exif_data"), KVWindow.mouse_pos or mouse_pos)

    def _on_window_mouse_pos(self, _window, pos, force=False):
        if force:
            self._hover_index = None
        index = self.hover_index_at_window_pos(pos)
        if index is None:
            self._hover_index = None
            self._cancel_hover_hint()
            self.hide_file_hint()
            return
        if self._hover_index != index:
            self._hover_index = index
            self.hide_file_hint()
            self._schedule_hover_hint(index, pos)
            return
        if self._file_hint is not None and self._file_hint.opacity > 0:
            self.move_file_hint(pos)

    def _schedule_hover_recheck(self, delay=0.05):
        if self._hover_recheck_event is not None:
            self._hover_recheck_event.cancel()
        self._hover_recheck_event = KVClock.schedule_once(self._recheck_hover_cards, delay)

    def _recheck_hover_cards(self, _dt):
        self._hover_recheck_event = None
        self._on_window_mouse_pos(KVWindow, KVWindow.mouse_pos, force=True)

    def _on_viewer_scroll_position(self, *_args):
        self._hover_index = None
        self._cancel_hover_hint()
        self.hide_file_hint()
        self._schedule_hover_recheck()

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
            with self._watch_directory_lock:
                watch_directory = self.watch_directory
                stop_event = self._watch_stop_event
            if watch_directory is None or stop_event is None:
                time.sleep(1)
                continue
            try:
                for changes in watch(watch_directory, stop_event=stop_event):
                    if stop_event.is_set():
                        break
                    for action, path in changes:
                        if action in action_type_map:
                            action_type_map[action](path)
            except Exception:
                pass
            time.sleep(1)

    def _set_watch_directory(self, directory):
        directory = os.path.abspath(directory) if directory else None
        with self._watch_directory_lock:
            if self.watch_directory and self._norm_path_key(self.watch_directory) == self._norm_path_key(directory):
                return
            if self._watch_stop_event is not None:
                self._watch_stop_event.set()
            self.watch_directory = directory
            self._watch_stop_event = threading.Event() if directory else None

    def _new_image_item(self, file_path):
        return {
            'file_path': file_path,
            'thumb_source': None,
            'exif_data': None,
            'load_pending': True,
            'selected': False,
            'ctx': self,
            'rating': 0,
            'pmck_exists': os.path.exists(file_path + ".pmck"),
            'ai_job_state': "",
            'ai_job_progress': "",
        }

    def _data_index_for_path(self, file_path):
        want = self._norm_path_key(file_path)
        for i, d in enumerate(self.data):
            if self._norm_path_key(d.get("file_path") or "") == want:
                return i
        return None

    def _is_in_current_watch_directory(self, file_path):
        if not self.watch_directory:
            return False
        try:
            file_dir = os.path.dirname(os.path.abspath(file_path))
            watch_dir = os.path.abspath(self.watch_directory)
        except OSError:
            return False
        return self._norm_path_key(file_dir) == self._norm_path_key(watch_dir)

    def _insert_image_item_sorted(self, file_path):
        idx = self._data_index_for_path(file_path)
        if idx is not None:
            return idx, False

        new_item = self._new_image_item(file_path)
        file_key = self._norm_path_key(file_path)
        idx = len(self.data)
        for i, d in enumerate(self.data):
            if self._norm_path_key(d.get('file_path') or "") > file_key:
                idx = i
                break

        self.data.insert(idx, new_item)
        self.selected_indices = {
            selected_idx + 1 if selected_idx >= idx else selected_idx
            for selected_idx in self.selected_indices
        }
        if self.last_selected_index is not None and self.last_selected_index >= idx:
            self.last_selected_index += 1
        self.cols = max(1, len(self.data))
        return idx, True

    def _mapped_or_current_index(self, file_path_dict, file_path):
        idx = file_path_dict.get(file_path)
        if idx is not None and idx < len(self.data):
            if self._norm_path_key(self.data[idx].get('file_path') or "") == self._norm_path_key(file_path):
                return idx
        return self._data_index_for_path(file_path)

    def refresh_exported_paths(self, file_paths):
        paths = []
        seen = set()
        for file_path in file_paths or []:
            if not file_path or not self.is_visible_image(file_path):
                continue
            if not self._is_in_current_watch_directory(file_path):
                continue
            key = self._norm_path_key(file_path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(file_path)

        if not paths:
            return False

        changed = False
        for file_path in sorted(paths, key=self._norm_path_key):
            _, inserted = self._insert_image_item_sorted(file_path)
            changed = changed or inserted

        file_path_dict = {}
        for file_path in paths:
            idx = self._data_index_for_path(file_path)
            if idx is not None:
                file_path_dict[self.data[idx]["file_path"]] = idx

        if changed:
            self.refresh_from_data()
        self.load_images(file_path_dict)
        return bool(file_path_dict)

    @kvmainthread
    def _added_file(self, file_path):
        pmck_image_path = self._image_path_for_pmck_sidecar(file_path)
        if pmck_image_path is not None:
            self.set_pmck_indicator_for_path(pmck_image_path, True)
            return
        if self.is_visible_image(file_path):
            self.refresh_exported_paths([file_path])

    @kvmainthread
    def _deleted_file(self, file_path):
        pmck_image_path = self._image_path_for_pmck_sidecar(file_path)
        if pmck_image_path is not None:
            self.set_pmck_indicator_for_path(pmck_image_path, False)
            return
        if not self._is_in_current_watch_directory(file_path):
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
        if not self.is_visible_image(file_path):
            return
        self.refresh_exported_paths([file_path])

    def set_path(self, directory):
        self._hover_index = None
        self._cancel_hover_hint()
        self.hide_file_hint()
        preset_utils.cleanup_pmck_backup_files(directory)
        self.data = []
        self.selected_indices.clear()
        self.last_selected_index = None

        file_list = os.listdir(directory)
        file_list.sort()
        
        new_data = []
        file_path_dict = {} # path -> index mapping for loader
        
        for i, file_name in enumerate(file_list):
            if self.is_visible_image(file_name):
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
                    'ai_job_state': "",
                    'ai_job_progress': "",
                })
                file_path_dict[file_path] = len(new_data) - 1

        self.data = new_data
        self.cols = max(1, len(self.data)) # Not used for logic, but might be used by UI binding?
        
        self.load_images(file_path_dict)
        self._set_watch_directory(directory)

    def load_images(self, file_path_dict):
        if len(file_path_dict) > 0:
            self._set_load_pending(file_path_dict, True)
            threading.Thread(target=self.load_images_thread, args=(file_path_dict, 16), daemon=True).start()

    @kvmainthread
    def _set_load_pending(self, file_path_dict, pending):
        changed = False
        for file_path in file_path_dict:
            idx = self._mapped_or_current_index(file_path_dict, file_path)
            if idx is not None and idx < len(self.data):
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
                    idx = self._mapped_or_current_index(file_path_dict, file_path)

                    if idx is not None and idx < len(self.data):
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
        self._hover_index = None
        self._cancel_hover_hint()
        self.hide_file_hint()
        for idx, item in updates.items():
            self.data[idx] = item
        self.refresh_from_data()
        self._schedule_hover_recheck()

    @kvmainthread
    def _finish_failed_chunk(self, chunk, file_path_dict):
        for file_path in chunk:
            idx = self._mapped_or_current_index(file_path_dict, file_path)
            if idx is None:
                continue
            if idx < len(self.data):
                self.data[idx]['load_pending'] = False
                if self.data[idx].get('exif_data') is None:
                    self.data[idx]['exif_data'] = {}
        self.refresh_from_data()

    def is_supported_image(self, file_name):
        return (file_name.lower().endswith(define.SUPPORTED_FORMATS_RGB)
                or file_name.lower().endswith(define.SUPPORTED_FORMATS_RAW)
                or file_name.lower().endswith(define.SUPPORTED_FORMATS_EXR))

    def is_visible_image(self, file_name):
        basename = os.path.basename(str(file_name or ""))
        return bool(basename) and not basename.startswith(".") and self.is_supported_image(file_name)

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

    @kvmainthread
    def set_ai_job_state_for_path(self, file_path, state, progress_text=""):
        if not file_path:
            return False
        want = self._norm_path_key(file_path)
        found = False
        clean_state = state if state in {"queued", "running", "error"} else ""
        clean_progress = str(progress_text or "") if clean_state == "running" else ""
        for d in self.data:
            if self._norm_path_key(d.get("file_path") or "") != want:
                continue
            if d.get("ai_job_state") == clean_state and d.get("ai_job_progress", "") == clean_progress:
                return True
            d["ai_job_state"] = clean_state
            d["ai_job_progress"] = clean_progress
            found = True
            break
        if found:
            self.refresh_from_data()
        return found

    def process_exif_data(self, file_path_list, exif_data_list):
        thumb_data_list = []
        for i in range(len(file_path_list)):
            file_path = file_path_list[i]
            try:
                exif_data = exif_data_list[i]

                thumb, thumb_source_key = self._decode_embedded_thumbnail(exif_data)
                if thumb is not None:
                    pass
                else:
                    if file_path.lower().endswith(define.SUPPORTED_FORMATS_RAW):
                        with lre.imread(file_path) as raw:
                            thumb = raw.postprocess(demosaic_algorithm=lre.DemosaicAlgorithm.Linear, output_bps=8)
                    elif file_path.lower().endswith(define.SUPPORTED_FORMATS_EXR):
                        # EXR は pyvips 非対応。OpenEXR で読み、表示用にトーンマップ済み float32[0,1] を得る。
                        import cores.exr_io as exr_io
                        thumb = exr_io.read_exr_thumbnail(file_path)
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
                if orientation is not None and self._should_apply_parent_orientation(thumb_source_key):
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

            except Exception:
                logging.exception("process_exif_data: thumbnail load failed for %s", file_path)
                thumb_data_list.append(None)

        return thumb_data_list

    def _should_apply_parent_orientation(self, embedded_key):
        return embedded_key not in _EMBEDDED_PREVIEW_KEYS

    def _decode_embedded_bytes(self, encoded):
        if isinstance(encoded, str) and encoded.startswith("base64:"):
            encoded = encoded[7:]
        elif isinstance(encoded, bytes) and encoded.startswith(b"base64:"):
            encoded = encoded[7:]
        return base64.b64decode(encoded)

    def _decode_embedded_preview(self, encoded):
        data = self._decode_embedded_bytes(encoded)
        with PILImage.open(io.BytesIO(data)) as img:
            img = PILImageOps.exif_transpose(img)
            img = img.convert("RGB")
            return np.array(img)

    def _decode_embedded_thumbnail_image(self, encoded):
        data = self._decode_embedded_bytes(encoded)
        image = np.frombuffer(data, dtype=np.uint8)
        thumb = cv2.imdecode(image, 1)
        if thumb is None:
            return None
        if thumb.ndim == 2:
            return cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        if thumb.shape[2] == 4:
            return cv2.cvtColor(thumb, cv2.COLOR_BGRA2RGB)
        if thumb.shape[2] > 4:
            return cv2.cvtColor(thumb[:, :, :3], cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

    def _decode_embedded_thumbnail(self, exif_data):
        for key in _EMBEDDED_PREVIEW_KEYS:
            encoded = exif_data.get(key, None)
            if not encoded:
                continue
            try:
                return self._decode_embedded_preview(encoded), key
            except Exception:
                continue

        for key in _EMBEDDED_THUMBNAIL_KEYS:
            encoded = exif_data.get(key, None)
            if not encoded:
                continue
            try:
                thumb = self._decode_embedded_thumbnail_image(encoded)
            except Exception:
                continue
            if thumb is None:
                continue
            return thumb, key
        return None, None

    def handle_selection(self, index, touch):
        # We also need to notify MainWidget about selection change
        if not self._is_item_ready(index):
            return
        should_notify = False
        
        if not touch.is_mouse_scrolling and touch.button == 'left':
            already_single_selected = (
                index in self.selected_indices
                and len(self.selected_indices) == 1
            )
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
                    if already_single_selected:
                        self.last_selected_index = index
                        return
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
        エクスポート直後: watch より前でも新規カードを追加し、メタ＆星を取り直す。
        """
        return self.refresh_exported_paths([file_path])

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
            self._hover_index = None
            self._cancel_hover_hint()
            self.hide_file_hint()
            original_button = touch.button
            if touch.button == 'scrolldown':
                touch.button = 'scrollright'
            elif touch.button == 'scrollup':
                touch.button = 'scrollleft'
            result = super().on_scroll_start(touch, check_children)
            touch.button = original_button  # 他のウィジェットのために元の値に戻す
            self._schedule_hover_recheck()
            return result
        return super().on_scroll_start(touch, check_children)

    def on_touch_move(self, touch):
        if (
            not touch.is_mouse_scrolling
            and self.collide_point(*touch.pos)
            and not self.dragging
            and self.get_drag_files()
        ):
            self._hover_index = None
            self._cancel_hover_hint()
            self.hide_file_hint()
            self.dragging = True
            self.start_drag(touch)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        self.dragging = False
        return super().on_touch_up(touch)

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
