
from kivy.clock import Clock as KVClock
from kivy.core.window import Window as KVWindow
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty
from kivy.uix.behaviors import ButtonBehavior as KVButtonBehavior
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout

import utils.iconutils as iconutils
import utils.kvutils as kvutils


class CollapsibleHeader(KVButtonBehavior, KVBoxLayout):
    pass


class CollapsibleBox(KVBoxLayout):
    title = KVStringProperty("Title")
    icon = KVStringProperty("")
    icon_source = KVStringProperty("")
    icon_ref_size = KVNumericProperty(16)
    is_expanded = KVBooleanProperty(True)
    content_height = KVNumericProperty(0) 
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint_y = None
        self._content_height_event = None
        self._bound_content = None
        KVWindow.bind(size=lambda *_args: self._update_icon_source())
        self.bind(is_expanded=self._schedule_content_height_update)
        
    def on_kv_post(self, base_widget):
        kvutils.traverse_widget(self)
        self._bind_content_height()
        self._schedule_content_height_update()
        self._update_icon_source()

    def on_icon(self, *_args):
        self._update_icon_source()
        
    def toggle(self):
        self.is_expanded = not self.is_expanded
        self._apply_layout_state()
        self._schedule_parent_layout_update()

    def add_widget(self, widget, *args, **kwargs):
        if hasattr(self, 'ids') and 'content' in self.ids and self.ids.content is not widget:
            result = self.ids.content.add_widget(widget, *args, **kwargs)
            self._bind_child_height(widget)
            self._schedule_content_height_update()
            return result
        result = super().add_widget(widget, *args, **kwargs)
        self._schedule_content_height_update()
        return result

    def _bind_content_height(self):
        content = self.ids.get('content')
        if content is None or content is self._bound_content:
            return
        if self._bound_content is not None:
            self._bound_content.unbind(
                minimum_height=self._schedule_content_height_update,
                height=self._schedule_content_height_update,
            )
        self._bound_content = content
        content.bind(
            minimum_height=self._schedule_content_height_update,
            height=self._schedule_content_height_update,
        )
        for child in content.children:
            self._bind_child_height(child)

    def _bind_child_height(self, child):
        for prop in ("height", "minimum_height"):
            if prop in getattr(child, "properties", lambda: {})():
                child.fbind(prop, self._schedule_content_height_update)

    def _schedule_content_height_update(self, *_args):
        if self._content_height_event is None:
            self._content_height_event = KVClock.create_trigger(self._update_content_height, 0)
        self._content_height_event()
    
    def _update_content_height(self, *_args):
        """中身の高さをイベント駆動で更新する。"""
        content = self.ids.get('content')
        if content:
            self.content_height = content.minimum_height
        self._apply_layout_state()
        self._schedule_parent_layout_update()
        self._update_icon_source()

    def _apply_layout_state(self):
        header = self.ids.get('btn_header')
        scroll_view = self.ids.get('scroll_view')
        if header is None or scroll_view is None:
            return
        scroll_view.height = self.content_height if self.is_expanded else 0
        scroll_view.opacity = 1 if self.is_expanded else 0
        self.height = header.height + scroll_view.height

    def _schedule_parent_layout_update(self):
        KVClock.schedule_once(lambda _dt: self._update_local_parent_layout(), 0)
        KVClock.schedule_once(lambda _dt: self._update_local_parent_layout(), 0.02)

    def _update_local_parent_layout(self):
        parent = getattr(self, "parent", None)
        if parent is None:
            return
        if hasattr(parent, "do_layout"):
            parent.do_layout()
        if hasattr(parent, "minimum_height") and getattr(parent, "size_hint_y", None) is None:
            parent.height = parent.minimum_height
        grandparent = getattr(parent, "parent", None)
        if grandparent is not None and hasattr(grandparent, "do_layout"):
            grandparent.do_layout()

    def _update_icon_source(self):
        if not self.icon:
            self.icon_source = ""
            return
        desired = kvutils.dpi_scale_height(self.icon_ref_size)
        self.icon_source = iconutils.variant_source(self.icon, desired)
