
from kivy.clock import Clock as KVClock
from kivy.core.window import Window as KVWindow
from kivy.properties import StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.lang import Builder as KVBuilder

import utils.iconutils as iconutils
import utils.kvutils as kvutils

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

        KVClock.schedule_interval(self._update_content_height, 0.1)
        KVWindow.bind(size=lambda *_args: self._update_icon_source())
        
    def on_kv_post(self, base_widget):
        kvutils.traverse_widget(self)
        self._update_icon_source()

    def on_icon(self, *_args):
        self._update_icon_source()
        
    def toggle(self):
        self.is_expanded = not self.is_expanded

    def add_widget(self, widget, *args, **kwargs):
        if hasattr(self, 'ids') and 'content' in self.ids and self.ids.content is not widget:
            return self.ids.content.add_widget(widget, *args, **kwargs)
        return super().add_widget(widget, *args, **kwargs)
    
    def _update_content_height(self, dt):
        """中身の高さを監視（RecycleViewの変動対応）"""
        content = self.ids.content
        if content:
            self.content_height = content.height
        self._update_icon_source()

    def _update_icon_source(self):
        if not self.icon:
            self.icon_source = ""
            return
        desired = kvutils.dpi_scale_height(self.icon_ref_size)
        self.icon_source = iconutils.variant_source(self.icon, desired)
