
from kivy.clock import Clock
from kivy.properties import StringProperty, BooleanProperty, NumericProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder

import utils.kvutils as kvutils

class CollapsibleBox(BoxLayout):
    title = StringProperty("Title")
    icon = StringProperty("")
    is_expanded = BooleanProperty(True)
    content_height = NumericProperty(0) 
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint_y = None

        Clock.schedule_interval(self._update_content_height, 0.1)
        
    def on_kv_post(self, base_widget):
        kvutils.traverse_widget(self)
        
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