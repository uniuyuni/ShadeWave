import os

from kivy.lang import Builder as KVBuilder
from kivy.properties import StringProperty as KVStringProperty
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.uix.button import Button as KVButton
from kivy.uix.popup import Popup as KVPopup
from kivy.uix.recycleboxlayout import RecycleBoxLayout as KVRecycleBoxLayout
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivymd.app import MDApp

from utils import preset_utils


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
KVBuilder.load_file(os.path.join(CUR_DIR, "preset_content.kv"))


class PresetRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, KVRecycleBoxLayout):
    pass


class PresetItem(KVBoxLayout, RecycleDataViewBehavior):
    text = KVStringProperty("")
    path = KVStringProperty("")

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos):
            app = MDApp.get_running_app()
            if app and hasattr(app, "main_widget"):
                app.main_widget.apply_preset_path(self.path)
            return True
        return False

    def delete_item(self):
        app = MDApp.get_running_app()
        if app and hasattr(app, "main_widget"):
            app.main_widget.confirm_delete_preset(self.text, self.path)


class PresetContentPanel(KVBoxLayout):
    def on_kv_post(self, *args, **kwargs):
        super().on_kv_post(*args, **kwargs)
        self.refresh_list()

    def refresh_list(self, *args):
        data = []
        folder = preset_utils.ensure_preset_dir()
        for name in preset_utils.list_presets():
            data.append(
                {
                    "text": name,
                    "path": os.path.join(folder, name + ".json"),
                }
            )
        self.ids["preset_rv"].data = data

    def add_preset(self):
        app = MDApp.get_running_app()
        if app and hasattr(app, "main_widget"):
            app.main_widget.start_add_preset()


def create_preset_content_panel():
    panel = PresetContentPanel()
    panel.id = "preset_content_panel"
    return panel
