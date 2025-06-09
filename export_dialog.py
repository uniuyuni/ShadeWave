
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, DictProperty
from kivy.uix.popup import Popup
from kivy.uix.modalview import ModalView
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.metrics import dp
from functools import partial
import json

import macos
import utils

import param_slider
import spacer
import switchable_float_input

class PresetNameDialog(Popup):
    def __init__(self, save_callback, **kwargs):
        super().__init__(**kwargs)
        self.title = "Save Preset"
        self.size_hint = (0.6, 0.3)
        
        layout = BoxLayout(orientation='vertical', padding=dp(5), spacing=dp(5))
        self.preset_name = TextInput(multiline=False)
        save_button = Button(text='Save', size_hint_y=None, height=40)
        save_button.bind(on_press=lambda x: self.save_preset(save_callback))
        
        layout.add_widget(self.preset_name)
        layout.add_widget(save_button)
        self.content = layout

    def save_preset(self, callback):
        if self.preset_name.text:
            callback(self.preset_name.text)
            self.dismiss()

class ExportConfirmDialog(Popup):

    def __init__(self, callback, preset, **kwargs):
        super().__init__(**kwargs)

        self.title = "Target file already exsists"
        self.size_hint = (None, None)
        self.size = (dp(400), dp(300))
        
        layout = BoxLayout(orientation='vertical', padding=dp(5), spacing=dp(5))
        rename_button = Button(text='Rename')
        rename_button.bind(on_press=lambda x: self._on_callback(callback('Rename', preset)))
        layout.add_widget(rename_button)
        cancel_button = Button(text='Cancel')
        cancel_button.bind(on_press=lambda x: self._on_callback(None))
        layout.add_widget(cancel_button)
        overwrite_button = Button(text='Overwrite')
        overwrite_button.bind(on_press=lambda x: self._on_callback(callback('Overwrite', preset)))
        layout.add_widget(overwrite_button)

        self.content = layout

    def _on_callback(self, callback):
        self.dismiss()
        if callback is not None:
            callback()

class ExportDialog(ModalView):
    # File format properties
    format_value = StringProperty('.JPG')
    quality_value = NumericProperty(85)
    
    # Size properties
    size_mode = StringProperty('Original')
    size_value = StringProperty('')
    
    # Sharpening property
    sharpen_value = NumericProperty(50)
    
    # Metadata property
    include_metadata = BooleanProperty(True)
    
    # Output path
    output_path = StringProperty('')

    # Color Space
    color_space = StringProperty('sRGB')

    # Presets
    presets = DictProperty()

    def __init__(self, callback, **kwargs):
        super(ExportDialog, self).__init__(**kwargs)

        self.callback = callback
    
    def on_kv_post(self, *args, **kwargs):
        self.bind(on_dismiss=self.handle_dismiss)

        self._load_json()
        self._load_default_presets()

    def handle_dismiss(self, instance):
        self._save_json()

    def _save_json(self):
            file_path = "export_preset.json"
            with open(file_path, 'w') as f:
                json.dump(self.presets, f)

    def _load_json(self):
            file_path = "export_preset.json"
            try:
                with open(file_path, 'r') as f:
                    self.presets = json.load(f)
                    self.ids['preset_spinner'].values = list(self.presets.keys())
            except FileNotFoundError as e:
                pass

    def _load_default_presets(self):
        default_settings = {
            'format': '.JPG',
            'quality': 90,
            'size_mode': 'Original',
            'size_value': '',
            'sharpen': 50,
            'metadata': True,
            'output_path': '',
            'color_space': 'sRGB',
        }
        if self.presets.get('Default', None) is None:
            self.presets['Default'] = default_settings
        self.current_preset = 'Default'

    def on_format_value(self, instance, value):
        pass

    def on_size_mode(self, instance, value):
        pass

    def browse_output(self):
        macos.FileChooser(title="Select Folder", mode="dir", filters=[("Jpeg Files", "*.jpg")], on_selection=self._handle_for_dir_selection).run()

    def cancel(self):
        self.dismiss()

    def export(self):
        # エクスポート処理の実装
        print(f"Exporting with settings:")
        print(f"Format: {self.format_value}")
        print(f"Quality: {self.quality_value}")
        print(f"Size: {self.size_mode} - {self.size_value}")
        print(f"Sharpen: {self.sharpen_value}")
        print(f"Metadata: {self.include_metadata}")
        print(f"Output: {self.output_path}")
        print(f"Color Space: {self.color_space}")

        self.dismiss()
        if self.callback is not None:
            preset = {
                'format': self.format_value,
                'quality': self.quality_value,
                'size_mode': self.size_mode,
                'size_value': self.size_value,
                'sharpen': self.sharpen_value,
                'metadata': self.include_metadata,
                'output_path': self.output_path,
                'color_space': self.color_space,
            }            
            self.callback(preset)

    def _handle_for_dir_selection(self, selection):
        if selection is not None:
            self.output_path = selection[0].decode()

    def save_preset(self):
        # プリセット保存ダイアログを表示
        #dialog = ExportConfirmDialog(None, None)
        dialog = PresetNameDialog(self._save_preset_with_name)
        dialog.open()

    def _save_preset_with_name(self, preset_name):
        if preset_name and preset_name != 'Default':
            self.presets[preset_name] = {
                'format': self.format_value,
                'quality': self.quality_value,
                'size_mode': self.size_mode,
                'size_value': self.size_value,
                'sharpen': self.sharpen_value,
                'metadata': self.include_metadata,
                'output_path': self.output_path,
                'color_space': self.color_space,
            }
            # Spinnerの値を更新
            preset_spinner = self.ids['preset_spinner']
            preset_spinner.values = list(self.presets.keys())
            preset_spinner.text = preset_name

    def delete_preset(self):
        if self.current_preset != 'Default':
            del self.presets[self.current_preset]
            preset_spinner = self.ids['preset_spinner']
            preset_spinner.values = list(self.presets.keys())
            preset_spinner.text = 'Default'
            self.load_preset('Default')

    def load_preset(self, preset_name):
        if preset_name in self.presets:
            settings = self.presets[preset_name]
            self.format_value = settings['format']
            self.quality_value = settings['quality']
            self.ids['slider_quality'].set_slider_value(self.quality_value)
            self.size_mode = settings['size_mode']
            self.size_value = settings['size_value']
            self.sharpen_value = settings['sharpen']
            self.ids['slider_sharpen'].set_slider_value(self.sharpen_value)
            self.include_metadata = settings['metadata']
            self.output_path = settings['output_path']
            self.color_space = settings['color_space']
            self.current_preset = preset_name



class DummyWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dialog = ExportDialog(None)
        dialog.bind(pos=MDApp.get_running_app().on_window_resize)
        dialog.open()

class Export_DialogApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = 'Dark'
        self.theme_cls.primary_palette = 'Blue'

        Window.size = (dp(300), dp(200))

        return DummyWidget()

    def on_start(self):
        #Window.bind(on_resize=self.on_window_resize)
        return super().on_start()

    def on_window_resize(self, root, pos):
        # すべてのスケールが必要なウィジェットを更新
        if root:
            for child in utils.get_entire_widget_tree(root):
                if hasattr(child, 'ref_width'):
                    child.width = utils.dpi_scale_width(child.ref_width)
                if hasattr(child, 'ref_height'):
                    child.height = utils.dpi_scale_height(child.ref_height)
                if hasattr(child, 'ref_padding'):
                    child.padding = utils.dpi_scale_width(child.ref_padding)
                if hasattr(child, 'ref_spacing'):
                    child.spacing = utils.dpi_scale_width(child.ref_spacing)
                if hasattr(child, 'ref_tab_width'):
                    child.tab_width = utils.dpi_scale_width(child.ref_tab_width)
                if hasattr(child, 'ref_tab_height'):
                    child.tab_height = utils.dpi_scale_height(child.ref_tab_height)

if __name__ == '__main__':
    Export_DialogApp().run()