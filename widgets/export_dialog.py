
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.core.window import Window as KVWindow
from kivy.properties import StringProperty as KVStringProperty, NumericProperty as KVNumericProperty, BooleanProperty as KVBooleanProperty, DictProperty as KVDictProperty
from kivy.uix.popup import Popup as KVPopup
from kivy.uix.modalview import ModalView as KVModalView
from kivy.uix.textinput import TextInput as KVTextInput
from kivy.uix.button import Button as KVButton
from kivy.metrics import dp as kv_dp
from functools import partial
import json

import utils.dialogutils as dialogutils
import utils.kvutils as kvutils
import macos as device

import widgets.param_slider
import widgets.float_input
import widgets.modern_checkbox

class PresetNameDialog(KVPopup):
    def __init__(self, save_callback, **kwargs):
        super().__init__(**kwargs)
        self.title = "Save Preset"
        self.size_hint = (None, None)
        self.ref_width = 280
        self.ref_height = 120
        
        layout = KVBoxLayout(orientation='vertical')
        #layout.pos_hint = {'left': 0, 'top': 0}
        layout.ref_padding = 5
        layout.ref_spacing = 5

        self.preset_name = KVTextInput(multiline=False, size_hint_y=None)
        self.preset_name.ref_height = 28

        button_layout = KVBoxLayout(orientation='horizontal', size_hint_y=None)
        button_layout.ref_height = 30
        button_layout.ref_spacing = 5

        cancel_button = KVButton(text='Cancel', size_hint_y=None)
        cancel_button.ref_height = 30
        cancel_button.bind(on_press=lambda x: self.dismiss())
        button_layout.add_widget(cancel_button)

        save_button = KVButton(text='Save', size_hint_y=None)
        save_button.ref_height = 30
        save_button.bind(on_press=lambda x: self.save_preset(save_callback))        
        button_layout.add_widget(save_button)

        layout.add_widget(self.preset_name)
        layout.add_widget(button_layout)
        self.content = layout
        dialogutils.install_ref_scaling(self)

    def save_preset(self, callback):
        if self.preset_name.text:
            callback(self.preset_name.text)
            self.dismiss()
        
    def on_popup_resize(self, instance, value):
        kvutils.traverse_widget(instance)

class ExportConfirmDialog(KVPopup):

    def __init__(self, callback, preset, **kwargs):
        super().__init__(**kwargs)

        self.title = "Target file already exsists"
        self.size_hint = (None, None)
        self.ref_width = 400
        self.ref_height = 300
        
        layout = KVBoxLayout(orientation='vertical')
        layout.ref_padding = 5
        layout.ref_spacing = 5
        rename_button = KVButton(text='Rename')
        rename_button.bind(on_press=lambda x: self._on_callback(callback('Rename', preset)))
        layout.add_widget(rename_button)
        cancel_button = KVButton(text='Cancel')
        cancel_button.bind(on_press=lambda x: self._on_callback(None))
        layout.add_widget(cancel_button)
        overwrite_button = KVButton(text='Overwrite')
        overwrite_button.bind(on_press=lambda x: self._on_callback(callback('Overwrite', preset)))
        layout.add_widget(overwrite_button)

        self.content = layout
        dialogutils.install_ref_scaling(self)

    def _on_callback(self, callback):
        self.dismiss()
        if callback is not None:
            callback()

class ExportDialog(KVModalView):
    # File format properties
    format_value = KVStringProperty('.JPG')
    quality_value = KVNumericProperty(90)
    
    # Size properties
    size_mode = KVStringProperty('Original')
    size_value = KVStringProperty('')
    
    # Sharpening property
    sharpen_value = KVNumericProperty(0)
    
    # Metadata property
    include_metadata = KVBooleanProperty(True)
    include_gps = KVBooleanProperty(True)

    # Dhithering property
    dithering = KVBooleanProperty(True)
    
    # Output path
    output_path = KVStringProperty('')

    # Color Space
    icc_profile = KVStringProperty('sRGB IEC61966-2.1')

    # Presets
    presets = KVDictProperty()

    def __init__(self, callback, **kwargs):
        super(ExportDialog, self).__init__(**kwargs)

        self.callback = callback
    
    def on_kv_post(self, *args, **kwargs):
        self.bind(on_dismiss=self.handle_dismiss)
        dialogutils.install_ref_scaling(self)

        self._load_default_presets()
        self._load_json()
        self.load_preset('Default')
    
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
            'gps': True,
            'dithering': True,
            'output_path': '',
            'icc_profile': 'sRGB IEC61966-2.1',
        }
        if self.presets.get('Default', None) is None:
            self.presets['Default'] = default_settings
        self.current_preset = 'Default'

    def on_format_value(self, instance, value):
        pass

    def on_size_mode(self, instance, value):
        pass

    def browse_output(self):
        device.FileChooser(title="Select Folder", mode="dir", filters=[("Jpeg Files", "*.jpg")], on_selection=self._handle_for_dir_selection).run()

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
        print(f"GPS: {self.include_gps}")
        print(f"Dithering: {self.dithering}")
        print(f"Output: {self.output_path}")
        print(f"ICC Profile: {self.icc_profile}")

        self.dismiss()
        if self.callback is not None:
            preset = {
                'format': self.format_value,
                'quality': self.quality_value,
                'size_mode': self.size_mode,
                'size_value': self.size_value,
                'sharpen': self.sharpen_value,
                'metadata': self.include_metadata,
                'gps': self.include_gps,
                'dithering': self.dithering,
                'output_path': self.output_path,
                'icc_profile': self.icc_profile,
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
                'gps': self.include_gps,
                'dithering': self.dithering,
                'output_path': self.output_path,
                'icc_profile': self.icc_profile,
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
            self.format_value = settings.get('format', self.format_value)
            self.quality_value = settings.get('quality', self.quality_value)
            self.ids['slider_quality'].set_slider_value(self.quality_value)
            self.size_mode = settings.get('size_mode', self.size_mode)
            self.size_value = settings.get('size_value', self.size_value)
            self.sharpen_value = settings.get('sharpen', self.sharpen_value)
            self.ids['slider_sharpen'].set_slider_value(self.sharpen_value)
            self.include_metadata = settings.get('metadata', self.include_metadata)
            self.include_gps = settings.get('gps', self.include_gps)
            self.dithering = settings.get('dithering', self.dithering)
            self.output_path = settings.get('output_path', self.output_path)
            self.icc_profile = settings.get('icc_profile', self.icc_profile)
            self.current_preset = preset_name



class DummyWidget(KVBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dialog = ExportDialog(None)
        dialog.bind(pos=MDApp.get_running_app().on_window_resize)
        dialog.open()

class Export_DialogApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = 'Dark'
        self.theme_cls.primary_palette = 'Blue'

        KVWindow.size = (kv_dp(300), kv_dp(200))

        return DummyWidget()

    def on_start(self):
        #Window.bind(on_resize=self.on_window_resize)
        return super().on_start()

    def on_window_resize(self, root, pos):
        # すべてのスケールが必要なウィジェットを更新
        if root:
            for child in kvutils.get_entire_widget_tree(root):
                if hasattr(child, 'ref_width'):
                    child.width = kvutils.dpi_scale_width(child.ref_width)
                if hasattr(child, 'ref_height'):
                    child.height = kvutils.dpi_scale_height(child.ref_height)
                if hasattr(child, 'ref_padding'):
                    child.padding = kvutils.dpi_scale_width(child.ref_padding)
                if hasattr(child, 'ref_spacing'):
                    child.spacing = kvutils.dpi_scale_width(child.ref_spacing)
                if hasattr(child, 'ref_tab_width'):
                    child.tab_width = kvutils.dpi_scale_width(child.ref_tab_width)
                if hasattr(child, 'ref_tab_height'):
                    child.tab_height = kvutils.dpi_scale_height(child.ref_tab_height)

if __name__ == '__main__':
    Export_DialogApp().run()
