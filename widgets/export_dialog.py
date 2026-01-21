
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

import utils.kvutils as kvutils
import macos

import widgets.param_slider
import widgets.float_input

class PresetNameDialog(Popup):
    def __init__(self, save_callback, **kwargs):
        super().__init__(**kwargs)
        self.title = "Save Preset"
        self.size_hint = (0.3, None)
        self.ref_height = kvutils.dpi_scale_height(64)
        self.bind(pos=self.on_popup_resize)
        #self.bind(size=self.on_popup_resize)
        
        layout = BoxLayout(orientation='vertical')
        #layout.pos_hint = {'left': 0, 'top': 0}
        layout.ref_padding = kvutils.dpi_scale_width(5)

        self.preset_name = TextInput(multiline=False, size_hint_y=None)
        self.preset_name.ref_height = kvutils.dpi_scale_height(14)

        button_layout = BoxLayout(orientation='horizontal')

        cancel_button = Button(text='Cancel', size_hint_y=None)
        cancel_button.ref_height = kvutils.dpi_scale_height(15)
        cancel_button.bind(on_press=lambda x: self.dismiss())
        button_layout.add_widget(cancel_button)

        save_button = Button(text='Save', size_hint_y=None)
        save_button.ref_height = kvutils.dpi_scale_height(15)
        save_button.bind(on_press=lambda x: self.save_preset(save_callback))        
        button_layout.add_widget(save_button)

        layout.add_widget(self.preset_name)
        layout.add_widget(button_layout)
        self.content = layout

    def save_preset(self, callback):
        if self.preset_name.text:
            callback(self.preset_name.text)
            self.dismiss()
        
    def on_popup_resize(self, instance, value):
        kvutils.traverse_widget(instance)

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
    quality_value = NumericProperty(90)
    
    # Size properties
    size_mode = StringProperty('Original')
    size_value = StringProperty('')
    
    # Sharpening property
    sharpen_value = NumericProperty(0)
    
    # Metadata property
    include_metadata = BooleanProperty(True)
    include_gps = BooleanProperty(True)

    # Dhithering property
    dithering = BooleanProperty(True)
    
    # Output path
    output_path = StringProperty('')

    # Color Space
    icc_profile = StringProperty('sRGB IEC61966-2.1')

    # Presets
    presets = DictProperty()

    def __init__(self, callback, **kwargs):
        super(ExportDialog, self).__init__(**kwargs)

        self.callback = callback
    
    def on_kv_post(self, *args, **kwargs):
        self.bind(on_dismiss=self.handle_dismiss)

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