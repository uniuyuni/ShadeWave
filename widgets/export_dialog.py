
from kivy.app import App as KVApp
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.core.window import Window as KVWindow
from kivy.properties import StringProperty as KVStringProperty, NumericProperty as KVNumericProperty, BooleanProperty as KVBooleanProperty, DictProperty as KVDictProperty, ListProperty as KVListProperty
from kivy.uix.popup import Popup as KVPopup
from kivy.uix.modalview import ModalView as KVModalView
from kivy.uix.button import Button as KVButton
from kivy.metrics import dp as kv_dp
from functools import partial
import json
import logging

import utils.dialogutils as dialogutils
import utils.kvutils as kvutils
from utils import paths
import macos as device

import widgets.param_slider
import widgets.float_input
import widgets.modern_checkbox


def _available_icc_profiles():
    try:
        import export
        return export.get_available_icc_profiles()
    except Exception:
        return ['sRGB IEC61966-2.1']

class ExportConfirmDialog(KVPopup):

    def __init__(self, callback, preset, **kwargs):
        super().__init__(**kwargs)

        self.title = "Target file already exsists"
        self.size_hint = (None, None)
        self.ref_width = 400
        self.ref_height = 300
        
        layout = KVBoxLayout(orientation='vertical')
        layout.ref_layout_padding = 5
        layout.ref_layout_spacing = 5
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
    icc_profile_values = KVListProperty(['sRGB IEC61966-2.1'])

    # Presets
    presets = KVDictProperty()

    def __init__(self, callback, **kwargs):
        super(ExportDialog, self).__init__(**kwargs)

        self.callback = callback
        self.icc_profile_values = _available_icc_profiles()
    
    def on_kv_post(self, *args, **kwargs):
        self.bind(on_dismiss=self.handle_dismiss)
        dialogutils.install_ref_scaling(self)

        self._load_default_presets()
        self._load_json()
        self.load_preset('Default')
    
    def handle_dismiss(self, instance):
        self._save_json()

    def _save_json(self):
            file_path = paths.export_presets_path()
            with open(file_path, 'w') as f:
                json.dump(self.presets, f)

    def _load_json(self):
            file_path = paths.export_presets_path()
            try:
                with open(file_path, 'r') as f:
                    self.presets = json.load(f)
                    self._normalize_default_preset()
                    self.ids['preset_spinner'].values = list(self.presets.keys())
            except FileNotFoundError as e:
                pass

    def _load_default_presets(self):
        default_settings = {
            'format': '.JPG',
            'quality': 90,
            'size_mode': 'Original',
            'size_value': '',
            'sharpen': 0,
            'metadata': True,
            'gps': True,
            'dithering': True,
            'output_path': '',
            'icc_profile': 'sRGB IEC61966-2.1',
        }
        if self.presets.get('Default', None) is None:
            self.presets['Default'] = default_settings
        self._normalize_default_preset()
        self.current_preset = 'Default'

    def _normalize_default_preset(self):
        default_preset = self.presets.get('Default')
        if isinstance(default_preset, dict):
            default_preset['sharpen'] = 0

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
        logging.info(
            "Exporting with settings: format=%s quality=%s size=%s-%s sharpen=%s "
            "metadata=%s gps=%s dithering=%s output=%s icc_profile=%s",
            self.format_value,
            self.quality_value,
            self.size_mode,
            self.size_value,
            self.sharpen_value,
            self.include_metadata,
            self.include_gps,
            self.dithering,
            self.output_path,
            self.icc_profile,
        )

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
        try:
            preset_name = device.prompt_native(
                message="Preset name",
                title="Save Preset",
                default="",
                show_cancel=True,
                ascii_only=False,
            )
        except Exception as e:
            logging.warning("export preset name prompt failed: %s", e)
            return
        if preset_name:
            self._save_preset_with_name(preset_name)

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
            if self.icc_profile not in self.icc_profile_values:
                self.icc_profile = 'sRGB IEC61966-2.1'
            self.current_preset = preset_name



class DummyWidget(KVBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dialog = ExportDialog(None)
        dialog.bind(pos=KVApp.get_running_app().on_window_resize)
        dialog.open()

class Export_DialogApp(KVApp):

    def build(self):
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
                if hasattr(child, 'ref_layout_padding'):
                    child.padding = kvutils.dpi_scale_width(child.ref_layout_padding)
                if hasattr(child, 'ref_layout_spacing'):
                    child.spacing = kvutils.dpi_scale_width(child.ref_layout_spacing)
                if hasattr(child, 'ref_tab_width'):
                    child.tab_width = kvutils.dpi_scale_width(child.ref_tab_width)
                if hasattr(child, 'ref_tab_height'):
                    child.tab_height = kvutils.dpi_scale_height(child.ref_tab_height)

if __name__ == '__main__':
    Export_DialogApp().run()
