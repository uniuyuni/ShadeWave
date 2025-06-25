
import os
import json

_config = {
    'preview_size': 1024,
    'raw_auto_exposure': True,
    'raw_depth_expansion': False,
    'scale_threshold': 0.5,
    'iopaint_model': "lama",
    'iopaint_resize_limit': 1280,
    'iopaint_use_realesrgan': True,
    'display_color_gamut': "sRGB",
    'gpu_device': "mps",
    'cat': "cat16",
    'base_resolution_scale': [4096, 4096],
}
_main_widget = None

def set_main_widget(widget):
    global _main_widget
    _main_widget = widget

def get_config(key):
    global _config

    # 暫定処置
    if key == 'preview_width' or key == 'preview_height':
        key = 'preview_size'

    return _config[key]

def set_config(key, value):
    _config[key] = value
    _apply_config(key)
    save_config()

def _apply_config(key):
    global _main_widget, _config
    if key == 'lut_path':
        _main_widget.set_lut_path(_config.get('lut_path', os.getcwd() + "/lut"))
    elif key == 'import_path':
        _main_widget.ids['viewer'].set_path(_config.get('import_path', os.getcwd() + "/picture"))

def apply_config():
    global _config
    for key in _config:
        _apply_config(key)

def save_config():
    global _config
    file_path = os.getcwd() + '/config.json'
    with open(file_path, 'w') as f:
        json.dump(_config, f)

def load_config():
    global _config
    file_path = os.getcwd() + '/config.json'
    try:
        with open(file_path, 'r') as f:
            _config.update(json.load(f))
            apply_config()
    except FileNotFoundError as e:
        pass
