
import os
import json
import multiprocessing

_config = None
_main_widget = None

def init_config(widget):
    global _config, _main_widget
    _config = multiprocessing.Manager().dict()
    _main_widget = widget

    _config.update(
    {
        'import_path': os.getcwd() + "/test_photos",
        'lut_path': os.getcwd() + "/lut",
        'preview_size': 1024,
        'raw_auto_exposure': True,
        'scale_threshold': 0.5,
        'inpaint_resize_limit': 1024,
        'inpaint_use_realesrgan': True,
        'display_color_gamut': "sRGB",
        'gpu_device': "mps",
        'cat': "cat16",
        'base_resolution_scale': [4096, 4096],
        'display_output_dither': False,
        'display_output_downscale': True
    })

    if not os.path.exists(os.getcwd() + '/config.json'):
        save_config()

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
        _main_widget.ids['viewer'].set_path(_config.get('import_path', os.getcwd() + "/test_photos"))
    elif key in ['display_output_dither', 'display_output_downscale']:
        _main_widget.texture = None

def apply_config():
    global _config
    for key in _config:
        _apply_config(key)

def save_config():
    global _config
    file_path = os.getcwd() + '/config.json'
    with open(file_path, 'w') as f:
        json.dump(dict(_config), f)

def load_config():
    global _config    
    file_path = os.getcwd() + '/config.json'
    try:
        with open(file_path, 'r') as f:
            _config.update(json.load(f))
            apply_config()
    except FileNotFoundError as e:
        pass
