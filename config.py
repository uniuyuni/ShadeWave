
import os
import json
import multiprocessing

_config = None
_main_widget = None
_preview_texture_size = None

def init_config(widget):
    global _config, _main_widget, _preview_texture_size
    _config = multiprocessing.Manager().dict()
    _main_widget = widget

    _config.update(
    {
        'import_path': os.getcwd() + "/test_photos",
        'lut_path': os.getcwd() + "/lut",
        'preview_size': 640,
        'ai_demosaic': False,
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
    _preview_texture_size = (_config['preview_size'], _config['preview_size'])

    if not os.path.exists(os.getcwd() + '/config.json'):
        save_config()


def get_preview_min_size():
    global _config
    if _config is None:
        return 640
    return int(_config['preview_size'])


def get_preview_texture_size():
    global _preview_texture_size
    if _preview_texture_size is None:
        size = get_preview_min_size()
        return (size, size)
    return _preview_texture_size


def get_preview_texture_side():
    width, height = get_preview_texture_size()
    return min(width, height)


def set_preview_texture_size(width, height):
    global _preview_texture_size
    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    _preview_texture_size = (width, height)

def get_config(key):
    global _config

    if key == 'preview_width':
        return get_preview_texture_size()[0]
    if key == 'preview_height':
        return get_preview_texture_size()[1]

    return _config[key]

def set_config(key, value):
    _config[key] = value
    if key == 'preview_size':
        width, height = get_preview_texture_size()
        min_size = get_preview_min_size()
        set_preview_texture_size(max(width, min_size), max(height, min_size))
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
    elif key == 'preview_size' and _main_widget is not None:
        if hasattr(_main_widget, "sync_preview_widget_min_size"):
            _main_widget.sync_preview_widget_min_size()

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
            min_size = get_preview_min_size()
            set_preview_texture_size(min_size, min_size)
            apply_config()
    except FileNotFoundError as e:
        pass
