import copy
import json
import logging
import os
from datetime import datetime as dt

import numpy as np

import define
import params
from cores import pmck_store
from utils import paths
from widgets.switch_reset_map import build_switch_reset_targets


CONFIG_EFFECT_SELECTOR_KEYS = "effect_selector_selected_switch_keys"
PRESET_VERSION = 1

_HEAVY_KEYS = set(getattr(params, "HEAVY_PRIMARY_PARAM_KEYS", ())) | {
    "heavy_saved_at_fidelity",
}
_IMAGE_LOCAL_KEYS = {
    "exif_data",
    "image_fidelity",
    "img_size",
    "original_img_size",
    "disp_info",
    "crop_rect",
    "rating",
}


def get_preset_dir():
    return str(paths.presets_dir())


def ensure_preset_dir():
    path = get_preset_dir()
    os.makedirs(path, exist_ok=True)
    return path


def preset_path_for_name(name):
    safe = "".join(c for c in str(name).strip() if c not in '/\\:*?"<>|')
    safe = safe.strip()
    if not safe:
        raise ValueError("Preset name is empty.")
    if not safe.lower().endswith(".json"):
        safe += ".json"
    return os.path.join(ensure_preset_dir(), safe)


def list_presets():
    folder = ensure_preset_dir()
    names = []
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(".json"):
            names.append(os.path.splitext(file_name)[0])
    names.sort(key=str.lower)
    return names


def get_saved_selector_switch_keys(config_module):
    try:
        value = config_module.get_config(CONFIG_EFFECT_SELECTOR_KEYS)
    except Exception:
        value = []
    if not isinstance(value, list):
        return []
    return [str(x) for x in value]


def save_selector_switch_keys(config_module, switch_keys):
    config_module.set_config(CONFIG_EFFECT_SELECTOR_KEYS, list(dict.fromkeys(switch_keys)))


def _clone_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(v) for v in value]
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return value


def _strip_unwanted(primary_param):
    result = {}
    for key, value in primary_param.items():
        if key in _HEAVY_KEYS or key in _IMAGE_LOCAL_KEYS:
            continue
        result[key] = _clone_value(value)
    return result


def _target_param_defaults(effects_stack, current_param, switch_keys):
    targets = build_switch_reset_targets()
    selected = {}
    for switch_id in switch_keys:
        target = targets.get(switch_id)
        if target is None:
            continue
        lv, effect, subname = target
        effect_names = effect if isinstance(effect, list) else [effect]
        if subname is not None:
            eff = effects_stack[lv][effect_names[0]]
            selected.update(eff.get_param_dict(current_param, subname))
            continue
        for effect_name in effect_names:
            selected.update(effects_stack[lv][effect_name].get_param_dict(current_param))
    return selected


def collect_selected_primary_param(effects_stack, current_param, switch_keys):
    defaults = _target_param_defaults(effects_stack, current_param, switch_keys)
    selected = {}
    for key, default in defaults.items():
        value = current_param.get(key, default)
        selected[key] = _clone_value(value)
    return _strip_unwanted(selected)


def apply_partial_primary_param(target_param, partial_param):
    clean = _strip_unwanted(partial_param or {})
    for key, value in clean.items():
        target_param[key] = _clone_value(value)


def build_preset_dict(partial_primary_param):
    return {
        "make": "Platypus",
        "date": dt.now().strftime("%Y/%m/%d"),
        "version": define.VERSION,
        "preset_version": PRESET_VERSION,
        "primary_param": _json_value(_strip_unwanted(partial_primary_param or {})),
        "mask2": None,
    }


def save_preset_json(file_path, preset_dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(preset_dict, f, ensure_ascii=False, indent=2)


def load_preset_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    primary_param = data.get("primary_param")
    if not isinstance(primary_param, dict):
        primary_param = {}
    if "matrix" in primary_param:
        primary_param["matrix"] = np.array(primary_param["matrix"])
    return _strip_unwanted(primary_param)


def read_pmck_dict(pmck_path):
    data = pmck_store.read_path(pmck_path, default_empty=True)
    return pmck_store.ensure_primary_param(data)


def write_pmck_dict(pmck_path, data):
    pmck_store.write_path(pmck_path, data)


def apply_partial_to_pmck_file(image_path, partial_param):
    pmck_path = pmck_store.image_pmck_path(image_path)
    def _apply(data):
        data = pmck_store.ensure_primary_param(data)
        primary_param = data.setdefault("primary_param", {})
        apply_partial_primary_param(primary_param, partial_param)
        return data

    pmck_store.update_path(pmck_path, _apply, default_empty=True)
    return pmck_path


def cleanup_pmck_backup_files(directory):
    if not directory or not os.path.isdir(directory):
        return
    for file_name in os.listdir(directory):
        if file_name.endswith(".pmck.bak") or file_name.endswith(".pmck.swap_tmp"):
            try:
                os.remove(os.path.join(directory, file_name))
            except OSError:
                logging.exception("failed to remove pmck backup file: %s", file_name)


def backup_pmck_for_batch(image_path):
    pmck_path = pmck_store.image_pmck_path(image_path)
    bak_path = pmck_path + ".bak"
    tmp_path = pmck_path + ".swap_tmp"
    for path in (bak_path, tmp_path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    had_pmck = os.path.exists(pmck_path)
    if had_pmck:
        pmck_store.copy_path_to_path(pmck_path, bak_path)
    return {
        "image_path": image_path,
        "pmck_path": pmck_path,
        "bak_path": bak_path,
        "tmp_path": tmp_path,
        "had_pmck": had_pmck,
        "bak_role": "before" if had_pmck else "none",
    }


def undo_batch_item(item):
    pmck_path = item["pmck_path"]
    bak_path = item["bak_path"]
    tmp_path = item["tmp_path"]
    if item.get("had_pmck"):
        pmck_store.swap_paths(pmck_path, bak_path, tmp_path)
    else:
        if os.path.exists(bak_path):
            os.remove(bak_path)
        pmck_store.move_path_to_path(pmck_path, bak_path)
    item["bak_role"] = "after"


def redo_batch_item(item):
    pmck_path = item["pmck_path"]
    bak_path = item["bak_path"]
    tmp_path = item["tmp_path"]
    if item.get("had_pmck"):
        pmck_store.swap_paths(pmck_path, bak_path, tmp_path)
    else:
        pmck_store.move_path_to_path(bak_path, pmck_path)
    item["bak_role"] = "before"


def finalize_batch_item(item):
    pmck_path = item["pmck_path"]
    bak_path = item["bak_path"]
    if item.get("bak_role") == "after":
        # The user left this operation undone. Remove the redo payload.
        if os.path.exists(bak_path):
            os.remove(bak_path)
        return
    if os.path.exists(pmck_path):
        # Operation is applied. The backup is only for in-session undo.
        if os.path.exists(bak_path):
            os.remove(bak_path)
    item["bak_role"] = "none"


def cleanup_batch_item(item):
    for key in ("bak_path", "tmp_path"):
        path = item.get(key)
        if path:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
