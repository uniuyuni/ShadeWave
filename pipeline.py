
import numpy as np
import logging
import json
import os
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

import config
import params
import effects
import cores.core as core
import utils.utils as utils
from enums import EffectMode, PipelineStatus

_PIPELINE_TIMING_ENABLED = os.getenv("PLATYPUS_PIPELINE_TIMING", "0") == "1"
_PIPELINE_TIMING_EVERY = max(1, int(os.getenv("PLATYPUS_PIPELINE_TIMING_EVERY", "1")))
_PIPELINE_TIMING_LOCK = threading.Lock()
_PIPELINE_TIMING_FRAME_SEQ = 0
_PIPELINE_TIMING_LOG_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
_DEBUG_MASK_GEOMETRY = os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_MASK_ZOOM_SYNC = os.getenv("PLATYPUS_DEBUG_MASK_ZOOM_SYNC", "0").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_PIPELINE_STATS = os.getenv("PLATYPUS_DEBUG_PIPELINE_STATS", "0").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_PIPELINE_STATS_VERBOSE = os.getenv("PLATYPUS_DEBUG_PIPELINE_STATS_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_PIPELINE_STATS_EFFECTS = {
    name.strip()
    for name in os.getenv(
        "PLATYPUS_DEBUG_PIPELINE_STATS_EFFECTS",
        "color_temperature,auto_exposure,input_lut,look_lut,color_separation,hls2rgb2,vignette",
    ).split(",")
    if name.strip()
}
_DEBUG_PIPELINE_STATS_LABELS = {
    name.strip()
    for name in os.getenv(
        "PLATYPUS_DEBUG_PIPELINE_STATS_LABELS",
        "primary crop,primary after lv2,primary after lv3",
    ).split(",")
    if name.strip()
}


def _mask_geom_debug(message, *args):
    if _DEBUG_MASK_GEOMETRY:
        logging.warning("[MASK_GEOM] " + message, *args)


def _mask_zoom_sync_debug(message, *args):
    if _DEBUG_MASK_ZOOM_SYNC:
        logging.warning("[MASK_ZOOM_SYNC] " + message, *args)


def _debug_pipeline_param_summary(effect_name, param, effect=None):
    if not _DEBUG_PIPELINE_STATS or param is None:
        return ""
    keys_by_effect = {
        "color_temperature": (
            "switch_white_balance",
            "color_temperature_reset",
            "color_temperature",
            "color_tint_reset",
            "color_tint",
            "color_Y",
        ),
        "auto_exposure": ("switch_lut", "rgb_or_raw", "auto_exposure", "lut_to_log", "lut_name"),
        "input_lut": ("switch_lut", "lut_name", "lut_intensity", "lut_to_log"),
        "look_lut": ("switch_lut", "lut_name", "lut_intensity", "lut_to_log"),
        "exposure": ("exposure",),
        "contrast": ("contrast",),
        "tone": ("high_light", "light", "dark", "shadow"),
        "level": ("black_level", "white_level", "gamma"),
        "color_separation": (
            "switch_global",
            "shadow_chroma_clean",
            "shadow_chroma_threshold",
            "color_separation",
            "chroma_clarity",
            "color_density",
            "subtractive_saturation",
            "detail_tonemap",
        ),
        "clahe": ("switch_clahe", "clahe_clip_limit", "clahe_tile_grid_size"),
        "hls": ("hue", "lightness", "saturation2"),
        "vs_and_saturation": ("saturation", "vibrance"),
        "curves": ("tonecurve", "tonecurve_red", "tonecurve_green", "tonecurve_blue"),
        "film_emulation": ("film_simulation", "film_simulation_intensity"),
        "solid_color": ("solid_color", "solid_color_intensity"),
        "unsharp_mask": ("unsharp_mask", "unsharp_mask_radius", "unsharp_mask_amount"),
        "grain": (
            "switch_grain",
            "grain_intensity",
            "grain_size",
            "grain_blue_bias",
            "grain_shadow_boost",
            "grain_color_noise_ratio",
        ),
        "vignette": (
            "switch_vignette",
            "vignette_intensity",
            "vignette_radius_percent",
            "vignette_softness",
        ),
    }
    keys = keys_by_effect.get(effect_name)
    if not keys:
        return ""
    values = []
    for key in keys:
        try:
            if effect is not None:
                value = effect._get_param(param, key)
                source = "param" if key in param else "default"
                values.append(f"{key}={value!r}({source})")
            elif key in param:
                values.append(f"{key}={param[key]!r}")
        except Exception:
            if key in param:
                values.append(f"{key}={param[key]!r}")
    return " params={" + ", ".join(values) + "}" if values else ""


def _debug_pipeline_image_stats(label, img, *, effect_name=None, param=None, effect=None, state=None):
    if not _DEBUG_PIPELINE_STATS:
        return
    if not _DEBUG_PIPELINE_STATS_VERBOSE:
        if effect_name is None and label not in _DEBUG_PIPELINE_STATS_LABELS:
            return
        if effect_name is not None and effect_name not in _DEBUG_PIPELINE_STATS_EFFECTS:
            return
    if img is None:
        logging.warning("[PIPELINE_STATS] %s image=None", label)
        return
    try:
        arr = np.asarray(img)
        if arr.size == 0:
            logging.warning("[PIPELINE_STATS] %s shape=%s empty", label, getattr(arr, "shape", None))
            return
        finite = np.isfinite(arr)
        finite_count = int(np.count_nonzero(finite))
        total_count = int(arr.size)
        if finite_count == 0:
            logging.warning(
                "[PIPELINE_STATS] %s shape=%s dtype=%s finite=0/%d",
                label,
                arr.shape,
                arr.dtype,
                total_count,
            )
            return

        finite_values = arr[finite]
        msg_parts = [
            f"[PIPELINE_STATS] {label}",
            f"shape={arr.shape}",
            f"dtype={arr.dtype}",
            f"finite={finite_count}/{total_count}",
            f"min={float(np.min(finite_values)):.6g}",
            f"max={float(np.max(finite_values)):.6g}",
            f"mean={float(np.mean(finite_values)):.6g}",
            f"neg={int(np.count_nonzero(arr < 0))}",
            f"over1={int(np.count_nonzero(arr > 1))}",
        ]
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3]
            ch_min = np.nanmin(rgb, axis=(0, 1))
            ch_max = np.nanmax(rgb, axis=(0, 1))
            ch_mean = np.nanmean(rgb, axis=(0, 1))
            msg_parts.extend([
                "ch_min=({:.6g},{:.6g},{:.6g})".format(*ch_min),
                "ch_max=({:.6g},{:.6g},{:.6g})".format(*ch_max),
                "ch_mean=({:.6g},{:.6g},{:.6g})".format(*ch_mean),
            ])
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            shadow = np.isfinite(luma) & (luma >= 0.0) & (luma < 0.03)
            if np.any(shadow):
                shadow_rgb = rgb[shadow]
                shadow_spread = np.max(shadow_rgb, axis=1) - np.min(shadow_rgb, axis=1)
                shadow_mean = np.nanmean(shadow_rgb, axis=0)
                msg_parts.extend([
                    f"shadow_px={int(np.count_nonzero(shadow))}",
                    f"shadow_spread_mean={float(np.nanmean(shadow_spread)):.6g}",
                    "shadow_ch_mean=({:.6g},{:.6g},{:.6g})".format(*shadow_mean),
                ])
        if state:
            msg_parts.append(f"state={state}")
        if effect_name:
            msg_parts.append(_debug_pipeline_param_summary(effect_name, param, effect=effect))
        logging.warning(" ".join(part for part in msg_parts if part))
    except Exception:
        logging.exception("[PIPELINE_STATS] failed to inspect %s", label)


def _mask_geom_id(mask):
    if mask is None:
        return None
    mask_id = getattr(mask, "mask_id", "")
    short_id = str(mask_id)[:8] if mask_id else "no-id"
    return f"{mask.__class__.__name__}:{short_id}@{id(mask):x}"


def _mask_geom_image_stats(image):
    if image is None:
        return None
    try:
        return {
            "shape": tuple(int(v) for v in image.shape),
            "hash": hash(np.ascontiguousarray(image).tobytes()),
            "min": float(np.nanmin(image)),
            "max": float(np.nanmax(image)),
            "sum": float(np.nansum(image)),
            "nonzero": int(np.count_nonzero(image)),
        }
    except Exception:
        return {"shape": getattr(image, "shape", None)}


def _mask_geom_weight_stats(weight):
    if weight is None:
        return None
    try:
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim == 3:
            weight = np.mean(np.abs(weight), axis=2)
        total = float(np.nansum(weight))
        stats = {
            "shape": tuple(int(v) for v in weight.shape),
            "sum": total,
            "max": float(np.nanmax(weight)),
            "nonzero": int(np.count_nonzero(weight)),
        }
        if total > 0.0:
            yy, xx = np.indices(weight.shape, dtype=np.float32)
            stats["cx"] = float(np.nansum(xx * weight) / total)
            stats["cy"] = float(np.nansum(yy * weight) / total)
        return stats
    except Exception:
        return {"shape": getattr(weight, "shape", None)}


def _mask_geom_param_summary(param):
    if not param:
        return {}
    keys = (
        "switch_mask_geometry",
        "mask_rotation",
        "mask_translation_x",
        "mask_translation_y",
        "mask_scale_x",
        "mask_scale_y",
        "mask_flip_mode",
    )
    return {key: param.get(key) for key in keys if key in param}

def _effective_mask2_draw_effect_param(composit_mask):
    """Draw Effects は Composit レイヤーの設定だけを使う。"""
    return composit_mask.effects_param


def _is_nan_inf_debug_enabled():
    env = os.getenv("PLATYPUS_DEBUG_NAN_INF")
    if env is not None:
        return env.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return bool(config.get_config("debug_nan_inf_check"))
    except Exception:
        return False


def _next_pipeline_frame_id():
    global _PIPELINE_TIMING_FRAME_SEQ
    with _PIPELINE_TIMING_LOCK:
        _PIPELINE_TIMING_FRAME_SEQ += 1
        return _PIPELINE_TIMING_FRAME_SEQ


def _pipeline_timing_log_path():
    root = os.getenv("PIXI_PROJECT_ROOT") or os.getcwd()
    out_dir = os.path.join(root, ".profiles")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"pipeline_timing-{_PIPELINE_TIMING_LOG_STAMP}.jsonl")


def _new_pipeline_timing(is_drag):
    if not _PIPELINE_TIMING_ENABLED:
        return None
    frame_id = _next_pipeline_frame_id()
    return {
        "frame_id": frame_id,
        "is_drag": bool(is_drag),
        "frame_start": time.perf_counter(),
        "frame_start_unix_ms": int(time.time() * 1000),
        "frame_start_iso": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        "effect_total_ms": 0.0,
        "loop_orchestration_ms": 0.0,
        "overhead_components_ms": {},
        "cache_events": {},
        "reeffect_reasons": {},
        "sections_ms": {},
        "effects": {},  # effect_name -> counters
    }


def _timing_add_section_ms(timing, section, elapsed_ms):
    if timing is None:
        return
    timing["sections_ms"][section] = timing["sections_ms"].get(section, 0.0) + float(elapsed_ms)


def _timing_count(bucket, key, value):
    bucket[key] = bucket.get(key, 0) + int(value)


def _timing_record_effect(
    timing,
    effect_name,
    make_ms,
    apply_ms,
    iteration_ms,
    overhead_components=None,
    cache_event=None,
    reeffect_reason=None,
):
    if timing is None:
        return
    make_ms = float(make_ms)
    apply_ms = float(apply_ms)
    iteration_ms = float(iteration_ms)
    effect_ms = make_ms + apply_ms
    loop_overhead_ms = max(0.0, iteration_ms - effect_ms)
    overhead_components = overhead_components or {}

    timing["effect_total_ms"] += effect_ms
    timing["loop_orchestration_ms"] += loop_overhead_ms

    bucket = timing["effects"].setdefault(
        effect_name,
        {
            "calls": 0,
            "make_ms": 0.0,
            "apply_ms": 0.0,
            "effect_ms": 0.0,
            "loop_overhead_ms": 0.0,
            "overhead_components_ms": {},
            "cache_events": {},
            "reeffect_reasons": {},
        },
    )
    bucket["calls"] += 1
    bucket["make_ms"] += make_ms
    bucket["apply_ms"] += apply_ms
    bucket["effect_ms"] += effect_ms
    bucket["loop_overhead_ms"] += loop_overhead_ms
    if cache_event:
        _timing_count(timing["cache_events"], cache_event, 1)
        _timing_count(bucket["cache_events"], cache_event, 1)
    if reeffect_reason:
        _timing_count(timing["reeffect_reasons"], reeffect_reason, 1)
        _timing_count(bucket["reeffect_reasons"], reeffect_reason, 1)
    known = 0.0
    for comp, value in overhead_components.items():
        value = float(value)
        known += value
        timing["overhead_components_ms"][comp] = timing["overhead_components_ms"].get(comp, 0.0) + value
        bucket["overhead_components_ms"][comp] = bucket["overhead_components_ms"].get(comp, 0.0) + value
    unknown = max(0.0, loop_overhead_ms - known)
    timing["overhead_components_ms"]["unknown_ms"] = timing["overhead_components_ms"].get("unknown_ms", 0.0) + unknown
    bucket["overhead_components_ms"]["unknown_ms"] = bucket["overhead_components_ms"].get("unknown_ms", 0.0) + unknown


def _timing_effect_name(efconfig, level, effect_name):
    layer = getattr(efconfig, "pipeline_layer_label", "primary")
    if layer:
        return f"{layer}.lv{level}.{effect_name}"
    return f"lv{level}.{effect_name}"


def _timing_mask_layer_label(mask, index):
    mask_id = str(getattr(mask, "mask_id", ""))[:8]
    return f"mask{index}:{mask_id}" if mask_id else f"mask{index}"


def _reset_effect_cache_event(effect):
    if hasattr(effect, "_last_cache_event"):
        effect._last_cache_event = None


def _classify_cache_event(effect, pre_hash, pre_diff, diff, reeffected):
    event = getattr(effect, "_last_cache_event", None)
    if event:
        return event
    post_hash = getattr(effect, "hash", None)
    post_diff = getattr(effect, "diff", None)
    if diff is None and post_diff is None:
        return "no_output_after_reset" if reeffected else "no_output"
    if reeffected:
        return "recomputed_after_reset"
    if pre_hash == post_hash and pre_diff is post_diff:
        return "hit"
    return "miss"


def _finalize_pipeline_timing(timing):
    if timing is None:
        return
    frame_total_ms = (time.perf_counter() - timing["frame_start"]) * 1000.0
    effect_total_ms = timing["effect_total_ms"]
    loop_orchestration_ms = timing["loop_orchestration_ms"]
    non_effect_non_loop_ms = max(0.0, frame_total_ms - effect_total_ms - loop_orchestration_ms)

    payload = {
        "frame_id": timing["frame_id"],
        "is_drag": timing["is_drag"],
        "frame_start_unix_ms": timing.get("frame_start_unix_ms"),
        "frame_start_iso": timing.get("frame_start_iso"),
        "logged_at_unix_ms": int(time.time() * 1000),
        "logged_at_iso": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        "frame_total_ms": round(frame_total_ms, 4),
        "effect_total_ms": round(effect_total_ms, 4),
        "loop_orchestration_ms": round(loop_orchestration_ms, 4),
        "non_effect_non_loop_ms": round(non_effect_non_loop_ms, 4),
        "overhead_components_ms": {k: round(v, 4) for k, v in timing["overhead_components_ms"].items()},
        "cache_events": dict(timing["cache_events"]),
        "reeffect_reasons": dict(timing["reeffect_reasons"]),
        "sections_ms": {k: round(v, 4) for k, v in timing["sections_ms"].items()},
        "effects": {},
    }
    for name, stats in timing["effects"].items():
        row = {}
        for key, value in stats.items():
            if key == "overhead_components_ms":
                row[key] = {k: round(v, 4) for k, v in value.items()}
            elif isinstance(value, float):
                row[key] = round(value, 4)
            else:
                row[key] = value
        payload["effects"][name] = row
    if payload["frame_id"] % _PIPELINE_TIMING_EVERY != 0:
        return
    try:
        with open(_pipeline_timing_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as e:
        logger.warning("pipeline timing write failed: %s", e)


def _configure_preview_effect_config(efconfig, current_tab, mask2_active):
    is_geometry_tab = current_tab == "Ge"
    mask2_geometry_full_preview = is_geometry_tab and mask2_active

    efconfig.current_tab = current_tab
    efconfig.full_preview = is_geometry_tab
    # Mask2 ON 中の Ge タブは「マスク Geometry モード」であって画像 crop の編集ではないので
    # CropEffect には crop_editing=False を渡し、CropEditor を開く/disp_info を永続更新する
    # 経路には入れない。表示だけは process_pipeline 内の local disp_info で full-preview にする。
    efconfig.crop_editing = is_geometry_tab and not mask2_active

    return mask2_geometry_full_preview


class AsyncPipelineManager:
    def __init__(self, worker):
        self.worker = worker
        # Cache for async results: (effect_name, param_hash) -> {'status': ..., 'result': ..., 'task_id': ...}
        self.cache = {} 
        self.current_pipeline_version = 0

    def set_pipeline_version(self, version):
        if self.current_pipeline_version != version:
            self.current_pipeline_version = version
            # Clear cache for the new version? or keep valid results?
            # For now, simplistic: we rely on param_hash.
            pass

    def get_pipeline_version(self):
        return self.current_pipeline_version

    def get_result(self, effect_name, param_hash):
        key = (effect_name, param_hash)
        res = self.cache.get(key)
        logger.debug(
            "get_result %s %s -> %s",
            effect_name,
            param_hash,
            res["status"] if res else "None",
        )
        return res

    def discard_result(self, effect_name, param_hash):
        self.cache.pop((effect_name, param_hash), None)

    def discard_effect_results(self, effect_name):
        for key in [key for key in self.cache if key[0] == effect_name]:
            del self.cache[key]

    def submit_task(self, effect_name, img, params, efconfig, param_hash):
        key = (effect_name, param_hash)
        
        # Check if already submitted
        if key in self.cache:
            if self.cache[key]['status'] in ['RUNNING', 'COMPLETE']:
                logger.debug(
                    "submit_task HIT %s %s status=%s",
                    effect_name,
                    param_hash,
                    self.cache[key]["status"],
                )
                return self.cache[key]
        
        logger.debug(
            "submit_task MISS %s %s. Existing keys: %s",
            effect_name,
            param_hash,
            [k for k in self.cache.keys() if k[0] == effect_name],
        )

        # Check if ANY task for this effect is running (to support cancellation/restart)
        # If we are submitting a NEW task for the same effect, it means parameters changed.
        # We should kill the old one to save resources.
        for existing_key, info in self.cache.items():
            if existing_key[0] == effect_name and info['status'] == 'RUNNING':
                # Found a running task for this effect but with different params (since we define key by param_hash)
                # Restart worker to kill it
                self.worker.restart()
                # Mark as cancelled in cache? 
                # Actually restart clears the worker, so all running tasks are dead.
                # We should probably clear 'RUNNING' status in cache?
                # But simple restart is enough, subsequent polls will fail or timed out?
                # Actually if we restart locally, we should clear the cache of running items.
                break
        
        # Also clean cache for running items if we restarted?
        # For simplicity, just submit.
                
        # Submit
        # We need to ensure 'img' is CPU accessible and ready
        if isinstance(img, np.ndarray):
            # Clone efconfig just in case
            # efconfig object might not be picklable? It seems simple enough.
            task_id = self.worker.submit_task(effect_name, img, params, efconfig)
            self.cache[key] = {
                'status': 'RUNNING',
                'task_id': task_id,
                'result': None
            }
            return self.cache[key]
        else:
            logging.error(f"Cannot submit task for {effect_name}: Image is not ndarray")
            return None

    def cancel_effect(self, effect_name):
        # Check if we have running tasks for this effect
        running = False
        keys_to_remove = []
        for key, info in self.cache.items():
            if key[0] == effect_name:
                # Only remove running tasks from cache? 
                # If we remove COMPLETE tasks, we lose cache when toggling visibility.
                if info['status'] == 'RUNNING':
                    keys_to_remove.append(key)
                    running = True
        
        if running:
            self.worker.cancel_effect(effect_name)
            
        # Also clean cache
        for k in keys_to_remove:
            del self.cache[k]

    def update_result(self, task_id, result_image):
        # Update cache with result
        for key, value in self.cache.items():
            if value.get('task_id') == task_id:
                value['status'] = 'COMPLETE'
                value['result'] = result_image
                return key
        return None
    
    def cancel_all(self):
        self.worker.cancel_all()
        self.cache.clear()

    def clear_completed_cache(self):
        keys = [
            key for key, value in self.cache.items()
            if isinstance(value, dict) and value.get("status") == "COMPLETE"
        ]
        for key in keys:
            del self.cache[key]
        return len(keys)


def process_pipeline(img, crop_image, is_zoomed, zoom_ratio, texture_width, texture_height, click_x, click_y, primary_effects, primary_param, mask_editor2, processor, pipeline_version, current_tab, loading_flag=-1, is_drag=False, center_pos=None, mask2_active=False):
    timing = _new_pipeline_timing(is_drag)
    if timing is not None:
        _t0 = time.perf_counter()
    if not params.has_original_img_size(primary_param):
        logging.warning("process_pipeline: original_img_size 未定義のため処理しません")
        if timing is not None:
            _timing_add_section_ms(timing, "guard", (time.perf_counter() - _t0) * 1000.0)
            _finalize_pipeline_timing(timing)
        return None, crop_image
    if timing is not None:
        _timing_add_section_ms(timing, "guard", (time.perf_counter() - _t0) * 1000.0)

    params.ensure_initial_crop_rect(primary_param)

    # クロップ情報を得る、ない場合元のクロップ情報から展開
    if timing is not None:
        _t0 = time.perf_counter()
    disp_info = params.get_disp_info(primary_param)
    if disp_info is None:
        disp_info = core.convert_rect_to_info(params.get_crop_rect(primary_param), config.get_preview_texture_side()/max(primary_param['original_img_size']))
        params.set_disp_info(primary_param, disp_info)
    if timing is not None:
        _timing_add_section_ms(timing, "disp_info", (time.perf_counter() - _t0) * 1000.0)

    # 環境設定
    if timing is not None:
        _t0 = time.perf_counter()
    efconfig = effects.EffectConfig()
    efconfig.disp_info = disp_info
    efconfig.is_zoomed = is_zoomed
    efconfig.zoom_ratio = zoom_ratio
    efconfig.mode = EffectMode.PREVIEW
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], 1.0)
    mask2_geometry_full_preview = _configure_preview_effect_config(efconfig, current_tab, mask2_active)
    
    # Initialize basic input hash
    efconfig.loading_flag = loading_flag
    efconfig.image_fidelity = primary_param.get('image_fidelity')
    efconfig.upstream_hash = hash(id(img))
    efconfig.pipeline_timing = timing
    efconfig.pipeline_layer_label = "primary"
    efconfig.debug_nan_inf_check = _is_nan_inf_debug_enabled()
    if timing is not None:
        _timing_add_section_ms(timing, "efconfig_setup", (time.perf_counter() - _t0) * 1000.0)

    if mask_editor2 is not None:
        mask_editor2.set_texture_size(texture_width, texture_height)

    # 背景レイヤー
    if timing is not None:
        _t0 = time.perf_counter()
    img0, lv1reset, pre_rotation_img, _ = pipeline_lv0(img, primary_effects, primary_param, efconfig, processor=processor)
    disp_info = params.get_disp_info(primary_param) # Cropによって値が更新されてるかも
    if mask2_geometry_full_preview:
        msize = max(primary_param['original_img_size'][0], primary_param['original_img_size'][1])
        disp_info = (0, 0, msize, msize, config.get_preview_texture_side() / msize)
    if timing is not None:
        _timing_add_section_ms(timing, "pipeline_lv0", (time.perf_counter() - _t0) * 1000.0)

    if timing is not None:
        _t0 = time.perf_counter()
    if crop_image is None or lv1reset == True:
        _mask_zoom_sync_debug(
            "pipeline crop_refresh frame=%s crop_none=%s lv1reset=%s is_zoomed=%s zoom_ratio=%.3f click=(%.2f,%.2f) center_pos=%s in_disp=%s texture=%sx%s",
            pipeline_version, crop_image is None, lv1reset, is_zoomed, zoom_ratio,
            click_x, click_y, center_pos, disp_info, texture_width, texture_height,
        )
        _mask_geom_debug(
            "process_pipeline crop refresh crop_image_none=%s lv1reset=%s center_pos=%s texture=%sx%s",
            crop_image is None,
            lv1reset,
            center_pos,
            texture_width,
            texture_height,
        )
        deferred_geometry = getattr(efconfig, "deferred_geometry_transform", None)
        if deferred_geometry is not None:
            try:
                if is_zoomed:
                    imgc, disp_info2 = core.transform_zoom_crop_image(
                        pre_rotation_img,
                        deferred_geometry["matrix"],
                        deferred_geometry["width"],
                        deferred_geometry["height"],
                        disp_info,
                        params.get_crop_rect(primary_param),
                        texture_width,
                        texture_height,
                        click_x,
                        click_y,
                        center_pos,
                        zoom_ratio=zoom_ratio,
                        border_mode=deferred_geometry.get("border_mode", "reflect"),
                        transform_type=deferred_geometry.get("transform_type", "affine"),
                        lens_strength=deferred_geometry.get("lens_strength", 0.0),
                        lens_scale=deferred_geometry.get("lens_scale", 1.0),
                        mesh_map_x=deferred_geometry.get("mesh_map_x"),
                        mesh_map_y=deferred_geometry.get("mesh_map_y"),
                    )
                else:
                    imgc, disp_info2 = core.transform_crop_image(
                        pre_rotation_img,
                        deferred_geometry["matrix"],
                        deferred_geometry["width"],
                        deferred_geometry["height"],
                        disp_info,
                        texture_width,
                        texture_height,
                        border_mode=deferred_geometry.get("border_mode", "reflect"),
                        transform_type=deferred_geometry.get("transform_type", "affine"),
                        lens_strength=deferred_geometry.get("lens_strength", 0.0),
                        lens_scale=deferred_geometry.get("lens_scale", 1.0),
                        mesh_map_x=deferred_geometry.get("mesh_map_x"),
                        mesh_map_y=deferred_geometry.get("mesh_map_y"),
                    )
            except Exception:
                logging.exception("deferred geometry preview failed; falling back to two-pass crop")
                imgc, disp_info2 = core.crop_image(img0, disp_info, params.get_crop_rect(primary_param), texture_width, texture_height, click_x, click_y, is_zoomed, center_pos, zoom_ratio=zoom_ratio)
        else:
            imgc, disp_info2 = core.crop_image(img0, disp_info, params.get_crop_rect(primary_param), texture_width, texture_height, click_x, click_y, is_zoomed, center_pos, zoom_ratio=zoom_ratio)
        _mask_zoom_sync_debug(
            "pipeline crop_result frame=%s out_disp=%s imgc_shape=%s primary_disp_before_set=%s",
            pipeline_version, disp_info2, getattr(imgc, "shape", None),
            params.get_disp_info(primary_param),
        )
        if not mask2_geometry_full_preview:
            params.set_disp_info(primary_param, disp_info2)
        # 新規クロップ生成時は下流を必ず更新
        lv1reset = True
        
    else:
        _mask_zoom_sync_debug(
            "pipeline crop_reuse frame=%s is_zoomed=%s zoom_ratio=%.3f disp=%s crop_shape=%s",
            pipeline_version, is_zoomed, zoom_ratio, disp_info, getattr(crop_image, "shape", None),
        )
        _mask_geom_debug(
            "process_pipeline crop reuse crop_shape=%s disp_info=%s",
            getattr(crop_image, "shape", None),
            disp_info,
        )
        imgc = crop_image
        disp_info2 = disp_info
    mask_editor2.set_primary_param(primary_param, disp_info2, redraw_mask=False)
    mask_editor2.set_ref_image(imgc, pre_rotation_img)
    _debug_pipeline_image_stats("primary crop", imgc, param=primary_param)
    #mask_editor2.update()
    if timing is not None:
        _timing_add_section_ms(timing, "crop_and_mask_ref", (time.perf_counter() - _t0) * 1000.0)

    # 環境設定更新
    if timing is not None:
        _t0 = time.perf_counter()
    efconfig.disp_info = disp_info2
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], disp_info2[4])
    if timing is not None:
        _timing_add_section_ms(timing, "efconfig_update", (time.perf_counter() - _t0) * 1000.0)
    
    if not is_drag:
        if timing is not None:
            _t0 = time.perf_counter()
        img2, lv4reset = pipeline2(imgc, None, primary_effects, primary_param, mask_editor2, efconfig, lv1reset, processor=processor)
        if timing is not None:
            _timing_add_section_ms(timing, "pipeline2", (time.perf_counter() - _t0) * 1000.0)
            _t0 = time.perf_counter()
        img2 = pipeline_last(img2, primary_effects, primary_param, efconfig, prev_reset=lv4reset, processor=processor)
        if timing is not None:
            _timing_add_section_ms(timing, "pipeline_last", (time.perf_counter() - _t0) * 1000.0)
    else:
        img2 = imgc
        # Drag/space preview intentionally skips lv1-lv4, but the crop may have
        # changed. Do not keep this crop cache for the next normal frame, because
        # many downstream effects cache full processed images and rely on
        # lv1reset=True to invalidate those caches.
        crop_image = None

    _finalize_pipeline_timing(timing)
    return img2, crop_image if is_drag else imgc

def export_pipeline(img, primary_effects, primary_param, mask_editor2):
    if not params.has_original_img_size(primary_param):
        logging.error("export_pipeline: original_img_size 未定義のため処理しません")
        return None

    params.ensure_initial_crop_rect(primary_param)

    # 環境設定
    disp_info = core.convert_rect_to_info(params.get_crop_rect(primary_param), 1) # 倍率１で作成
    params.set_disp_info(primary_param, disp_info) # コピーしとく
    efconfig = effects.EffectConfig()
    efconfig.disp_info = disp_info
    efconfig.is_zoomed = True
    efconfig.mode = EffectMode.EXPORT
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], disp_info[4])
    efconfig.current_tab = None
    efconfig.crop_editing = False
    efconfig.full_preview = False
    efconfig.image_fidelity = primary_param.get('image_fidelity')
    efconfig.debug_nan_inf_check = _is_nan_inf_debug_enabled()

    # 背景レイヤー
    img0, lv1reset, pre_rotation_img, _ = pipeline_lv0(img, primary_effects, primary_param, efconfig, processor=None)

    # ここでクロップ (Export: Apply Crop FIRST)
    x1, y1, x2, y2 = params.get_crop_rect(primary_param)
    imgc = img0[y1:y2, x1:x2]
    
    mask_editor2.set_texture_size(imgc.shape[1], imgc.shape[0])
    mask_editor2.set_primary_param(primary_param, disp_info)    
    mask_editor2.set_ref_image(imgc, pre_rotation_img) 
    mask_editor2.update()

    img2, lv4reset = pipeline2(imgc, None, primary_effects, primary_param, mask_editor2, efconfig, lv1reset, processor=None)
    img2 = pipeline_last(img2, primary_effects, primary_param, efconfig, prev_reset=lv4reset, processor=None)
    
    return img2

def pipeline2(imgc, crop, primary_effects, primary_param, mask_editor2, efconfig, lv1reset=False, processor=None):
    # Initial status is COMPLETE because we start fresh or with completed image
    upstream_status = PipelineStatus.COMPLETE

    previous_layer_label = getattr(efconfig, "pipeline_layer_label", "primary")
    efconfig.pipeline_layer_label = "primary"
    _debug_pipeline_image_stats("primary pipeline2 input", imgc, param=primary_param)
    img1, lv2reset, upstream_status = pipeline_lv1(imgc, primary_effects, primary_param, efconfig, lv1reset, upstream_status, processor)
    _debug_pipeline_image_stats("primary after lv1", img1, param=primary_param)
    img2, lv3reset, upstream_status = pipeline_lv2(img1, primary_effects, primary_param, efconfig, lv2reset, upstream_status, processor)
    _debug_pipeline_image_stats("primary after lv2", img2, param=primary_param)
    img3, lv1reset, upstream_status = pipeline_lv3(img2, primary_effects, primary_param, efconfig, lv3reset, upstream_status, processor)
    _debug_pipeline_image_stats("primary after lv3", img3, param=primary_param)

    # マスクレイヤー
    if mask_editor2 is not None:
        mask_list = mask_editor2.get_mask_list()
        for i, mask in enumerate(mask_list):
            if not mask.is_composit():
                continue
            
            # Masks usually are light-weight or handled within their own simple pipeline.
            # Allowing async for masks might be complex due to dependency on 'upstream_status' of the base image?
            # Actually, `img3` is the base.
            
            # Reset upstream status for mask pipeline? 
            # The base image `img3` might be a preview. 
            # If `img3` is preview, the result of mask composite will also be preview quality.
            # But the mask *itself* generation might be heavy.
            # For now, let's pass current upstream_status.
            
            efconfig.pipeline_layer_label = _timing_mask_layer_label(mask, i)
            img2, lv2reset, _ = pipeline_lv1(img3, mask.effects, mask.effects_param, efconfig, lv1reset, upstream_status, processor)
            img2, lv1reset, _ = pipeline_lv2(img2, mask.effects, mask.effects_param, efconfig, lv2reset, upstream_status, processor)

            img2 = core.type_convert(img2, np.ndarray)
            img3 = core.type_convert(img3, np.ndarray)

            if _DEBUG_MASK_GEOMETRY:
                _mask_geom_debug(
                    "pipeline2 before_mask_image layer=%s composit=%s params=%s img3=%s img2=%s",
                    _timing_mask_layer_label(mask, i),
                    _mask_geom_id(mask),
                    _mask_geom_param_summary(getattr(mask, 'effects_param', None)),
                    _mask_geom_image_stats(img3),
                    _mask_geom_image_stats(img2),
                )
            mask_image = mask.get_mask_image()
            if crop is not None:
                mask_image = mask_image[crop[1]:crop[3], crop[0]:crop[2]]
            mask2_param = _effective_mask2_draw_effect_param(mask)
            if _DEBUG_MASK_GEOMETRY:
                _mask_geom_debug(
                    "pipeline2 apply_mask_draw_effects layer=%s composit=%s mask=%s mask_weight=%s draw_params=%s resolution_scale=%s",
                    _timing_mask_layer_label(mask, i),
                    _mask_geom_id(mask),
                    _mask_geom_image_stats(mask_image),
                    _mask_geom_weight_stats(mask_image),
                    _mask_geom_param_summary(mask2_param),
                    getattr(efconfig, "resolution_scale", 1.0),
                )
            img3_before_mask_apply = img3 if not _DEBUG_MASK_GEOMETRY else img3.copy()
            img3 = core.apply_mask_draw_effects(
                img3, mask_image, img2, mask2_param,
                resolution_scale=getattr(efconfig, "resolution_scale", 1.0),
            )
            # Mask composition mutates the image between lv3 and pipeline_last.
            # Treat it as an upstream change so final-stage cached effects do
            # not reuse a result made from the previous mask position.
            lv1reset = True
            child_state = tuple(
                (
                    getattr(child, "mask_id", None),
                    maskop,
                    getattr(child, "image_mask_cache_hash", None),
                    getattr(child, "segment_mask_cache_hash", None),
                    getattr(child, "depth_map_mask_cache_hash", None),
                    getattr(child, "faces_mask_cache_hash", None),
                )
                for child, maskop in getattr(mask, "mask_list", [])
            )
            efconfig.upstream_hash = hash((
                efconfig.upstream_hash,
                "mask_composite",
                getattr(mask, "mask_id", None),
                child_state,
                hash(repr(sorted(mask2_param.items()))),
            ))
            if _DEBUG_MASK_GEOMETRY:
                _mask_geom_debug(
                    "pipeline2 after_apply layer=%s composit=%s img3=%s delta_weight=%s",
                    _timing_mask_layer_label(mask, i),
                    _mask_geom_id(mask),
                    _mask_geom_image_stats(img3),
                    _mask_geom_weight_stats(img3 - img3_before_mask_apply),
                )

    efconfig.pipeline_layer_label = previous_layer_label
    return img3, lv1reset

def pipeline_lv0(img, effects, param, efconfig, processor=None):
    lv0 = effects[0]
    lv1reset = False
    
    pre_rotation_img = None
    
    rgb = img
    upstream_status = PipelineStatus.COMPLETE # Initial input is complete
    
    efconfig.processor = processor
    efconfig.layer_status = PipelineStatus.COMPLETE
    
    for i, n in enumerate(lv0):
        _iter_t0 = time.perf_counter() if getattr(efconfig, "pipeline_timing", None) is not None else None
        overhead_components = {}
        reeffected = False
        reeffect_reason = None
        # Update upstream status for the effect
        if _iter_t0 is not None:
            _t = time.perf_counter()
            efconfig.upstream_status = upstream_status
            overhead_components["set_upstream_status_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            efconfig.upstream_status = upstream_status
        
        if lv1reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv0[n].reeffect()
                overhead_components["reeffect_ms"] = overhead_components.get("reeffect_ms", 0.0) + (time.perf_counter() - _t) * 1000.0
            else:
                lv0[n].reeffect()
            reeffected = True
            reeffect_reason = "previous_lv0_effect_changed"
            
        if n == 'geometry':
            if _iter_t0 is not None:
                _t = time.perf_counter()
                pre_rotation_img = rgb
                overhead_components["special_case_ms"] = overhead_components.get("special_case_ms", 0.0) + (time.perf_counter() - _t) * 1000.0
            else:
                pre_rotation_img = rgb
            
        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv0[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv0[n].diff
        pre_hash = getattr(lv0[n], "hash", None)
        
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        _reset_effect_cache_event(lv0[n])
        diff = lv0[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        cache_event = _classify_cache_event(lv0[n], pre_hash, pre_diff, diff, reeffected)
        apply_ms = 0.0
        if diff is not None:
            _apply_t0 = time.perf_counter() if _iter_t0 is not None else None
            rgb = lv0[n].apply_diff(rgb)
            apply_ms = (time.perf_counter() - _apply_t0) * 1000.0 if _apply_t0 is not None else 0.0
            
        # Update upstream hash
        if _iter_t0 is not None:
            _t = time.perf_counter()
            efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv0[n], 'hash', None)))
            overhead_components["upstream_hash_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv0[n], 'hash', None)))

        # Update upstream status if layer became preview
        if _iter_t0 is not None:
            _t = time.perf_counter()
            if efconfig.layer_status == PipelineStatus.PREVIEW:
                upstream_status = PipelineStatus.PREVIEW
            overhead_components["status_propagation_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            if efconfig.layer_status == PipelineStatus.PREVIEW:
                upstream_status = PipelineStatus.PREVIEW

        if _iter_t0 is not None:
            _t = time.perf_counter()
            if pre_diff is not diff or pre_hash != getattr(lv0[n], "hash", None):
                lv1reset = True
            overhead_components["reset_flag_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            if pre_diff is not diff or pre_hash != getattr(lv0[n], "hash", None):
                lv1reset = True
        if _iter_t0 is not None:
            _timing_record_effect(
                efconfig.pipeline_timing,
                _timing_effect_name(efconfig, 0, n),
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
                cache_event,
                reeffect_reason,
            )
            
    if pre_rotation_img is None:
        pre_rotation_img = rgb

    return rgb, lv1reset, pre_rotation_img, upstream_status

def pipeline_lv1(img, effects, param, efconfig, prev_reset=False, upstream_status=PipelineStatus.COMPLETE, processor=None):
    lv1 = effects[1]
    lv2reset = prev_reset

    # Set status for this layer
    efconfig.upstream_status = upstream_status
    efconfig.layer_status = PipelineStatus.COMPLETE
    efconfig.processor = processor

    rgb = img
    for i, n in enumerate(lv1):
        _iter_t0 = time.perf_counter() if getattr(efconfig, "pipeline_timing", None) is not None else None
        overhead_components = {}
        reeffected = False
        reeffect_reason = None
        if lv2reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv1[n].reeffect()
                overhead_components["reeffect_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                lv1[n].reeffect()
            reeffected = True
            reeffect_reason = "upstream_or_previous_lv1_effect_changed"
            
        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv1[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv1[n].diff
        pre_hash = getattr(lv1[n], "hash", None)
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        _reset_effect_cache_event(lv1[n])
        diff = lv1[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        cache_event = _classify_cache_event(lv1[n], pre_hash, pre_diff, diff, reeffected)
        apply_ms = 0.0
        if diff is not None:
            rgb = diff
        _debug_pipeline_image_stats(
            f"{getattr(efconfig, 'pipeline_layer_label', 'primary')} lv1 {n}",
            rgb,
            effect_name=n,
            param=param,
            effect=lv1[n],
            state="changed" if diff is not None else "noop",
        )
        if getattr(efconfig, "debug_nan_inf_check", False):
            if _iter_t0 is not None:
                _t = time.perf_counter()
                utils.print_nan_inf(rgb, f"lv1-{n}")
                overhead_components["nan_inf_check_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                utils.print_nan_inf(rgb, f"lv1-{n}")
        
        # Update upstream hash
        if _iter_t0 is not None:
            _t = time.perf_counter()
            efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv1[n], 'hash', None)))
            overhead_components["upstream_hash_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv1[n], 'hash', None)))

        if _iter_t0 is not None:
            _t = time.perf_counter()
            if pre_diff is not diff:
                lv2reset = True
            overhead_components["reset_flag_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            if pre_diff is not diff:
                lv2reset = True
        if _iter_t0 is not None:
            _timing_record_effect(
                efconfig.pipeline_timing,
                _timing_effect_name(efconfig, 1, n),
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
                cache_event,
                reeffect_reason,
            )
            
    return rgb, lv2reset, efconfig.layer_status

def pipeline_lv2(rgb, effects, param, efconfig, prev_reset=False, upstream_status=PipelineStatus.COMPLETE, processor=None):
    lv2 = effects[2]
    lv3reset = prev_reset

    efconfig.upstream_status = upstream_status
    efconfig.layer_status = PipelineStatus.COMPLETE
    efconfig.processor = processor

    for i, n in enumerate(lv2):
        _iter_t0 = time.perf_counter() if getattr(efconfig, "pipeline_timing", None) is not None else None
        overhead_components = {}
        reeffected = False
        reeffect_reason = None
        if lv3reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv2[n].reeffect()
                overhead_components["reeffect_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                lv2[n].reeffect()
            reeffected = True
            reeffect_reason = "upstream_or_previous_lv2_effect_changed"

        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv2[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv2[n].diff
        pre_hash = getattr(lv2[n], "hash", None)
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        _reset_effect_cache_event(lv2[n])
        diff = lv2[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        cache_event = _classify_cache_event(lv2[n], pre_hash, pre_diff, diff, reeffected)
        apply_ms = 0.0
        if diff is not None:
            _apply_t0 = time.perf_counter() if _iter_t0 is not None else None
            rgb = lv2[n].apply_diff(rgb)
            apply_ms = (time.perf_counter() - _apply_t0) * 1000.0 if _apply_t0 is not None else 0.0
        _debug_pipeline_image_stats(
            f"{getattr(efconfig, 'pipeline_layer_label', 'primary')} lv2 {n}",
            rgb,
            effect_name=n,
            param=param,
            effect=lv2[n],
            state="changed" if diff is not None else "noop",
        )
        if getattr(efconfig, "debug_nan_inf_check", False):
            if _iter_t0 is not None:
                _t = time.perf_counter()
                utils.print_nan_inf(rgb, f"lv2-{n}")
                overhead_components["nan_inf_check_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                utils.print_nan_inf(rgb, f"lv2-{n}")
        
        # Update upstream hash
        if _iter_t0 is not None:
            _t = time.perf_counter()
            efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv2[n], 'hash', None)))
            overhead_components["upstream_hash_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv2[n], 'hash', None)))

        if _iter_t0 is not None:
            _t = time.perf_counter()
            if pre_diff is not diff:
                lv3reset = True
            overhead_components["reset_flag_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            if pre_diff is not diff:
                lv3reset = True
        if _iter_t0 is not None:
            _timing_record_effect(
                efconfig.pipeline_timing,
                _timing_effect_name(efconfig, 2, n),
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
                cache_event,
                reeffect_reason,
            )

    return rgb, lv3reset, efconfig.layer_status

def pipeline_lv3(rgb, effects, param, efconfig, prev_reset=False, upstream_status=PipelineStatus.COMPLETE, processor=None):
    lv3 = effects[3]
    lv4reset = prev_reset

    efconfig.upstream_status = upstream_status
    efconfig.layer_status = PipelineStatus.COMPLETE
    efconfig.processor = processor

    for i, n in enumerate(lv3):            
        _iter_t0 = time.perf_counter() if getattr(efconfig, "pipeline_timing", None) is not None else None
        overhead_components = {}
        reeffected = False
        reeffect_reason = None
        if lv4reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv3[n].reeffect()
                overhead_components["reeffect_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                lv3[n].reeffect()
            reeffected = True
            reeffect_reason = "upstream_or_previous_lv3_effect_changed"

        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv3[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv3[n].diff
        pre_hash = getattr(lv3[n], "hash", None)
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        _reset_effect_cache_event(lv3[n])
        diff = lv3[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        cache_event = _classify_cache_event(lv3[n], pre_hash, pre_diff, diff, reeffected)
        apply_ms = 0.0
        if diff is not None:
            _apply_t0 = time.perf_counter() if _iter_t0 is not None else None
            rgb = lv3[n].apply_diff(rgb)
            apply_ms = (time.perf_counter() - _apply_t0) * 1000.0 if _apply_t0 is not None else 0.0
        _debug_pipeline_image_stats(
            f"{getattr(efconfig, 'pipeline_layer_label', 'primary')} lv3 {n}",
            rgb,
            effect_name=n,
            param=param,
            effect=lv3[n],
            state="changed" if diff is not None else "noop",
        )
        if getattr(efconfig, "debug_nan_inf_check", False):
            if _iter_t0 is not None:
                _t = time.perf_counter()
                utils.print_nan_inf(rgb, f"lv3-{n}")
                overhead_components["nan_inf_check_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                utils.print_nan_inf(rgb, f"lv3-{n}")
        
        # Update upstream hash
        if _iter_t0 is not None:
            _t = time.perf_counter()
            efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv3[n], 'hash', None)))
            overhead_components["upstream_hash_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv3[n], 'hash', None)))
        
        if _iter_t0 is not None:
            _t = time.perf_counter()
            if pre_diff is not diff:
                lv4reset = True
            overhead_components["reset_flag_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            if pre_diff is not diff:
                lv4reset = True
        if _iter_t0 is not None:
            _timing_record_effect(
                efconfig.pipeline_timing,
                _timing_effect_name(efconfig, 3, n),
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
                cache_event,
                reeffect_reason,
            )

    return rgb, lv4reset, efconfig.layer_status

def pipeline_last(rgb, effects, param, efconfig, prev_reset=False, processor=None):
    lv4 = effects[4]
    lv5reset = prev_reset
    
    # Last layer usually tone curve etc, fast. No async needed probably.
    # But for consistency:
    efconfig.upstream_status = PipelineStatus.COMPLETE # Last layer doesn't propagate? 
    # Actually pipeline_last is usually blocking/sync.
    efconfig.processor = processor

    for i, n in enumerate(lv4):            
        _iter_t0 = time.perf_counter() if getattr(efconfig, "pipeline_timing", None) is not None else None
        overhead_components = {}
        reeffected = False
        reeffect_reason = None
        if lv5reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv4[n].reeffect()
                overhead_components["reeffect_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                lv4[n].reeffect()
            reeffected = True
            reeffect_reason = "upstream_or_previous_lv4_effect_changed"

        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv4[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv4[n].diff
        pre_hash = getattr(lv4[n], "hash", None)
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        _reset_effect_cache_event(lv4[n])
        diff = lv4[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        cache_event = _classify_cache_event(lv4[n], pre_hash, pre_diff, diff, reeffected)
        apply_ms = 0.0
        if diff is not None:
            _apply_t0 = time.perf_counter() if _iter_t0 is not None else None
            rgb = lv4[n].apply_diff(rgb)
            apply_ms = (time.perf_counter() - _apply_t0) * 1000.0 if _apply_t0 is not None else 0.0
        _debug_pipeline_image_stats(
            f"{getattr(efconfig, 'pipeline_layer_label', 'primary')} lv4 {n}",
            rgb,
            effect_name=n,
            param=param,
            effect=lv4[n],
            state="changed" if diff is not None else "noop",
        )

        if _iter_t0 is not None:
            _t = time.perf_counter()
            if pre_diff is not diff:
                lv5reset = True
            overhead_components["reset_flag_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            if pre_diff is not diff:
                lv5reset = True
        if _iter_t0 is not None:
            _timing_record_effect(
                efconfig.pipeline_timing,
                _timing_effect_name(efconfig, 4, n),
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
                cache_event,
                reeffect_reason,
            )

    return rgb

def pipeline_curve(rgb, effects, param, efconfig):
    rgb2 = None

    def ensure_output():
        nonlocal rgb2
        if rgb2 is None:
            rgb2 = rgb.copy()
        return rgb2

    # トーンカーブ
    diff = effects['tonecurve'].make_diff(rgb, param, efconfig)
    if diff is not None:
        rgb2 = effects['tonecurve'].apply_diff(ensure_output())
    diff = effects['tonecurve_red'].make_diff(rgb, param, efconfig)
    if diff is not None:
        rgb2 = ensure_output()
        rgb2[..., 0:1] = effects['tonecurve_red'].apply_diff(rgb2[..., 0:1])
    diff = effects['tonecurve_green'].make_diff(rgb, param, efconfig)
    if diff is not None:
        rgb2 = ensure_output()
        rgb2[..., 1:2] = effects['tonecurve_green'].apply_diff(rgb2[..., 1:2])
    diff = effects['tonecurve_blue'].make_diff(rgb, param, efconfig)
    if diff is not None:
        rgb2 = ensure_output()
        rgb2[..., 2:3] = effects['tonecurve_blue'].apply_diff(rgb2[..., 2:3])
    
    # グレーディング
    diff = effects['grading1'].make_diff(rgb, param, efconfig)
    if diff is not None:
        rgb2 = effects['grading1'].apply_diff(ensure_output())
    diff = effects['grading2'].make_diff(rgb, param, efconfig)
    if diff is not None:
        rgb2 = effects['grading2'].apply_diff(ensure_output())

    return rgb2

def pipeline_vs_and_saturation(hls, effects, param, efconfig):

    hls_h = hls2_h = hls[..., 0]
    hls_l = hls2_l = hls[..., 1]
    hls_s = hls2_s = hls[..., 2]
    changed = False

    # Hのみ
    diff = effects['HuevsHue'].make_diff([hls_h, hls2_h], param, efconfig)
    if diff is not None:
        hls2_h = effects['HuevsHue'].apply_diff(hls2_h)
        changed = True

    #　Lのみ
    lum_list = [('HuevsLum', hls_h), ('LumvsLum', hls_l), ('SatvsLum', hls_s)]
    lum_reset = False
    for n, src in lum_list:
        if lum_reset == True:
            effects[n].reeffect()

        pre_diff = effects[n].diff
        # 最新の hls2_l を使用して引数を構築
        diff = effects[n].make_diff([src, hls2_l], param, efconfig)
        if diff is not None:
            hls2_l = effects[n].apply_diff(hls2_l)
            changed = True

        if pre_diff is not diff:
            lum_reset = True

    # Sのみ
    sat_list = [('HuevsSat', hls_h), ('LumvsSat', hls_l), ('SatvsSat', hls_s), ('saturation', None)]
    sat_reset = False
    for n, src in sat_list:
        if sat_reset == True:
            effects[n].reeffect()

        pre_diff = effects[n].diff
        diff = effects[n].make_diff([src, hls2_s] if n != 'saturation' else hls2_s, param, efconfig)
        if diff is not None:
            hls2_s = effects[n].apply_diff(hls2_s)
            changed = True

        if pre_diff is not diff:
            sat_reset = True
    
    if not changed:
        return None

    # チャンネル数が4以上の場合（Gainマップ等）、残りのチャンネルを結合
    channels = [hls2_h, hls2_l, hls2_s]
    if hls.shape[-1] > 3:
        for i in range(3, hls.shape[-1]):
            channels.append(hls[..., i])

    return np.stack(channels, axis=-1)
