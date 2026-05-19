
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
        "sections_ms": {},
        "effects": {},  # effect_name -> counters
    }


def _timing_add_section_ms(timing, section, elapsed_ms):
    if timing is None:
        return
    timing["sections_ms"][section] = timing["sections_ms"].get(section, 0.0) + float(elapsed_ms)


def _timing_record_effect(timing, effect_name, make_ms, apply_ms, iteration_ms, overhead_components=None):
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
        },
    )
    bucket["calls"] += 1
    bucket["make_ms"] += make_ms
    bucket["apply_ms"] += apply_ms
    bucket["effect_ms"] += effect_ms
    bucket["loop_overhead_ms"] += loop_overhead_ms
    known = 0.0
    for comp, value in overhead_components.items():
        value = float(value)
        known += value
        timing["overhead_components_ms"][comp] = timing["overhead_components_ms"].get(comp, 0.0) + value
        bucket["overhead_components_ms"][comp] = bucket["overhead_components_ms"].get(comp, 0.0) + value
    unknown = max(0.0, loop_overhead_ms - known)
    timing["overhead_components_ms"]["unknown_ms"] = timing["overhead_components_ms"].get("unknown_ms", 0.0) + unknown
    bucket["overhead_components_ms"]["unknown_ms"] = bucket["overhead_components_ms"].get("unknown_ms", 0.0) + unknown


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


def process_pipeline(img, crop_image, is_zoomed, texture_width, texture_height, click_x, click_y, primary_effects, primary_param, mask_editor2, processor, pipeline_version, current_tab, loading_flag=-1, is_drag=False, center_pos=None):
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
    efconfig.mode = EffectMode.PREVIEW
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], 1.0)
    efconfig.current_tab = current_tab
    efconfig.crop_editing = current_tab == "Ge"
    
    # Initialize basic input hash
    efconfig.loading_flag = loading_flag
    efconfig.image_fidelity = primary_param.get('image_fidelity')
    efconfig.upstream_hash = hash(id(img))
    efconfig.pipeline_timing = timing
    efconfig.debug_nan_inf_check = _is_nan_inf_debug_enabled()
    if timing is not None:
        _timing_add_section_ms(timing, "efconfig_setup", (time.perf_counter() - _t0) * 1000.0)

    # 背景レイヤー
    if timing is not None:
        _t0 = time.perf_counter()
    img0, lv1reset, pre_rotation_img, _ = pipeline_lv0(img, primary_effects, primary_param, efconfig, processor=processor)
    disp_info = params.get_disp_info(primary_param) # Cropによって値が更新されてるかも
    if timing is not None:
        _timing_add_section_ms(timing, "pipeline_lv0", (time.perf_counter() - _t0) * 1000.0)

    if timing is not None:
        _t0 = time.perf_counter()
    if crop_image is None or lv1reset == True:
        imgc, disp_info2 = core.crop_image(img0, disp_info, params.get_crop_rect(primary_param), texture_width, texture_height, click_x, click_y, is_zoomed, center_pos)
        mask_editor2.set_primary_param(primary_param, disp_info2)
        mask_editor2.set_ref_image(imgc, pre_rotation_img)
        params.set_disp_info(primary_param, disp_info2)
        # 新規クロップ生成時は下流を必ず更新
        lv1reset = True
        
    else:
        imgc = crop_image
        disp_info2 = disp_info
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

    _finalize_pipeline_timing(timing)
    return img2, imgc

def export_pipeline(img, primary_effects, primary_param, mask_editor2):
    if not params.has_original_img_size(primary_param):
        logging.error("export_pipeline: original_img_size 未定義のため処理しません")
        return None

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

    img1, lv2reset, upstream_status = pipeline_lv1(imgc, primary_effects, primary_param, efconfig, lv1reset, upstream_status, processor)
    img2, lv3reset, upstream_status = pipeline_lv2(img1, primary_effects, primary_param, efconfig, lv2reset, upstream_status, processor)
    img3, lv1reset, upstream_status = pipeline_lv3(img2, primary_effects, primary_param, efconfig, lv3reset, upstream_status, processor)

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
            
            img2, lv2reset, _ = pipeline_lv1(img3, mask.effects, mask.effects_param, efconfig, lv1reset, upstream_status, processor)
            img2, lv1reset, _ = pipeline_lv2(img2, mask.effects, mask.effects_param, efconfig, lv2reset, upstream_status, processor)

            img2 = core.type_convert(img2, np.ndarray)
            img3 = core.type_convert(img3, np.ndarray)

            mask_image = mask.get_mask_image()
            if crop is not None:
                mask_image = mask_image[crop[1]:crop[3], crop[0]:crop[2]]
            mask2_param = _effective_mask2_draw_effect_param(mask)
            img3 = core.apply_mask_draw_effects(
                img3, mask_image, img2, mask2_param,
                resolution_scale=getattr(efconfig, "resolution_scale", 1.0),
            )

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
        
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        diff = lv0[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
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
            if pre_diff is not diff:
                lv1reset = True
            overhead_components["reset_flag_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            if pre_diff is not diff:
                lv1reset = True
        if _iter_t0 is not None:
            _timing_record_effect(
                efconfig.pipeline_timing,
                f"lv0.{n}",
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
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
        if lv2reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv1[n].reeffect()
                overhead_components["reeffect_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                lv1[n].reeffect()
            
        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv1[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv1[n].diff
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        diff = lv1[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        apply_ms = 0.0
        if diff is not None:
            rgb = diff
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
                f"lv1.{n}",
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
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
        if lv3reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv2[n].reeffect()
                overhead_components["reeffect_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                lv2[n].reeffect()

        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv2[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv2[n].diff
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        diff = lv2[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        apply_ms = 0.0
        if diff is not None:
            _apply_t0 = time.perf_counter() if _iter_t0 is not None else None
            rgb = lv2[n].apply_diff(rgb)
            apply_ms = (time.perf_counter() - _apply_t0) * 1000.0 if _apply_t0 is not None else 0.0
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
                f"lv2.{n}",
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
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
        if lv4reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv3[n].reeffect()
                overhead_components["reeffect_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                lv3[n].reeffect()

        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv3[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv3[n].diff
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        diff = lv3[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        apply_ms = 0.0
        if diff is not None:
            _apply_t0 = time.perf_counter() if _iter_t0 is not None else None
            rgb = lv3[n].apply_diff(rgb)
            apply_ms = (time.perf_counter() - _apply_t0) * 1000.0 if _apply_t0 is not None else 0.0
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
                f"lv3.{n}",
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
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
        if lv5reset == True:
            if _iter_t0 is not None:
                _t = time.perf_counter()
                lv4[n].reeffect()
                overhead_components["reeffect_ms"] = (time.perf_counter() - _t) * 1000.0
            else:
                lv4[n].reeffect()

        if _iter_t0 is not None:
            _t = time.perf_counter()
            pre_diff = lv4[n].diff
            overhead_components["pre_diff_read_ms"] = (time.perf_counter() - _t) * 1000.0
        else:
            pre_diff = lv4[n].diff
        _make_t0 = time.perf_counter() if _iter_t0 is not None else None
        diff = lv4[n].make_diff(rgb, param, efconfig)
        make_ms = (time.perf_counter() - _make_t0) * 1000.0 if _make_t0 is not None else 0.0
        apply_ms = 0.0
        if diff is not None:
            _apply_t0 = time.perf_counter() if _iter_t0 is not None else None
            rgb = lv4[n].apply_diff(rgb)
            apply_ms = (time.perf_counter() - _apply_t0) * 1000.0 if _apply_t0 is not None else 0.0

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
                f"lv4.{n}",
                make_ms,
                apply_ms,
                (time.perf_counter() - _iter_t0) * 1000.0,
                overhead_components,
            )

    return rgb

def pipeline_curve(rgb, effects, param, efconfig):
    rgb2 = rgb.copy()

    # トーンカーブ
    diff = effects['tonecurve'].make_diff(rgb, param, efconfig)
    if diff is not None: rgb2 = effects['tonecurve'].apply_diff(rgb2)
    diff = effects['tonecurve_red'].make_diff(rgb, param, efconfig)
    if diff is not None: rgb2[..., 0:1] = effects['tonecurve_red'].apply_diff(rgb2[..., 0:1])
    diff = effects['tonecurve_green'].make_diff(rgb, param, efconfig)
    if diff is not None: rgb2[..., 1:2] = effects['tonecurve_green'].apply_diff(rgb2[..., 1:2])
    diff = effects['tonecurve_blue'].make_diff(rgb, param, efconfig)
    if diff is not None: rgb2[..., 2:3] = effects['tonecurve_blue'].apply_diff(rgb2[..., 2:3])
    
    # グレーディング
    diff = effects['grading1'].make_diff(rgb, param, efconfig)
    if diff is not None: rgb2 = effects['grading1'].apply_diff(rgb2)
    diff = effects['grading2'].make_diff(rgb, param, efconfig)
    if diff is not None: rgb2 = effects['grading2'].apply_diff(rgb2)

    return rgb2

def pipeline_vs_and_saturation(hls, effects, param, efconfig):

    hls_h = hls2_h = hls[..., 0]
    hls_l = hls2_l = hls[..., 1]
    hls_s = hls2_s = hls[..., 2]

    # Hのみ
    diff = effects['HuevsHue'].make_diff([hls_h, hls2_h], param, efconfig)
    if diff is not None:
        hls2_h = effects['HuevsHue'].apply_diff(hls2_h)

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

        if pre_diff is not diff:
            sat_reset = True
    
    # チャンネル数が4以上の場合（Gainマップ等）、残りのチャンネルを結合
    channels = [hls2_h, hls2_l, hls2_s]
    if hls.shape[-1] > 3:
        for i in range(3, hls.shape[-1]):
            channels.append(hls[..., i])

    return np.stack(channels, axis=-1)
