
import numpy as np
import logging

import config
import params
import effects
import cores.core as core
import utils.utils as utils
from enums import EffectMode, ExecutionMode, PipelineStatus

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
        print(f"DEBUG: get_result {effect_name} {param_hash} -> {res['status'] if res else 'None'}")
        return res

    def submit_task(self, effect_name, img, params, efconfig, param_hash):
        key = (effect_name, param_hash)
        
        # Check if already submitted
        if key in self.cache:
            if self.cache[key]['status'] in ['RUNNING', 'COMPLETE']:
                print(f"DEBUG: submit_task HIT {effect_name} {param_hash} status={self.cache[key]['status']}")
                return self.cache[key]
        
        print(f"DEBUG: submit_task MISS {effect_name} {param_hash}. Existing keys: {[k for k in self.cache.keys() if k[0]==effect_name]}")

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
        
        print(f"DEBUG: cancel_effect {effect_name}. Removing {len(keys_to_remove)} RUNNING tasks. Kept: {[k for k,v in self.cache.items() if k[0]==effect_name and v['status']=='COMPLETE']}")
        
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
    
    # クロップ情報を得る、ない場合元のクロップ情報から展開
    disp_info = params.get_disp_info(primary_param)
    if disp_info is None:
        disp_info = core.convert_rect_to_info(params.get_crop_rect(primary_param), config.get_config('preview_size')/max(primary_param['original_img_size']))
        params.set_disp_info(primary_param, disp_info)

    # 環境設定
    efconfig = effects.EffectConfig()
    efconfig.disp_info = disp_info
    efconfig.is_zoomed = is_zoomed
    efconfig.mode = EffectMode.PREVIEW
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], 1.0)
    
    # Initialize basic input hash
    efconfig.loading_flag = loading_flag
    efconfig.upstream_hash = hash(id(img))

    # 背景レイヤー
    img0, lv1reset, pre_rotation_img, _ = pipeline_lv0(img, primary_effects, primary_param, efconfig, processor=processor)
    disp_info = params.get_disp_info(primary_param) # Cropによって値が更新されてるかも

    if crop_image is None or lv1reset == True:
        imgc, disp_info2 = core.crop_image(img0, disp_info, params.get_crop_rect(primary_param), texture_width, texture_height, click_x, click_y, is_zoomed, center_pos)
        mask_editor2.set_primary_param(primary_param, disp_info2)
        mask_editor2.set_ref_image(imgc, pre_rotation_img)
        params.set_disp_info(primary_param, disp_info2)
        
        # Crop performed, so upstream (imgc) changed.
        # But lv1reset might be False if lv0 didn't change.
        # We MUST set lv1reset=True if we generated a NEW imgc that is different from cached crop result?
        # Actually core.crop_image runs every time `process_pipeline` is called if we don't cache `crop_image` outside?
        # `process_pipeline` argument `crop_image` IS the cache from `MainWidget`.
        # If `crop_image` was None, we created it. So it IS new (to the pipeline flow).
        # We should set lv1reset=True to force downstream updates.
        lv1reset = True
        
    else:
        imgc = crop_image
        disp_info2 = disp_info
    #mask_editor2.update()

    # 環境設定更新
    efconfig.disp_info = disp_info2
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], disp_info2[4])
    
    # 並列処理
    # Async Manager update
    if processor is not None:
        if processor.get_pipeline_version() - pipeline_version > 2:
            pass
            #return None, imgc        

    if not is_drag:
        img2, lv4reset = pipeline2(imgc, None, primary_effects, primary_param, mask_editor2, efconfig, lv1reset, processor=processor)
        img2 = pipeline_last(img2, primary_effects, primary_param, efconfig, prev_reset=lv4reset, processor=processor)
    else:
        img2 = imgc

    if processor is not None:
        if processor.get_pipeline_version() - pipeline_version > 2:
            pass
            #return None, imgc        

    return img2, imgc

def export_pipeline(img, primary_effects, primary_param, mask_editor2):
    
    # 環境設定
    disp_info = core.convert_rect_to_info(params.get_crop_rect(primary_param), 1) # 倍率１で作成
    params.set_disp_info(primary_param, disp_info) # コピーしとく
    efconfig = effects.EffectConfig()
    efconfig.disp_info = disp_info
    efconfig.is_zoomed = True
    efconfig.mode = EffectMode.EXPORT
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], disp_info[4])

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

            if crop is None:
                img3 = core.apply_mask(img3, mask.get_mask_image(), img2)
            else:
                img3 = core.apply_mask(img3, mask.get_mask_image()[crop[1]:crop[3], crop[0]:crop[2], :], img2)

    return img3, lv1reset

def _process_effect(effect, name, rgb, param, efconfig, upstream_status, processor):
    # Determine execution mode
    mode = getattr(effect, 'execution_mode', ExecutionMode.SYNC)
    if efconfig.mode == EffectMode.EXPORT:
        mode = ExecutionMode.SYNC # Force sync (or blocking) on export
    
    # 1. SYNC or BLOCKING
    if mode != ExecutionMode.ASYNC:
        # Just execute
        diff = effect.make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = effect.apply_diff(rgb)
        utils.print_nan_inf(rgb, f"{name}")
        return rgb, upstream_status, True # Always return True (executed) to imply potentially dirty? 
        # Actually `make_diff` internally manages `self.diff` and `self.hash`?
        # The caller checks `pre_diff is not diff`. We need to preserve that logic.
        # But here we are refactoring.
        # Let's keep the caller logic for diff check if possible, OR return 'changed'.
        
    # 2. ASYNC
    # Check upstream
    if upstream_status == PipelineStatus.PREVIEW:
        # Upstream is dirty/preview, so we MUST skip heavy calc or do preview
        # Return simple pass-through or preview if available
        # For now, pass-through (no diff applied)
        # But we must update `diff` to None or similar so next calls know it changed?
        # Actually, if we skip, `diff` remains what it was? API says `make_diff` updates `self.diff`.
        # We should call `make_diff` with a flag? No.
        
        # Strategies:
        # A. Call `make_diff` but it knows to be fast?
        # B. Don't call `make_diff`.
        
        # User said: "simple preview mode (or skip)".
        # To skip: `diff = None`.
        # But `effect.hash` must be managed?
        
        # Simplest: Just return rgb as is.
        # But we need to signal that we are "PROCESSING".
        # And we need to ensure `lvXreset` logic works.
        pass
        
    # Check cache / Submit
    if processor is not None:
        # Calculate param hash (complex, `make_diff` usually does it)
        # We need a way to get hash WITHOUT running heavy code.
        # BUT current `make_diff` does both: check hash, run if needed.
        # We need to separate hash calculation?
        # Many effects in `effects.py` do: `param_hash = hash(...)`.
        pass

    return rgb, upstream_status, False

# Helper for async logic reused in lv1..4
def _execute_layer_async(layer_effects, rgb, param, efconfig, reset_flag, upstream_status, processor, layer_name):
    # This replaces the loop in pipeline_lvX
    # BUT `lvX` functions have specific returns (lvXreset).
    # And logic is slightly different (e.g. `pre_rotation` in lv0).
    pass
    # It is hard to extract a common function because of specific variable binding.
    # We will modify each execution loop in place.

def pipeline_lv0(img, effects, param, efconfig, processor=None):
    lv0 = effects[0]
    lv1reset = False
    
    pre_rotation_img = None
    
    # lv0 is usually sync (Lens correction, rotation).
    # But if we had async...
    
    rgb = img
    upstream_status = PipelineStatus.COMPLETE # Initial input is complete
    
    efconfig.processor = processor
    efconfig.layer_status = PipelineStatus.COMPLETE
    
    for i, n in enumerate(lv0):
        # Update upstream status for the effect
        efconfig.upstream_status = upstream_status
        
        if lv1reset == True:
            lv0[n].reeffect()
            
        if n == 'geometry':
            pre_rotation_img = rgb
            
        pre_diff = lv0[n].diff
        
        # Check Mode
        mode = getattr(lv0[n], 'execution_mode', ExecutionMode.SYNC)
        if efconfig.mode == EffectMode.EXPORT:
            mode = ExecutionMode.SYNC
            
        if mode == ExecutionMode.ASYNC and processor is not None:
            # ASYNC LOGIC
            # 1. If upstream is preview, skip
            if upstream_status == PipelineStatus.PREVIEW:
                diff = None # Skip
                # Effectively we act as if diff is None.
                # But we should NOT update hash? 
                # If we don't update hash, next time it will run.
                # But if we skipp, `diff` becomes None.
                # If `pre_diff` was something, `pre_diff != diff` -> True -> lv1reset=True.
                pass
            else:
                # 2. Upstream Complete. Check/Submit.
                pass
                # We need param_hash. 
                # We must modify `make_diff` to return hash? Or access internal `hash` method?
                # Most effects don't expose hash calculation separately.
                # We might need to rely on `make_diff` to be fast enough to just check hash?
                # "Heavy" part is usually inside `if self.hash != param_hash:`.
                # So `make_diff` IS the check.
                # IF the effect is written correctly, `make_diff` is fast if hash matches.
                # BUT if hash mismatches, it runs heavy code.
                # We need to INTERCEPT this.
                
                # We can't easily intercept without changing Effect classes.
                # BUT user said "ExecMode" enum.
                # Maybe we change `make_diff` signature?
                # `make_diff(..., async_check_only=True)`?
                
                # Let's assume we can call `make_diff`? No.
                # If we call `make_diff`, it blocks.
                
                # WE NEED TO REFLECT ON THE PLAN.
                # "Cache check: Is there a result?"
                # To check cache, we need hash.
                # To get hash, we need to calculate it.
                # Code in `make_diff` calculates hash.
                
                # I should MODIFY `Effect` class or individual effects to expose `get_params_hash(param)`?
                # Or just assume `make_diff` is refactored?
                # Refactoring all effects is huge.
                
                # Alternative: `processor` wraps the heavy call?
                # The `make_diff` calls `core.heavy_func`.
                # We can wrap `core.heavy_func`? No.
                
                # Let's implement a wrapper in `pipeline.py` that calculates hash FOR the specific async effects we know?
                # We only have `AsyncWorker` logic for generic effect?
                
                # User request: "Execute Sync if lvXreset=True".
                
                # Let's look at `effects.py` again.
                # Example `AINoiseReductonEffect.make_diff`:
                # `nr = ...`, `param_hash = hash((nr))`
                # `if self.hash != param_hash: heavy...`
                
                # If I want to async this:
                # I need `param_hash`.
                
                # I will wrap the logic here in pipeline:
                # We assume for ASYNC effects, `make_diff` simply returns the cached diff if available?
                # No.
                
                # I'll stick to: Pipeline calls `make_diff` for everything.
                # BUT for Async effects, I need to modify `make_diff` to NOT run heavy calc.
                # I will modify `Effect.make_diff` signature to accept `async_manager`?
                # `make_diff(img, param, efconfig, async_manager=None)`
                
                # If `async_manager` is passed:
                # It calculates hash.
                # Calls `async_manager.submit_or_get(self.name, img, ...)`
                # If returns None (running), set `self.diff = None` (or existing?), return.
                
                # This seems invasive but necessary.
                # I need to modify `Effect.make_diff` signature in `effects.py`.
                # And update all overrides?
                # Most overrides use `*args, **kwargs`? No, explicit arguments.
                
                # Wait, `make_diff` is called in `pipeline.py`.
                # `diff = lv0[n].make_diff(rgb, param, efconfig)`
                
                # If I change arguments, I break everything if I don't update all classes.
                # There are ~20 effects. Updates are feasible.
                
                # OR use `kwargs` in `make_diff`?
                # `def make_diff(self, img, param, efconfig, **kwargs):`
                # Base `Effect` has it.
                # Subclasses might not accept kwargs.
                
                # Let's check `effects.py` again.
                # `class LensModifierEffect(Effect): def make_diff(self, img, param, efconfig):`
                # It does NOT accept `**kwargs`.
                
                # I cannot easily pass `processor` to `make_diff`.
                
                # Alternative:
                # In `pipeline.py`, I manually check cache if I know how to calc hash.
                # But I don't know internal hash logic of each effect.
                
                # Alternative 2:
                # `processor` is set to `efconfig`?!
                # `efconfig` is passed to `make_diff`.
                # I can attach `processor` and `upstream_status` to `efconfig`.
                # Then modify `make_diff` of target ASYNC effects (e.g. `AINoiseReductonEffect`) to use it.
                # This avoids changing signature of ALL effects.
                # I only need to touch the Heavy effects.
                
                # Plan Check:
                # 1. Add `processor` and `upstream_status` to `EffectConfig`.
                # 2. In `pipeline.py`, set these in `efconfig`.
                # 3. In `pipeline.py`, still loop as usual.
                # 4. In `effects.py`, modify Heavy effects (e.g. `AINoiseReductonEffect`) to:
                #    - Extract `processor`, `upstream_status` from `efconfig`.
                #    - If `processor` present and `execution_mode == ASYNC`:
                #      - Calc hash.
                #      - Check `upstream_status`. IF PREVIEW, return None (or cached).
                #      - IF COMPLETE, `processor.submit_task(...)`.
                #      - If result ready, use it.
                
                # This isolates changes to `effects.py` (heavy classes) and `pipeline.py` (setup).
                # `pipeline.py` changes are minimal (just passing context via efconfig).
                
                # Wait, `pipeline_lvX` loops are:
                # `diff = lv1[n].make_diff(...)`
                # `if pre_diff is not diff: lv2reset = True`
                
                # If `make_diff` returns `None` (skipped/running), `diff` is `None`.
                # If `pre_diff` was `Some`, then `reset=True`.
                # Next frame, `diff` is `None`. `pre_diff` is `None`. `reset=False`. Stable.
                
                # If task completes:
                # `make_diff` returns `Result`.
                # `pre_diff` was `None`. `reset=True`. Stable.
                
                # This logic works!
                
                # But what about `upstream_status`?
                # `pipeline.py` needs to know if ANY effect in this layer is in "PROCESSING" state to update `upstream_status` for next layer.
                # `efconfig` is shared? No, created in `process_pipeline`.
                # But `upstream_status` changes PER LAYER.
                # `efconfig.upstream_status`?
                # We update `efconfig.upstream_status` before calling `lv1`?
                # And inside `lv1`, if an effect returns "PROCESSING", we need to flag `current_layer_status = PREVIEW`.
                # But `make_diff` returns `diff`. It doesn't return status.
                
                # How do we know if it's "PROCESSING" vs "Effect Disabled (None)"?
                # Effect Disabled -> diff is None.
                # Processing -> diff is None (or preview).
                # Using `None` for processing implies "Disabled" to the image flow (no change).
                # That's fine for "Preview".
                
                # But how does `pipeline.py` know to propagate `PREVIEW` status to next layer?
                # We need to query the effect?
                # `failed_to_complete = ...`?
                
                # I can populate `efconfig` with `layer_status = COMPLETE` initially.
                # If an async effect runs and is NOT complete, it sets `efconfig.layer_status = PREVIEW`.
                # Then `pipeline.py` reads `efconfig.layer_status` and passes it to next layer.
                
                # Excellent.
                
                # Changes Plan Revised:
                # 1. Update `EffectConfig` in `effects.py` to hold `processor` and `upstream_status` and `current_status`.
                # 2. Implement `AsyncPipelineManager` in `pipeline.py` (done).
                # 3. Update `pipeline.py` to set `efconfig` fields.
                #    - Pass `upstream_status` to `pipeline_lvX`.
                #    - Should update `efconfig` inside `pipeline_lvX` or before?
                #    - `efconfig.upstream_status = upstream_status`
                #    - `efconfig.layer_status = PipelineStatus.COMPLETE` (reset for this layer)
                #    - Call effects.
                #    - Return `efconfig.layer_status` as `upstream_status` for next layer.
                # 4. Update `effects.py`:
                #    - Modify `AINoiseReductonEffect` (and others) `make_diff` to use async logic.
                
                # Let's clean up `process_pipeline` first.
                
    
        diff = lv0[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv0[n].apply_diff(rgb)
            
        # Update upstream hash
        efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv0[n], 'hash', None)))

        # Update upstream status if layer became preview
        if efconfig.layer_status == PipelineStatus.PREVIEW:
            upstream_status = PipelineStatus.PREVIEW

        if pre_diff is not diff:
            lv1reset = True
            
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
        if lv2reset == True:
            lv1[n].reeffect()
            
        pre_diff = lv1[n].diff
        diff = lv1[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = diff
        utils.print_nan_inf(rgb, f"lv1-{n}")
        
        # Update upstream hash
        efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv1[n], 'hash', None)))

        if pre_diff is not diff:
            lv2reset = True
            
    return rgb, lv2reset, efconfig.layer_status

def pipeline_lv2(rgb, effects, param, efconfig, prev_reset=False, upstream_status=PipelineStatus.COMPLETE, processor=None):
    lv2 = effects[2]
    lv3reset = prev_reset

    efconfig.upstream_status = upstream_status
    efconfig.layer_status = PipelineStatus.COMPLETE
    efconfig.processor = processor

    for i, n in enumerate(lv2):
        if lv3reset == True:
            lv2[n].reeffect()

        pre_diff = lv2[n].diff
        diff = lv2[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv2[n].apply_diff(rgb)
        utils.print_nan_inf(rgb, f"lv2-{n}")
        
        # Update upstream hash
        efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv2[n], 'hash', None)))

        if pre_diff is not diff:
            lv3reset = True

    return rgb, lv3reset, efconfig.layer_status

def pipeline_lv3(rgb, effects, param, efconfig, prev_reset=False, upstream_status=PipelineStatus.COMPLETE, processor=None):
    lv3 = effects[3]
    lv4reset = prev_reset

    efconfig.upstream_status = upstream_status
    efconfig.layer_status = PipelineStatus.COMPLETE
    efconfig.processor = processor

    for i, n in enumerate(lv3):            
        if lv4reset == True:
            lv3[n].reeffect()

        diff = lv3[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv3[n].apply_diff(rgb)
        utils.print_nan_inf(rgb, f"lv3-{n}")
        
        # Update upstream hash
        efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv3[n], 'hash', None)))

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
        if lv5reset == True:
            lv4[n].reeffect()

        diff = lv4[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv4[n].apply_diff(rgb)

    return rgb

def pipeline_hls(hls, effects, param, efconfig):
    efconfig.hls_reference = hls.copy()
    for i, n in enumerate(effects):
        diff = effects[n].make_diff(hls, param, efconfig)
        if diff is not None:
            hls = effects[n].apply_diff(hls)

    return hls

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

    hls_h = hls[..., 0]
    hls2_h = hls_h.copy()
    hls_l = hls[..., 1]
    hls2_l = hls_l.copy()
    hls_s = hls[..., 2]
    hls2_s = hls_s.copy()

    # Hのみ
    diff = effects['HuevsHue'].make_diff(hls_h, param, efconfig)
    if diff is not None: hls2_h = effects['HuevsHue'].apply_diff(hls2_h)

    #　Lのみ
    diff = effects['HuevsLum'].make_diff(hls_l, param, efconfig)
    if diff is not None: hls2_l = effects['HuevsLum'].apply_diff([hls_h, hls2_l])
    diff = effects['LumvsLum'].make_diff(hls_l, param, efconfig)
    if diff is not None: hls2_l = effects['LumvsLum'].apply_diff(hls2_l)
    diff = effects['SatvsLum'].make_diff(hls_l, param, efconfig)
    if diff is not None: hls2_l = effects['SatvsLum'].apply_diff([hls_s, hls2_l])

    # Sのみ
    diff = effects['HuevsSat'].make_diff(hls_s, param, efconfig)
    if diff is not None: hls2_s = effects['HuevsSat'].apply_diff([hls2_h, hls2_s])
    diff = effects['LumvsSat'].make_diff(hls_s, param, efconfig)
    if diff is not None: hls2_s = effects['LumvsSat'].apply_diff([hls_l, hls2_s])
    diff = effects['SatvsSat'].make_diff(hls_s, param, efconfig)
    if diff is not None: hls2_s = effects['SatvsSat'].apply_diff(hls2_s)
    diff = effects['saturation'].make_diff(hls_s, param, efconfig)
    if diff is not None: hls2_s = effects['saturation'].apply_diff(hls2_s)
    
    # チャンネル数が4以上の場合（Gainマップ等）、残りのチャンネルを結合
    channels = [hls2_h, hls2_l, hls2_s]
    if hls.shape[-1] > 3:
        for i in range(3, hls.shape[-1]):
            channels.append(hls[..., i])

    return np.stack(channels, axis=-1)
