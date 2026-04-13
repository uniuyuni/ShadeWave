
import numpy as np
import logging

logger = logging.getLogger(__name__)

import config
import params
import effects
import cores.core as core
import utils.utils as utils
from enums import EffectMode, PipelineStatus

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
    if not params.has_original_img_size(primary_param):
        logging.warning("process_pipeline: original_img_size 未定義のため処理しません")
        return None, crop_image

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
    efconfig.image_fidelity = primary_param.get('image_fidelity')
    efconfig.upstream_hash = hash(id(img))

    # 背景レイヤー
    img0, lv1reset, pre_rotation_img, _ = pipeline_lv0(img, primary_effects, primary_param, efconfig, processor=processor)
    disp_info = params.get_disp_info(primary_param) # Cropによって値が更新されてるかも

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

    # 環境設定更新
    efconfig.disp_info = disp_info2
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], disp_info2[4])
    
    if not is_drag:
        img2, lv4reset = pipeline2(imgc, None, primary_effects, primary_param, mask_editor2, efconfig, lv1reset, processor=processor)
        img2 = pipeline_last(img2, primary_effects, primary_param, efconfig, prev_reset=lv4reset, processor=processor)
    else:
        img2 = imgc

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
    efconfig.image_fidelity = primary_param.get('image_fidelity')

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

def pipeline_lv0(img, effects, param, efconfig, processor=None):
    lv0 = effects[0]
    lv1reset = False
    
    pre_rotation_img = None
    
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

        pre_diff = lv3[n].diff
        diff = lv3[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv3[n].apply_diff(rgb)
        utils.print_nan_inf(rgb, f"lv3-{n}")
        
        # Update upstream hash
        efconfig.upstream_hash = hash((efconfig.upstream_hash, n, getattr(lv3[n], 'hash', None)))
        
        if pre_diff is not diff:
            lv4reset = True

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

        pre_diff = lv4[n].diff
        diff = lv4[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv4[n].apply_diff(rgb)

        if pre_diff is not diff:
            lv5reset = True

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
