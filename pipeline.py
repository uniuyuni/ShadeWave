
import numpy as np

import cores.core as core
import config
import params
import effects
import time
import splitimage

def process_pipeline(img, offset, crop_image, is_zoomed, texture_width, texture_height, click_x, click_y, primary_effects, primary_param, mask_editor2, processor, pipeline_version):
    
    # クロップ情報を得る、ない場合元のクロップ情報から展開
    disp_info = params.get_disp_info(primary_param)
    if disp_info is None:
        disp_info = core.convert_rect_to_info(params.get_crop_rect(primary_param), config.get_config('preview_size')/max(primary_param['original_img_size']))
        params.set_disp_info(primary_param, disp_info)

    # 環境設定
    efconfig = effects.EffectConfig()
    efconfig.disp_info = disp_info
    efconfig.is_zoomed = is_zoomed
    efconfig.mode = effects.EffectMode.PREVIEW
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], 1.0)

    # 背景レイヤー
    img0, lv1reset = pipeline_lv0(img, primary_effects, primary_param, efconfig)
    disp_info = params.get_disp_info(primary_param) # Cropによって値が更新されてるかも

    if crop_image is None or lv1reset == True:
        imgc, disp_info2 = core.crop_image(img0, disp_info, params.get_crop_rect(primary_param), texture_width, texture_height, click_x, click_y, offset, is_zoomed)
        #mask_editor2.set_orientation(primary_param.get('rotation', 0), primary_param.get('rotation2', 0), primary_param.get('flip_mode', 0))
        #mask_editor2.set_texture_size(texture_width, texture_height)
        mask_editor2.set_primary_param(primary_param, disp_info2)
        mask_editor2.set_ref_image(imgc, img0)
        params.set_disp_info(primary_param, disp_info2)
    else:
        imgc = crop_image
        disp_info2 = disp_info
    #mask_editor2.update()

    # 環境設定更新
    efconfig.disp_info = disp_info2
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], disp_info2[4])
    
    # 並列処理
    if False:
        process_params = {
            "param": primary_param,
            "efconfig": efconfig,
        }
        processor.submit_tiles(imgc, process_params, pipeline_version, lv1reset)
        results = processor.collect_results(pipeline_version)
        
        # 画像の再構築
        if len(results) > 0:
            results = sorted(results, key=lambda x: x[0][0])
            blocks = [tile_result for (tile_id, split_info), tile_result in results]
            img2 = splitimage.combine_image_with_overlap(blocks, results[0][0][1])
        else:
            img2 = imgc
    else:
        img2 = pipeline2(imgc, None, primary_effects, primary_param, mask_editor2, efconfig, lv1reset)
        
    img2 = pipeline_last(img2, primary_effects, primary_param, efconfig)

    return img2, imgc

def export_pipeline(img, primary_effects, primary_param, mask_editor2):
    
    # 環境設定
    disp_info = core.convert_rect_to_info(params.get_crop_rect(primary_param), 1)
    efconfig = effects.EffectConfig()
    efconfig.disp_info = disp_info
    efconfig.is_zoomed = True
    efconfig.mode = effects.EffectMode.EXPORT
    efconfig.resolution_scale = core.calc_resolution_scale(primary_param['original_img_size'], disp_info[4])

    # 背景レイヤー
    img0, lv1reset = pipeline_lv0(img, primary_effects, primary_param, efconfig)
    imgc = img0
    #imgc, disp_info2 = core.crop_image(img0, disp_info, *primary_param['original_img_size'], 0, 0, (0, 0), False)
    #mask_editor2.set_orientation(primary_param.get('rotation', 0), primary_param.get('rotation2', 0), primary_param.get('flip_mode', 0))
    imax = max(imgc.shape[1], imgc.shape[0])
    mask_editor2.set_texture_size(imax, imax)
    mask_editor2.set_primary_param(primary_param, disp_info)
    mask_editor2.set_ref_image(imgc, img0)
    mask_editor2.update()

    img2 = pipeline2(imgc, None, primary_effects, primary_param, mask_editor2, efconfig, lv1reset)

    img2 = pipeline_last(img2, primary_effects, primary_param, efconfig)
    
    # ここでクロップ
    x1, y1, x2, y2 = params.get_crop_rect(primary_param)
    img2 = img2[y1:y2, x1:x2] # ただのクロップ

    return img2

def pipeline2(imgc, crop, primary_effects, primary_param, mask_editor2, efconfig, lv1reset=False):
    img1, lv2reset = pipeline_lv1(imgc, primary_effects, primary_param, efconfig, lv1reset)
    img2, lv3reset = pipeline_lv2(img1, primary_effects, primary_param, efconfig, lv2reset)
    img3, lv1reset = pipeline_lv3(img2, primary_effects, primary_param, efconfig, lv3reset)

    # マスクレイヤー
    if mask_editor2 is not None:
        mask_list = mask_editor2.get_mask_list()
        for mask in mask_list:
            if not mask.is_composit():
                continue
            
            img2, lv2reset = pipeline_lv1(img3, mask.effects, mask.effects_param, efconfig, lv1reset)
            img2, lv1reset = pipeline_lv2(img2, mask.effects, mask.effects_param, efconfig, lv2reset)

            img2 = core.type_convert(img2, np.ndarray)
            if crop is None:
                img3 = core.apply_mask(img3, mask.get_mask_image(), img2)
            else:
                img3 = core.apply_mask(img3, mask.get_mask_image()[crop[1]:crop[3], crop[0]:crop[2], :], img2)
        mask_editor2.set_rotation_changed_flag(False)

    return img3

def pipeline_lv0(img, effects, param, efconfig):
    lv0 = effects[0]
    lv1reset = False

    rgb = img
    for i, n in enumerate(lv0):
        if lv1reset == True:
            lv0[n].reeffect()
            
        pre_diff = lv0[n].diff
        diff = lv0[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv0[n].apply_diff(rgb)

        if pre_diff is not diff:
            lv1reset = True

    if lv1reset == True:
        for v in effects[1].values():
            v.reeffect()
        for v in effects[2].values():
            v.reeffect()
        for v in effects[3].values():
            v.reeffect()
        for v in effects[4].values():
            v.reeffect()

    return rgb, lv1reset

def pipeline_lv1(img, effects, param, efconfig, prev_reset=False):
    lv1 = effects[1]
    lv2reset = prev_reset

    rgb = img.copy()
    for i, n in enumerate(lv1):
        if lv2reset == True:
            lv1[n].reeffect()
            
        pre_diff = lv1[n].diff
        diff = lv1[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = diff

        if pre_diff is not diff:
            lv2reset = True
            
    return rgb, lv2reset

def pipeline_lv2(rgb, effects, param, efconfig, prev_reset=False):
    lv2 = effects[2]
    lv3reset = prev_reset

    for i, n in enumerate(lv2):
        if lv3reset == True:
            lv2[n].reeffect()
        """
        f1 = rgb[..., 0] < 0.0
        f2 = rgb[..., 1] < 0.0
        f3 = rgb[..., 2] < 0.0
        jax.debug.print("{nn} minus = {x1}, {x2}, {x3}", nn=n, x1=jnp.sum(f1), x2=jnp.sum(f2), x3=jnp.sum(f3))
        """    
        pre_diff = lv2[n].diff
        diff = lv2[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv2[n].apply_diff(rgb)

        if pre_diff is not diff:
            lv3reset = True

    return rgb, lv3reset

def pipeline_lv3(rgb, effects, param, efconfig, prev_reset=False):
    lv3 = effects[3]
    lv4reset = prev_reset

    for i, n in enumerate(lv3):            
        if lv4reset == True:
            lv3[n].reeffect()

        diff = lv3[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv3[n].apply_diff(rgb)

    return rgb, lv4reset

def pipeline_last(rgb, effects, param, efconfig, prev_reset=False):
    lv4 = effects[4]
    lv5reset = prev_reset

    for i, n in enumerate(lv4):            
        if lv5reset == True:
            lv4[n].reeffect()

        diff = lv4[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv4[n].apply_diff(rgb)

    return rgb

def pipeline_hls(hls, effects, param, efconfig):
    hls2 = hls.copy()
    for i, n in enumerate(effects):
        diff = effects[n].make_diff(hls2, param, efconfig)
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

    return np.stack([hls2_h, hls2_l, hls2_s], axis=-1)
