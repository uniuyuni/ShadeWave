
import numpy as np

import core
import config
import crop_editor
import params
import effects

def process_pipeline(img, offset, crop_image, is_zoomed, texture_width, texture_height, click_x, click_y, primary_effects, primary_param, mask_editor2):
    
    # クロップ情報を得る、ない場合元のクロップ情報から展開
    disp_info = params.get_disp_info(primary_param)
    if disp_info is None:
        disp_info = crop_editor.CropEditor.convert_rect_to_info(params.get_crop_rect(primary_param), primary_param['original_img_size'], config.get_config('preview_size')/max(primary_param['original_img_size']))
        params.set_disp_info(primary_param, disp_info)

    # 環境設定
    efconfig = effects.EffectConfig()
    efconfig.disp_info = disp_info
    efconfig.is_zoomed = is_zoomed
    efconfig.mode = effects.EffectMode.PREVIEW

    # 背景レイヤー
    img0, reset = pipeline_lv0(img, primary_effects, primary_param, efconfig)
    disp_info = params.get_disp_info(primary_param) # Cropによって値が更新されてるかも

    if crop_image is None or reset == True:
        imgc, disp_info2 = core.crop_image(img0, disp_info, params.get_crop_rect(primary_param), texture_width, texture_height, click_x, click_y, offset, is_zoomed)
        mask_editor2.set_orientation(primary_param.get('rotation', 0), primary_param.get('rotation2', 0), primary_param.get('flip_mode', 0))
        #mask_editor2.set_texture_size(texture_width, texture_height)
        mask_editor2.set_image(primary_param['original_img_size'], disp_info2)
        params.set_disp_info(primary_param, disp_info2)
    else:
        imgc = crop_image
        disp_info2 = disp_info
    #mask_editor2.set_ref_image(effects.ColorTemperatureEffect.apply_color_temperature(imgc, primary_param),
    #                           effects.ColorTemperatureEffect.apply_color_temperature(img0, primary_param))
    mask_editor2.set_ref_image(imgc, img0)
    mask_editor2.update()
    efconfig.disp_info = disp_info2
    # 並列処理
    #split_img = []
    #split_img.extend(np.vsplit(imgc, 4))
    #height = 1024//4
    #for i, img in enumerate(split_img):
    #    split_img[i] = MainWidget._process_pipeline2(img, height*i, height, primary_effects, primary_param, mask_editor2)
    #result_img = joblib.Parallel(n_jobs=-1, require='sharedmem')(joblib.delayed(MainWidget._process_pipeline2)(img, height*i, height, primary_effects, primary_param, mask_editor2) for i, img in enumerate(split_img))
    #img2 = np.vstack(result_img)        
    img2 = pipeline2(imgc, 0, 1024, primary_effects, primary_param, mask_editor2, efconfig)

    img2 = pipeline_last(img2, primary_effects, primary_param, efconfig)

    return img2, imgc

def export_pipeline(img, primary_effects, primary_param, mask_editor2):
    
    # 環境設定
    disp_info = crop_editor.CropEditor.convert_rect_to_info(params.get_crop_rect(primary_param), primary_param['original_img_size'], 1)
    efconfig = effects.EffectConfig()
    efconfig.disp_info = disp_info
    efconfig.is_zoomed = True
    efconfig.mode = effects.EffectMode.EXPORT

    # 背景レイヤー
    img0, _ = pipeline_lv0(img, primary_effects, primary_param, efconfig)
    x1, y1, x2, y2 = params.get_crop_rect(primary_param)
    imgc = img0[y1:y2, x1:x2] # ただのクロップ
    #imgc, disp_info2 = core.crop_image(img0, disp_info, *primary_param['original_img_size'], 0, 0, (0, 0), False)
    mask_editor2.set_orientation(primary_param.get('rotation', 0), primary_param.get('rotation2', 0), primary_param.get('flip_mode', 0))
    imax = max(imgc.shape[1], imgc.shape[0])
    mask_editor2.set_texture_size(imax, imax)
    mask_editor2.set_image(primary_param['original_img_size'], disp_info)
    #mask_editor2.set_ref_image(effects.ColorTemperatureEffect.apply_color_temperature(imgc, primary_param),
    #                           effects.ColorTemperatureEffect.apply_color_temperature(img0, primary_param))
    mask_editor2.set_ref_image(imgc, img0)
    mask_editor2.update()

    img2 = pipeline2(imgc, 0, imgc.shape[0], None, primary_effects, primary_param, mask_editor2, efconfig)

    img2 = pipeline_last(img2, primary_effects, primary_param, efconfig)
    
    return img2

def pipeline2(imgc, slice_y, slice_h, primary_effects, primary_param, mask_editor2, efconfig):
    img1 = pipeline_lv1(imgc, primary_effects, primary_param, efconfig)
    img2 = pipeline_lv2(img1, primary_effects, primary_param, efconfig)
    img3 = pipeline_lv3(img2, primary_effects, primary_param, efconfig)

    # マスクレイヤー
    mask_list = mask_editor2.get_layers_list()
    for mask in mask_list:
        
        img2 = pipeline_lv1(img3, mask.effects, mask.effects_param, efconfig)
        img2 = pipeline_lv2(img2, mask.effects, mask.effects_param, efconfig)

        img3 = core.apply_mask(img2, mask.get_mask_image()[slice_y:slice_y+slice_h, :], img3)

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

def pipeline_lv1(img, effects, param, efconfig):
    lv1 = effects[1]
    lv2reset = False

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
            
    if lv2reset == True:
        for l in effects[2].values():
            l.reeffect()
        for l in effects[3].values():
            l.reeffect()
        for l in effects[4].values():
            l.reeffect()

    return rgb

def pipeline_lv2(rgb, effects, param, efconfig):
    lv2 = effects[2]

    lv1reset = False

    for i, n in enumerate(lv2):
        if lv1reset == True:
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
            lv1reset = True

    if lv1reset == True:
        for v in effects[3].values():
            v.reeffect()
        for v in effects[4].values():
            v.reeffect()

    return rgb

def pipeline_lv3(rgb, effects, param, efconfig):
    lv3 = effects[3]

    for i, n in enumerate(lv3):            
        diff = lv3[n].make_diff(rgb, param, efconfig)
        if diff is not None:
            rgb = lv3[n].apply_diff(rgb)

    return rgb

def pipeline_last(rgb, effects, param, efconfig):
    lv4 = effects[4]

    for i, n in enumerate(lv4):            
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

    # Hのみ
    hls_h = hls[..., 0]
    hls2_h = hls_h
    diff = effects['HuevsHue'].make_diff(hls_h, param, efconfig)
    if diff is not None: hls2_h = effects['HuevsHue'].apply_diff(hls_h)

    #　Lのみ
    hls_l = hls[..., 1]
    hls2_l = hls_l.copy()
    diff = effects['HuevsLum'].make_diff(hls_l, param, efconfig)
    if diff is not None: hls2_l = effects['HuevsLum'].apply_diff(hls2_l)
    diff = effects['LumvsLum'].make_diff(hls_l, param, efconfig)
    if diff is not None: hls2_l = effects['LumvsLum'].apply_diff(hls2_l)
    diff = effects['SatvsLum'].make_diff(hls_l, param, efconfig)
    if diff is not None: hls2_l = effects['SatvsLum'].apply_diff(hls2_l)

    # Sのみ
    hls_s = hls[..., 2]
    hls2_s = hls_s.copy()
    diff = effects['HuevsSat'].make_diff(hls_s, param, efconfig)
    if diff is not None: hls2_s = effects['HuevsSat'].apply_diff(hls2_s)
    diff = effects['LumvsSat'].make_diff(hls_s, param, efconfig)
    if diff is not None: hls2_s = effects['LumvsSat'].apply_diff(hls2_s)
    diff = effects['SatvsSat'].make_diff(hls_s, param, efconfig)
    if diff is not None: hls2_s = effects['SatvsSat'].apply_diff(hls2_s)
    diff = effects['saturation'].make_diff(hls_s, param, efconfig)
    if diff is not None: hls2_s = effects['saturation'].apply_diff(hls2_s)

    return np.stack([hls2_h, hls2_l, hls2_s], axis=-1)
