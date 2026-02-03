
import cv2
import numpy as np
from enum import Enum
import os
import logging

import cores.core as core
import cores.cubelut as cubelut
import cores.subpixel_shift as subpixel_shift
import cores.film_emulator as film_emulator
import cores.lens_simulator as lens_simulator
import linear_to_log.linear_to_log_lut as linear_to_log
import cores.filters as filters
import cores.local_contrast as local_contrast
import cores.highlight_recovery as highlight_recovery
import cores.hlsrgb as hlsrgb
from cores.fringe_removal.fringe_removal import remove_chromatic_aberration
from cores.distortion_correction import (
    correct_lens_distortion, correct_trapezoid, correct_four_points, correct_with_lines, warp_mesh,
    calculate_trapezoid_homography, calculate_four_point_homography, calculate_lines_homography
)
import cores.cross_filter as cross_filter
import config
import pipeline
import params
import utils.utils as utils
import utils.aiutils as aiutils
import macos as device
from enums import EffectMode, ExecutionMode

class EffectConfig():

    def __init__(self, **kwargs):
        self.disp_info = None
        self.is_zoom = False
        self.mode = EffectMode.PREVIEW
        self.resolution_scale = 1.0
        self.processor = None
        self.upstream_status = None
        self.layer_status = None
        self.upstream_hash = 0
        self.loading_flag = -1

# 補正基底クラス
class Effect():

    def __init__(self, **kwargs):
        self.diff = None
        self.hash = None
        self.execution_mode = ExecutionMode.SYNC
    
    def try_async_execution(self, img, param, efconfig, param_hash):
        """
        Attempts to execute the effect asynchronously.
        Returns:
            (bool, object): 
                - handled (bool): True if async handling logic was executed (cached returned or task submitted). 
                  If True, the caller should return `result` immediately.
                - result (object): The value to return if handled is True (usually self.diff).
        """
        if self.execution_mode == ExecutionMode.ASYNC and efconfig.processor is not None and efconfig.mode != EffectMode.EXPORT:
            from enums import PipelineStatus
            
            # Mix upstream hash into the key to ensure cache validity depends on input content
            combined_hash = hash((param_hash, efconfig.upstream_hash))

            # 1. Check Upstream Status FIRST
            # If upstream is not complete, skip heavy processing (return None or Preview)
            if efconfig.upstream_status == PipelineStatus.PREVIEW:
                # Upstream is dirty/preview, so we cannot trust input `img` for heavy calc.
                # Use preview (None for now)
                self.diff = None 
                self.hash = None # Upstream is unstable, so we are unstable
                if efconfig.layer_status is not None:
                        efconfig.layer_status = PipelineStatus.PREVIEW
                return True, self.diff
            
            # 2. Check cache with combined hash
            # We use ClassName + ParamHash + UpstreamHash as key
            cached = efconfig.processor.get_result(self.__class__.__name__, combined_hash)
            
            if cached and cached['status'] == 'COMPLETE':
                self.diff = cached['result']
                self.hash = combined_hash 
                return True, self.diff

            # Upstream complete, check if we are already running
            if cached and cached['status'] == 'RUNNING':
                if efconfig.layer_status is not None:
                    efconfig.layer_status = PipelineStatus.PREVIEW
                self.hash = None # Running
                return True, None # Return None as preview while running
                    
            # Submit new task
            efconfig.processor.submit_task(self.__class__.__name__, img, param, efconfig, combined_hash)
            if efconfig.layer_status is not None:
                    efconfig.layer_status = PipelineStatus.PREVIEW
            
            self.hash = None # Submitted
            return True, None # Submitted
            
        return False, None

    def check_sync_necessity(self, param_hash, efconfig):
        """
        Check if synchronous recalculation is needed based on params and upstream status.
        Also handles upstream hash mixing validation.
        Returns:
            (bool, int): (needed, combined_hash)
        """
        combined_hash = hash((param_hash, efconfig.upstream_hash))
        if self.hash != combined_hash:
            return True, combined_hash
        return False, combined_hash

    def reeffect(self):
        self.diff = None
        self.hash = None

    def set2widget(self, widget, param):
        pass

    def set2param(self, param, widget):
        pass

    # 差分の作成
    def make_diff(self, img, param, efconfig):
        self.diff = img

    def apply_diff(self, img):
        if self.diff is not None:
            return self.diff
        return img

    def finalize(self, param, widget):
        pass

    def get_param_dict(self, param):
        return {}

    def _get_param(self, param, key):
        return param.get(key, self.get_param_dict(param)[key])

    def delete_default_param(self, param):
        for p in self.get_param_dict(param).items():
            try:
                if param[p[0]] == p[1]:
                    del param[p[0]]
            except:
                pass

# ロード待ちエフェクト
class LoadingWaitEffect(Effect):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC

    def make_diff(self, img, param, efconfig):
        # Flag check: Wait until flag is -1
        # If flag != -1, we are loading.
        if efconfig.loading_flag != -1:
            # We are waiting for load completion.
            # Block downstream heavy effects.
            if efconfig.layer_status is not None:
                from enums import PipelineStatus
                efconfig.layer_status = PipelineStatus.PREVIEW
            
            # Since this is an ASYNC effect (conceptually), we assume we return None (Preview/NoOp)
            # while waiting.
            # We DONT submit a task because the Worker cannot see the main thread's loading flag.
            # We simply block here in the main thread pipeline logic.
            # This satisfies "Prevent subsequent heavy processing from starting".
            self.diff = None
            self.hash = None
            return None
        
        # If flag == -1, Loading Complete.
        # Pass through.
        self.diff = None
        self.hash = None
        return None

class RemoveChromaticAberrationEffect(Effect):
    def get_param_dict(self, param):
        return {
            'switch_fringe_removal': True,
            'rca_enabled': False,
            'rca_purple_amount': 20,
            'rca_fringe_width': 20,
            'rca_edge_threshold': 10,
        }
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC

    def set2widget(self, widget, param):
        widget.ids["switch_fringe_removal"].enabled = self._get_param(param, 'switch_fringe_removal')
        widget.ids["switch_rca"].active = self._get_param(param, 'rca_enabled')
        widget.ids["slider_rca_purple_amount"].set_slider_value(self._get_param(param, 'rca_purple_amount'))
        widget.ids["slider_rca_fringe_width"].set_slider_value(self._get_param(param, 'rca_fringe_width'))
        widget.ids["slider_rca_edge_threshold"].set_slider_value(self._get_param(param, 'rca_edge_threshold'))
    
    def set2param(self, param, widget):
        param['switch_fringe_removal'] = widget.ids["switch_fringe_removal"].enabled
        param['rca_enabled'] = widget.ids["switch_rca"].active
        param['rca_purple_amount'] = widget.ids["slider_rca_purple_amount"].value
        param['rca_fringe_width'] = widget.ids["slider_rca_fringe_width"].value
        param['rca_edge_threshold'] = widget.ids["slider_rca_edge_threshold"].value

    def make_diff(self, img, param, efconfig):
        switch_fringe_removal = self._get_param(param, 'switch_fringe_removal')
        rca_enabled = self._get_param(param, 'rca_enabled')
        rca_purple_amount = self._get_param(param, 'rca_purple_amount')
        rca_fringe_width = self._get_param(param, 'rca_fringe_width')
        rca_edge_threshold = self._get_param(param, 'rca_edge_threshold')
        if switch_fringe_removal == False or rca_enabled == False or efconfig.loading_flag != -1:
            if efconfig.processor is not None:
                efconfig.processor.cancel_effect(self.__class__.__name__)
            
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((rca_enabled, rca_purple_amount, rca_fringe_width, rca_edge_threshold))

            # Async Processing Logic
            handled, result = self.try_async_execution(img, param, efconfig, param_hash)
            if handled:
                return result

            needed, combined_hash = self.check_sync_necessity(param_hash, efconfig)
            if needed:
                self.hash = combined_hash
                self.diff = remove_chromatic_aberration(img, purple_amount=rca_purple_amount/10, fringe_width=rca_fringe_width, edge_threshold=rca_edge_threshold/1000, min_saturation=0.1)
        
        return self.diff

# レンズモディファイア
class LensModifierEffect(Effect):
    
    def __init__(self, lens_modifier_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.callback = lens_modifier_callback

    def get_param_dict(self, param):
        return {
            'switch_lens_modifier': True,
            'lens_modifier': True,
            'color_modification': True,
            'subpixel_distortion': True,
            'geometry_distortion': True,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_lens_modifier"].enabled = self._get_param(param, 'switch_lens_modifier')
        widget.ids["checkbox_color_modification"].active = self._get_param(param, 'color_modification')
        widget.ids["checkbox_subpixel_distortion"].active = self._get_param(param, 'subpixel_distortion')
        widget.ids["checkbox_geometry_distortion"].active = self._get_param(param, 'geometry_distortion')

    def set2param(self, param, widget):
        param['switch_lens_modifier'] = widget.ids["switch_lens_modifier"].enabled
        param['color_modification'] = widget.ids["checkbox_color_modification"].active
        param['subpixel_distortion'] = widget.ids["checkbox_subpixel_distortion"].active
        param['geometry_distortion'] = widget.ids["checkbox_geometry_distortion"].active

    def make_diff(self, img, param, efconfig):
        switch_lm = self._get_param(param, 'switch_lens_modifier')        
        lm = self._get_param(param, 'lens_modifier')
        cd = self._get_param(param, 'color_modification')
        sd = self._get_param(param, 'subpixel_distortion')
        gd = self._get_param(param, 'geometry_distortion')
        if switch_lm == False or lm == False or (cd == False and sd == False and gd == False) or efconfig.loading_flag != -1:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((cd, sd, gd))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff, is_cm, is_sd, is_gd = core.modify_lensfun(img, cd, sd, gd)

                # 適用されなかったパラメータをUIに反映
                param['color_modification'] = is_cm
                param['subpixel_distortion'] = is_sd
                param['geometry_distortion'] = is_gd
                if self.callback:
                    self.callback()
        
        return self.diff
    

# サブピクセルシフト合成
class SubpixelShiftEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_details': True,
            'subpixel_shift': False,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_details"].active = self._get_param(param, 'switch_details')
        widget.ids["switch_subpixel_shift"].active = self._get_param(param, 'subpixel_shift')

    def set2param(self, param, widget):
        param['switch_details'] = widget.ids["switch_details"].active
        param['subpixel_shift'] = widget.ids["switch_subpixel_shift"].active

    def make_diff(self, img, param, efconfig):
        switch_details = self._get_param(param, 'switch_details')
        ss = self._get_param(param, 'subpixel_shift')
        if switch_details == False or ss == False:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ss))
            if self.hash != param_hash:
                self.hash = param_hash
                
                self.diff = subpixel_shift.create_enhanced_image(img)
        
        return self.diff
    

class InpaintDiff:
    def __init__(self, **kwargs):
        self.type = kwargs.get('type', "mask")
        self.disp_info = kwargs.get('disp_info', None)
        self.image = kwargs.get('image', None)

    def image2list(self):
        if type(self.image) is np.ndarray:
            self.image = utils.convert_image_to_list(self.image)

    def list2image(self):
        if type(self.image) is list or type(self.image) is tuple:
            self.image = utils.convert_image_from_list(self.image)

class InpaintEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.inpaint_diff_list = []
        self.inpaint_mask_list = []
        self.mask_editor = None

    def get_param_dict(self, param):
        return {
            'switch_details': True,
            'inpaint': False,
            'inpaint_predict': False,
            'inpaint_diff_list': [],
            'inpaint_mask_list': [],
        }

    def set2widget(self, widget, param):
        widget.ids["switch_details"].active = self._get_param(param, 'switch_details')
        widget.ids["switch_inpaint"].active = self._get_param(param, 'inpaint')
        widget.ids["button_inpaint_predict"].state = "normal" if self._get_param(param, 'inpaint_predict') == False else "down"

        # 履歴描画
        if self.mask_editor is not None:
            self.mask_editor.clear_mask()
            self.inpaint_mask_list = self._get_param(param, 'inpaint_mask_list')
            for inpaint_mask in self.inpaint_mask_list:
                self.mask_editor.add_mask(inpaint_mask.disp_info, inpaint_mask.image)
            self.mask_editor.delay_update_canvas()

    def set2param(self, param, widget):
        param['switch_details'] = widget.ids["switch_details"].active
        param['inpaint'] = widget.ids["switch_inpaint"].active
        param['inpaint_predict'] = widget.ids["button_inpaint_predict"].state == "down"

        if param['inpaint'] == True:
            if self.mask_editor is None:
                from widgets.mask_editor import MaskEditor
                
                self.mask_editor = MaskEditor(param,
                                              effect_ctrl_param=(0, 'inpaint'),
                                              touch_up_callback=self.mask_editor_touch_up)
                self.mask_editor.zoom = params.get_disp_info(param)[4]
                self.mask_editor.pos = [0, 0]
                widget.ids["preview_widget"].add_widget(self.mask_editor)
                param['inpaint_mask_list'] = self.inpaint_mask_list = []
            
        if param['inpaint'] == False:
            if self.mask_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor = None
                param['inpaint_mask_list'] = self.inpaint_mask_list = []

    def make_diff(self, img, param, efconfig):
        self.inpaint_diff_list = self._get_param(param, 'inpaint_diff_list')
        self.inpaint_mask_list = self._get_param(param, 'inpaint_mask_list')

        switch_details = self._get_param(param, 'switch_details')
        ip = self._get_param(param, 'inpaint')
        ipp = self._get_param(param, 'inpaint_predict')
        if switch_details == True and (ip == True and ipp == True):
            import helpers.qwen_image_helper as qih
            
            param['inpaint_predict'] = False # なぜか二重起動するときがあるので予防

            mask = self.mask_editor.get_mask().astype(np.float32) / 255.0

            for inpaint_mask in self.inpaint_mask_list:
                proc_x, proc_y, proc_w, proc_h = inpaint_mask.disp_info

                #img2 = qih.predict_helper(img, mask, (proc_x, proc_y, proc_w, proc_h), qih.predict_erace)
                img2 = img

                # 範囲を記録
                self.inpaint_diff_list.append(
                    InpaintDiff(type="image",
                                disp_info=(proc_x, proc_y, proc_w, proc_h),
                                image=img2[proc_y:proc_y+proc_h, proc_x:proc_x+proc_w]))

            param['inpaint_diff_list'] = self.inpaint_diff_list
            
            # マスク消去
            self.mask_editor.clear_mask()
            param['inpaint_mask_list'] = self.inpaint_mask_list = []
            self.mask_editor.delay_update_canvas()
        
        param_hash = hash((len(self.inpaint_diff_list)))
        if self.hash != param_hash:
            self.hash = param_hash

            if len(self.inpaint_diff_list) > 0:
                img2 = img.copy()
                for inpaint_diff in self.inpaint_diff_list:
                    if inpaint_diff.type == "image":
                        inpaint_diff.list2image()   # データを変換する必要があるときがある
                        cx, cy, cw, ch = inpaint_diff.disp_info
                        img2[cy:cy+ch, cx:cx+cw] = inpaint_diff.image
                self.diff = img2
            else:
                self.diff = None

        return self.diff

    def mask_editor_touch_up(self, param, mask):
        
        # イメージが四角く処理されていた場合のオフセット計算
        w, h = param['original_img_size']
        eh, ew = mask.shape[:2]
        x, y = (ew-w)//2, (eh-h)//2

        # 処理
        self.inpaint_mask_list = []
        bboxes = core.get_multiple_mask_bbox(mask)
        for bbox in bboxes:
            proc_x, proc_y, proc_w, proc_h = aiutils.calculate_expanded_crop(
                                                mask.shape[1], mask.shape[0],
                                                bbox[0] + x, bbox[1] + y, bbox[2], bbox[3],
                                                32, 32)

            # 範囲を記録
            self.inpaint_mask_list.append(
                InpaintDiff(type="mask",
                            disp_info=(proc_x, proc_y, proc_w, proc_h),
                            image=mask[proc_y:proc_y+proc_h, proc_x:proc_x+proc_w]))

        param['inpaint_mask_list'] = self.inpaint_mask_list


class CrossFilterEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_cross_filter': True,
            'cross_filter_num_points': 0,
            'cross_filter_length': 2000,
            'cross_filter_angle': 0,
            'cross_filter_threshold': 70,
            'cross_filter_intensity': 15,
            'cross_filter_spectral': 25,
            'cross_filter_thickness': 1,
            'cross_filter_distance': 100,
            'cross_filter_random': 50
        }

    def set2widget(self, widget, param):
        widget.ids["switch_cross_filter"].enabled = self._get_param(param, 'switch_cross_filter')
        widget.ids["slider_cross_filter_num_points"].set_slider_value(self._get_param(param, 'cross_filter_num_points'))
        widget.ids["slider_cross_filter_length"].set_slider_value(self._get_param(param, 'cross_filter_length'))
        widget.ids["slider_cross_filter_angle"].set_slider_value(self._get_param(param, 'cross_filter_angle'))
        widget.ids["slider_cross_filter_threshold"].set_slider_value(self._get_param(param, 'cross_filter_threshold'))
        widget.ids["slider_cross_filter_intensity"].set_slider_value(self._get_param(param, 'cross_filter_intensity'))
        widget.ids["slider_cross_filter_spectral"].set_slider_value(self._get_param(param, 'cross_filter_spectral'))
        widget.ids["slider_cross_filter_thickness"].set_slider_value(self._get_param(param, 'cross_filter_thickness'))
        widget.ids["slider_cross_filter_distance"].set_slider_value(self._get_param(param, 'cross_filter_distance'))
        widget.ids["slider_cross_filter_random"].set_slider_value(self._get_param(param, 'cross_filter_random'))

    def set2param(self, param, widget):
        param['switch_cross_filter'] = widget.ids["switch_cross_filter"].enabled
        param['cross_filter_num_points'] = widget.ids["slider_cross_filter_num_points"].value
        param['cross_filter_length'] = widget.ids["slider_cross_filter_length"].value
        param['cross_filter_angle'] = widget.ids["slider_cross_filter_angle"].value
        param['cross_filter_threshold'] = widget.ids["slider_cross_filter_threshold"].value
        param['cross_filter_intensity'] = widget.ids["slider_cross_filter_intensity"].value
        param['cross_filter_spectral'] = widget.ids["slider_cross_filter_spectral"].value
        param['cross_filter_thickness'] = widget.ids["slider_cross_filter_thickness"].value
        param['cross_filter_distance'] = widget.ids["slider_cross_filter_distance"].value
        param['cross_filter_random'] = widget.ids["slider_cross_filter_random"].value

    def make_diff(self, rgb, param, efconfig):
        switch_cross_filter = self._get_param(param, 'switch_cross_filter')
        num_points = self._get_param(param, 'cross_filter_num_points')
        length = self._get_param(param, 'cross_filter_length') #* efconfig.disp_info[4]
        angle = self._get_param(param, 'cross_filter_angle')
        threshold = self._get_param(param, 'cross_filter_threshold')
        intensity = self._get_param(param, 'cross_filter_intensity') #/ max(0.01, efconfig.disp_info[4])
        spectral = self._get_param(param, 'cross_filter_spectral')
        thickness = max(1.0, self._get_param(param, 'cross_filter_thickness')) #* efconfig.disp_info[4])
        distance = self._get_param(param, 'cross_filter_distance') #* efconfig.disp_info[4]
        random = self._get_param(param, 'cross_filter_random')
        if switch_cross_filter is False or num_points == 0 or length <= 1 or intensity == 0 or efconfig.loading_flag > -1:
            if efconfig.processor is not None:
                efconfig.processor.cancel_effect(self.__class__.__name__)

            self.diff = None
            self.hash = None
        else:
            param_hash = hash((num_points, length, angle, threshold, intensity, spectral, thickness, distance, random))

           # Async Processing Logic
            handled, result = self.try_async_execution(rgb, param, efconfig, param_hash)
            if handled:
                return result

            needed, combined_hash = self.check_sync_necessity(param_hash, efconfig)
            if needed:
                self.hash = combined_hash

                self.diff = cross_filter.apply_cross_filter(
                                rgb,
                                num_points=int(num_points),
                                length=int(length),
                                angle_deg=angle,
                                threshold=threshold/50.0,
                                intensity=intensity/100.0,
                                spectral_strength=spectral/100.0,
                                line_thickness=thickness,
                                min_distance=distance,
                                randomness=random/100.0,
                                speed_factor=4)

        return self.diff

# 変形描画
class DistortionEffect(Effect):

    def __init__(self, distortion_callback=None, **kwargs):
        super().__init__(**kwargs)
        
        self.distortion_painter = None
        self.is_initial_open = 0
        self.effect_type = 'forward_warp'
        self.set_distortion_callback(distortion_callback)

    def set_distortion_callback(self, callback):
        self.distortion_callback = callback

    def get_param_dict(self, param):
        return {
            'switch_distortion': True,
            'distortion_recorded': [],
            'distortion_brush_size': 300,
            'distortion_strength': 50,
        }    

    def set2widget(self, widget, param):
        widget.ids["switch_distortion"].enabled = self._get_param(param, 'switch_distortion')
        widget.ids["slider_distortion_brush_size"].set_slider_value(self._get_param(param, 'distortion_brush_size'))
        widget.ids["slider_distortion_strength"].set_slider_value(self._get_param(param, 'distortion_strength'))
        
        if self.distortion_painter is not None:
            self.distortion_painter.set_recorded(self._get_param(param, 'distortion_recorded'))
            self.distortion_painter.remap_recorded()

    def set2param(self, param, widget):
        param['switch_distortion'] = widget.ids["switch_distortion"].enabled
        distortion_enable = False if widget.ids["effects"].current_tab.text != "Li" else True
        param['distortion_brush_size'] = widget.ids["slider_distortion_brush_size"].value
        param['distortion_strength'] = widget.ids["slider_distortion_strength"].value

        # エディタを開く
        if distortion_enable == True:
            self._open_distortion_painter(param, widget)

        # エディタを閉じる
        elif distortion_enable == False:
            self._close_distortion_painter(param, widget)

        if self.distortion_painter is not None:
            self.distortion_painter.set_brush_size(param['distortion_brush_size'])
            self.distortion_painter.set_strength(param['distortion_strength'])

            # クロップ範囲をリセット
            if widget.ids["button_distortion_reset"].state == "down":
                widget.ids["button_distortion_reset"].state = "normal" # 無限ルーぷ防止
                self.distortion_painter.reset_image()


    def set2param2(self, param, arg):
        if self.distortion_painter is not None:
            self.distortion_painter.set_effect(arg)
            self.effect_type = arg

    def make_diff(self, img, param, efconfig):
        if self.is_initial_open > 0:
            if self.distortion_painter is not None and efconfig.loading_flag != None:
                self.distortion_painter.set_effect(self.effect_type)
                self.distortion_painter.set_primary_param(param)
                self.distortion_painter.set_ref_image(img, True)
                self.distortion_painter.set_recorded(self._get_param(param, 'distortion_recorded'))
                self.distortion_painter.remap_recorded()

                if efconfig.loading_flag == -1:
                    self.is_initial_open = 0
        
        switch_distortion = self._get_param(param, 'switch_distortion')
        if switch_distortion == True and self.distortion_painter is not None:
            self.diff = self.distortion_painter.get_current_image()
            self.hash = hash((len(self.distortion_painter.get_recorded())))

        else:
            dr = self._get_param(param, 'distortion_recorded')

            if switch_distortion == False or len(dr) == 0:
                self.diff = None
                self.hash = None
            else:
                param_hash = hash((len(dr)))
                if self.hash != param_hash:
                    from widgets.distortion_painter import DistortionCanvas

                    tcg_info = params.param_to_tcg_info(param)
                    self.diff = DistortionCanvas.replay_recorded(img, dr, tcg_info)
                self.hash = param_hash
        
        return self.diff

    def apply_diff(self, img):
        if self.diff is not None:
            if self.distortion_painter is not None:
                self.diff = self.distortion_painter.get_current_image()
                if self.diff is not None:
                    return self.diff
            else:
                return self.diff
        return img

    def finalize(self, param, widget):
        self._close_distortion_painter(param, widget)

    def _open_distortion_painter(self, param, widget):
        if self.distortion_painter is None:
            from widgets.distortion_painter import DistortionCanvas

            self.distortion_painter = DistortionCanvas(#image_widget=widget.ids["preview_widget"],
                    recorded=self._get_param(param, 'distortion_recorded'),
                    callback=self._painter_callback,
                    effect_type=self.effect_type,
                    brush_size=widget.ids["slider_distortion_brush_size"].value,
                    strength=widget.ids["slider_distortion_strength"].value)
            self.distortion_painter.set_primary_param(param)
            widget.ids["preview_widget"].add_widget(self.distortion_painter)

            self.is_initial_open = 1

    def _close_distortion_painter(self, param, widget):
        if self.distortion_painter is not None:
            widget.ids["preview_widget"].remove_widget(self.distortion_painter)
            param['distortion_recorded'] = self.distortion_painter.get_recorded()
            self.distortion_painter = None

    def _painter_callback(self, proc, widget):
        if self.distortion_callback is not None:
            self.distortion_callback(proc, widget)

# 画像回転、反転、変形
class GeometryEffect(Effect):

    def __init__(self, geometry_callback=None, **kwargs):
        super().__init__(**kwargs)
        
        self.geometry_editor = None
        self.geometry_editor_callback = geometry_callback

    def _editor_update_callback(self, type, widget):
        if self.geometry_editor_callback:
            self.geometry_editor_callback(type, widget)

    def get_param_dict(self, param):
        return {
            'rotation': 0,
            'rotation2': 0,
            'flip_mode': 0,
            'crop_enable': False,
            'switch_distortion_correction': True,
            'lens_distortion_strength': 0,
            'lens_distortion_scale': 0,
            'correct_horizontal': 0,
            'correct_vertical': 0,
            'focal_length': 20,
            'four_points': [],
            'reference_lines': [],
            'mesh_size': [4, 4],
            'control_points': {},
            'matrix': np.eye(3),
        }

    def set2widget(self, widget, param):
        widget.ids["slider_rotation"].set_slider_value(self._get_param(param, 'rotation'))
        widget.ids["switch_distortion_correction"].active = self._get_param(param, 'switch_distortion_correction')
        widget.ids["slider_lens_distortion_strength"].set_slider_value(self._get_param(param, 'lens_distortion_strength'))
        widget.ids["slider_lens_distortion_scale"].set_slider_value(self._get_param(param, 'lens_distortion_scale'))
        widget.ids["slider_correct_trapezoid_h"].set_slider_value(self._get_param(param, 'correct_horizontal'))
        widget.ids["slider_correct_trapezoid_v"].set_slider_value(self._get_param(param, 'correct_vertical'))
        widget.ids["slider_focal_length"].set_slider_value(self._get_param(param, 'focal_length'))

        if self.geometry_editor is not None:
            self.geometry_editor.set_correction_params(param)

    def set2param(self, param, widget):
        param['rotation'] = widget.ids["slider_rotation"].value
        param['switch_distortion_correction'] = widget.ids["switch_distortion_correction"].active
        param['lens_distortion_strength'] = widget.ids["slider_lens_distortion_strength"].value
        param['lens_distortion_scale'] = widget.ids["slider_lens_distortion_scale"].value
        param['correct_horizontal'] = widget.ids["slider_correct_trapezoid_h"].value
        param['correct_vertical'] = widget.ids["slider_correct_trapezoid_v"].value
        param['focal_length'] = widget.ids["slider_focal_length"].value
    
        # crop_rect がないのはマスク
        if params.get_crop_rect(param) is not None:

            def get_selected():
                """1行で全確認"""
                for btn_name in ['btn_lens', 'btn_trapezoid', 'btn_four_points', 'btn_mesh', 'btn_lines']:
                    btn = getattr(widget.ids, btn_name)
                    if btn.state == 'down':
                        return btn.text
                return None
            
            # Update params from editor if active (BEFORE opening/syncing)
            if self.geometry_editor is not None:
                    param.update(self.geometry_editor.get_correction_params())

            # Update Matrix Param based on current params (Visual Fix)
            self._update_matrix_param(param)

            type = get_selected()
            self._open_geometry_editor(widget, type, param)

    def _update_matrix_param(self, param):
        """
        画像処理を行わずにパラメータのみからマトリックスを計算・更新する
        """
        params.set_matrix(param, None)

        if self._get_param(param, 'switch_distortion_correction') == False:
            return

        # パラメータ取得
        correct_horizontal = self._get_param(param, 'correct_horizontal')
        correct_vertical = self._get_param(param, 'correct_vertical')
        focal_length = self._get_param(param, 'focal_length')
        #ang = self._get_param(param, 'rotation')
        #ang2 = self._get_param(param, 'rotation2')
        four_points = self._get_param(param, 'four_points')
        reference_lines = self._get_param(param, 'reference_lines')
        
        # 基準サイズ（回転後を想定して max(w, h) の正方形）
        w_org, h_org = param['original_img_size']
        size = max(w_org, h_org)
        half_size = size / 2

        # 台形補正
        if correct_horizontal != 0 or correct_vertical != 0:
            multiplier = 0.5 + (focal_length * 0.025)
            f_pixel = size * multiplier
            
            H = calculate_trapezoid_homography(
                size, size,
                horizontal=correct_horizontal * 0.5,
                vertical=correct_vertical * 0.5,
                focal_length=f_pixel,
            )
            params.add_matrix(param, H, offset=(half_size, half_size))
        
        # 4点補正
        reset_points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        if four_points != [] and four_points != reset_points:
            # 座標変換用のダミー画像サイズ
            # Note: tcg_info内のmatrixはここまで（台形補正）の結果を含んでいる必要がある
            # params.add_matrixでparam['matrix']は更新されている
            tcg_info = params.param_to_tcg_info(param)
            
            class DummyShape:
                def __init__(self, s): self.shape = (s, s, 3)
            dummy_img = DummyShape(size)

            src_point = []
            for cx, cy in four_points:
                src_point.append(params.tcg_to_ref_image(cx, cy, dummy_img, tcg_info))
            dst_point = []
            for cx, cy in reset_points:
                dst_point.append(params.tcg_to_ref_image(cx, cy, dummy_img, tcg_info))

            # dst -> src (Inverse)
            H_inv = calculate_four_point_homography(src_point, dst_point)
            # src -> dst (Forward)
            H = np.linalg.inv(H_inv)
            
            params.add_matrix(param, H, offset=(half_size, half_size))

        # Lines
        if len(reference_lines) > 0:
            tcg_info = params.param_to_tcg_info(param)
            H = calculate_lines_homography(reference_lines, size, size, tcg_info=tcg_info)
            if H is not None:
                params.add_matrix(param, H, offset=(half_size, half_size))

    def set2param2(self, param, arg):
        if arg == 'hflip':
            param['flip_mode'] = self._get_param(param, 'flip_mode') ^ 1

        elif arg == 'vflip':
            param['flip_mode'] = self._get_param(param, 'flip_mode') ^ 2

        elif arg == 90:
            rot = self._get_param(param, 'rotation2') + 90.0
            if rot >= 90*4:
                rot = 0
            param['rotation2'] = rot

        elif arg == -90:
            rot = self._get_param(param, 'rotation2') - 90.0
            if rot < 0:
                rot = 90*3
            param['rotation2'] = rot


    def make_diff(self, img, param, efconfig):
        ang = self._get_param(param, 'rotation')
        ang2 = self._get_param(param, 'rotation2')
        flp = self._get_param(param, 'flip_mode')
        crop_enable = self._get_param(param, 'crop_enable')
        switch_distortion_correction = self._get_param(param, 'switch_distortion_correction')
        lens_distortion_strength = self._get_param(param, 'lens_distortion_strength')
        lens_distortion_scale = self._get_param(param, 'lens_distortion_scale')
        correct_horizontal = self._get_param(param, 'correct_horizontal')
        correct_vertical = self._get_param(param, 'correct_vertical')
        focal_length = self._get_param(param, 'focal_length')
        four_points = self._get_param(param, 'four_points')
        reference_lines = self._get_param(param, 'reference_lines')
        mesh_size = self._get_param(param, 'mesh_size')
        control_points = self._get_param(param, 'control_points') # dict

        # list, convert to tuple for hashing
        fps_hash = tuple(tuple(x) for x in four_points) if four_points else None
        lines_hash = tuple(tuple(tuple(p) for p in line) for line in reference_lines) if reference_lines else None
        cp_hash = tuple(sorted((k, tuple(v)) for k, v in control_points.items())) if control_points else None
        mesh_hash = tuple(mesh_size)
        
        param_hash = hash((switch_distortion_correction, ang, ang2, flp, crop_enable, lens_distortion_strength, lens_distortion_scale, correct_horizontal, correct_vertical, focal_length, fps_hash, lines_hash, mesh_hash, cp_hash))
        if self.hash != param_hash:
            self.hash = param_hash
            
            params.set_matrix(param, None)

            # レンズ歪み補正
            if switch_distortion_correction == True and lens_distortion_strength != 0:
                img = correct_lens_distortion(
                        img,
                        strength=lens_distortion_strength,
                        scale=lens_distortion_scale / 100.0 + 1.0,
                        interpolation='bicubic' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                        grid_size=2 if efconfig.mode == EffectMode.EXPORT else 4,
                )

            # 回転
            img = core.rotation(img, ang + ang2, flp,
                    inter_mode='bicubic' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                    border_mode="reflect" if crop_enable == False else "constant")

            tcg_info = params.param_to_tcg_info(param)
            # 基準サイズ（回転後を想定して max(w, h) の正方形）
            size = max(img.shape[0], img.shape[1])
            half_size = size / 2

            if switch_distortion_correction == True:
                # 台形補正
                if correct_horizontal != 0 or correct_vertical != 0:
                    # Focal Length Mapping:
                    # 0-100 Slider -> Multiplier.
                    # Assuming 0 is Wide (High persp), 100 is Tele (Low persp).
                    # Previous default was max(w,h) which corresponds to freq standard lens.
                    # Let's say Slider=20 -> 1.0x (Standard)
                    # Slider=0 -> 0.5x (Super Wide)
                    # Slider=100 -> 5.0x (Super Tele)
                    
                    # Using a linear mapping for simplicity first:
                    # val 20 -> 1.0
                    # val 0 -> 0.5 (delta -20 -> -0.5 => 1 unit = 0.5/20 = 0.025)
                    # val 100 -> 1.0 + (80 * 0.025) = 1.0 + 2.0 = 3.0
                    
                    # So: multiplier = 0.5 + (focal_length* 0.025)
                    # 0 -> 0.5
                    # 20 -> 1.0
                    # 100 -> 3.0
                    # --- Trapezoid Correction ---
                    base_f = np.max(img.shape[:2])
                    multiplier = 0.5 + (focal_length * 0.025)
                    f_pixel = base_f * multiplier # Focal len in pixels

                    img, H = correct_trapezoid(
                            img,
                            horizontal=correct_horizontal * 0.5, 
                            vertical=correct_vertical * 0.5,
                            focal_length=f_pixel,
                            interpolation='bicubic' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                    )
                    params.add_matrix(param, H, offset=(half_size, half_size))
                                
                # 4点補正
                reset_points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
                if four_points != [] and four_points != reset_points:

                    # 座標をテクスチャ座標へ変換
                    src_point = []
                    for cx, cy in four_points:
                        src_point.append(params.tcg_to_ref_image(cx, cy, img, tcg_info))
                    dst_point = []
                    for cx, cy in reset_points:
                        dst_point.append(params.tcg_to_ref_image(cx, cy, img, tcg_info))

                    img, H = correct_four_points(
                            img,
                            src_point,
                            dst_point,
                            interpolation='lanczos' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                    )
                    params.add_matrix(param, H, offset=(half_size, half_size))
                    
                # Lines
                if len(reference_lines) > 0: 
                    img, H = correct_with_lines(
                        img,
                        reference_lines,
                        tcg_info=tcg_info, # correct_with_lines内部でtcg_info使うので渡す
                        interpolation='lanczos' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                    )
                    params.add_matrix(param, H, offset=(half_size, half_size))

                # Mesh           
                if control_points:
                    # Ensure keys are tuples
                    cp = {}
                    for k, v in control_points.items():
                        if isinstance(k, str):
                            try:
                                parts = k.strip('()').split(',')
                                key = (int(parts[0]), int(parts[1]))
                            except:
                                continue
                        else:
                            key = tuple(k)
                        cp[key] = tuple(v)
                        
                    img = warp_mesh(
                        img,
                        mesh_size if mesh_size else (4, 4),
                        cp,
                        tcg_info=tcg_info,
                        interpolation='lanczos' if efconfig.mode == EffectMode.EXPORT else 'bilinear'
                    )

            self.diff = img
        
        return self.diff

    def finalize(self, param, widget):
        self._open_geometry_editor(widget, None) # 閉じる

    def _open_geometry_editor(self, widget, type, param=None):
        from widgets.distortion_correction import (
            LensDistortionWidget, LineGuideCorrectionWidget, TrapezoidCorrectionWidget, FourPointCorrectionWidget, MeshWarpWidget
        )
        
        # Check if we can reuse the existing editor
        current_editor_class = self.geometry_editor.__class__.__name__ if self.geometry_editor else None
        target_class = None
        match type:
            case 'Lens': target_class = 'LensDistortionWidget'
            case 'Trapezoid': target_class = 'TrapezoidCorrectionWidget'
            case 'Four Points': target_class = 'FourPointCorrectionWidget'
            case 'Mesh': target_class = 'MeshWarpWidget'
            case 'Lines': target_class = 'LineGuideCorrectionWidget'
            case 'Points': target_class = 'PointWarpWidget'

        # 前のを削除
        if current_editor_class != target_class:
            if self.geometry_editor is not None:
                widget.ids['preview_widget'].remove_widget(self.geometry_editor)
                self.geometry_editor = None

        # 作成
        if self.geometry_editor is None:
            texture_size = (config.get_config('preview_width'), config.get_config('preview_height'))
            match type:
                case 'Lens': self.geometry_editor = LensDistortionWidget(texture_size, param)
                case 'Trapezoid': self.geometry_editor = TrapezoidCorrectionWidget(texture_size, param)
                case 'Four Points': self.geometry_editor = FourPointCorrectionWidget(texture_size, param)
                case 'Mesh': self.geometry_editor = MeshWarpWidget(texture_size, param)
                case 'Lines': self.geometry_editor = LineGuideCorrectionWidget(texture_size, param)

            if self.geometry_editor is not None:
                self.geometry_editor.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
                self.geometry_editor.type = type # 消す時必要
                widget.ids['preview_widget'].add_widget(self.geometry_editor)

        # Update parameters and image if applicable
        if self.geometry_editor is not None and param is not None:
            if type == 'Lens':
                #self.geometry_editor.set_image(widget.imgset.img) # なんか重い
                # Sync Params
                self.geometry_editor.set_correction_params(param)

            elif type == 'Trapezoid':
                pass

            elif type == 'Four Points':                
                # Sync Params
                self.geometry_editor.set_correction_params(param)
                # Bind callback
                self.geometry_editor.set_callback(self._editor_update_callback)

            elif type == 'Lines':
                # Sync Params
                self.geometry_editor.set_correction_params(param)
                # Bind callback
                self.geometry_editor.set_callback(self._editor_update_callback)

            elif type == 'Mesh':
                # Sync Params
                self.geometry_editor.set_correction_params(param)
                # Bind callback
                self.geometry_editor.set_callback(self._editor_update_callback)

    def close_geometry_editor(self, widget):
        if self.geometry_editor is not None:
            btn_id = f"btn_{self.geometry_editor.type.lower().replace(' ', '_')}"
            widget.ids[btn_id].state = 'normal'
            widget.ids['preview_widget'].remove_widget(self.geometry_editor)
            self.geometry_editor = None

# クロップ
class CropEffect(Effect):

    def __init__(self, crop_callback=None, **kwargs):
        super().__init__(**kwargs)
        
        self.backup_img = None

        self.crop_editor = None
        self.crop_editor_callback = crop_callback

    def set_editing_callback(self, callback):
        self.crop_editor_callback = callback

    def _param_to_aspect_ratio(self, param):
        ar = self._get_param(param, 'aspect_ratio')
        return eval(ar if ar != "None" else "0")

    def get_param_dict(self, param):
        param2 = param.copy()
        params.set_crop_rect(param2, core.get_initial_crop_rect(*param['original_img_size']))
        #params.set_disp_info(param2, core.get_initial_disp_info(*param['original_img_size'], config.get_config('preview_size')/max(param['original_img_size'])))
        return {
            'rotation': 0,
            'rotation2': 0,
            'crop_enable': False,
            'crop_rect': param2['crop_rect'],
            'aspect_ratio': "None",
            'auto_crop': False,
        }

    def set2widget(self, widget, param):
        widget.ids["spinner_acpect_ratio"].set_text(param.get('aspect_ratio', "None"))

        if self.crop_editor is not None:
            self.crop_editor.set_to_local_crop_rect(params.get_crop_rect(param))

    def set2param(self, param, widget):
        param['crop_enable'] = False if widget.ids["effects"].current_tab.text != "Ge" else True
        param['aspect_ratio'] = widget.ids["spinner_acpect_ratio"].text

        # crop_rect がないのはマスク
        if params.get_crop_rect(param) is not None:

            # クロップエディタを開く
            if param['crop_enable'] == True:
                self._open_crop_editor(param, widget)

            # クロップエディタを閉じる
            elif param['crop_enable'] == False:
                self._close_crop_editor(param, widget)

            # クロップ範囲をリセット
            if widget.ids["button_crop_reset"].state == "down":
                self.reset_crop_editor()

            self.reset2_crop_editor(param)

            # 自動クロップ
            if widget.ids["button_crop_auto"].state == "down":
                self.auto_crop_editor(self.backup_img)

            # クロップ情報を更新
            if self.crop_editor is not None:
                params.set_crop_rect(param, self.crop_editor.get_crop_rect())

    def make_diff(self, img, param, efconfig):
        ce = self._get_param(param, 'crop_enable')
        disp_info = params.get_disp_info(param)

        self.backup_img = img

        if ce == True or disp_info is None:
            self.diff = None
            self.hash = None
            param['img_size'] = (param['original_img_size'][0], param['original_img_size'][1])
            msize = max(param['original_img_size'][0], param['original_img_size'][1])
            scale = config.get_config('preview_size')/msize
            params.set_disp_info(param, (0, 0, msize, msize, scale))
        else:
            param_hash = hash((ce))
            if self.hash != param_hash:
                self.diff = disp_info
                self.hash = param_hash
                param['img_size'] = (disp_info[2], disp_info[3])
        return self.diff

    def apply_diff(self, img):
        return img

    def _open_crop_editor(self, param, widget):
        if self.crop_editor is None:
            from widgets.crop_editor import CropEditor

            input_width, input_height = param['original_img_size']
            x1, y1, x2, y2 = params.get_crop_rect(param)
            scale = config.get_config('preview_size') * device.dpi_scale() / max(input_width, input_height)
            self.crop_editor = CropEditor(input_width=input_width, input_height=input_height, scale=scale, crop_rect=[x1, y1, x2, y2], aspect_ratio=self._param_to_aspect_ratio(param))
            self.crop_editor.set_editing_callback(self._crop_editing)
            widget.ids["preview_widget"].add_widget(self.crop_editor)

            # 編集中は一時的に変更
            params.set_disp_info(param, core.get_initial_disp_info(input_width, input_height, scale))

            # 保存しておく
            self.param = param

    def _close_crop_editor(self, param, widget):
        if self.crop_editor is not None:
            params.set_crop_rect(param, self.crop_editor.get_crop_rect())
            params.set_disp_info(param, self.crop_editor.get_disp_info())
            widget.ids["preview_widget"].remove_widget(self.crop_editor)
            self.crop_editor = None

    def _crop_editing(self, proc, widget):
        if self.crop_editor_callback is not None:
            self.crop_editor_callback(proc, widget)

    def reset_crop_editor(self):
        if self.crop_editor is not None:
            self.crop_editor.set_to_local_crop_rect((0, 0, 0, 0))
            self.crop_editor.update_crop_size()

    def reset2_crop_editor(self, param):
        if self.crop_editor is not None:
            self.crop_editor.input_angle = self._get_param(param, 'rotation') + self._get_param(param, 'rotation2')
            self.crop_editor.set_aspect_ratio(self._param_to_aspect_ratio(param))

    # 自動クロップ
    def auto_crop_editor(self, img):
        import cores.find_bounding_box as find_bounding_box

        if img is not None:
            # クロップエディタのアスペクト比設定を取得
            aspect_ratio = None
            if self.crop_editor is not None:
                ar = self.crop_editor.aspect_ratio
                # aspect_ratioが0でない場合のみ使用
                if ar is not None and ar > 0:
                    aspect_ratio = ar
            
            bbox = find_bounding_box.find_bounding_box(
                img, 
                threshold=0.0001, 
                aspect_ratio=aspect_ratio,
                verbose=True
            )
            self.crop_editor.set_to_local_crop_rect(bbox)

    def finalize(self, param, widget):
        self._close_crop_editor(param, widget)


# AI ノイズ除去
class AINoiseReductonEffect(Effect):
    __net = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC

    def get_param_dict(self, param):
        return {
            'switch_ai_noise_reduction': True,
            'ai_noise_reduction': False,
            'ai_noise_reduction_intensity': 70,
            'ai_noise_reduction_result': None,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_ai_noise_reduction"].active = self._get_param(param, 'switch_ai_noise_reduction')
        widget.ids["chip_ai_noise_reduction"].active = self._get_param(param, 'ai_noise_reduction')
        widget.ids["slider_ai_noise_reduction_intensity"].value = self._get_param(param, 'ai_noise_reduction_intensity')

    def set2param(self, param, widget):
        param['switch_ai_noise_reduction'] = widget.ids["switch_ai_noise_reduction"].active
        param['ai_noise_reduction'] = widget.ids["chip_ai_noise_reduction"].active
        param['ai_noise_reduction_intensity'] = widget.ids["slider_ai_noise_reduction_intensity"].value

    def make_diff(self, img, param, efconfig):
        switch_ai_noise_reduction = self._get_param(param, 'switch_ai_noise_reduction')
        nr = self._get_param(param, 'ai_noise_reduction')
        nr_intensity = self._get_param(param, 'ai_noise_reduction_intensity') 
        nr_result = self._get_param(param, 'ai_noise_reduction_result')         
        if switch_ai_noise_reduction == False or nr == False:
            if efconfig.processor is not None:
                efconfig.processor.cancel_effect(self.__class__.__name__)

            self.diff = None
            self.hash = None
            # param['ai_noise_reduction_result'] = None
            # Also clear result from params if disabled? User might want to keep it. 
            # But usually disabled means no result.
            # param['ai_noise_reduction_result'] = None 
        else:
            # Hash only parameters. try_async_execution will mix upstream_hash.
            param_hash = hash(nr)
            
            # Additional hash for rendering (includes intensity)
            render_hash = hash((param_hash, efconfig.upstream_hash, nr_intensity))

            # Debug Log for Cache Verification
            print(f"DEBUG: AINoiseReducton make_diff. nr={nr}, upstream={efconfig.upstream_hash}, param_hash={param_hash}, render_hash={render_hash}, self.hash={self.hash}")

            # Optimization: Skip if already rendered for this state
            if self.hash == render_hash and self.diff is not None:
                # print("DEBUG: Short-circuit return self.diff")
                return self.diff

            # Async Processing Logic: Always try async first
            handled, result = self.try_async_execution(img, param, efconfig, param_hash)
            if handled:
                print(f"DEBUG: try_async handled. result ID={id(result) if result is not None else 'None'}. CombinedHash={hash((param_hash, efconfig.upstream_hash))}")
                # If we got a result (cached or newly computed), update nr_result
                if result is not None:
                    param['ai_noise_reduction_result'] = result
                    
                    # Blend with intensity
                    alpha = nr_intensity / 100.0
                    # Optimization: Use cv2 for faster blending
                    # self.diff = result * alpha + img * (1.0 - alpha)
                    if alpha <= 0.0:
                        self.diff = img
                    elif alpha >= 1.0:
                        self.diff = result
                    else:
                        self.diff = cv2.addWeighted(result, alpha, img, 1.0 - alpha, 0.0)
                    
                    # Store render hash as current state hash
                    self.hash = render_hash
                    return self.diff
                
                # If running/waiting (result is None), try to use preserved result as preview
                if nr_result is not None:
                    # Blend preview too
                    alpha = nr_intensity / 100.0
                    # self.diff = nr_result * alpha + img * (1.0 - alpha)
                    if alpha <= 0.0:
                        self.diff = img
                    elif alpha >= 1.0:
                        self.diff = nr_result
                    else:
                        # Ensure shapes match (upstream might have changed size?)
                        if nr_result.shape == img.shape:
                            self.diff = cv2.addWeighted(nr_result, alpha, img, 1.0 - alpha, 0.0)
                        else:
                            self.diff = None # Cannot blend mismatch
                    
                    # Use a distinct hash for preview (add 'preview')
                    self.hash = hash((render_hash, 'preview')) 
                    return self.diff
                    
                return None

            # Sync Fallback (Main Thread)
            needed, combined_hash = self.check_sync_necessity(param_hash, efconfig)
            
            if needed:
                import helpers.scunet_helper as scunet_helper
                if AINoiseReductonEffect.__net is None:
                    AINoiseReductonEffect.__net = scunet_helper.setup_scunet(device=config.get_config('gpu_device'), is_half=False)
                
                raw_diff = scunet_helper.predict_scunet_helper(AINoiseReductonEffect.__net, img)
                AINoiseReductonEffect.__net = None
                param['ai_noise_reduction_result'] = raw_diff
            else:
                raw_diff = param.get('ai_noise_reduction_result')
                
            if raw_diff is not None:
                 alpha = nr_intensity / 100.0
                 # self.diff = raw_diff * alpha + img * (1.0 - alpha)
                 if alpha <= 0.0:
                     self.diff = img
                 elif alpha >= 1.0:
                     self.diff = raw_diff
                 else:
                     self.diff = cv2.addWeighted(raw_diff, alpha, img, 1.0 - alpha, 0.0)
                     
                 self.hash = render_hash
            else:
                 self.diff = None
                 self.hash = None
        
        return self.diff


# BM3Dノイズ除去
class BM3DNoiseReductionEffect(Effect):

    def get_param_dict(self, param):
        return {
            'bm3d_noise_reduction': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_bm3d_noise_reduction"].set_slider_value(self._get_param(param, 'bm3d_noise_reduction'))

    def set2param(self, param, widget):
        param['bm3d_noise_reduction'] = widget.ids["slider_bm3d_noise_reduction"].value

    def make_diff(self, img, param, efconfig):
        bm3d = int(self._get_param(param, 'bm3d_noise_reduction'))
        if bm3d == 0 or efconfig.disp_info[4] < config.get_config('scale_threshold'):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((bm3d))
            if self.hash != param_hash:
                import bm3dcl
                self.diff = bm3dcl.bm3d_denoise(img, bm3d/100.0 * efconfig.disp_info[4])
                self.hash = param_hash

        return self.diff

class LightNoiseReductionEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_light_noise_reduction': True,
            'light_noise_reduction': 0,
            'light_color_noise_reduction': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_light_noise_reduction"].active = self._get_param(param, 'switch_light_noise_reduction')
        widget.ids["slider_light_noise_reduction"].set_slider_value(self._get_param(param, 'light_noise_reduction'))
        widget.ids["slider_light_color_noise_reduction"].set_slider_value(self._get_param(param, 'light_color_noise_reduction'))

    def set2param(self, param, widget):
        param['switch_light_noise_reduction'] = widget.ids["switch_light_noise_reduction"].active
        param['light_noise_reduction'] = widget.ids["slider_light_noise_reduction"].value
        param['light_color_noise_reduction'] = widget.ids["slider_light_color_noise_reduction"].value

    def make_diff(self, img, param, efconfig):
        switch_light_noise_reduction = self._get_param(param, 'switch_light_noise_reduction')
        its = int(self._get_param(param, 'light_noise_reduction'))
        col = int(self._get_param(param, 'light_color_noise_reduction'))
        if switch_light_noise_reduction == False or its == 0 and col == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((its, col))
            if self.hash != param_hash:  
                self.hash = param_hash

                self.diff = core.light_denoise(img, its * efconfig.disp_info[4], col * efconfig.disp_info[4])

        return self.diff

# デブラーフィルタ
class DeblurFilterEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_focus': True,
            'deblur_filter': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_focus"].active = self._get_param(param, 'switch_focus')
        widget.ids["slider_deblur_filter"].set_slider_value(self._get_param(param, 'deblur_filter'))

    def set2param(self, param, widget):
        param['switch_focus'] = widget.ids["switch_focus"].active
        param['deblur_filter'] = widget.ids["slider_deblur_filter"].value

    def make_diff(self, img, param, efconfig):
        switch_focus = self._get_param(param, 'switch_focus')
        dbfr = int(self._get_param(param, 'deblur_filter'))
        if switch_focus == False or dbfr == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((dbfr))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.lucy_richardson_gauss(img, dbfr)

        return self.diff


class DefocusEffect(Effect):
    __net = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_param_dict(self, param):
        return {
            'switch_focus': True,
            'defocus': False,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_focus"].active = self._get_param(param, 'switch_focus')
        widget.ids["switch_defocus"].active = self._get_param(param, 'defocus')

    def set2param(self, param, widget):
        param['switch_focus'] = widget.ids["switch_focus"].active
        param['defocus'] = widget.ids["switch_defocus"].active

    def make_diff(self, img, param, efconfig):
        switch_focus = self._get_param(param, 'switch_focus')
        df = self._get_param(param, 'defocus')
        if switch_focus == False or df == False:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((df))
            if self.hash != param_hash:
                self.hash = param_hash

                import DRBNet
                if DefocusEffect.__net is None:
                    DefocusEffect.__net = DRBNet.setup_predict()

                self.diff = DRBNet.predict(img, DefocusEffect.__net, config.get_config('gpu_device'))

        return self.diff


class LensblurFilterEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_filters': True,
            'lensblur_filter': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_filters"].active = self._get_param(param, 'switch_filters')
        widget.ids["slider_lensblur_filter"].set_slider_value(self._get_param(param, 'lensblur_filter'))

    def set2param(self, param, widget):
        param['switch_filters'] = widget.ids["switch_filters"].active
        param['lensblur_filter'] = widget.ids["slider_lensblur_filter"].value

    def make_diff(self, img, param, efconfig):
        switch_filters = self._get_param(param, 'switch_filters')
        lpfr = int(self._get_param(param, 'lensblur_filter'))
        if switch_filters == False or lpfr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((lpfr))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = filters.lensblur_filter(img, int(round(lpfr-1) * 4 * efconfig.resolution_scale))

        return self.diff

class ScratchEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_filters': True,
            'scratch': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_filters"].active = self._get_param(param, 'switch_filters')
        widget.ids["slider_scratch"].set_slider_value(self._get_param(param, 'scratch'))

    def set2param(self, param, widget):
        param['switch_filters'] = widget.ids["switch_filters"].active
        param['scratch'] = widget.ids["slider_scratch"].value

    def make_diff(self, img, param, efconfig):
        switch_filters = self._get_param(param, 'switch_filters')
        fr = int(self._get_param(param, 'scratch'))
        if switch_filters == False or fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = filters.scratch_effect(img, 1.0, fr / 100 * efconfig.resolution_scale)

        return self.diff

class FrostedGlassEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_filters': True,
            'frosted_glass': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_filters"].active = self._get_param(param, 'switch_filters')
        widget.ids["slider_frosted_glass"].set_slider_value(self._get_param(param, 'frosted_glass'))

    def set2param(self, param, widget):
        param['switch_filters'] = widget.ids["switch_filters"].active
        param['frosted_glass'] = widget.ids["slider_frosted_glass"].value

    def make_diff(self, img, param, efconfig):
        switch_filters = self._get_param(param, 'switch_filters')
        fr = int(self._get_param(param, 'frosted_glass'))
        if switch_filters == False or fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = filters.frosted_glass_effect(img, fr / 100 * efconfig.resolution_scale, fr / 1000 * efconfig.resolution_scale)

        return self.diff

class MosaicEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_filters': True,
            'mosaic': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_filters"].active = self._get_param(param, 'switch_filters')
        widget.ids["slider_mosaic"].set_slider_value(self._get_param(param, 'mosaic'))

    def set2param(self, param, widget):
        param['switch_filters'] = widget.ids["switch_filters"].active
        param['mosaic'] = widget.ids["slider_mosaic"].value

    def make_diff(self, img, param, efconfig):
        switch_filters = self._get_param(param, 'switch_filters')
        fr = int(self._get_param(param, 'mosaic'))
        if switch_filters == False or fr == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = filters.mosaic_effect(img, int(fr * efconfig.resolution_scale))

        return self.diff

class OrtonEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_orton_effect': True,
            'orton_radius': 30,
            'orton_opacity': 75,
            'orton_intensity': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_orton_effect"].active = self._get_param(param, 'switch_orton_effect')
        widget.ids["slider_orton_radius"].set_slider_value(self._get_param(param, 'orton_radius'))
        widget.ids["slider_orton_opacity"].set_slider_value(self._get_param(param, 'orton_opacity'))
        widget.ids["slider_orton_intensity"].set_slider_value(self._get_param(param, 'orton_intensity'))

    def set2param(self, param, widget):
        param['switch_orton_effect'] = widget.ids["switch_orton_effect"].active
        param['orton_radius'] = widget.ids["slider_orton_radius"].value
        param['orton_opacity'] = widget.ids["slider_orton_opacity"].value
        param['orton_intensity'] = widget.ids["slider_orton_intensity"].value

    def make_diff(self, img, param, efconfig):
        switch_orton_effect = self._get_param(param, 'switch_orton_effect')
        oradius = int(self._get_param(param, 'orton_radius'))
        oopacity = int(self._get_param(param, 'orton_opacity'))
        ointensity = int(self._get_param(param, 'orton_intensity'))
        if switch_orton_effect == False or ointensity == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((oradius, oopacity, ointensity))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = filters.orton_effect(img, oradius * efconfig.disp_info[4], oopacity / 100, ointensity / 100)

        return self.diff

class GlowEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_glow_effect': True,
            'glow_black': 0,
            'glow_gauss': 0,
            'glow_opacity': 0,
        }    

    def set2widget(self, widget, param):
        widget.ids["switch_glow_effect"].active = self._get_param(param, 'switch_glow_effect')
        widget.ids["slider_glow_black"].set_slider_value(self._get_param(param, 'glow_black'))
        widget.ids["slider_glow_gauss"].set_slider_value(self._get_param(param, 'glow_gauss'))
        widget.ids["slider_glow_opacity"].set_slider_value(self._get_param(param, 'glow_opacity'))

    def set2param(self, param, widget):
        param['switch_glow_effect'] = widget.ids["switch_glow_effect"].active
        param['glow_black'] = widget.ids["slider_glow_black"].value
        param['glow_gauss'] = widget.ids["slider_glow_gauss"].value
        param['glow_opacity'] = widget.ids["slider_glow_opacity"].value

    def make_diff(self, rgb, param, efconfig):
        switch_glow_effect = self._get_param(param, 'switch_glow_effect')
        gb = self._get_param(param, 'glow_black')
        gg = int(self._get_param(param, 'glow_gauss'))
        go = self._get_param(param, 'glow_opacity')
        if switch_glow_effect == False or (gb == 0 and gg == 0 and go == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((gb, gg, go))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                hls = hlsrgb.rgb_to_hlc_gain(rgb)
                hls[:,:,1] = core.apply_level_adjustment(hls[:,:,1], gb, 127+gg/2, 255)
                rgb2 = hlsrgb.hlc_gain_to_rgb(hls)
                if gg > 0:
                    radius = gg * 10 * efconfig.resolution_scale
                    rgb2 = filters.lensblur_filter(rgb2, 1 if radius <= 0 else radius) 
                go = go/100.0
                self.diff = cv2.addWeighted(rgb, 1.0-go, core.blend_screen(rgb, rgb2), go, 0)

        return self.diff

class FaceEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_face': True,
            'jawline_scale': 0,
            'jaw_scale': 0,
            'left_eye_scale': 0,
            'right_eye_scale': 0,
            'lips_scale': 0,
        }    

    def set2widget(self, widget, param):
        widget.ids["switch_face"].active = self._get_param(param, 'switch_face')
        widget.ids["slider_jawline_scale"].set_slider_value(self._get_param(param, 'jawline_scale'))
        widget.ids["slider_jaw_scale"].set_slider_value(self._get_param(param, 'jaw_scale'))
        widget.ids["slider_left_eye_scale"].set_slider_value(self._get_param(param, 'left_eye_scale'))
        widget.ids["slider_right_eye_scale"].set_slider_value(self._get_param(param, 'right_eye_scale'))
        widget.ids["slider_lips_scale"].set_slider_value(self._get_param(param, 'lips_scale'))

    def set2param(self, param, widget):
        param['switch_face'] = widget.ids["switch_face"].active
        param['jawline_scale'] = widget.ids["slider_jawline_scale"].value
        param['jaw_scale'] = widget.ids["slider_jaw_scale"].value
        param['left_eye_scale'] = widget.ids["slider_left_eye_scale"].value
        param['right_eye_scale'] = widget.ids["slider_right_eye_scale"].value
        param['lips_scale'] = widget.ids["slider_lips_scale"].value

    def make_diff(self, rgb, param, efconfig):
        switch_face = self._get_param(param, 'switch_face')
        jls = self._get_param(param, 'jawline_scale')
        js = self._get_param(param, 'jaw_scale')
        ls = self._get_param(param, 'left_eye_scale')
        rs = self._get_param(param, 'right_eye_scale')
        lipss = self._get_param(param, 'lips_scale')
        if switch_face == False or (ls == 0 and rs == 0 and jls == 0 and js == 0 and lipss == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((jls, js, ls, rs, lipss))
            if self.hash != param_hash:
                self.hash = param_hash

                import helpers.mediapipe_helper
                fms = helpers.mediapipe_helper.setup_face_mesh(rgb)
                rgb = helpers.mediapipe_helper.adjust_face_jawline(fms, rgb, jls/100, False) #efconfig.mode == EffectMode.PREVIEW)
                rgb = helpers.mediapipe_helper.adjust_face_jaw(fms, rgb, js/100, False)
                rgb = helpers.mediapipe_helper.adjust_left_eye(fms, rgb, ls/100, False)
                rgb = helpers.mediapipe_helper.adjust_right_eye(fms, rgb, rs/100, False)
                rgb = helpers.mediapipe_helper.adjust_lips(fms, rgb, lipss/100, False)
                helpers.mediapipe_helper.clear_face_mesh(fms)
                self.diff = rgb

        return self.diff

class ColorTemperatureEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_white_balance': True,
            'color_temperature_reset': 5000,
            'color_temperature': param.get('color_temperature_reset', 5000),
            'color_tint_reset': 0,
            'color_tint': param.get('color_tint_reset', 0),
            'color_Y': 1.0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_white_balance'].enabled = self._get_param(param, 'switch_white_balance')
        widget.ids["slider_color_temperature"].set_slider_value(self._get_param(param, 'color_temperature'))
        widget.ids["slider_color_tint"].set_slider_value(self._get_param(param, 'color_tint'))
        widget.ids["slider_color_temperature"].set_slider_reset(self._get_param(param, 'color_temperature_reset'))
        widget.ids["slider_color_tint"].set_slider_reset(self._get_param(param, 'color_tint_reset'))
 
    def set2param(self, param, widget):
        param['switch_white_balance'] = widget.ids['switch_white_balance'].enabled
        param['color_temperature'] = widget.ids["slider_color_temperature"].value
        param['color_tint'] = widget.ids["slider_color_tint"].value

    @staticmethod
    def apply_color_temperature(rgb, param):
        temp = param.get('color_temperature', param.get('color_temperature_reset', 5000))
        tint = param.get('color_tint', param.get('color_tint_reset', 0))
        Y = param.get('color_Y', 1.0)
        return rgb * core.invert_TempTint2RGB(temp, tint, Y, 5000)

    def make_diff(self, rgb, param, efconfig):
        switch_white_balance = self._get_param(param, 'switch_white_balance')
        temp = self._get_param(param, 'color_temperature')
        tint = self._get_param(param, 'color_tint')
        Y = self._get_param(param, 'color_Y')
        if switch_white_balance == False:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((temp, tint))
            if self.hash != param_hash:
                trgb = core.convert_TempTint2RGB(param['color_temperature_reset'], param['color_tint_reset'], self._get_param(param, 'color_Y'))
                self.diff = rgb * (trgb / core.convert_TempTint2RGB(temp, tint, Y))
                self.hash = param_hash

        return self.diff

class DehazeEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_precence': True,
            'dehaze': 0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_precence'].enabled = self._get_param(param, 'switch_precence')
        widget.ids["slider_dehaze"].set_slider_value(self._get_param(param, 'dehaze'))

    def set2param(self, param, widget):
        param['switch_precence'] = widget.ids['switch_precence'].enabled
        param['dehaze'] = widget.ids["slider_dehaze"].value

    def make_diff(self, rgb, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        de = self._get_param(param, 'dehaze')
        if switch_precence == False or de == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((de))
            if self.hash != param_hash:
                self.hash = param_hash
                
                self.diff = core.dehaze_image(rgb, de/100)

        return self.diff

class RGB2HLSEffect(Effect):

    def make_diff(self, rgb, param, efconfig):
        if self.diff is None:
            rgb = core.type_convert(rgb, np.ndarray)
            self.diff = hlsrgb.rgb_to_hlc_gain(rgb)
        return self.diff

class HLS2RGBEffect(Effect):

    def make_diff(self, hls, param, efconfig):
        if self.diff is None:
            hls = core.type_convert(hls, np.ndarray)
            self.diff = hlsrgb.hlc_gain_to_rgb(hls)
        return self.diff

    
class HLSEffect(Effect):

    def get_param_dict(self, param):
        return {
            "switch_color_mixer": True,
            "switch_hls_red": True,
            "switch_hls_orange": True,
            "switch_hls_yellow": True,
            "switch_hls_green": True,
            "switch_hls_cyan": True,
            "switch_hls_blue": True,
            "switch_hls_purple": True,
            "switch_hls_magenta": True,
            "hls_red_hue": 0,
            "hls_red_lum": 0,
            "hls_red_sat": 0,
            "hls_orange_hue": 0,
            "hls_orange_lum": 0,
            "hls_orange_sat": 0,
            "hls_yellow_hue": 0,
            "hls_yellow_lum": 0,
            "hls_yellow_sat": 0,
            "hls_green_hue": 0,
            "hls_green_lum": 0,
            "hls_green_sat": 0,
            "hls_cyan_hue": 0,
            "hls_cyan_lum": 0,
            "hls_cyan_sat": 0,
            "hls_blue_hue": 0,
            "hls_blue_lum": 0,
            "hls_blue_sat": 0,
            "hls_purple_hue": 0,
            "hls_purple_lum": 0,
            "hls_purple_sat": 0,
            "hls_magenta_hue": 0,
            "hls_magenta_lum": 0,
            "hls_magenta_sat": 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_color_mixer"].active = self._get_param(param, "switch_color_mixer")
        widget.ids["switch_hls_red"].active = self._get_param(param, "switch_hls_red")
        widget.ids["switch_hls_orange"].active = self._get_param(param, "switch_hls_orange")
        widget.ids["switch_hls_yellow"].active = self._get_param(param, "switch_hls_yellow")
        widget.ids["switch_hls_green"].active = self._get_param(param, "switch_hls_green")
        widget.ids["switch_hls_cyan"].active = self._get_param(param, "switch_hls_cyan")
        widget.ids["switch_hls_blue"].active = self._get_param(param, "switch_hls_blue")
        widget.ids["switch_hls_purple"].active = self._get_param(param, "switch_hls_purple")
        widget.ids["switch_hls_magenta"].active = self._get_param(param, "switch_hls_magenta")
        widget.ids["slider_hls_red_hue"].set_slider_value(self._get_param(param, "hls_red_hue"))
        widget.ids["slider_hls_red_lum"].set_slider_value(self._get_param(param, "hls_red_lum"))
        widget.ids["slider_hls_red_sat"].set_slider_value(self._get_param(param, "hls_red_sat"))
        widget.ids["slider_hls_orange_hue"].set_slider_value(self._get_param(param, "hls_orange_hue"))
        widget.ids["slider_hls_orange_lum"].set_slider_value(self._get_param(param, "hls_orange_lum"))
        widget.ids["slider_hls_orange_sat"].set_slider_value(self._get_param(param, "hls_orange_sat"))
        widget.ids["slider_hls_yellow_hue"].set_slider_value(self._get_param(param, "hls_yellow_hue"))
        widget.ids["slider_hls_yellow_lum"].set_slider_value(self._get_param(param, "hls_yellow_lum"))
        widget.ids["slider_hls_yellow_sat"].set_slider_value(self._get_param(param, "hls_yellow_sat"))
        widget.ids["slider_hls_green_hue"].set_slider_value(self._get_param(param, "hls_green_hue"))
        widget.ids["slider_hls_green_lum"].set_slider_value(self._get_param(param, "hls_green_lum"))
        widget.ids["slider_hls_green_sat"].set_slider_value(self._get_param(param, "hls_green_sat"))
        widget.ids["slider_hls_cyan_hue"].set_slider_value(self._get_param(param, "hls_cyan_hue"))
        widget.ids["slider_hls_cyan_lum"].set_slider_value(self._get_param(param, "hls_cyan_lum"))
        widget.ids["slider_hls_cyan_sat"].set_slider_value(self._get_param(param, "hls_cyan_sat"))
        widget.ids["slider_hls_blue_hue"].set_slider_value(self._get_param(param, "hls_blue_hue"))
        widget.ids["slider_hls_blue_lum"].set_slider_value(self._get_param(param, "hls_blue_lum"))
        widget.ids["slider_hls_blue_sat"].set_slider_value(self._get_param(param, "hls_blue_sat"))
        widget.ids["slider_hls_purple_hue"].set_slider_value(self._get_param(param, "hls_purple_hue"))
        widget.ids["slider_hls_purple_lum"].set_slider_value(self._get_param(param, "hls_purple_lum"))
        widget.ids["slider_hls_purple_sat"].set_slider_value(self._get_param(param, "hls_purple_sat"))
        widget.ids["slider_hls_magenta_hue"].set_slider_value(self._get_param(param, "hls_magenta_hue"))
        widget.ids["slider_hls_magenta_lum"].set_slider_value(self._get_param(param, "hls_magenta_lum"))
        widget.ids["slider_hls_magenta_sat"].set_slider_value(self._get_param(param, "hls_magenta_sat"))

    def set2param(self, param, widget):
        param["switch_color_mixer"] = widget.ids["switch_color_mixer"].active
        param["switch_hls_red"] = widget.ids["switch_hls_red"].active
        param["switch_hls_orange"] = widget.ids["switch_hls_orange"].active
        param["switch_hls_yellow"] = widget.ids["switch_hls_yellow"].active
        param["switch_hls_green"] = widget.ids["switch_hls_green"].active
        param["switch_hls_cyan"] = widget.ids["switch_hls_cyan"].active
        param["switch_hls_blue"] = widget.ids["switch_hls_blue"].active
        param["switch_hls_purple"] = widget.ids["switch_hls_purple"].active
        param["switch_hls_magenta"] = widget.ids["switch_hls_magenta"].active
        param["hls_red_hue"] = widget.ids["slider_hls_red_hue"].value
        param["hls_red_lum"] = widget.ids["slider_hls_red_lum"].value
        param["hls_red_sat"] = widget.ids["slider_hls_red_sat"].value
        param["hls_orange_hue"] = widget.ids["slider_hls_orange_hue"].value
        param["hls_orange_lum"] = widget.ids["slider_hls_orange_lum"].value
        param["hls_orange_sat"] = widget.ids["slider_hls_orange_sat"].value
        param["hls_yellow_hue"] = widget.ids["slider_hls_yellow_hue"].value
        param["hls_yellow_lum"] = widget.ids["slider_hls_yellow_lum"].value
        param["hls_yellow_sat"] = widget.ids["slider_hls_yellow_sat"].value
        param["hls_green_hue"] = widget.ids["slider_hls_green_hue"].value
        param["hls_green_lum"] = widget.ids["slider_hls_green_lum"].value
        param["hls_green_sat"] = widget.ids["slider_hls_green_sat"].value
        param["hls_cyan_hue"] = widget.ids["slider_hls_cyan_hue"].value
        param["hls_cyan_lum"] = widget.ids["slider_hls_cyan_lum"].value
        param["hls_cyan_sat"] = widget.ids["slider_hls_cyan_sat"].value
        param["hls_blue_hue"] = widget.ids["slider_hls_blue_hue"].value
        param["hls_blue_lum"] = widget.ids["slider_hls_blue_lum"].value
        param["hls_blue_sat"] = widget.ids["slider_hls_blue_sat"].value
        param["hls_purple_hue"] = widget.ids["slider_hls_purple_hue"].value
        param["hls_purple_lum"] = widget.ids["slider_hls_purple_lum"].value
        param["hls_purple_sat"] = widget.ids["slider_hls_purple_sat"].value
        param["hls_magenta_hue"] = widget.ids["slider_hls_magenta_hue"].value
        param["hls_magenta_lum"] = widget.ids["slider_hls_magenta_lum"].value
        param["hls_magenta_sat"] = widget.ids["slider_hls_magenta_sat"].value

    def make_diff(self, hls, param, efconfig):
        switch_color_mixer = self._get_param(param, "switch_color_mixer")
        switch_hls_red = self._get_param(param, "switch_hls_red")
        switch_hls_orange = self._get_param(param, "switch_hls_orange")
        switch_hls_yellow = self._get_param(param, "switch_hls_yellow")
        switch_hls_green = self._get_param(param, "switch_hls_green")
        switch_hls_cyan = self._get_param(param, "switch_hls_cyan")
        switch_hls_blue = self._get_param(param, "switch_hls_blue")
        switch_hls_purple = self._get_param(param, "switch_hls_purple")
        switch_hls_magenta = self._get_param(param, "switch_hls_magenta")
        red_hue = self._get_param(param, "hls_red_hue")
        red_lum = self._get_param(param, "hls_red_lum")
        red_sat = self._get_param(param, "hls_red_sat")
        orange_hue = self._get_param(param, "hls_orange_hue")
        orange_lum = self._get_param(param, "hls_orange_lum")
        orange_sat = self._get_param(param, "hls_orange_sat")
        yellow_hue = self._get_param(param, "hls_yellow_hue")
        yellow_lum = self._get_param(param, "hls_yellow_lum")
        yellow_sat = self._get_param(param, "hls_yellow_sat")
        green_hue = self._get_param(param, "hls_green_hue")
        green_lum = self._get_param(param, "hls_green_lum")
        green_sat = self._get_param(param, "hls_green_sat")
        cyan_hue = self._get_param(param, "hls_cyan_hue")
        cyan_lum = self._get_param(param, "hls_cyan_lum")
        cyan_sat = self._get_param(param, "hls_cyan_sat")
        blue_hue = self._get_param(param, "hls_blue_hue")
        blue_lum = self._get_param(param, "hls_blue_lum")
        blue_sat = self._get_param(param, "hls_blue_sat")
        purple_hue = self._get_param(param, "hls_purple_hue")
        purple_lum = self._get_param(param, "hls_purple_lum")
        purple_sat = self._get_param(param, "hls_purple_sat")
        magenta_hue = self._get_param(param, "hls_magenta_hue")
        magenta_lum = self._get_param(param, "hls_magenta_lum")
        magenta_sat = self._get_param(param, "hls_magenta_sat")
        if (   switch_color_mixer == False
            or (    (switch_hls_red == False or (red_hue == 0 and red_lum == 0 and red_sat == 0))
                and (switch_hls_orange == False or (orange_hue == 0 and orange_lum == 0 and orange_sat == 0))
                and (switch_hls_yellow == False or (yellow_hue == 0 and yellow_lum == 0 and yellow_sat == 0))
                and (switch_hls_green == False or (green_hue == 0 and green_lum == 0 and green_sat == 0))
                and (switch_hls_cyan == False or (cyan_hue == 0 and cyan_lum == 0 and cyan_sat == 0))
                and (switch_hls_blue == False or (blue_hue == 0 and blue_lum == 0 and blue_sat == 0))
                and (switch_hls_purple == False or (purple_hue == 0 and purple_lum == 0 and purple_sat == 0))
                and (switch_hls_magenta == False or (magenta_hue == 0 and magenta_lum == 0 and magenta_sat == 0))
            )):
            self.diff = None
            self.hash = None        
        else:
            param_hash = hash((
                switch_hls_red, red_hue, red_lum, red_sat,
                switch_hls_orange, orange_hue, orange_lum, orange_sat,
                switch_hls_yellow, yellow_hue, yellow_lum, yellow_sat,
                switch_hls_green, green_hue, green_lum, green_sat,
                switch_hls_cyan, cyan_hue, cyan_lum, cyan_sat,
                switch_hls_blue, blue_hue, blue_lum, blue_sat,
                switch_hls_purple, purple_hue, purple_lum, purple_sat,
                switch_hls_magenta, magenta_hue, magenta_lum, magenta_sat
            ))
            if self.hash != param_hash:
                self.hash = param_hash

                # 全色の設定を作成
                color_settings = []
                colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']
                
                # ローカル変数から値を取得するためのマップ
                # (変数名が動的ではないので、明示的にマッピングするか、条件分岐で記述する)
                # ここではスマートに実装するため、変数名をマップするのではなく、
                # 上記で取得済みの変数をリスト化して処理する。
                
                # パラメータ取得済み変数のリスト化
                params_map = {
                    'red': (switch_hls_red, red_hue, red_lum, red_sat),
                    'orange': (switch_hls_orange, orange_hue, orange_lum, orange_sat),
                    'yellow': (switch_hls_yellow, yellow_hue, yellow_lum, yellow_sat),
                    'green': (switch_hls_green, green_hue, green_lum, green_sat),
                    'cyan': (switch_hls_cyan, cyan_hue, cyan_lum, cyan_sat),
                    'blue': (switch_hls_blue, blue_hue, blue_lum, blue_sat),
                    'purple': (switch_hls_purple, purple_hue, purple_lum, purple_sat),
                    'magenta': (switch_hls_magenta, magenta_hue, magenta_lum, magenta_sat),
                }

                for color_name in colors:
                    switch, h, l, s = params_map[color_name]
                    
                    if not switch:
                        continue
                    
                    if h == 0 and l == 0 and s == 0:
                        continue
                        
                    # 設定のコピーと調整値の反映
                    # core.HLS_COLOR_SETTING はグローバル定数なのでコピーして使う
                    if color_name in core.HLS_COLOR_SETTING:
                        setting = core.HLS_COLOR_SETTING[color_name].copy()
                        # core.adjust_hls_color_oneの実装に合わせる (Lum/Satは%指定なので /100)
                        setting['adjust'] = [h, l/100.0, s/100.0]
                        color_settings.append(setting)

                if not color_settings:
                     self.diff = None
                else:
                    self.diff = core.adjust_hls_colors(hls, color_settings, efconfig.resolution_scale)

        return self.diff

class ExposureEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_exposure_contrast': True,
            'exposure': 0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_exposure_contrast'].enabled = self._get_param(param, 'switch_exposure_contrast')
        widget.ids["slider_exposure"].set_slider_value(self._get_param(param, 'exposure'))

    def set2param(self, param, widget):
        param['switch_exposure_contrast'] = widget.ids['switch_exposure_contrast'].enabled
        param['exposure'] = widget.ids["slider_exposure"].value

    def make_diff(self, rgb, param, efconfig):
        switch_exposure_contrast = self._get_param(param, 'switch_exposure_contrast')
        ev = self._get_param(param, 'exposure')
        if switch_exposure_contrast == False or ev == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ev))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.adjust_exposure(rgb, ev)

        return self.diff
    
class ContrastEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_exposure_contrast': True,
            'contrast': 0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_exposure_contrast'].enabled = self._get_param(param, 'switch_exposure_contrast')
        widget.ids["slider_contrast"].set_slider_value(self._get_param(param, 'contrast'))

    def set2param(self, param, widget):
        param['switch_exposure_contrast'] = widget.ids['switch_exposure_contrast'].enabled
        param['contrast'] = widget.ids["slider_contrast"].value

    def make_diff(self, rgb, param, efconfig):
        switch_exposure_contrast = self._get_param(param, 'switch_exposure_contrast')
        con = self._get_param(param, 'contrast')
        if switch_exposure_contrast == False or con == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((con))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.adjust_tone(rgb, con, -con, disp_scale=efconfig.disp_info[4], resolution_scale=efconfig.resolution_scale)

        return self.diff

class ClarityEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_precence': True,
            'clarity': 0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_precence'].enabled = self._get_param(param, 'switch_precence')
        widget.ids["slider_clarity"].set_slider_value(self._get_param(param, 'clarity'))

    def set2param(self, param, widget):
        param['switch_precence'] = widget.ids['switch_precence'].enabled
        param['clarity'] = widget.ids["slider_clarity"].value

    def make_diff(self, rgb, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        con = self._get_param(param, 'clarity')
        if switch_precence == False or con == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((con))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = local_contrast.apply_clarity_luminance(rgb, (con * 2 * efconfig.resolution_scale) / 100)

        return self.diff

class TextureEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_precence': True,
            'texture': 0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_precence'].enabled = self._get_param(param, 'switch_precence')
        widget.ids["slider_texture"].set_slider_value(self._get_param(param, 'texture'))

    def set2param(self, param, widget):
        param['switch_precence'] = widget.ids['switch_precence'].enabled
        param['texture'] = widget.ids["slider_texture"].value

    def make_diff(self, rgb, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        con = self._get_param(param, 'texture')
        if switch_precence == False or con == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((con))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = local_contrast.apply_texture_advanced(rgb, (con * 0.5 * efconfig.resolution_scale) / 100)

        return self.diff
    
class MicroContrastEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_precence': True,
            'microcontrast': 0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_precence'].enabled = self._get_param(param, 'switch_precence')
        widget.ids["slider_microcontrast"].set_slider_value(self._get_param(param, 'microcontrast'))

    def set2param(self, param, widget):
        param['switch_precence'] = widget.ids['switch_precence'].enabled
        param['microcontrast'] = widget.ids["slider_microcontrast"].value

    def make_diff(self, rgb, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        con = self._get_param(param, 'microcontrast')
        if switch_precence == False or con == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((con))
            if self.hash != param_hash:
                self.hash = param_hash
                
                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = local_contrast.apply_microcontrast(rgb, (con * 0.5 * efconfig.resolution_scale) / 100)

        return self.diff
    
class ToneEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_tone': True,
            'shadow': 0,
            'highlight': 0,
            'midtone': 0,
            'white': 0,
            'black': 0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_tone'].active = self._get_param(param, 'switch_tone')
        widget.ids["slider_shadow"].set_slider_value(self._get_param(param, 'shadow'))
        widget.ids["slider_highlight"].set_slider_value(self._get_param(param, 'highlight'))
        widget.ids["slider_midtone"].set_slider_value(self._get_param(param, 'midtone'))
        widget.ids["slider_white"].set_slider_value(self._get_param(param, 'white'))
        widget.ids["slider_black"].set_slider_value(self._get_param(param, 'black'))
        

    def set2param(self, param, widget):
        param['switch_tone'] = widget.ids['switch_tone'].active
        param['shadow'] = widget.ids["slider_shadow"].value
        param['highlight'] = widget.ids["slider_highlight"].value
        param['midtone'] = widget.ids["slider_midtone"].value
        param['white'] = widget.ids["slider_white"].value
        param['black'] = widget.ids["slider_black"].value

    def make_diff(self, rgb, param, efconfig):
        switch_tone = self._get_param(param, 'switch_tone')
        shadow = self._get_param(param, 'shadow')
        highlight = self._get_param(param, 'highlight')
        mt = self._get_param(param, 'midtone')
        white = self._get_param(param, 'white')
        black = self._get_param(param, 'black')
        if switch_tone == False or (shadow == 0 and highlight == 0 and mt == 0 and white == 0 and black == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((shadow, highlight, mt, white, black))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.adjust_tone(rgb, highlight, shadow, mt, white, black, disp_scale=efconfig.disp_info[4], resolution_scale=efconfig.resolution_scale)

        return self.diff
    
class HighlightCompressEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_global': True,
            'highlight_compress': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_global"].active = self._get_param(param, 'switch_global')
        widget.ids["switch_highlight_compress"].active = self._get_param(param, 'highlight_compress')

    def set2param(self, param, widget):
        param['switch_global'] = widget.ids["switch_global"].active
        param['highlight_compress'] = widget.ids["switch_highlight_compress"].active

    def make_diff(self, rgb, param, efconfig):
        switch_global = self._get_param(param, 'switch_global')
        hc = self._get_param(param, 'highlight_compress')
        if switch_global == False or hc == False:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((hc))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.highlight_compress(rgb)

        return self.diff

class LevelEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_level': True,
            'black_level': 0,
            'mid_level': 127,
            'white_level': 255,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_level'].active = self._get_param(param, 'switch_level')
        widget.ids["slider_black_level"].set_slider_value(self._get_param(param, 'black_level'))
        widget.ids["slider_mid_level"].set_slider_value(self._get_param(param, 'mid_level'))
        widget.ids["slider_white_level"].set_slider_value(self._get_param(param, 'white_level'))

    def set2param(self, param, widget):
        param['switch_level'] = widget.ids['switch_level'].active
        param['black_level'] = widget.ids["slider_black_level"].value
        param['mid_level'] = widget.ids["slider_mid_level"].value
        param['white_level'] = widget.ids["slider_white_level"].value

    def make_diff(self, rgb, param, efconfig):
        switch_level = self._get_param(param, 'switch_level')
        bl = self._get_param(param, 'black_level')
        ml = self._get_param(param, 'mid_level')
        wl = self._get_param(param, 'white_level')
        if switch_level == False or (bl == 0 and wl == 255 and ml == 127):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((bl, ml, wl))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.apply_level_adjustment(rgb, bl, ml, wl)

        return self.diff
    
class CLAHEEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_precence': True,
            'clahe': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_precence"].active = self._get_param(param, 'switch_precence')
        widget.ids["slider_clahe"].set_slider_value(self._get_param(param, 'clahe'))

    def set2param(self, param, widget):
        param['switch_precence'] = widget.ids["switch_precence"].active
        param['clahe'] = widget.ids["slider_clahe"].value

    def make_diff(self, img, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        ci = self._get_param(param, 'clahe')
        if switch_precence == False or ci == 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((ci))
            if self.hash != param_hash:
                self.hash = param_hash

                img = core.type_convert(img, np.ndarray)
                img_min, img_max = img.min(), img.max()
                img = (img - img_min) / (img_max - img_min)
                r, g, b = cv2.split(img)
                target = np.empty_like(img)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                for i, n in enumerate([r, g, b]):
                    n = (n * 65535).astype(np.uint16)
                    n = clahe.apply(n)
                    n = n.astype(np.float32) / 65535
                    target[..., i] = n
                target = target * (img_max - img_min) + img_min
                ci = ci / 100
                self.diff = cv2.addWeighted(target, ci, img, 1.0 - ci, 0)

        return self.diff
    
class CurvesEffect(Effect):

    def get_param_dict(self, param, subname=None):
        if subname is None:
            return {
                'switch_tone_curves': True,
                'switch_color_gradings': True,
            }
        return self.effects[subname].get_param_dict(param)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['tonecurve'] = TonecurveEffect()
        effecs['tonecurve_red'] = TonecurveRedEffect()
        effecs['tonecurve_green'] = TonecurveGreenEffect()
        effecs['tonecurve_blue'] = TonecurveBlueEffect()
        effecs['grading1'] = GradingEffect("1")
        effecs['grading2'] = GradingEffect("2")
        self.effects = effecs

    def delete_default_param(self, param):
        super().delete_default_param(param)
        for n in self.effects.values():
            n.delete_default_param(param)

    def reeffect(self):
        for n in self.effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        widget.ids["switch_tone_curves"].active = self._get_param(param, 'switch_tone_curves')
        widget.ids["switch_color_gradings"].active = self._get_param(param, 'switch_color_gradings')
        for n in self.effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        param['switch_tone_curves'] = widget.ids["switch_tone_curves"].active
        param['switch_color_gradings'] = widget.ids["switch_color_gradings"].active
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, rgb, param, efconfig):
        self.diff = pipeline.pipeline_curve(rgb, self.effects, param, efconfig)

        return self.diff
    
class TonecurveEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_tone_curves': True,
            'tonecurve': None,
        }

    def set2widget(self, widget, param):
        widget.ids["tonecurve"].set_point_list(self._get_param(param, 'tonecurve'))

    def set2param(self, param, widget):
        param['tonecurve'] = widget.ids["tonecurve"].get_point_list()

    def make_diff(self, rgb, param, efconfig):
        switch_tone_curves = self._get_param(param, 'switch_tone_curves')
        pl = self._get_param(param, 'tonecurve')
        if switch_tone_curves == False or pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.calc_point_list_to_lut(pl)

        return self.diff
    
    def apply_diff(self, rgb):
        rgb =  core.type_convert(rgb, np.ndarray)
        return core.apply_lut(rgb, self.diff)

class TonecurveRedEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_tone_curves': True,
            'tonecurve_red': None,
        }

    def set2widget(self, widget, param):
        widget.ids["tonecurve_red"].set_point_list(self._get_param(param, 'tonecurve_red'))

    def set2param(self, param, widget):
        param['tonecurve_red'] = widget.ids["tonecurve_red"].get_point_list()

    def make_diff(self, rgb_r, param, efconfig):
        switch_tone_curves = self._get_param(param, 'switch_tone_curves')
        pl = self._get_param(param, 'tonecurve_red')
        if switch_tone_curves == False or pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.calc_point_list_to_lut(pl)

        return self.diff

    def apply_diff(self, rgb_r):
        rgb_r =  core.type_convert(rgb_r, np.ndarray)
        return core.apply_lut(rgb_r, self.diff)

class TonecurveGreenEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_tone_curves': True,
            'tonecurve_green': None,
        }

    def set2widget(self, widget, param):
        widget.ids["tonecurve_green"].set_point_list(self._get_param(param, 'tonecurve_green'))

    def set2param(self, param, widget):
        param['tonecurve_green'] = widget.ids["tonecurve_green"].get_point_list()

    def make_diff(self, rgb_g, param, efconfig):   
        switch_tone_curves = self._get_param(param, 'switch_tone_curves')
        pl = self._get_param(param, 'tonecurve_green')
        if switch_tone_curves == False or pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.calc_point_list_to_lut(pl)

        return self.diff

    def apply_diff(self, rgb_g):
        rgb_g =  core.type_convert(rgb_g, np.ndarray)
        return core.apply_lut(rgb_g, self.diff)

class TonecurveBlueEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_tone_curves': True,
            'tonecurve_blue': None,
        }

    def set2widget(self, widget, param):
        widget.ids["tonecurve_blue"].set_point_list(self._get_param(param, 'tonecurve_blue'))

    def set2param(self, param, widget):
        param['tonecurve_blue'] = widget.ids["tonecurve_blue"].get_point_list()

    def make_diff(self, rgb_b, param, efconfig):
        switch_tone_curves = self._get_param(param, 'switch_tone_curves')
        pl = self._get_param(param, 'tonecurve_blue')
        if switch_tone_curves == False or pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.calc_point_list_to_lut(pl)

        return self.diff

    def apply_diff(self, rgb_b):
        rgb_b =  core.type_convert(rgb_b, np.ndarray)
        return core.apply_lut(rgb_b, self.diff)

class GradingEffect(Effect):

    def __init__(self, numstr, **kwargs):
        super().__init__(**kwargs)

        self.numstr = numstr

    def get_param_dict(self, param):
        return {
            'switch_color_gradings': True,
            'grading' + self.numstr: None,
            'grading' + self.numstr + '_hue': 0,
            'grading' + self.numstr + '_lum': 50,
            'grading' + self.numstr + '_sat': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["grading" + self.numstr].set_point_list(self._get_param(param, 'grading' + self.numstr))
        widget.ids["grading" + self.numstr + "_color_picker"].set_slider_value(
            [self._get_param(param, 'grading' + self.numstr + '_hue'),
             self._get_param(param, 'grading' + self.numstr + '_lum'),
             self._get_param(param, 'grading' + self.numstr + '_sat')]
        )

    def set2param(self, param, widget):
        param["grading" + self.numstr] = widget.ids["grading" + self.numstr].get_point_list(True)
        param["grading" + self.numstr + "_hue"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_hue'].value
        param["grading" + self.numstr + "_lum"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_lum'].value
        param["grading" + self.numstr + "_sat"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_sat'].value

    def make_diff(self, rgb, param, efconfig):
        switch_color_gradings = self._get_param(param, "switch_color_gradings")
        pl = self._get_param(param, "grading" + self.numstr)
        gh = self._get_param(param, "grading" + self.numstr + "_hue")
        gl = self._get_param(param, "grading" + self.numstr + "_lum")
        gs = self._get_param(param, "grading" + self.numstr + "_sat")
        if switch_color_gradings == False or (gh == 0 and gl == 50 and gs == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((np.sum(pl), gh, gl, gs))
            if self.hash != param_hash:
                self.hash = param_hash

                import colorsys
                lut = core.calc_point_list_to_lut(pl)
                rgbs = np.array(colorsys.hls_to_rgb(gh/360.0, gl/100.0, gs/100.0), dtype=np.float32)
                self.diff = (lut, rgbs)

        return self.diff
    
    def apply_diff(self, rgb):
        lut, rgbs = self.diff
        rgb = core.type_convert(rgb, np.ndarray)
        gray = core.cvtColorRGB2Gray(rgb)
        blend = core.apply_lut(gray, lut)
        blend = np.array(blend)
        return core.apply_mask(rgb, blend, rgb * rgbs)

class VSandSaturationEffect(Effect):

    def get_param_dict(self, param, subname=None):
        if subname is None:
            return {
                'switch_color_curves': True,
                'switch_saturation': True,
            }
        return self.effects[subname].get_param_dict(param)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['HuevsHue'] = HuevsHueEffect()
        effecs['HuevsLum'] = HuevsLumEffect()
        effecs['LumvsLum'] = LumvsLumEffect()
        effecs['SatvsLum'] = SatvsLumEffect()
        effecs['HuevsSat'] = HuevsSatEffect()
        effecs['LumvsSat'] = LumvsSatEffect()
        effecs['SatvsSat'] = SatvsSatEffect()
        effecs['saturation'] = SaturationEffect()
        self.effects = effecs

    def delete_default_param(self, param):
        super().delete_default_param(param)
        for n in self.effects.values():
            n.delete_default_param(param)

    def reeffect(self):
        for n in self.effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        widget.ids["switch_color_curves"].active = self._get_param(param, 'switch_color_curves')
        widget.ids["switch_saturation"].active = self._get_param(param, 'switch_saturation')
        for n in self.effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        param['switch_color_curves'] = widget.ids["switch_color_curves"].active
        param['switch_saturation'] = widget.ids["switch_saturation"].active
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, hls, param, efconfig):
        self.diff = pipeline.pipeline_vs_and_saturation(hls, self.effects, param, efconfig)

        return self.diff
    
class HuevsHueEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_color_curves': True,
            'HuevsHue': None,
        }

    def set2widget(self, widget, param):
        widget.ids["HuevsHue"].set_point_list(self._get_param(param, 'HuevsHue'))

    def set2param(self, param, widget):
        param['HuevsHue'] = widget.ids["HuevsHue"].get_point_list()

    def make_diff(self, hls_hh, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        hh = self._get_param(param, "HuevsHue")
        if switch_color_curves == False or hh is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hh))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(hh)
                lut = ((lut - 0.5) * 2.0) * 360
                self.diff = core.apply_lut(hls_hh[0] / 360, lut, 1.0) + hls_hh[1]

        return self.diff

class HuevsLumEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_color_curves': True,
            'HuevsLum': None,
        }

    def set2widget(self, widget, param):
        widget.ids["HuevsLum"].set_point_list(self._get_param(param, 'HuevsLum'))

    def set2param(self, param, widget):
        param['HuevsLum'] = widget.ids["HuevsLum"].get_point_list()

    def make_diff(self, hls_hl, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        hl = self._get_param(param, "HuevsLum")
        if switch_color_curves == False or hl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hl))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(hl)
                lut = 2.0 ** ((lut - 0.5) * 4.0)
                self.diff = core.apply_lut(hls_hl[0] / 360, lut, 1.0) * hls_hl[1]

        return self.diff

class HuevsSatEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_color_curves': True,
            'HuevsSat': None,
        }

    def set2widget(self, widget, param):
        widget.ids["HuevsSat"].set_point_list(self._get_param(param, 'HuevsSat'))

    def set2param(self, param, widget):
        param['HuevsSat'] = widget.ids["HuevsSat"].get_point_list()

    def make_diff(self, hls_hs, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        hs = self._get_param(param, "HuevsSat")
        if switch_color_curves == False or hs is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hs))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(hs)
                lut = (lut - 0.5) * 2.0 + 1.0
                self.diff = core.apply_lut(hls_hs[0] / 360.0, lut, 1.0) * hls_hs[1]

        return self.diff

class LumvsLumEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_color_curves': True,
            'LumvsLum': None,
        }

    def set2widget(self, widget, param):
        widget.ids["LumvsLum"].set_point_list(self._get_param(param, 'LumvsLum'))

    def set2param(self, param, widget):
        param['LumvsLum'] = widget.ids["LumvsLum"].get_point_list()

    def make_diff(self, hls_ll, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        ll = self._get_param(param, "LumvsLum")
        if switch_color_curves == False or ll is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ll))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(ll)
                lut = 2.0 ** ((lut - 0.5) * 4.0)
                self.diff = core.apply_lut(hls_ll[0], lut, 1.0) * hls_ll[1]

        return self.diff

class LumvsSatEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_color_curves': True,
            'LumvsSat': None,
        }

    def set2widget(self, widget, param):
        widget.ids["LumvsSat"].set_point_list(self._get_param(param, 'LumvsSat'))

    def set2param(self, param, widget):
        param['LumvsSat'] = widget.ids["LumvsSat"].get_point_list()

    def make_diff(self, hls_ls, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        ls = self._get_param(param, "LumvsSat")
        if switch_color_curves == False or ls is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ls))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(ls)
                lut = (lut - 0.5) * 2.0 + 1.0
                self.diff = core.apply_lut(hls_ls[0], lut, 1.0) * hls_ls[1]

        return self.diff

class SatvsLumEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_color_curves': True,
            'SatvsLum': None,
        }

    def set2widget(self, widget, param):
        widget.ids["SatvsLum"].set_point_list(self._get_param(param, 'SatvsLum'))

    def set2param(self, param, widget):
        param['SatvsLum'] = widget.ids["SatvsLum"].get_point_list()

    def make_diff(self, hls_sl, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        sl = self._get_param(param, "SatvsLum")
        if switch_color_curves == False or sl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(sl))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(sl)
                lut = 2.0 ** ((lut - 0.5) * 4.0)
                self.diff = core.apply_lut(hls_sl[0], lut, 1.0) * hls_sl[1]

        return self.diff

class SatvsSatEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_color_curves': True,
            'SatvsSat': None,
        }

    def set2widget(self, widget, param):
        widget.ids["SatvsSat"].set_point_list(self._get_param(param, 'SatvsSat'))

    def set2param(self, param, widget):
        param['SatvsSat'] = widget.ids["SatvsSat"].get_point_list()

    def make_diff(self, hls_ss, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        ss = self._get_param(param, "SatvsSat")
        if switch_color_curves == False or ss is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ss))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(ss)
                lut = (lut - 0.5) * 2.0 + 1.0
                self.diff = core.apply_lut(hls_ss[0], lut, 1.0) * hls_ss[1]

        return self.diff

class SaturationEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_saturation': True,
            'saturation': 0,
            'vibrance': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["slider_saturation"].set_slider_value(self._get_param(param, 'saturation'))
        widget.ids["slider_vibrance"].set_slider_value(self._get_param(param, 'vibrance'))

    def set2param(self, param, widget):
        param['saturation'] = widget.ids["slider_saturation"].value
        param['vibrance'] = widget.ids["slider_vibrance"].value

    def make_diff(self, hls_s, param, efconfig):
        switch_saturation = self._get_param(param, 'switch_saturation')
        sat = self._get_param(param, 'saturation')
        vib = self._get_param(param, 'vibrance')
        if switch_saturation == False or (sat == 0 and vib == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((sat, vib))
            if self.hash != param_hash:
                self.hash = param_hash

                hls_s = core.type_convert(hls_s, np.ndarray)
                self.diff = core.calc_saturation(hls_s, sat, vib)
        
        return self.diff

class AutoExposureEffect(Effect):
    # rgb_or_rawがrawの場合
    #   lut_to_logが設定されていたら、auto_exposureを適用する
    # rgb_or_rawがrgbの場合
    #   lut_to_logが設定されていたら、auto_exposureを適用する

    def get_param_dict(self, param):
        return {
            'switch_lut': True,
            'rgb_or_raw': 'raw',
            'auto_exposure': 0,
            'lut_name': 'None',
            'lut_to_log': 'None'
        }

    def make_diff(self, rgb, param, efconfig):
        switch_lut = self._get_param(param, 'switch_lut')
        rgb_or_raw = self._get_param(param, 'rgb_or_raw')
        ae = self._get_param(param, 'auto_exposure')
        lut_to_log = self._get_param(param, 'lut_to_log')
        lut_name = self._get_param(param, 'lut_name')
        if switch_lut == False or (rgb_or_raw == 'raw' and lut_to_log == 'None') or (rgb_or_raw == 'rgb' and lut_to_log == 'None'):
            self.diff = None
            self.hash = None
        
        else:
            param_hash = hash((rgb_or_raw, ae, lut_to_log, lut_name))
            if self.hash != param_hash:
                self.hash = param_hash
                
                # 自動コントラスト補正
                #rgb = core.auto_contrast_tonemap(rgb)

                # 明るさ補正適用
                rgb = core.adjust_exposure(rgb, ae)
                
                # 超ハイライト領域のコントラストを上げてディティールをはっきりさせるなどする
                #rgb = highlight_recovery.reconstruct_highlight_details(rgb, False)

                #hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
                #hls[..., 2] = core.calc_saturation(hls[..., 2], 0, 60)
                #rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
                self.diff = rgb

        return self.diff

class LUTEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lut = None

    def get_param_dict(self, param):
        return {
            'switch_lut': True,
            'lut_name': 'None',
            'lut_intensity': 100,
            'lut_to_log': 'None',
        }

    def set2widget(self, widget, param):
        widget.ids["switch_lut"].active = self._get_param(param, 'switch_lut')
        widget.ids["lut_spinner"].set_text(self._get_param(param, 'lut_name'))
        widget.ids["lut_to_log_spinner"].set_text(self._get_param(param, 'lut_to_log'))
        widget.ids["slider_lut_intensity"].set_slider_value(self._get_param(param, 'lut_intensity'))

    def set2param(self, param, widget):
        param['switch_lut'] = widget.ids["switch_lut"].active
        spinner = widget.ids["lut_spinner"]
        name = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        if self._get_param(param, 'lut_name') != name:
            self.lut = None
        param['lut_name'] = name
        param['lut_intensity'] = widget.ids["slider_lut_intensity"].value
        param['lut_to_log'] = widget.ids["lut_to_log_spinner"].text

    def make_diff(self, rgb, param, efconfig):
        switch_lut = self._get_param(param, 'switch_lut')
        lut_name = self._get_param(param, 'lut_name')
        lut_to_log = self._get_param(param, 'lut_to_log')
        lut_intensity = self._get_param(param, 'lut_intensity')
        lut_path = config.get_config('lut_path')
        if switch_lut == False or lut_path is None or lut_name == 'None' or lut_intensity == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((lut_name, lut_path, lut_to_log, lut_intensity))
            if self.hash != param_hash:
                if self.lut is None:
                    path = os.path.join(lut_path, lut_name)
                    self.lut = cubelut.read_lut(path)

                if self.lut is not None:
                    self.hash = param_hash

                    rgb = core.type_convert(rgb, np.ndarray)
                    if lut_to_log != 'None':
                        rgb = linear_to_log.process_image(rgb, lut_to_log)

                    apply_rgb = cubelut.apply_lut(rgb, self.lut)
                    self.diff = rgb * (1-lut_intensity/100) + apply_rgb * lut_intensity/100
                else:
                    self.diff = None
                    self.hash = None

        return self.diff

class LensSimulatorEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_lens_simulator': True,
            'lens_preset': 'None',
            'lens_intensity': 100,
        }
 
    def set2widget(self, widget, param):
        widget.ids["switch_lens_simulator"].enabled = self._get_param(param, 'switch_lens_simulator')
        widget.ids["spinner_lens_preset"].set_text(self._get_param(param, 'lens_preset'))
        widget.ids["slider_lens_intensity"].set_slider_value(self._get_param(param, 'lens_intensity'))

    def set2param(self, param, widget):
        param['switch_lens_simulator'] = widget.ids["switch_lens_simulator"].enabled
        spinner = widget.ids["spinner_lens_preset"]
        param['lens_preset'] = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        param['lens_intensity'] = widget.ids["slider_lens_intensity"].value

    def make_diff(self, rgb, param, efconfig):
        switch_lens_simulator = self._get_param(param, 'switch_lens_simulator')
        preset = self._get_param(param, 'lens_preset')
        intensity = self._get_param(param, 'lens_intensity')
        if switch_lens_simulator == False or preset == 'None' or intensity <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((preset, intensity))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                lens = lens_simulator.process_image(rgb, preset)
                per = intensity / 100.0
                self.diff = cv2.addWeighted(lens, per, rgb, 1 - per, 0)

        return self.diff
    
class FilmSimulationEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_film_simulation': True,
            'film_preset': 'None',
            'film_intensity': 100,
            'film_expired': 0,
        }
 
    def set2widget(self, widget, param):
        widget.ids["switch_film_simulation"].enabled = self._get_param(param, 'switch_film_simulation')
        widget.ids["spinner_film_preset"].set_text(self._get_param(param, 'film_preset'))
        widget.ids["slider_film_intensity"].set_slider_value(self._get_param(param, 'film_intensity'))
        widget.ids["slider_film_expired"].set_slider_value(self._get_param(param, 'film_expired'))

    def set2param(self, param, widget):
        param['switch_film_simulation'] = widget.ids["switch_film_simulation"].enabled
        spinner = widget.ids["spinner_film_preset"]
        param['film_preset'] = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        param['film_intensity'] = widget.ids["slider_film_intensity"].value
        param['film_expired'] = widget.ids["slider_film_expired"].value

    def make_diff(self, rgb, param, efconfig):
        switch_film_simulation = self._get_param(param, 'switch_film_simulation')
        preset = self._get_param(param, 'film_preset')
        intensity = self._get_param(param, 'film_intensity')
        expired = self._get_param(param, 'film_expired')
        if switch_film_simulation == False or preset == 'None' or intensity <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((preset, intensity, expired))
            if self.hash != param_hash:
                self.hash = param_hash
                
                rgb = core.type_convert(rgb, np.ndarray)
                film = film_emulator.emulator.apply_film_effect(rgb, preset, expired)
                per = intensity / 100.0
                self.diff = cv2.addWeighted(film, per, rgb, 1 - per, 0)

        return self.diff

class SolidColorEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_solid_color': True,
            'solid_color': 0,
            'solid_color_hue': 0,
            'solid_color_lum': 50,
            'solid_color_sat': 0,
            'solid_opacity': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_solid_color"].active = self._get_param(param, 'switch_solid_color')
        h, l, s = self._get_param(param, 'solid_color_hue'), self._get_param(param, 'solid_color_lum'), self._get_param(param, 'solid_color_sat')
        widget.ids["cp_solid_color"].ids['slider_hue'].set_slider_value(h)
        widget.ids["cp_solid_color"].ids['slider_lum'].set_slider_value(l)
        widget.ids["cp_solid_color"].ids['slider_sat'].set_slider_value(s)
        widget.ids["slider_solid_color"].set_slider_value(self._get_param(param, 'solid_opacity'))
        # これを後にしないと値が上書きされる
        widget.ids["cp_solid_color"].set_slider_value((h, l, s))

    def set2param(self, param, widget):
        param['switch_solid_color'] = widget.ids["switch_solid_color"].active
        param["solid_color_hue"] = widget.ids["cp_solid_color"].ids['slider_hue'].value
        param["solid_color_lum"] = widget.ids["cp_solid_color"].ids['slider_lum'].value
        param["solid_color_sat"] = widget.ids["cp_solid_color"].ids['slider_sat'].value
        param["solid_opacity"] = widget.ids["slider_solid_color"].value

    def make_diff(self, rgb, param, efconfig):
        switch_solid_color = self._get_param(param, 'switch_solid_color')
        coh = self._get_param(param, "solid_color_hue")
        col = self._get_param(param, "solid_color_lum")
        cos = self._get_param(param, "solid_color_sat")
        coao = self._get_param(param, "solid_opacity")
        if switch_solid_color == False or coao <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((coh, cos, col, coao))
            if self.hash != param_hash:
                self.hash = param_hash

                import colorsys
                r, g, b = colorsys.hls_to_rgb(coh/360, col/100, cos/100)
                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.apply_solid_color(rgb, solid_color=(r, g, b), opacity=coao/100)

        return self.diff

class UnsharpMaskEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_unsharp_mask': True,
            'unsharp_mask_amount': 0,
            'unsharp_mask_sigma': 50,
        }
 
    def set2widget(self, widget, param):
        widget.ids['switch_unsharp_mask'].active = self._get_param(param, 'switch_unsharp_mask')
        widget.ids["slider_unsharp_mask_amount"].set_slider_value(self._get_param(param, 'unsharp_mask_amount'))
        widget.ids["slider_unsharp_mask_sigma"].set_slider_value(self._get_param(param, 'unsharp_mask_sigma'))

    def set2param(self, param, widget):
        param['switch_unsharp_mask'] = widget.ids['switch_unsharp_mask'].active
        param['unsharp_mask_amount'] = widget.ids["slider_unsharp_mask_amount"].value
        param['unsharp_mask_sigma'] = widget.ids["slider_unsharp_mask_sigma"].value

    def make_diff(self, rgb, param, efconfig):
        switch_unsharp_mask = self._get_param(param, 'switch_unsharp_mask')
        amount = self._get_param(param, 'unsharp_mask_amount')
        sigma = self._get_param(param, 'unsharp_mask_sigma')
        if switch_unsharp_mask == False or amount == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((amount, sigma))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                amount = amount / 100.0 * 1.5
                sigma = sigma / 100.0 * 3.0
                self.diff = core.unsharp_mask(rgb, amount, sigma)

        return self.diff


class Mask2Effect(Effect):

    @staticmethod
    def get_param(param, key, default=None):
        if default is not None:
            return param.get(key, default)
        
        return param.get(key, Mask2Effect.get_param_dict(param)[key])

    @staticmethod
    def get_param_dict(param):
        return {
            'switch_mask2_settings': True,
            'mask2_invert': False,
            'switch_mask2_depth': True,
            'mask2_depth_min': 0,
            'mask2_depth_max': 255,
            'switch_mask2_hue': True,
            'mask2_hue_distance': 179,
            'mask2_hue_min': 0,
            'mask2_hue_max': 359,
            'switch_mask2_lum': True,
            'mask2_lum_distance': 127,
            'mask2_lum_min': 0,
            'mask2_lum_max': 255,
            'switch_mask2_sat': True,
            'mask2_sat_distance': 127,
            'mask2_sat_min': 0,
            'mask2_sat_max': 255,
            'switch_mask2_options': True,
            'mask2_blur': 0,
            'mask2_open_space': 0,
            'mask2_close_space': 0,
            'switch_mask2_face': True,            
            'mask2_face_face': True,
            'mask2_face_brows': True,
            'mask2_face_eyes': True,
            'mask2_face_nose': True,
            'mask2_face_mouth': True,
            'mask2_face_lips': True,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_mask2_settings"].active = self._get_param(param, 'switch_mask2_settings')
        widget.ids["checkbox_mask2_invert"].active = self._get_param(param, 'mask2_invert')
        widget.ids["switch_mask2_depth"].active = self._get_param(param, 'switch_mask2_depth')
        widget.ids["slider_mask2_depth_min"].set_slider_value(self._get_param(param, 'mask2_depth_min'))
        widget.ids["slider_mask2_depth_max"].set_slider_value(self._get_param(param, 'mask2_depth_max'))
        widget.ids["switch_mask2_hue"].active = self._get_param(param, 'switch_mask2_hue')
        widget.ids["slider_mask2_hue_distance"].set_slider_value(self._get_param(param, 'mask2_hue_distance'))
        widget.ids["slider_mask2_hue_min"].set_slider_value(self._get_param(param, 'mask2_hue_min'))
        widget.ids["slider_mask2_hue_max"].set_slider_value(self._get_param(param, 'mask2_hue_max'))
        widget.ids["switch_mask2_lum"].active = self._get_param(param, 'switch_mask2_lum')
        widget.ids["slider_mask2_lum_distance"].set_slider_value(self._get_param(param, 'mask2_lum_distance'))
        widget.ids["slider_mask2_lum_min"].set_slider_value(self._get_param(param, 'mask2_lum_min'))
        widget.ids["slider_mask2_lum_max"].set_slider_value(self._get_param(param, 'mask2_lum_max'))
        widget.ids["switch_mask2_sat"].active = self._get_param(param, 'switch_mask2_sat')
        widget.ids["slider_mask2_sat_distance"].set_slider_value(self._get_param(param, 'mask2_sat_distance'))
        widget.ids["slider_mask2_sat_min"].set_slider_value(self._get_param(param, 'mask2_sat_min'))
        widget.ids["slider_mask2_sat_max"].set_slider_value(self._get_param(param, 'mask2_sat_max'))
        widget.ids["switch_mask2_options"].active = self._get_param(param, 'switch_mask2_options')
        widget.ids["slider_mask2_blur"].set_slider_value(self._get_param(param, 'mask2_blur'))
        widget.ids["slider_mask2_open_space"].set_slider_value(self._get_param(param, 'mask2_open_space'))
        widget.ids["slider_mask2_close_space"].set_slider_value(self._get_param(param, 'mask2_close_space'))
        widget.ids["switch_mask2_face"].active = self._get_param(param, 'switch_mask2_face')
        widget.ids["checkbox_mask2_face_face"].active = self._get_param(param, 'mask2_face_face')
        widget.ids["checkbox_mask2_face_brows"].active = self._get_param(param, 'mask2_face_brows')
        widget.ids["checkbox_mask2_face_eyes"].active = self._get_param(param, 'mask2_face_eyes')
        widget.ids["checkbox_mask2_face_nose"].active = self._get_param(param, 'mask2_face_nose')
        widget.ids["checkbox_mask2_face_mouth"].active = self._get_param(param, 'mask2_face_mouth')
        widget.ids["checkbox_mask2_face_lips"].active = self._get_param(param, 'mask2_face_lips')

    def set2param(self, param, widget):
        param['switch_mask2_settings'] = widget.ids["switch_mask2_settings"].active
        param['mask2_invert'] = widget.ids["checkbox_mask2_invert"].active
        param['switch_mask2_depth'] = widget.ids["switch_mask2_depth"].active
        param['mask2_depth_min'] = widget.ids["slider_mask2_depth_min"].value
        param['mask2_depth_max'] = widget.ids["slider_mask2_depth_max"].value
        param['switch_mask2_hue'] = widget.ids["switch_mask2_hue"].active
        param['mask2_hue_distance'] = widget.ids["slider_mask2_hue_distance"].value
        param['mask2_hue_min'] = widget.ids["slider_mask2_hue_min"].value
        param['mask2_hue_max'] = widget.ids["slider_mask2_hue_max"].value
        param['switch_mask2_lum'] = widget.ids["switch_mask2_lum"].active
        param['mask2_lum_distance'] = widget.ids["slider_mask2_lum_distance"].value
        param['mask2_lum_min'] = widget.ids["slider_mask2_lum_min"].value
        param['mask2_lum_max'] = widget.ids["slider_mask2_lum_max"].value
        param['switch_mask2_sat'] = widget.ids["switch_mask2_sat"].active
        param['mask2_sat_distance'] = widget.ids["slider_mask2_sat_distance"].value
        param['mask2_sat_min'] = widget.ids["slider_mask2_sat_min"].value
        param['mask2_sat_max'] = widget.ids["slider_mask2_sat_max"].value
        param['switch_mask2_options'] = widget.ids["switch_mask2_options"].active
        param['mask2_blur'] = widget.ids["slider_mask2_blur"].value
        param['mask2_open_space'] = widget.ids["slider_mask2_open_space"].value
        param['mask2_close_space'] = widget.ids["slider_mask2_close_space"].value
        param['switch_mask2_face'] = widget.ids["switch_mask2_face"].active
        param['mask2_face_face'] = widget.ids["checkbox_mask2_face_face"].active
        param['mask2_face_brows'] = widget.ids["checkbox_mask2_face_brows"].active
        param['mask2_face_eyes'] = widget.ids["checkbox_mask2_face_eyes"].active
        param['mask2_face_nose'] = widget.ids["checkbox_mask2_face_nose"].active
        param['mask2_face_mouth'] = widget.ids["checkbox_mask2_face_mouth"].active
        param['mask2_face_lips'] = widget.ids["checkbox_mask2_face_lips"].active

    def make_diff(self, rgb, param, efconfig):
        """
        invert = self._get_param(param, 'mask2_invert')
        dmin = self._get_param(param, 'mask2_depth_min')
        dmax = self._get_param(param, 'mask2_depth_max')
        hdis = self._get_param(param, 'mask2_hue_distance')
        hmin = self._get_param(param, 'mask2_hue_min')
        hmax = self._get_param(param, 'mask2_hue_max')
        ldis = self._get_param(param, 'mask2_lum_distance')
        lmin = self._get_param(param, 'mask2_lum_min')
        lmax = self._get_param(param, 'mask2_lum_max')
        sdis = self._get_param(param, 'mask2_sat_distance')
        smin = self._get_param(param, 'mask2_sat_min')
        smax = self._get_param(param, 'mask2_sat_max')
        blur = self._get_param(param, 'mask2_blur')
        face_face = self._get_param(param, 'mask2_face_face')
        face_brows = self._get_param(param, 'mask2_face_brows')
        face_eyes = self._get_param(param, 'mask2_face_eyes')
        face_nose = self._get_param(param, 'mask2_face_nose')
        face_mouth = self._get_param(param, 'mask2_face_mouth')
        face_lips = self._get_param(param, 'mask2_face_lips')
        open_space = self._get_param(param, 'mask2_open_space')
        close_space = self._get_param(param, 'mask2_close_space')
        if  (invert == False and dmin == 0 and dmax == 255 and
             hdis == 179 and hmin == 0 and hmax == 359 and
             ldis == 127 and lmin == 0 and lmax == 255 and
             sdis == 127 and smin == 0 and smax == 255 and
             blur == 0):
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((invert, dmin, dmax, hdis, hmin, hmax, ldis, lmin, lmax, sdis, smin, smax, blur))
            if self.hash != param_hash:
                self.hash = param_hash
                
                self.diff = None
        """
        return self.diff

class GrainEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_grain': True,
            'grain_intensity': 0,
            'grain_size': 0,
            'grain_blue_bias': 0,
            'grain_shadow_boost': 0,
            'grain_color_noise_ratio': 0
        }

    def set2widget(self, widget, param):
        widget.ids["switch_grain"].active = self._get_param(param, 'switch_grain')
        widget.ids["slider_grain_intensity"].set_slider_value(self._get_param(param, 'grain_intensity'))
        widget.ids["slider_grain_size"].set_slider_value(self._get_param(param, 'grain_size'))
        widget.ids["slider_grain_blue_bias"].set_slider_value(self._get_param(param, 'grain_blue_bias'))
        widget.ids["slider_grain_shadow_boost"].set_slider_value(self._get_param(param, 'grain_shadow_boost'))
        widget.ids["slider_grain_color_noise_ratio"].set_slider_value(self._get_param(param, 'grain_color_noise_ratio'))

    def set2param(self, param, widget):
        param['switch_grain'] = widget.ids["switch_grain"].active
        param['grain_intensity'] = widget.ids["slider_grain_intensity"].value
        param['grain_size'] = widget.ids["slider_grain_size"].value
        param['grain_blue_bias'] = widget.ids["slider_grain_blue_bias"].value
        param['grain_shadow_boost'] = widget.ids["slider_grain_shadow_boost"].value        
        param['grain_color_noise_ratio'] = widget.ids["slider_grain_color_noise_ratio"].value

    def make_diff(self, rgb, param, efconfig):
        switch_grain = self._get_param(param, 'switch_grain')
        gi = self._get_param(param, 'grain_intensity')
        gs = self._get_param(param, 'grain_size')
        gbb = self._get_param(param, 'grain_blue_bias')
        gsb = self._get_param(param, 'grain_shadow_boost')
        gcnr = self._get_param(param, 'grain_color_noise_ratio')
        if switch_grain == False or gi == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((gi, gs, gbb, gsb, gcnr))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                gi = gi / 100.0                 # 0.0-1.0
                gs = gs / 100.0 * 4.0 + 1.0     # 1.0-5.0
                gbb = gbb / 100.0 + 1.0         # 1.0-2.0
                gsb = gsb / 100.0 * 1.5 + 0.5   # 0.5-2.0          
                gcnr = gcnr / 100.0             # 0.0-1.0
                self.diff = core.apply_film_grain(rgb, gi * efconfig.disp_info[4], gs * efconfig.resolution_scale , gbb, gsb, gcnr)
        
        return self.diff
    
class VignetteEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_vignette': True,
            'vignette_intensity': 0,
            'vignette_radius_percent': 0,
            'vignette_softness': 100,
            'crop_enable': False,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_vignette"].active = self._get_param(param, 'switch_vignette')
        widget.ids["slider_vignette_intensity"].set_slider_value(self._get_param(param, 'vignette_intensity'))
        widget.ids["slider_vignette_radius_percent"].set_slider_value(self._get_param(param, 'vignette_radius_percent'))
        widget.ids["slider_vignette_softness"].set_slider_value(self._get_param(param, 'vignette_softness'))

    def set2param(self, param, widget):
        param['switch_vignette'] = widget.ids["switch_vignette"].active
        param['vignette_intensity'] = widget.ids["slider_vignette_intensity"].value
        param['vignette_radius_percent'] = widget.ids["slider_vignette_radius_percent"].value
        param['vignette_softness'] = widget.ids["slider_vignette_softness"].value

    def make_diff(self, rgb, param, efconfig):
        switch_vignette = self._get_param(param, 'switch_vignette')
        vi = self._get_param(param, 'vignette_intensity')
        vr = self._get_param(param, 'vignette_radius_percent')
        vs = self._get_param(param, 'vignette_softness')
        pce = self._get_param(param, 'crop_enable')
        if switch_vignette == False or (vi == 0 and vr == 0) or pce == True:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((vi, vr, vs))
            if self.hash != param_hash:
                self.hash = param_hash

                if efconfig.mode == EffectMode.EXPORT:
                    offset_x, offset_y = 0, 0
                else:
                    _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(config.get_config('preview_width'), config.get_config('preview_height'), efconfig.disp_info)
                
                rgb = core.type_convert(rgb, np.ndarray)
                vs = (100 - vs) / 100.0 * 3.0 + 1.0  # 1.0-4.0
                self.diff = core.apply_vignette(rgb, vi, vr, efconfig.disp_info, params.get_crop_rect(param), (offset_x, offset_y), vs)
        
        return self.diff
    

def create_effects(lens_modifier_callback=None, geometry_callback=None, distortion_callback=None, crop_callback=None):
    effects = [{}, {}, {}, {}, {}]

    lv0 = effects[0]
    lv0['loading_wait'] = LoadingWaitEffect()
    lv0['ai_noise_reduction'] = AINoiseReductonEffect()
    lv0['remove_chromatic_aberration'] = RemoveChromaticAberrationEffect()
    lv0['lens_modifier'] = LensModifierEffect(lens_modifier_callback=lens_modifier_callback)
    lv0['subpixel_shift'] = SubpixelShiftEffect()
    lv0['inpaint'] = InpaintEffect()
    lv0['cross_filter'] = CrossFilterEffect()
    lv0['geometry'] = GeometryEffect(geometry_callback=geometry_callback)
    lv0['crop'] = CropEffect(crop_callback=crop_callback)

    lv1 = effects[1]
    lv1['distortion'] = DistortionEffect(distortion_callback=distortion_callback)
    lv1['deblur_filter'] = DeblurFilterEffect()
    lv1['defocus'] = DefocusEffect()
    lv1['orton'] = OrtonEffect()
    lv1['lensblur_filter'] = LensblurFilterEffect()
    lv1['scratch'] = ScratchEffect()
    lv1['frosted_glass'] = FrostedGlassEffect()
    lv1['mosaic'] = MosaicEffect()
    lv1['face'] = FaceEffect()
    
    lv2 = effects[2]
    lv2['color_temperature'] = ColorTemperatureEffect()
    
    lv2['auto_exposure'] = AutoExposureEffect()
    lv2['lut'] = LUTEffect()

    lv2['dehaze'] = DehazeEffect()

    lv2['exposure'] = ExposureEffect()
    lv2['contrast'] = ContrastEffect()
    lv2['light_noise_reduction'] = LightNoiseReductionEffect()
    lv2['clarity'] = ClarityEffect()
    lv2['texture'] = TextureEffect()
    lv2['microcontrast'] = MicroContrastEffect()
    lv2['tone'] = ToneEffect()
    lv2['level'] = LevelEffect()

    lv2['glow'] = GlowEffect()
    lv2['highlight_compress'] = HighlightCompressEffect()

    # ここでクリッピング

    #lv2['rgb2hls1'] = RGB2HLSEffect()
    #lv2['hls2rgb1'] = HLS2RGBEffect()

    lv2['clahe'] = CLAHEEffect()

    lv2['rgb2hls2'] = RGB2HLSEffect()
    lv2['hls'] = HLSEffect()
    lv2['vs_and_saturation'] = VSandSaturationEffect()
    lv2['hls2rgb2'] = HLS2RGBEffect()

    lv2['curves'] = CurvesEffect()

    lv2['lens_simulator'] = LensSimulatorEffect()
    lv2['film_emulation'] = FilmSimulationEffect()
    lv2['solid_color'] = SolidColorEffect()
    lv2['unsharp_mask'] = UnsharpMaskEffect()

    lv3 = effects[3]
    lv3['mask2'] = Mask2Effect()

    lv4 = effects[4]
    lv4['grain'] = GrainEffect()
    lv4['vignette'] = VignetteEffect()

    return effects

def set2widget_all(widget, effects, param):
    for dict in effects:
        for l in dict.values():
            l.set2widget(widget, param)
            l.reeffect()

def set2param_all(effects, param, widget):
    for dict in effects:
        for l in dict.values():
            l.set2param(param, widget)
            l.reeffect()

def reeffect_all(effects, lv=0):
    for i, dict in enumerate(effects):
        if i >= lv:
            for l in dict.values():
               l.reeffect()

def finalize_all(effects, param, widget):
    for dict in effects:
        for l in dict.values():
            l.finalize(param, widget)

def delete_default_param_all(effects, param):
    param2 = param.copy()
    for dict in effects:
        for l in dict.values():
            l.delete_default_param(param2)
    return param2

def get_default_param(effects, key, param):
    for dict in effects:
        for l in dict.values():
            if hasattr(l, 'effects'):
                for l2 in l.effects.values():
                    if key in l2.get_param_dict(param):
                        return l2.get_param_dict(param)[key]
            if key in l.get_param_dict(param):
                return l.get_param_dict(param)[key]
    return None
