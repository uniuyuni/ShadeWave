
from re import L
import scipy
from typing_extensions import ItemsView
import cv2
import numpy as np
import importlib
import bz2
from enum import Enum

#import colorsys
#import skimage

#import noise2void
#import DRBNet
#import iopaint.predict
#import dehazing.dehaze

import core
import cubelut
import mask_editor
import crop_editor
import subpixel_shift
import film_simulation
import lens_simulator
import config
import pipeline
import filter
import local_contrast
import params

class EffectMode(Enum):
    PREVIEW = 0
    LOUPE = 1
    EXPORT = 2

class EffectConfig():

    def __init__(self, **kwargs):
        self.disp_info = None
        self.is_zoom = False
        self.mode = EffectMode.PREVIEW
        self.dpi_scale = 1.0

# 補正基底クラス
class Effect():

    def __init__(self, **kwargs):
        self.diff = None
        self.hash = None

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


# レンズモディファイア
class LensModifierEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_color_modification"].active = False if param.get('color_modification', 0) == 0 else True
        widget.ids["switch_subpixel_distortion"].active = False if param.get('subpixel_distortion', 0) == 0 else True
        widget.ids["switch_geometry_distortion"].active = False if param.get('geometry_distortion', 0) == 0 else True

    def set2param(self, param, widget):
        param['color_modification'] = 0 if widget.ids["switch_color_modification"].active == False else 1
        param['subpixel_distortion'] = 0 if widget.ids["switch_subpixel_distortion"].active == False else 1
        param['geometry_distortion'] = 0 if widget.ids["switch_geometry_distortion"].active == False else 1

    def make_diff(self, img, param, efconfig):
        cd = param.get('color_modification', 0)
        sd = param.get('subpixel_distortion', 0)
        gd = param.get('geometry_distortion', 0)
        if cd <= 0 and sd <= 0 and gd <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((cd, sd, gd))
            if self.hash != param_hash:
                self.diff = core.modify_lensimage(img, None, cd > 0, sd > 0, gd > 0)
                self.hash = param_hash
        
        return self.diff
    

# サブピクセルシフト合成
class SubpixelShiftEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_subpixel_shift"].active = False if param.get('subpixel_shift', 0) == 0 else True

    def set2param(self, param, widget):
        param['subpixel_shift'] = 0 if widget.ids["switch_subpixel_shift"].active == False else 1

    def make_diff(self, img, param, efconfig):
        ss = param.get('subpixel_shift', 0)
        if ss <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ss))
            if self.hash != param_hash:
                self.diff = subpixel_shift.create_enhanced_image(img)
                self.hash = param_hash
        
        return self.diff
    
import io
import imageio as iio

class InpaintDiff:
    def __init__(self, **kwargs):
        self.disp_info = kwargs.get('disp_info', None)
        self.image = kwargs.get('image', None)

    def image2list(self):
        if type(self.image) is np.ndarray:
            self.image = (self.image.shape, list(bz2.compress(self.image.tobytes(), 1)))
            #output = io.BytesIO()
            #iio.imwrite(output, self.image, plugin="pillow", extension=".avif")
            #self.image = (self.image.shape, list(output.getvalue()))

    def list2image(self):
        if type(self.image) is list:
            self.image = np.reshape(np.frombuffer(bz2.decompress(bytearray(self.image[1])), dtype=np.float32), self.image[0])
            #ary = np.frombuffer(bytearray(self.image[1]))
            #bgr = cv2.imdecode(ary)
            #self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

class InpaintEffect(Effect):
    __iopaint = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.inpaint_diff_list = []
        self.mask_editor = None

    @staticmethod
    def dump(param):
        inpaint_diff_list = param.get('inpaint_diff_list', None)
        if inpaint_diff_list is not None:
            inpaint_diff_list_dumps = []
            for inpaint_diff in inpaint_diff_list:
                inpaint_diff.image2list()
                inpaint_diff_list_dumps.append((inpaint_diff.disp_info, inpaint_diff.image))
            param['inpaint_diff_list'] = inpaint_diff_list_dumps

    @staticmethod
    def load(param):
        inpaint_diff_list_dumps = param.get('inpaint_diff_list', None)
        if inpaint_diff_list_dumps is not None:
            inpaint_diff_list = []
            for inpaint_diff_dump in inpaint_diff_list_dumps:
                inpaint_diff = InpaintDiff(disp_info=inpaint_diff_dump[0], image=inpaint_diff_dump[1])
                inpaint_diff.list2image()
                inpaint_diff_list.append(inpaint_diff)
            param['inpaint_diff_list'] = inpaint_diff_list

    def set2widget(self, widget, param):
        widget.ids["switch_inpaint"].active = False if param.get('inpaint', 0) == 0 else True
        widget.ids["button_inpaint_predict"].state = "normal" if param.get('inpaint_predict', 0) == 0 else "down"

    def set2param(self, param, widget):
        param['inpaint'] = 0 if widget.ids["switch_inpaint"].active == False else 1
        param['inpaint_predict'] = 0 if widget.ids["button_inpaint_predict"].state == "normal" else 1

        if param['inpaint'] > 0:
            if self.mask_editor is None:
                self.mask_editor = mask_editor.MaskEditor(param['img_size'][0], param['img_size'][1])
                self.mask_editor.zoom = param.get_disp_info(param)[4]
                self.mask_editor.pos = [0, 0]
                widget.ids["preview_widget"].add_widget(self.mask_editor)
            
        if param['inpaint'] <= 0:
            if self.mask_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor = None

    def make_diff(self, img, param, efconfig):
        self.inpaint_diff_list = param.get('inpaint_diff_list', [])
        ip = param.get('inpaint', 0)
        ipp = param.get('inpaint_predict', 0)
        if (ip > 0 and ipp > 0) is True:
            if InpaintEffect.__iopaint is None:
                InpaintEffect.__iopaint = importlib.import_module('iopaint.predict')

            mask = cv2.GaussianBlur(self.mask_editor.get_mask(), (63, 63), 0)
            w, h = param['original_img_size']
            eh, ew = img.shape[:2]
            x, y = (ew-w)//2, (eh-h)//2
            img2 = InpaintEffect.__iopaint.predict(img[y:y+h, x:x+w], mask, model=config.get_config('iopaint_model'), resize_limit=config.get_config('iopaint_resize_limit'), use_realesrgan=config.get_config('iopaint_use_realesrgan'))
            img2 = img2 #/ param.get('white_balance', [1, 1, 1])
            bboxes = core.get_multiple_mask_bbox(self.mask_editor.get_mask())
            for bbox in bboxes:
                self.inpaint_diff_list.append(InpaintDiff(disp_info=(bbox[0] + x, bbox[1] + y, bbox[2], bbox[3]), image=img2[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]))
            param['inpaint_diff_list'] = self.inpaint_diff_list
            self.mask_editor.clear_mask()
            self.mask_editor.update_canvas()
        
        if len(self.inpaint_diff_list) > 0:
            img2 = img.copy()
            for inpaint_diff in self.inpaint_diff_list:
                cx, cy, cw, ch = inpaint_diff.disp_info
                img2[cy:cy+ch, cx:cx+cw] = inpaint_diff.image
            self.diff = img2
        else:
            self.diff = None

        return self.diff
    

# 画像回転、反転
class RotationEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_rotation"].set_slider_value(param.get('rotation', 0))

    def set2param(self, param, widget):
        param['rotation'] = widget.ids["slider_rotation"].value
        
    def set2param2(self, param, arg):
        if arg == 'hflip':
            param['flip_mode'] = param.get('flip_mode', 0) ^ 1

        elif arg == 'vflip':
            param['flip_mode'] = param.get('flip_mode', 0) ^ 2

        elif arg == 90:
            rot = param.get('rotation2', 0) + 90.0
            if rot >= 90*4:
                rot = 0
            param['rotation2'] = rot

        elif arg == -90:
            rot = param.get('rotation2', 0) - 90.0
            if rot < 0:
                rot = 90*3
            param['rotation2'] = rot

        else:
            pass

    def make_diff(self, img, param, efconfig):
        ang = param.get('rotation', 0)
        ang2 = param.get('rotation2', 0)
        flp = param.get('flip_mode', 0)
        if ang == 0 and ang2 == 0 and flp == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ang, ang2, flp))
            if self.hash != param_hash:
                self.diff = core.rotation(img, ang + ang2, flp)
                self.hash = param_hash
        
        return self.diff

# クロップ
class CropEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.crop_editor = None
        self.crop_editor_callback = None

    def set_editing_callback(self, callback):
        self.crop_editor_callback = callback

    def _param_to_aspect_ratio(self, param):
        ar = param.get('aspect_ratio', "None")
        return eval(ar if ar != "None" else "0")

    def set2widget(self, widget, param):
        widget.ids["spinner_acpect_ratio"].text = param.get('aspect_ratio', "None")

    def set2param(self, param, widget):
        param['crop_enable'] = False if widget.ids["effects"].current_tab.text != "Geometry" else True
        param['aspect_ratio'] = widget.ids["spinner_acpect_ratio"].text

        # disp_info がないのはマスク
        if params.get_crop_rect(param) is not None:

            # クロップエディタを開く
            if param['crop_enable'] == True:
                self._open_crop_editor(param, widget)

            # クロップエディタを閉じる
            elif param['crop_enable'] == False:
                self._close_crop_editor(param, widget)

            # クロップ範囲をリセット
            if self.crop_editor is not None:
                if widget.ids["button_crop_reset"].state == "down":
                    self.crop_editor._set_to_local_crop_rect([0, 0, 0, 0])
                    self.crop_editor.update_crop_size()

                self.crop_editor.input_angle = param.get('rotation', 0) + param.get('rotation2', 0)
                self.crop_editor.set_aspect_ratio(self._param_to_aspect_ratio(param))


    def make_diff(self, img, param, efconfig):
        ce = param.get('crop_enable', False)
        disp_info = params.get_disp_info(param)
        if ce == True or disp_info is None:
            self.diff = None
            self.hash = None
            param['img_size'] = (param['original_img_size'][0], param['original_img_size'][1])
            msize = max(param['original_img_size'][0], param['original_img_size'][1])
            params.set_disp_info(param, (0, 0, msize, msize, disp_info[4]))
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
            input_width, input_height = param['original_img_size']
            x1, y1, x2, y2 = params.get_crop_rect(param)
            scale = config.get_config('preview_size')/max(input_width, input_height)
            self.crop_editor = crop_editor.CropEditor(input_width=input_width, input_height=input_height, scale=scale, crop_rect=[x1, y1, x2, y2], aspect_ratio=self._param_to_aspect_ratio(param))
            self.crop_editor.set_editing_callback(self._crop_editing)
            widget.ids["preview_widget"].add_widget(self.crop_editor)

            # 編集中は一時的に変更
            params.set_disp_info(param, crop_editor.CropEditor.get_initial_disp_info(input_width, input_height, scale))

            # 保存しておく
            self.param = param

    def _close_crop_editor(self, param, widget):
        if self.crop_editor is not None:
            params.set_crop_rect(param, self.crop_editor.get_crop_rect())
            params.set_disp_info(param, self.crop_editor.get_disp_info())

            widget.ids["preview_widget"].remove_widget(self.crop_editor)
            self.crop_editor = None

    def _crop_editing(self):
        params.set_crop_rect(self.param, self.crop_editor.get_crop_rect())
        if self.crop_editor_callback is not None:
            self.crop_editor_callback()

    def finalize(self, param, widget):
        self._close_crop_editor(param, widget)


# AI ノイズ除去
class AINoiseReductonEffect(Effect):
    __net = None
    __module = None

    def set2widget(self, widget, param):
        widget.ids["switch_ai_noise_reduction"].active = False if param.get('ai_noise_reduction', 0) == 0 else True

    def set2param(self, param, widget):
        param['ai_noise_reduction'] = 0 if widget.ids["switch_ai_noise_reduction"].active == False else 1

    def make_diff(self, img, param, efconfig):
        nr = param.get('ai_noise_reduction', 0)
        if nr <= 0 or efconfig.disp_info[4] < config.get_config('scale_threshold'):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((nr))
            if self.hash != param_hash:
                if AINoiseReductonEffect.__module is None:
                    AINoiseReductonEffect.__module = importlib.import_module('SCUNet')
                if AINoiseReductonEffect.__net is None:
                    AINoiseReductonEffect.__net = AINoiseReductonEffect.__module.setup_model(device=config.get_config('gpu_type'))

                #img = np.clip(img, 0, 1)
                self.diff = AINoiseReductonEffect.__module.denoise_image_helper(AINoiseReductonEffect.__net, img, config.get_config('gpu_type'))
                self.hash = param_hash
        
        return self.diff


# NLMノイズ除去
class NLMNoiseReductionEffect(Effect):
    __skimage = None

    def set2widget(self, widget, param):
        widget.ids["slider_nlm_noise_reduction"].set_slider_value(param.get('nlm_noise_reduction', 0))

    def set2param(self, param, widget):
        param['nlm_noise_reduction'] = widget.ids["slider_nlm_noise_reduction"].value

    def make_diff(self, img, param, efconfig):
        nlm = int(param.get('nlm_noise_reduction', 0))
        if nlm == 0 or efconfig.disp_info[4] < config.get_config('scale_threshold'):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((nlm))
            if self.hash != param_hash:
                if NLMNoiseReductionEffect.__skimage is None:
                    NLMNoiseReductionEffect.__skimage = importlib.import_module('bm3d')

                #noisy_img0 = img[..., 0]
                #basic_img0 = NLMNoiseReductionEffect.__skimage.BM3D(noisy_img0)
                #noisy_img1 = img[..., 1]
                #basic_img1 = NLMNoiseReductionEffect.__skimage.BM3D(noisy_img1)
                #noisy_img2 = img[..., 2]
                #basic_img2 = NLMNoiseReductionEffect.__skimage.BM3D(noisy_img2)
                #self.diff = np.stack([basic_img0, basic_img1, basic_img2], axis=-1)
                
                self.diff = NLMNoiseReductionEffect.__skimage.bm3d(img, nlm/1000.0 * efconfig.disp_info[4])
                #sigma_est = np.mean(NLMNoiseReductionEffect.__skimage.restoration.estimate_sigma(img, channel_axis=2))
                #self.diff = NLMNoiseReductionEffect.__skimage.restoration.denoise_nl_means(img, h=nlm/100.0*sigma_est, sigma=sigma_est, fast_mode=True, channel_axis=2)
                self.hash = param_hash

        return self.diff

class LightNoiseReductionEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_light_noise_reduction"].set_slider_value(param.get('light_noise_reduction', 0))
        widget.ids["slider_light_color_noise_reduction"].set_slider_value(param.get('light_color_noise_reduction', 0))

    def set2param(self, param, widget):
        param['light_noise_reduction'] = widget.ids["slider_light_noise_reduction"].value
        param['light_color_noise_reduction'] = widget.ids["slider_light_color_noise_reduction"].value

    def make_diff(self, img, param, efconfig):
        its = int(param.get('light_noise_reduction', 0))
        col = int(param.get('light_color_noise_reduction', 0))
        if its == 0 and col == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((its, col))
            if self.hash != param_hash:  
                its = its * efconfig.disp_info[4]
                col = col * efconfig.disp_info[4]

                # Lab色空間に変換（L: 輝度, a,b: 色度）
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
                l, a, b = cv2.split(lab)
                
                # 輝度チャンネル(L)のノイズ除去 - エッジ保持フィルタ
                if its > 0:
                    d_l = max(1, min(15, int(1 + its * 0.05)))
                    d_l = d_l + 1 if d_l % 2 == 0 else d_l
                    sigma_l = 10 + its * 0.5

                    # ノイズ低減処理付きSobel
                    gray = l / 100.0
                    denoised = cv2.bilateralFilter(gray, 5, 0.1, 5)
                    sobel_x = cv2.Sobel(denoised, cv2.CV_32F, 1, 0)
                    sobel_y = cv2.Sobel(denoised, cv2.CV_32F, 0, 1)
                    mag = np.sqrt(sobel_x**2 + sobel_y**2)
                    
                    # ノイズ閾値処理
                    noise_threshold = 0.05
                    mag[mag < noise_threshold] = 0
                    #cv2.imwrite("mag.jpg", (mag * 255).astype(np.uint8))
                    l_filtered = cv2.ximgproc.jointBilateralFilter(
                        mag, gray, 
                        d_l, sigma_l / 10, sigma_l
                    ) * 100.0
                    print(f"jbf: {d_l}, {sigma_l/10}, {sigma_l}")

                    #l_filtered = cv2.bilateralFilter(l, d_l, sigma_l, sigma_l / 2)
                else:
                    l_filtered = l
                
                if col > 0:
                    # 色度チャンネル(a,b)のノイズ除去 - 強力な平滑化
                    ksize = max(3, min(51, int(3 + col * 0.5)))
                    ksize = ksize + 1 if ksize % 2 == 0 else ksize
                    a_filtered = core.fast_median_filter(a, ksize)
                    b_filtered = core.fast_median_filter(b, ksize)
                else:
                    a_filtered = a
                    b_filtered = b
                
                # チャンネルを結合
                filtered_lab = cv2.merge([l_filtered, a_filtered, b_filtered])
                self.diff = cv2.cvtColor(filtered_lab, cv2.COLOR_Lab2RGB)
    

                self.hash = param_hash

        return self.diff

# デブラーフィルタ
class DeblurFilterEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_deblur_filter"].set_slider_value(param.get('deblur_filter', 0))

    def set2param(self, param, widget):
        param['deblur_filter'] = widget.ids["slider_deblur_filter"].value

    def make_diff(self, img, param, efconfig):
        dbfr = int(param.get('deblur_filter', 0))
        if dbfr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((dbfr))
            if self.hash != param_hash:
                self.diff = core.lucy_richardson_gauss(img, dbfr)
                self.hash = param_hash

        return self.diff


class DefocusEffect(Effect):
    __net = None
    __DRBNet = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set2widget(self, widget, param):
        widget.ids["switch_defocus"].active = False if param.get('defocus', 0) == 0 else True

    def set2param(self, param, widget):
        param['defocus'] = 0 if widget.ids["switch_defocus"].active == False else 1

    def make_diff(self, img, param, efconfig):
        df = param.get('defocus', 0)
        if df <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((df))
            if self.hash != param_hash:
                if DefocusEffect.__DRBNet is None:
                    DefocusEffect.__DRBNet = importlib.import_module('DRBNet')

                if DefocusEffect.__net is None:
                    DefocusEffect.__net = DefocusEffect.__DRBNet.setup_predict()

                self.diff = DefocusEffect.__DRBNet.predict(img, DefocusEffect.__net, 'mps')
                self.hash = param_hash

        return self.diff


class LensblurFilterEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_lensblur_filter"].set_slider_value(param.get('lensblur_filter', 0))

    def set2param(self, param, widget):
        param['lensblur_filter'] = widget.ids["slider_lensblur_filter"].value

    def make_diff(self, img, param, efconfig):
        lpfr = int(param.get('lensblur_filter', 0))
        if lpfr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((lpfr))
            if self.hash != param_hash:
                self.diff = filter.lensblur_filter(img, int(round(lpfr-1) * 4 * efconfig.disp_info[4]))
                self.hash = param_hash

        return self.diff

class ScratchEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_scratch"].set_slider_value(param.get('scratch', 0))

    def set2param(self, param, widget):
        param['scratch'] = widget.ids["slider_scratch"].value

    def make_diff(self, img, param, efconfig):
        fr = int(param.get('scratch', 0))
        if fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.diff = filter.scratch_effect(img, 1.0, fr / 100 * efconfig.disp_info[4])
                self.hash = param_hash

        return self.diff

class FrostedGlassEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_frosted_glass"].set_slider_value(param.get('frosted_glass', 0))

    def set2param(self, param, widget):
        param['frosted_glass'] = widget.ids["slider_frosted_glass"].value

    def make_diff(self, img, param, efconfig):
        fr = int(param.get('frosted_glass', 0))
        if fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.diff = filter.frosted_glass_effect(img, fr / 10, fr / 1000 * efconfig.disp_info[4])
                self.hash = param_hash

        return self.diff

class MosaicEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_mosaic"].set_slider_value(param.get('mosaic', 0))

    def set2param(self, param, widget):
        param['mosaic'] = widget.ids["slider_mosaic"].value

    def make_diff(self, img, param, efconfig):
        fr = int(param.get('mosaic', 0))
        if fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.diff = filter.mosaic_effect(img, int(fr * efconfig.disp_info[4]))
                self.hash = param_hash

        return self.diff

class GlowEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_glow_black"].set_slider_value(param.get('glow_black', 0))
        widget.ids["slider_glow_gauss"].set_slider_value(param.get('glow_gauss', 0))
        widget.ids["slider_glow_opacity"].set_slider_value(param.get('glow_opacity',0))

    def set2param(self, param, widget):
        param['glow_black'] = widget.ids["slider_glow_black"].value
        param['glow_gauss'] = widget.ids["slider_glow_gauss"].value
        param['glow_opacity'] = widget.ids["slider_glow_opacity"].value

    def make_diff(self, rgb, param, efconfig):
        gb = param.get('glow_black', 0)
        gg = int(param.get('glow_gauss', 0))
        go = param.get('glow_opacity', 0)
        if gb == 0 and gg == 0 and go == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((gb, gg, go))
            if self.hash != param_hash:
                hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
                hls[:,:,1] = core.apply_level_adjustment(hls[:,:,1], gb, 127+gg/2, 255)
                rgb2 = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
                if gg > 0:
                    rgb2 = filter.lensblur_filter(rgb2, gg*2-1)
                go = go/100.0
                self.diff = cv2.addWeighted(rgb, 1.0-go, core.blend_screen(rgb, rgb2), go, 0)
                self.hash = param_hash

        return self.diff

class ColorTemperatureEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_color_temperature"].set_slider_value(param.get('color_temperature', 5000))
        widget.ids["slider_color_tint"].set_slider_value(param.get('color_tint', 0))
        widget.ids["slider_color_temperature"].set_slider_reset(param.get('color_temperature_reset', 5000))
        widget.ids["slider_color_tint"].set_slider_reset(param.get('color_tint_reset', 0))
 
    def set2param(self, param, widget):
        param['color_temperature'] = widget.ids["slider_color_temperature"].value
        param['color_tint'] = widget.ids["slider_color_tint"].value

    @staticmethod
    def apply_color_temperature(rgb, param):
        temp = param.get('color_temperature', param.get('color_temperature_reset', 5000))
        tint = param.get('color_tint', param.get('color_tint_reset', 0))
        Y = param.get('color_Y', 1.0)
        return rgb * core.invert_TempTint2RGB(temp, tint, Y, 5000)

    def make_diff(self, rgb, param, efconfig):
        temp = param.get('color_temperature', param.get('color_temperature_reset', 5000))
        tint = param.get('color_tint', param.get('color_tint_reset', 0))
        Y = param.get('color_Y', 1.0)
        if False:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((temp, tint))
            if self.hash != param_hash:
                trgb = core.convert_TempTint2RGB(param['color_temperature_reset'], param['color_tint_reset'], param['color_Y'])
                self.diff = rgb * (trgb / core.convert_TempTint2RGB(temp, tint, Y))
                #self.diff = rgb * np.array(core.invert_TempTint2RGB(temp, tint, Y, 5000), dtype=np.float32)
                self.hash = param_hash

        return self.diff

class DehazeEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_dehaze"].set_slider_value(param.get('dehaze', 0))

    def set2param(self, param, widget):
        param['dehaze'] = widget.ids["slider_dehaze"].value

    def make_diff(self, rgb, param, efconfig):
        de = param.get('dehaze', 0)
        if de == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((de))
            if self.hash != param_hash:
                self.diff = core.dehaze_image(rgb, de/100)
                self.hash = param_hash

        return self.diff

class RGB2HLSEffect(Effect):

    def make_diff(self, rgb, param, efconfig):
        if self.diff is None:
            self.diff = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2HLS_FULL)
            #self.diff = hlsrgb.rgb_to_hls(np.array(rgb))
        return self.diff

class HLS2RGBEffect(Effect):

    def make_diff(self, hls, param, efconfig):
        if self.diff is None:
            self.diff = cv2.cvtColor(np.array(hls), cv2.COLOR_HLS2RGB_FULL)
            #self.diff = hlsrgb.hls_to_rgb(np.array(hls))
        return self.diff

class HLSEffect(Effect):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['hls_red'] = HLSColorEffect('red')
        effecs['hls_orange'] = HLSColorEffect('orange')
        effecs['hls_yellow'] = HLSColorEffect('yellow')
        effecs['hls_green'] = HLSColorEffect('green')
        effecs['hls_cyan'] = HLSColorEffect('cyan')
        effecs['hls_blue'] = HLSColorEffect('blue')
        effecs['hls_purple'] = HLSColorEffect('purple')
        effecs['hls_magenta'] = HLSColorEffect('magenta')
        self.hls_effects = effecs

    def reeffect(self):
        for n in self.hls_effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.hls_effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.hls_effects.values():
            n.set2param(param, widget)

    def make_diff(self, hls, param, efconfig):
        self.diff = pipeline.pipeline_hls(hls, self.hls_effects, param, efconfig)

        return self.diff
    
class HLSColorEffect(Effect):

    def __init__(self, color_name, **kwargs):
        super().__init__(**kwargs)

        self.color_name = color_name

    def set2widget(self, widget, param):
        widget.ids["slider_hls_" + self.color_name + "_hue"].set_slider_value(param.get("hls_" + self.color_name + "_hue", 0))
        widget.ids["slider_hls_" + self.color_name + "_lum"].set_slider_value(param.get("hls_" + self.color_name + "_lum", 0))
        widget.ids["slider_hls_" + self.color_name + "_sat"].set_slider_value(param.get("hls_" + self.color_name + "_sat", 0))

    def set2param(self, param, widget):
        param["hls_" + self.color_name + "_hue"] = widget.ids["slider_hls_" + self.color_name + "_hue"].value
        param["hls_" + self.color_name + "_lum"] = widget.ids["slider_hls_" + self.color_name + "_lum"].value
        param["hls_" + self.color_name + "_sat"] = widget.ids["slider_hls_" + self.color_name + "_sat"].value

    def make_diff(self, hls, param, efconfig):
        hue = param.get("hls_" + self.color_name + "_hue", 0)
        lum = param.get("hls_" + self.color_name + "_lum", 0)
        sat = param.get("hls_" + self.color_name + "_sat", 0)
        param_hash = hash((hue, lum, sat))
        if hue == 0 and lum == 0 and sat == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = core.adjust_hls_color_one(hls, self.color_name, hue, lum/100, sat/100) - hls
            self.hash = param_hash

        return self.diff
    
    def apply_diff(self, hls):
        return hls + self.diff

class ExposureEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_exposure"].set_slider_value(param.get('exposure', 0))

    def set2param(self, param, widget):
        param['exposure'] = widget.ids["slider_exposure"].value

    def make_diff(self, rgb, param, efconfig):
        ev = param.get('exposure', 0)
        param_hash = hash((ev))
        if ev == 0:
            self.diff = None
            self.hash = None
        
        elif self.hash != param_hash:
            self.diff = core.adjust_exposure(rgb, ev)
            self.hash = param_hash

        return self.diff
    
class ContrastEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_contrast"].set_slider_value(param.get('contrast', 0))

    def set2param(self, param, widget):
        param['contrast'] = widget.ids["slider_contrast"].value

    def make_diff(self, rgb, param, efconfig):
        con = param.get('contrast', 0)
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff, _ = core.adjust_tone(rgb, con, -con)
            self.hash = param_hash

        return self.diff

class ClarityEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_clarity"].set_slider_value(param.get('clarity', 0))

    def set2param(self, param, widget):
        param['clarity'] = widget.ids["slider_clarity"].value

    def make_diff(self, rgb, param, efconfig):
        con = param.get('clarity', 0)
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = local_contrast.apply_clarity_luminance(rgb, con * efconfig.disp_info[4])
            self.hash = param_hash

        return self.diff

class TextureEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_texture"].set_slider_value(param.get('texture', 0))

    def set2param(self, param, widget):
        param['texture'] = widget.ids["slider_texture"].value

    def make_diff(self, rgb, param, efconfig):
        con = param.get('texture', 0)
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = local_contrast.apply_texture_advanced(rgb, con * efconfig.disp_info[4])
            self.hash = param_hash

        return self.diff
    
class MicroContrastEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_microcontrast"].set_slider_value(param.get('microcontrast', 0))

    def set2param(self, param, widget):
        param['microcontrast'] = widget.ids["slider_microcontrast"].value

    def make_diff(self, rgb, param, efconfig):
        con = param.get('microcontrast', 0)
        param_hash = hash((con))
        if con == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = local_contrast.apply_microcontrast(rgb, con * efconfig.disp_info[4])
            self.hash = param_hash

        return self.diff
    
class ToneEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_shadow"].set_slider_value(param.get('shadow', 0))
        widget.ids["slider_highlight"].set_slider_value(param.get('highlight', 0))
        widget.ids["slider_midtone"].set_slider_value(param.get('midtone', 0))

    def set2param(self, param, widget):
        param['shadow'] = widget.ids["slider_shadow"].value
        param['highlight'] = widget.ids["slider_highlight"].value
        param['midtone'] = widget.ids["slider_midtone"].value

    def make_diff(self, rgb, param, efconfig):
        shadow = param.get('shadow', 0)
        highlight =  param.get('highlight', 0)
        mt = param.get('midtone', 0)
        param_hash = hash((shadow, highlight, mt))
        if shadow == 0 and highlight == 0 and mt == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff, masks = core.adjust_tone(rgb, highlight, shadow, mt, 0, 0)
            """
            if masks[0] is not None:
                mask = np.array(masks[0])
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                mask = 1 - mask
                threshold = np.max(mask)
                mask[mask > threshold / 2] = 0.0
                source = np.array(self.diff)
                #cv2.imwrite("mask.jpg", cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                #cv2.imwrite("mask.jpg", (mask * 255).astype(np.uint8))
                target = local_contrast.apply_microcontrast(source, 150)
                mask = mask[..., np.newaxis]
                self.diff = source * (1-mask) + target * mask
            """
            if masks[1] is not None:
                mask = np.array(masks[1])
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                threshold = np.max(mask)
                mask[mask < threshold * 3 / 4] = 0.0
                source = np.array(self.diff)
                #cv2.imwrite("mask.jpg", cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                #cv2.imwrite("mask.jpg", (mask * 255).astype(np.uint8))
                target = local_contrast.apply_microcontrast(source, 400)
                mask = mask[..., np.newaxis]
                self.diff = source * (1-mask) + target * mask

            self.hash = param_hash
        return self.diff
    
class HighlightCompressEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_highlight_compress"].active = True if param.get('highlight_compress', 0) == 1 else False

    def set2param(self, param, widget):
        param['highlight_compress'] = 1 if widget.ids["switch_highlight_compress"].active == True else 0

    def make_diff(self, rgb, param, efconfig):
        hc = param.get('highlight_compress', 0)
        if hc <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((hc))
            if self.hash != param_hash:
                self.diff = core.highlight_compress(rgb)
                self.hash = param_hash

        return self.diff

class LevelEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_black_level"].set_slider_value(param.get('black_level', 0))
        widget.ids["slider_white_level"].set_slider_value(param.get('white_level', 255))
        widget.ids["slider_mid_level"].set_slider_value(param.get('mid_level',127))

    def set2param(self, param, widget):
        bl = widget.ids["slider_black_level"].value
        wl = widget.ids["slider_white_level"].value
        ml = widget.ids["slider_mid_level"].value
        param['black_level'] = bl
        param['white_level'] = wl
        param['mid_level'] = ml

    def make_diff(self, rgb, param, efconfig):
        bl = param.get('black_level', 0)
        wl = param.get('white_level', 255)
        ml = param.get('mid_level', 127)
        param_hash = hash((bl, wl, ml))
        if bl == 0 and wl == 255 and ml == 127:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            self.diff = core.apply_level_adjustment(rgb, bl, ml, wl)
            self.hash = param_hash

        return self.diff
    
class CLAHEEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_clahe_intensity"].set_slider_value(param.get('clahe_intensity', 0))

    def set2param(self, param, widget):
        param['clahe_intensity'] = widget.ids["slider_clahe_intensity"].value

    def make_diff(self, img, param, efconfig):
        ci = param.get('clahe_intensity', 0)
        if ci <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((ci))
            if self.hash != param_hash:
                clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
                target = np.zeros_like(img, dtype=np.uint16)
                img2 = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
                for i in range(3):
                    target[..., i] = clahe.apply(img2[..., i])
                target = target.astype(np.float32) / 65535
                ci = ci / 100
                self.diff = target * ci + img * (1-ci)
                self.hash = param_hash

        return self.diff
    
class CurveEffect(Effect):

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

    def reeffect(self):
        for n in self.effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, rgb, param, efconfig):
        self.diff = pipeline.pipeline_curve(rgb, self.effects, param, efconfig)

        return self.diff
    
class TonecurveEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve"].set_point_list(param.get('tonecurve', None))

    def set2param(self, param, widget):
        param['tonecurve'] = widget.ids["tonecurve"].get_point_list()

    def make_diff(self, rgb, param, efconfig):
        pl = param.get('tonecurve', None)
        if pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return core.apply_lut(rgb, self.diff)

class TonecurveRedEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_red"].set_point_list(param.get('tonecurve_red', None))

    def set2param(self, param, widget):
        param['tonecurve_red'] = widget.ids["tonecurve_red"].get_point_list()

    def make_diff(self, rgb_r, param, efconfig):
        pl = param.get('tonecurve_red', None)
        if pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, rgb_r):
        return core.apply_lut(rgb_r, self.diff)

class TonecurveGreenEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_green"].set_point_list(param.get('tonecurve_green', None))

    def set2param(self, param, widget):
        param['tonecurve_green'] = widget.ids["tonecurve_green"].get_point_list()

    def make_diff(self, rgb_g, param, efconfig):   
        pl = param.get('tonecurve_green', None)
        if pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, rgb_g):
        return core.apply_lut(rgb_g, self.diff)

class TonecurveBlueEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["tonecurve_blue"].set_point_list(param.get('tonecurve_blue', None))

    def set2param(self, param, widget):
        param['tonecurve_blue'] = widget.ids["tonecurve_blue"].get_point_list()

    def make_diff(self, rgb_b, param, efconfig):
        pl = param.get('tonecurve_blue', None)
        if pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash(np.sum(pl))
            if self.hash != param_hash:
                self.diff = core.calc_point_list_to_lut(pl)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, rgb_g):
        return core.apply_lut(rgb_g, self.diff)

class GradingEffect(Effect):
    __colorsys = None

    def __init__(self, numstr, **kwargs):
        super().__init__(**kwargs)

        self.numstr = numstr

    def set2widget(self, widget, param):
        widget.ids["grading" + self.numstr].set_point_list(param.get('grading' + self.numstr, None))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_hue'].set_slider_value(param.get('grading' + self.numstr + '_hue', 0))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_lum'].set_slider_value(param.get('grading' + self.numstr + '_lum', 50))
        widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_sat'].set_slider_value(param.get('grading' + self.numstr + '_sat', 0))

    def set2param(self, param, widget):
        param["grading" + self.numstr] = widget.ids["grading" + self.numstr].get_point_list(True)
        param["grading" + self.numstr + "_hue"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_hue'].value
        param["grading" + self.numstr + "_lum"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_lum'].value
        param["grading" + self.numstr + "_sat"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_sat'].value

    def make_diff(self, rgb, param, efconfig):
        pl = param.get("grading" + self.numstr, None)
        gh = param.get("grading" + self.numstr + "_hue", 0)
        gl = param.get("grading" + self.numstr + "_lum", 50)
        gs = param.get("grading" + self.numstr + "_sat", 0)
        if gh == 0 and gl == 50 and gs == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((np.sum(pl), gh, gl, gs))
            if self.hash != param_hash:
                if GradingEffect.__colorsys is None:
                    GradingEffect.__colorsys = importlib.import_module('colorsys')

                lut = core.calc_point_list_to_lut(pl)
                rgbs = np.array(GradingEffect.__colorsys.hls_to_rgb(gh/360.0, gl/100.0, gs/100.0), dtype=np.float32)
                self.diff = (lut, rgbs)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        lut, rgbs = self.diff
        blend = core.apply_lut(rgb, lut)
        blend_inv = 1-blend
        return (rgb*blend_inv + rgb*rgbs*blend)

class VSandSaturationEffect(Effect):

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

    def reeffect(self):
        for n in self.effects.values():
            n.reeffect()

    def set2widget(self, widget, param):
        for n in self.effects.values():
            n.set2widget(widget, param)

    def set2param(self, param, widget):
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, hls, param, efconfig):
        self.diff = pipeline.pipeline_vs_and_saturation(hls, self.effects, param, efconfig)

        return self.diff
    
class HuevsHueEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["HuevsHue"].set_point_list(param.get('HuevsHue', None))

    def set2param(self, param, widget):
        param['HuevsHue'] = widget.ids["HuevsHue"].get_point_list()

    def make_diff(self, hls_h, param, efconfig):
        hh = param.get("HuevsHue", None)
        if hh is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hh))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(hh)
                self.diff = ((lut-0.5)*2.0)*360
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_h):
        return core.apply_lut(hls_h, self.diff, 359) + hls_h

class HuevsLumEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["HuevsLum"].set_point_list(param.get('HuevsLum', None))

    def set2param(self, param, widget):
        param['HuevsLum'] = widget.ids["HuevsLum"].get_point_list()

    def make_diff(self, hls_l, param, efconfig):
        hl = param.get("HuevsLum", None)
        if hl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hl))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(hl)
                self.diff = 2.0**((lut-0.5)*2.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_l):
        return core.apply_lut(hls_l, self.diff, 1.0) * hls_l

class HuevsSatEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["HuevsSat"].set_point_list(param.get('HuevsSat', None))

    def set2param(self, param, widget):
        param['HuevsSat'] = widget.ids["HuevsSat"].get_point_list()

    def make_diff(self, hls_s, param, efconfig):
        hs = param.get("HuevsSat", None)
        if hs is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(hs))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(hs)
                self.diff = (lut-0.5)*2.0+1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_s):
        return core.apply_lut(hls_s, self.diff, 1.0) * hls_s

class LumvsLumEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["LumvsLum"].set_point_list(param.get('LumvsLum', None))

    def set2param(self, param, widget):
        param['LumvsLum'] = widget.ids["LumvsLum"].get_point_list()

    def make_diff(self, hls_l, param, efconfig):
        ll = param.get("LumvsLum", None)
        if ll is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ll))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(ll)
                self.diff = 2.0**((lut-0.5)*2.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_l):
        return core.apply_lut(hls_l, self.diff, 1.0) * hls_l

class LumvsSatEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["LumvsSat"].set_point_list(param.get('LumvsSat', None))

    def set2param(self, param, widget):
        param['LumvsSat'] = widget.ids["LumvsSat"].get_point_list()

    def make_diff(self, hls_l, param, efconfig):
        ls = param.get("LumvsSat", None)
        if ls is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ls))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(ls)
                self.diff = (lut-0.5)*2.0+1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_s):
        return core.apply_lut(hls_s, self.diff, 1.0) * hls_s

class SatvsLumEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["SatvsLum"].set_point_list(param.get('SatvsLum', None))

    def set2param(self, param, widget):
        param['SatvsLum'] = widget.ids["SatvsLum"].get_point_list()

    def make_diff(self, hls_s, param, efconfig):
        sl = param.get("SatvsLum", None)
        if sl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(sl))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(sl)
                self.diff = 2.0**((lut-0.5)*2.0)
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_l):
        return core.apply_lut(hls_l, self.diff, 1.0) * hls_l

class SatvsSatEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["SatvsSat"].set_point_list(param.get('SatvsSat', None))

    def set2param(self, param, widget):
        param['SatvsSat'] = widget.ids["SatvsSat"].get_point_list()

    def make_diff(self, hls_s, param, efconfig):
        ss = param.get("SatvsSat", None)
        if ss is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash(np.sum(ss))
            if self.hash != param_hash:
                lut = core.calc_point_list_to_lut(ss)
                self.diff = (lut-0.5)*2.0+1.0
                self.hash = param_hash

        return self.diff

    def apply_diff(self, hls_s):
        return core.apply_lut(hls_s, self.diff, 1.0) + hls_s

class SaturationEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_saturation"].set_slider_value(param.get('saturation', 0))
        widget.ids["slider_vibrance"].set_slider_value(param.get('vibrance', 0))

    def set2param(self, param, widget):
        param['saturation'] = widget.ids["slider_saturation"].value
        param['vibrance'] = widget.ids["slider_vibrance"].value

    def make_diff(self, hls_s, param, efconfig):
        sat = param.get('saturation', 0)
        vib = param.get('vibrance', 0)
        param_hash = hash((sat, vib))
        if sat == 0 and vib == 0:
            self.diff = None
            self.hash = None

        elif self.hash != param_hash:
            hls2_s = core.calc_saturation(hls_s, sat, vib)
            self.diff = np.divide(hls2_s, hls_s, where=hls_s!=0.0)    # Sのみ保存
            self.hash = param_hash
        
        return self.diff
    
    def apply_diff(self, hls_s):
        return hls_s * self.diff

class LUTEffect(Effect):
    file_pathes = { '---': None }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lut = None

    def set2widget(self, widget, param):
        widget.ids["lut_spinner"].text = param.get('lut_name', 'None')

    def set2param(self, param, widget):
        spinner = widget.ids["lut_spinner"]
        name = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        if param.get('lut_name', "") != name:
            self.lut = None
        param['lut_name'] = name
        param['lut_path'] = LUTEffect.file_pathes.get(param['lut_name'], None)

    def make_diff(self, rgb, param, efconfig):
        lut_path = param.get('lut_path', None)
        if lut_path is None:
            self.diff = None
            self.hash = None
        
        else:
            param_hash = hash((lut_path))
            if self.hash != param_hash:
                if self.lut is None:
                    self.lut = cubelut.read_lut(lut_path)
                self.diff = self.lut
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        return cubelut.process_image(rgb, self.diff)

class LensSimulatorEffect(Effect):
 
    def set2widget(self, widget, param):
        widget.ids["spinner_lens_preset"].text = param.get('lens_preset', 'None')
        widget.ids["slider_lens_intensity"].set_slider_value(param.get('lens_intensity', 100))

    def set2param(self, param, widget):
        spinner = widget.ids["spinner_lens_preset"]
        param['lens_preset'] = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        param['lens_intensity'] = widget.ids["slider_lens_intensity"].value

    def make_diff(self, rgb, param, efconfig):
        preset = param.get('lens_preset', 'None')
        intensity = param.get('lens_intensity', 100)
        if preset == 'None' or intensity <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((preset, intensity))
            if self.hash != param_hash:
                self.diff = (preset, intensity)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        lens = lens_simulator.process_image(rgb, self.diff[0])
        #lens = lens_simulator.apply_old_lens_effect(rgb, self.diff[0])
        per = self.diff[1] / 100.0
        return lens * per + rgb * (1-per)

class FilmSimulationEffect(Effect):
 
    def set2widget(self, widget, param):
        widget.ids["spinner_film_preset"].text = param.get('film_preset', 'None')
        widget.ids["slider_film_intensity"].set_slider_value(param.get('film_intensity', 100))

    def set2param(self, param, widget):
        spinner = widget.ids["spinner_film_preset"]
        param['film_preset'] = spinner.text if spinner.hovered_item is None else spinner.hovered_item.text
        param['film_intensity'] = widget.ids["slider_film_intensity"].value

    def make_diff(self, rgb, param, efconfig):
        preset = param.get('film_preset', 'None')
        intensity = param.get('film_intensity', 100)
        if preset == 'None' or intensity <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((preset, intensity))
            if self.hash != param_hash:
                self.diff = (preset, intensity)
                self.hash = param_hash

        return self.diff
    
    def apply_diff(self, rgb):
        film = film_simulation.simulator.apply_simulation(rgb, self.diff[0])
        per = self.diff[1] / 100.0
        return film * per + rgb * (1-per)

class Mask2Effect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_mask2_hue_distance"].set_slider_value(param.get('mask2_hue_distance', 359))
        widget.ids["slider_mask2_lum_min"].set_slider_value(param.get('mask2_lum_min', 0))
        widget.ids["slider_mask2_lum_max"].set_slider_value(param.get('mask2_lum_max', 255))
        widget.ids["slider_mask2_sat_min"].set_slider_value(param.get('mask2_sat_min', 0))
        widget.ids["slider_mask2_sat_max"].set_slider_value(param.get('mask2_sat_max', 255))
        widget.ids["slider_mask2_blur"].set_slider_value(param.get('mask2_blur', 0))

    def set2param(self, param, widget):
        param['mask2_hue_distance'] = widget.ids["slider_mask2_hue_distance"].value
        param['mask2_lum_min'] = widget.ids["slider_mask2_lum_min"].value
        param['mask2_lum_max'] = widget.ids["slider_mask2_lum_max"].value
        param['mask2_sat_min'] = widget.ids["slider_mask2_sat_min"].value
        param['mask2_sat_max'] = widget.ids["slider_mask2_sat_max"].value
        param['mask2_blur'] = widget.ids["slider_mask2_blur"].value

    def make_diff(self, rgb, param, efconfig):
        hdis = param.get('mask2_hue_distance', 359)
        lmin = param.get('mask2_lum_min', 0)
        lmax = param.get('mask2_lum_max', 255)
        smin = param.get('mask2_sat_min', 0)
        smax = param.get('mask2_sat_max', 255)
        blur = param.get('mask2_blur', 0)
        if  hdis == 359 and lmin == 0 and lmax == 255 and smin == 0 and smax == 255 and blur == 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((hdis, lmin, lmax, smin, smax, blur))
            if self.hash != param_hash:
                self.diff = None
                self.hash = param_hash

        return self.diff

class SolidColorEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["switch_solid_color"].active = False if param.get('solid_color', 0) == 0 else True
        widget.ids["cp_solid_color"].ids['slider_red'].set_slider_value(param.get('solid_color_red', 0))
        widget.ids["cp_solid_color"].ids['slider_green'].set_slider_value(param.get('solid_color_green', 0))
        widget.ids["cp_solid_color"].ids['slider_blue'].set_slider_value(param.get('solid_color_blue', 0))

    def set2param(self, param, widget):
        param['solid_color'] = 0 if widget.ids["switch_solid_color"].active == False else 1
        param["solid_color_red"] = widget.ids["cp_solid_color"].ids['slider_red'].value
        param["solid_color_green"] = widget.ids["cp_solid_color"].ids['slider_green'].value
        param["solid_color_blue"] = widget.ids["cp_solid_color"].ids['slider_blue'].value

    def make_diff(self, rgb, param, efconfig):
        coa = param.get('solid_color', 0)
        coar = param.get("solid_color_red", 0)
        coag = param.get("solid_color_green", 0)
        coab = param.get("solid_color_blue", 0)
        if coa <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((coa, coar, coag, coab))
            if self.hash != param_hash:
                self.diff = core.apply_solid_color(rgb, solid_color=(coar/255, coag/255, coab/255))
                self.hash = param_hash

        return self.diff
    
class VignetteEffect(Effect):

    def set2widget(self, widget, param):
        widget.ids["slider_vignette_intensity"].set_slider_value(param.get('vignette_intensity', 0))
        widget.ids["slider_vignette_radius_percent"].set_slider_value(param.get('vignette_radius_percent', 0))

    def set2param(self, param, widget):
        param['vignette_intensity'] = widget.ids["slider_vignette_intensity"].value
        param['vignette_radius_percent'] = widget.ids["slider_vignette_radius_percent"].value

    def make_diff(self, rgb, param, efconfig):
        vi = param.get('vignette_intensity', 0)
        vr = param.get('vignette_radius_percent', 0)
        pce = param.get('crop_enable', False)
        if (vi == 0 and vr == 0) or pce == True:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((vi, vr))
            if self.hash != param_hash:
                _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(config.get_config('preview_size'), config.get_config('preview_size'), efconfig.disp_info)
                self.diff = core.apply_vignette(rgb, vi, vr, efconfig.disp_info, params.get_crop_rect(param), (offset_x, offset_y))
                self.hash = param_hash
        
        return self.diff
    

def create_effects():
    effects = [{}, {}, {}, {}, {}]

    lv0 = effects[0]
    #lv0['lens_modifier'] = LensModifierEffect()
    lv0['subpixel_shift'] = SubpixelShiftEffect()
    lv0['inpaint'] = InpaintEffect()
    lv0['rotation'] = RotationEffect()
    lv0['crop'] = CropEffect()

    lv1 = effects[1]
    lv1['ai_noise_reduction'] = AINoiseReductonEffect()
    lv1['nlm_noise_reduction'] = NLMNoiseReductionEffect()
    lv1['light_noise_reduction'] = LightNoiseReductionEffect()
    lv1['deblur_filter'] = DeblurFilterEffect()
    lv1['defocus'] = DefocusEffect()
    lv1['lensblur_filter'] = LensblurFilterEffect()
    lv1['scratch'] = ScratchEffect()
    lv1['frosted_glass'] = FrostedGlassEffect()
    lv1['mosaic'] = MosaicEffect()
    lv1['glow'] = GlowEffect()
    
    lv2 = effects[2]
    lv2['color_temperature'] = ColorTemperatureEffect()
    lv2['dehaze'] = DehazeEffect()
 
    lv2['exposure'] = ExposureEffect()
    lv2['contrast'] = ContrastEffect()
    lv2['clarity'] = ClarityEffect()
    lv2['texture'] = TextureEffect()
    lv2['microcontrast'] = MicroContrastEffect()
    lv2['tone'] = ToneEffect()

    lv2['highlight_compress'] = HighlightCompressEffect()

    # ここでクリッピング

    #lv2['rgb2hls1'] = RGB2HLSEffect()
    #lv2['hls2rgb1'] = HLS2RGBEffect()

    lv2['level'] = LevelEffect()
    lv2['clahe'] = CLAHEEffect()

    lv2['rgb2hls2'] = RGB2HLSEffect()
    lv2['hls'] = HLSEffect()
    lv2['vs_and_saturation'] = VSandSaturationEffect()
    lv2['hls2rgb2'] = HLS2RGBEffect()

    lv2['curve'] = CurveEffect()

    lv2['lut'] = LUTEffect()
    lv2['lens_simulator'] = LensSimulatorEffect()
    lv2['film_simulation'] = FilmSimulationEffect()
    lv2['solid_color'] = SolidColorEffect()

    lv3 = effects[3]
    lv3['mask2'] = Mask2Effect()

    lv4 = effects[4]
    lv4['vignette'] = VignetteEffect()

    return effects

def set2widget_all(widget, effects, param):
    for dict in effects:
        for l in dict.values():
            l.set2widget(widget, param)
            #l.set2param(param, self)
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
