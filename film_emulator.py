import json
from unittest.case import enterModuleContext
import numpy as np
import cv2
from scipy.interpolate import interp1d

import core

class FilmEmulator:

    def __init__(self, params_path):
        with open(params_path, 'r') as f:
            self.film_presets = json.load(f)

    def get_presets(self):
        return self.film_presets
        
    def apply_film_effect(self, img_rgb, film_name, expired=0):
        import utils
        
        # パラメータ取得
        params = self.film_presets[film_name]
        img = img_rgb.copy()
        utils.print_nan_inf(img)
        
        # 処理ステップごとにクリッピングを追加
        img = self.apply_tone_curves(img, params['tone_curves'])
        utils.print_nan_inf(img)

        img = self.apply_color_adjustment(img, params['color_adjustment'])
        utils.print_nan_inf(img)
        
        img = np.clip(img, 0, 1) # NaNを防ぐためにクリッピング
        img = self.apply_contrast_compression(img, params['contrast'])
        utils.print_nan_inf(img)
        
        img = self.apply_base_color(img, params['base_color'])
        utils.print_nan_inf(img)
        
        if expired > 0:
            img = self.apply_expired_effects(img, params.get('expired_effects', {}), expired)
            utils.print_nan_inf(img)
        
        return img
    
    def apply_tone_curves(self, img, curves):
        for ch, curve in enumerate(curves):
            # 制御点をx値でソート
            sorted_curve = sorted(curve, key=lambda x: x[0])
            x = np.array([p[0] for p in sorted_curve], dtype=np.float32)
            y = np.array([p[1] for p in sorted_curve], dtype=np.float32)
            
            # 外挿を防ぐため範囲制限
            interpolator = interp1d(
                x, y, 
                kind='linear',  # 線形補間に変更
                bounds_error=False, 
                fill_value=(y[0], y[-1])
            )
            img[..., ch] = interpolator(img[..., ch])
        return img
    
    def apply_color_adjustment(self, img, adjustment):
        # チャンネル別ゲイン (安全な範囲に制限)
        gains = np.clip(adjustment['channel_gain'], 0.5, 1.5)
        img *= np.array(gains, dtype=np.float32)
        
        # 色相シフトの修正 (RGB空間で直接処理)
        h_shift = adjustment['hue_shift'] / 180.0  # -1.0〜1.0に正規化
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        
        # 色相シフト行列
        matrix = np.array([
            [1.0 - abs(h_shift), h_shift if h_shift < 0 else 0, 0],
            [0, 1.0 - abs(h_shift), h_shift if h_shift > 0 else 0],
            [h_shift if h_shift > 0 else 0, 0, 1.0 - abs(h_shift)]
        ])
        
        # 行列適用
        img = np.stack([
            matrix[0,0]*r + matrix[0,1]*g + matrix[0,2]*b,
            matrix[1,0]*r + matrix[1,1]*g + matrix[1,2]*b,
            matrix[2,0]*r + matrix[2,1]*g + matrix[2,2]*b
        ], axis=-1)
        
        # 彩度調整 (安全な範囲に制限)
        sat_factor = np.clip(adjustment['saturation'], 0.5, 2.0)
        img = self.adjust_saturation(img, sat_factor)
        
        return img
    
    def adjust_saturation(self, img, factor):
        # グレースケール変換を正確化
        gray = core.cvtColorRGB2Gray(img)
        gray = np.stack([gray, gray, gray], axis=-1)
        return gray + factor * (img - gray)
    
    def apply_contrast_compression(self, img, params):
        # 安全なパラメータ範囲に制限
        hl_thresh = np.clip(params['highlight_threshold'], 0.7, 0.95)
        hl_power = np.clip(params['highlight_power'], 1.0, 3.0)
        sd_thresh = np.clip(params['shadow_threshold'], 0.05, 0.3)
        sd_power = np.clip(params['shadow_power'], 1.0, 3.0)
        
        # ハイライト圧縮の修正
        hl_mask = img > hl_thresh
        if np.any(hl_mask):
            # 正規化して圧縮
            normalized = (img[hl_mask] - hl_thresh) / (1 - hl_thresh)
            compressed = 1 - (1 - normalized) ** hl_power
            img[hl_mask] = hl_thresh + compressed * (1 - hl_thresh)
        
        # シャドウ圧縮の修正
        sd_mask = img < sd_thresh
        if np.any(sd_mask):
            normalized = img[sd_mask] / sd_thresh
            compressed = normalized ** (1 / sd_power)
            img[sd_mask] = compressed * sd_thresh
        
        return img
    
    def apply_base_color(self, img, base_color):
        if len(base_color) != 4:
            raise ValueError("base_color must have 4 values [R, G, B, alpha]")
        
        alpha = min(max(base_color[3], 0.0), 0.2)  # アルファ値を安全な範囲に制限
        rgb_color = np.array(base_color[:3], dtype=np.float32)
        
        # オーバーレイの代わりにソフトライトブレンド
        return img * (1 - alpha) + rgb_color * alpha
    
    def apply_expired_effects(self, img, params, expired):
        # 色シフトの強度を制限
        shift_range = params.get('color_shift_range', [-0.05, 0.05])
        min_shift = max(shift_range[0], -0.1)
        max_shift = min(shift_range[1], 0.1)
        np.random.seed(int(expired) * 2) # 乱数指定
        img += np.random.uniform(min_shift, max_shift, size=3)
        
        # 化学的シミの強度制限
        if 'stain_intensity' in params:
            intensity = min(params['stain_intensity'], 0.3)
            stains = self.generate_stains(img.shape[:2], params)
            img = np.where(
                stains[..., np.newaxis] > 0, 
                img * (1 - stains[..., np.newaxis] * intensity), 
                img
            )
        
        return img
    
    def create_degradation_map(self, h, w, params):
        scale = params.get('uneven_scale', 100)
        intensity = min(params.get('uneven_intensity', 0.2), 0.3)
        
        x = np.linspace(0, 4*np.pi, w)
        y = np.linspace(0, 4*np.pi, h)
        xx, yy = np.meshgrid(x, y)
        
        pattern = np.sin(xx) * np.cos(yy) * 0.5 + 0.5
        return 0.9 + 0.1 * pattern  # 常に0.9-1.0の範囲
    
    def generate_stains(self, shape, params):
        h, w = shape
        stains = np.zeros(shape, dtype=np.float32)
        density = min(params.get('stain_density', 0.0001), 0.001)
        max_size = min(params.get('stain_max_size', 30), 50)
        
        num_stains = int(h * w * density)
        for _ in range(num_stains):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            radius = np.random.randint(5, max_size)
            stain = np.zeros(shape, dtype=np.float32)
            cv2.circle(stain, (x, y), radius, 1.0, -1)
            stain = core.gaussian_blur_cv(stain, (0, 0), radius/3)
            stains = np.maximum(stains, stain)
        
        return stains
    
    def generate_cosmic_spots(self, shape, params):
        density = min(params.get('cosmic_spots', 0.0001), 0.001)
        spots = np.random.random(shape) < density
        return spots

emulator = FilmEmulator("film_presets.json")

