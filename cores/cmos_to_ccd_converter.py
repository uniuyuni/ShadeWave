
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d
import warnings
warnings.filterwarnings('ignore')

class CMOSToCCDConverter:
    """
    CMOSセンサーで撮影した画像をCCDセンサーのような画像に変換する包括的なパイプライン
    """
    
    def __init__(self, img_rgb_float32):
        """
        Parameters:
        -----------
        img_rgb_float32 : numpy.ndarray
            RGB色空間のfloat32画像 (値域: 0.0-1.0 or 0-255)
        """
        # 入力値域の正規化（0-1に統一）
        self.original = img_rgb_float32.copy()
        if self.original.max() > 1.5:
            self.original = self.original / 255.0
        
        self.img = self.original.copy()
        self.height, self.width = self.img.shape[:2]
        
    def convert_to_lab(self, img_rgb):
        """RGB画像をLab色空間に変換"""
        img_uint8 = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return img_lab / 255.0
    
    def convert_from_lab(self, img_lab):
        """Lab画像をRGB色空間に変換"""
        img_lab_uint8 = np.clip(img_lab * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_lab_uint8, cv2.COLOR_LAB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb.astype(np.float32) / 255.0
    
    def bilateral_filter(self, img, d, sigma_color, sigma_spatial):
        """バイラテラルフィルタを適用"""
        return cv2.bilateralFilter(img, d, sigma_color, sigma_spatial)
    
    def chroma_denoise(self, strength=1.5, window_size=5):
        """
        ステップ1: クロマデノイジング
        色情報（クロマ）のノイズを軽減する処理
        ルミナンス（明度）は保持
        """
        print("Processing: Chroma Denoising...")
        
        # Lab色空間に変換（L: ルミナンス, a,b: クロマ）
        img_lab = self.convert_to_lab(self.img)
        
        # 分離
        L = img_lab[:, :, 0]
        a = img_lab[:, :, 1]
        b = img_lab[:, :, 2]
        
        # a, b チャンネルにバイラテラルフィルタを適用
        # 両側フィルタは色情報を保持しながらノイズを除去
        a_filtered = self.bilateral_filter(a, 9, sigma_color=0.1 * strength, 
                                     sigma_spatial=window_size * strength)
        b_filtered = self.bilateral_filter(b, 9, sigma_color=0.1 * strength, 
                                     sigma_spatial=window_size * strength)
        
        # 再構成
        img_lab[:, :, 0] = L
        img_lab[:, :, 1] = a_filtered
        img_lab[:, :, 2] = b_filtered
        
        self.img = self.convert_from_lab(img_lab)
        return self
    
    def hot_pixel_removal(self, threshold_percentile=99.5):
        """
        ステップ2: ホットピクセル除去
        CMOS固有のホットピクセル（異常に明るいノイズ）を除去
        """
        print("Processing: Hot Pixel Removal...")
        
        # 各チャンネルごと処理
        for c in range(3):
            channel = self.img[:, :, c]
            
            # 局所的な異常値検出（3x3近傍との比較）
            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    neighborhood = channel[i-1:i+2, j-1:j+2]
                    local_mean = np.mean(neighborhood)
                    local_std = np.std(neighborhood)
                    
                    if channel[i, j] > local_mean + 3.0 * local_std:
                        # ホットピクセルの可能性が高い
                        channel[i, j] = np.median(neighborhood)
            
            self.img[:, :, c] = channel
        
        return self
    
    def dark_current_correction(self, dark_level=0.01):
        """
        ステップ3: 暗電流補正
        CMOSセンサーの暗電流（オフセット）を補正
        """
        print("Processing: Dark Current Correction...")
        
        # 暗電流の推定（画像全体の最小値付近）
        min_val = np.percentile(self.img, 0.5)
        correction_offset = max(0, dark_level - min_val)
        
        # オフセット除去
        self.img = np.clip(self.img - correction_offset, 0, 1.0)
        
        # 明度の再正規化
        self.img = self.img / (self.img.max() + 1e-8)
        
        return self
    
    def color_uniformity_correction(self, grid_size=8):
        """
        ステップ4: 色均一性補正
        CMOSの色ムラ（各画素のゲイン差）を補正
        """
        print("Processing: Color Uniformity Correction...")
        
        # グリッド分割による局所的な色統計計算
        block_height = self.height // grid_size
        block_width = self.width // grid_size
        
        # 色温度マップ（各ブロックの色バランス）の計算
        gain_map_r = np.ones_like(self.img[:, :, 0])
        gain_map_g = np.ones_like(self.img[:, :, 1])
        gain_map_b = np.ones_like(self.img[:, :, 2])
        
        reference_mean_r = np.mean(self.img[:, :, 0])
        reference_mean_g = np.mean(self.img[:, :, 1])
        reference_mean_b = np.mean(self.img[:, :, 2])
        
        # 各ブロックのゲイン補正マップを作成
        for bi in range(grid_size):
            for bj in range(grid_size):
                yi_start = bi * block_height
                yi_end = (bi + 1) * block_height if bi < grid_size - 1 else self.height
                xi_start = bj * block_width
                xi_end = (bj + 1) * block_width if bj < grid_size - 1 else self.width
                
                block_r = self.img[yi_start:yi_end, xi_start:xi_end, 0]
                block_g = self.img[yi_start:yi_end, xi_start:xi_end, 1]
                block_b = self.img[yi_start:yi_end, xi_start:xi_end, 2]
                
                mean_r = np.mean(block_r) + 1e-8
                mean_g = np.mean(block_g) + 1e-8
                mean_b = np.mean(block_b) + 1e-8
                
                gain_map_r[yi_start:yi_end, xi_start:xi_end] = reference_mean_r / mean_r
                gain_map_g[yi_start:yi_end, xi_start:xi_end] = reference_mean_g / mean_g
                gain_map_b[yi_start:yi_end, xi_start:xi_end] = reference_mean_b / mean_b
        
        # ゲインマップを平滑化（グリッド境界のアーティファクト除去）
        gain_map_r = gaussian_filter(gain_map_r, sigma=block_height/2)
        gain_map_g = gaussian_filter(gain_map_g, sigma=block_height/2)
        gain_map_b = gaussian_filter(gain_map_b, sigma=block_height/2)
        
        # ゲイン適用
        self.img[:, :, 0] = np.clip(self.img[:, :, 0] * gain_map_r, 0, 1.0)
        self.img[:, :, 1] = np.clip(self.img[:, :, 1] * gain_map_g, 0, 1.0)
        self.img[:, :, 2] = np.clip(self.img[:, :, 2] * gain_map_b, 0, 1.0)
        
        return self
    
    def tone_mapping_expansion(self, strength=1.2):
        """
        ステップ5: ダイナミックレンジ拡張
        見かけ上のダイナミックレンジを拡大し、暗部の階調を活かす
        """
        print("Processing: Tone Mapping Expansion...")
        
        # 非線形トーン曲線の適用
        # ガンマ補正をベースに、暗部を持ち上げる
        gamma = 1.0 / strength
        self.img = np.power(self.img, gamma)
        
        # 局所的なコントラスト強化（アンシャープマスク）
        self.img_blurred = gaussian_filter(self.img, sigma=2.0)
        local_contrast = self.img - self.img_blurred
        
        # 局所コントラストを追加（強度を制御）
        self.img = self.img + 0.3 * local_contrast
        self.img = np.clip(self.img, 0, 1.0)
        
        return self
    
    def color_cast_correction(self, target_temperature=6500):
        """
        ステップ6: 色かぶり補正
        色温度を調整してCCDのような色再現を実現
        """
        print("Processing: Color Cast Correction...")
        
        # ホワイトバランス統計（グレーの領域を参照）
        img_rgb = np.clip(self.img, 0, 1.0)
        
        # 色温度による色相変化を模擬（簡易的なホワイトバランス）
        # 6500Kは昼光色（標準）
        # より暖色（低色温度）にするにはRを強調、青を弱める
        
        mean_r = np.mean(self.img[:, :, 0])
        mean_g = np.mean(self.img[:, :, 1])
        mean_b = np.mean(self.img[:, :, 2])
        
        luminance = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b
        
        # 色温度補正係数
        temp_factor = (target_temperature - 6500) / 1000.0
        r_corr = 1.0 + 0.05 * temp_factor
        b_corr = 1.0 - 0.05 * temp_factor
        g_corr = 1.0
        
        self.img[:, :, 0] = np.clip(self.img[:, :, 0] * r_corr, 0, 1.0)
        self.img[:, :, 1] = np.clip(self.img[:, :, 1] * g_corr, 0, 1.0)
        self.img[:, :, 2] = np.clip(self.img[:, :, 2] * b_corr, 0, 1.0)
        
        # 明度の再正規化
        new_luminance = np.mean(self.img)
        if new_luminance > 1e-8:
            self.img = self.img * (luminance / new_luminance)
        
        self.img = np.clip(self.img, 0, 1.0)
        
        return self
    
    def color_saturation_adjustment(self, saturation_boost=1.1):
        """
        ステップ7: 彩度調整
        CCDのようなナチュラルな彩度にする
        """
        print("Processing: Saturation Adjustment...")
        
        img_lab = self.convert_to_lab(self.img)
        
        # a, b チャンネル（彩度情報）を調整
        # グレー色（L=50, a=0.5, b=0.5）からの距離を彩度として扱う
        a = img_lab[:, :, 1]
        b = img_lab[:, :, 2]
        
        # 彩度を調整（a=0.5, b=0.5が中立）
        a_adjusted = 0.5 + (a - 0.5) * saturation_boost
        b_adjusted = 0.5 + (b - 0.5) * saturation_boost
        
        img_lab[:, :, 1] = np.clip(a_adjusted, 0, 1.0)
        img_lab[:, :, 2] = np.clip(b_adjusted, 0, 1.0)
        
        self.img = self.convert_from_lab(img_lab)
        
        return self
    
    def noise_profile_learning_simulation(self, strength=0.8):
        """
        ステップ8: ノイズプロファイル学習の模擬
        CMOSのノイズパターンを分析して除去
        """
        print("Processing: Noise Profile Learning Simulation...")
        
        # ラプラシアンフィルタでエッジを検出
        kernel_laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        
        for c in range(3):
            channel = self.img[:, :, c]
            edges = cv2.filter2D(channel, -1, kernel_laplacian)
            
            # エッジ領域のマスク生成
            edge_mask = np.abs(edges) > np.percentile(np.abs(edges), 80)
            
            # エッジ以外の領域にノイズ除去を適用
            channel_denoised = medfilt2d(channel, kernel_size=3)
            
            # マスクベースのブレンド
            self.img[:, :, c] = np.where(
                edge_mask,
                channel * (1 - strength * 0.3) + channel_denoised * (strength * 0.3),
                channel * (1 - strength) + channel_denoised * strength
            )
        
        self.img = np.clip(self.img, 0, 1.0)
        
        return self
    
    def micro_contrast_enhancement(self, strength=0.15):
        """
        ステップ9: マイクロコントラスト強化
        細部のコントラストを自然に強化
        """
        print("Processing: Micro Contrast Enhancement...")
        
        # 複数スケールでのアンシャープマスク
        original_img = self.img.copy()
        
        for scale in [1.0, 2.0, 4.0]:
            blurred = gaussian_filter(self.img, sigma=scale)
            micro_contrast = self.img - blurred
            self.img = self.img + strength * micro_contrast / 3.0
        
        self.img = np.clip(self.img, 0, 1.0)
        
        # 元画像とのブレンド（自然さの維持）
        self.img = original_img * 0.7 + self.img * 0.3
        self.img = np.clip(self.img, 0, 1.0)
        
        return self
    
    def final_color_grading(self):
        """
        ステップ10: 最終的なカラーグレーディング
        CCDのような温かみのある色再現を実現
        """
        print("Processing: Final Color Grading...")
        
        img_lab = self.convert_to_lab(self.img)
        
        # わずかなコントラスト調整（Lab色空間）
        L = img_lab[:, :, 0]
        L_min = np.percentile(L, 2)
        L_max = np.percentile(L, 98)
        
        # コントラスト拡張
        L_expanded = (L - L_min) / (L_max - L_min + 1e-8)
        L_expanded = np.clip(L_expanded, 0, 1.0)
        
        # S字カーブを軽く適用
        L_graded = 0.5 + 1.2 * (L_expanded - 0.5)
        L_graded = np.clip(L_graded, 0, 1.0)
        
        img_lab[:, :, 0] = L_graded
        
        self.img = self.convert_from_lab(img_lab)
        
        return self
    
    def process(self, chroma_strength=1.5, saturation_boost=1.08, 
                tone_mapping_strength=1.15, noise_removal_strength=0.75):
        """
        フル処理パイプラインの実行
        
        Parameters:
        -----------
        chroma_strength : float
            クロマデノイジングの強度（1.0-3.0）
        saturation_boost : float
            彩度調整の強度（0.8-1.3）
        tone_mapping_strength : float
            トーンマッピングの強度（1.0-1.5）
        noise_removal_strength : float
            ノイズ除去の強度（0.0-1.0）
        
        Returns:
        --------
        numpy.ndarray : 処理済みのRGB画像（float32, 0-1範囲）
        """
        print("\n=== CMOS to CCD-like Conversion Pipeline ===\n")
        
        (self.chroma_denoise(strength=chroma_strength)
#             .hot_pixel_removal()
             .dark_current_correction(dark_level=0.01)
             .color_uniformity_correction(grid_size=8)
             .tone_mapping_expansion(strength=tone_mapping_strength)
             .color_cast_correction(target_temperature=6500)
             .color_saturation_adjustment(saturation_boost=saturation_boost)
             .noise_profile_learning_simulation(strength=noise_removal_strength)
             .micro_contrast_enhancement(strength=0.15)
             .final_color_grading())
        
        print("\n=== Processing Complete ===\n")
        
        return np.clip(self.img, 0, 1.0)


# 使用例
if __name__ == "__main__":
    # テスト画像の生成または読み込み
    test_img = cv2.imread("your_image.jpg")
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    
    # 変換処理の実行
    converter = CMOSToCCDConverter(test_img)
    result = converter.process(
        chroma_strength=1.5,
        saturation_boost=1.08,
        tone_mapping_strength=1.15,
        noise_removal_strength=0.75
    )
    
    test_img = cv2.cvtColor((test_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("your_image_out.jpg", test_img)
