import logging

import numpy as np
import cv2
from typing import Dict, Tuple, Optional


class CoatingSimulator:
    """
    レンズコーティング特性シミュレーター
    
    コーティングの種類による「色乗り」「フレア耐性」「コントラスト」の違いを再現します。
    """
    
    def __init__(self):
        # プリセット定義（行列は RGB に対する影響、float32 固定）
        f32 = np.float32
        self.presets = {
            # 単層コーティング〜無コーティング（オールドレンズ風）
            'VINTAGE_NO_COAT': {
                'color_matrix': np.array([
                    [1.05, 0.05, 0.05],  # 赤：少し強調
                    [0.05, 0.95, 0.05],  # 緑：少し減衰（黄ばみ）
                    [0.05, 0.05, 0.85]   # 青：大きく減衰（青抜け）
                ], dtype=f32),
                'flare_factor': 0.15,    # フレアが多い
                'contrast_factor': 0.85, # コントラスト低め
                'saturation_factor': 0.9,
                'name': "Vintage No-Coat"
            },
            # 現代のマルチコーティング（ニュートラル）
            'MODERN_MULTI_COAT': {
                'color_matrix': np.array([
                    [1.00, 0.00, 0.00],
                    [0.00, 1.00, 0.00],
                    [0.00, 0.00, 1.00]
                ], dtype=f32),
                'flare_factor': 0.02,    # フレア极少
                'contrast_factor': 1.05, # コントラスト高め
                'saturation_factor': 1.0,
                'name': "Modern Multi-Coat"
            },
            # ライカ風（赤の発色が良く、微コントラストが高い）
            'LEICA_CLASSIC': {
                'color_matrix': np.array([
                    [1.08, 0.02, 0.02],  # 赤：豊かに
                    [0.02, 0.98, 0.02],  # 緑：自然
                    [0.02, 0.02, 0.95]   # 青：少し抑えめ
                ], dtype=f32),
                'flare_factor': 0.05,    # 適度な耐性
                'contrast_factor': 1.10, # マイクロコントラスト高
                'saturation_factor': 1.05,
                'name': "Leica Classic"
            },
            # ツァイス T* コーティング風（青みがかり、コントラスト鋭い）
            'ZEISS_TSTAR': {
                'color_matrix': np.array([
                    [0.95, 0.02, 0.02],
                    [0.02, 1.02, 0.02],
                    [0.02, 0.02, 1.05]   # 青：強調
                ], dtype=f32),
                'flare_factor': 0.03,
                'contrast_factor': 1.15,
                'saturation_factor': 1.1,
                'name': "Zeiss T*"
            },
            # キヤノン風（暖色系、柔らかい）
            'CANON_L': {
                'color_matrix': np.array([
                    [1.05, 0.03, 0.03],
                    [0.03, 1.00, 0.03],
                    [0.03, 0.03, 0.95]
                ], dtype=f32),
                'flare_factor': 0.04,
                'contrast_factor': 0.95,
                'saturation_factor': 1.05,
                'name': "Canon L"
            }
        }
    
    def apply_color_matrix(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        透過スペクトルによる色キャストを適用
        
        Args:
            image: float32, RGB, HDR (H, W, 3)
            matrix: 3x3 カラーマトリクス
        
        Returns:
            色変換された画像
        """
        # 行列演算で RGB チャンネルを混合（float32 行列で float32 を維持）
        m = np.asarray(matrix, dtype=np.float32)
        return np.matmul(image, m.T)
    
    def apply_veiling_glare(self, image: np.ndarray, flare_factor: float, resolution_scale: float = 1.0) -> np.ndarray:
        """
        ベーリングフレア（光の散乱）をシミュレート
        
        コーティング性能が悪いと、強い光が画面全体に散乱し、
        黒が浮いてコントラストが低下する現象を再現。
        
        Args:
            image: float32, RGB, HDR
            flare_factor: 0.0(無し)-1.0(最強)
        
        Returns:
            フレアを適用した画像
        """
        if flare_factor <= 0.0:
            return image
        
        # 画像の平均輝度を計算（光源の強さの代わり）
        luminance = np.mean(image, axis=2, keepdims=True, dtype=np.float32)
        
        # 輝度の高い部分ほどフレアが強くなるようにマスク作成
        # 簡易的にガウシアンブラーで光の広がりを表現(縮尺対応)
        glow = cv2.GaussianBlur(luminance, (0, 0), sigmaX=max(1.0, 50.0 * float(resolution_scale)))
        if glow.ndim == 2:
            glow = glow[:, :, np.newaxis]
        glow = np.asarray(glow, dtype=np.float32)
        
        # フレアの色（通常は白っぽいが、コーティング残留色で少し色づく）
        flare_color = np.array([1.0, 0.95, 0.9], dtype=np.float32).reshape(1, 1, 3) # 暖色系のフレア
        
        # 元の画像にフレア成分を足し込む
        # 黒レベルを持ち上げる効果を含む
        flare_intensity = np.float32(flare_factor * 0.2)
        result = image + (glow * flare_color * flare_intensity)
        
        return result
    
    def apply_micro_contrast(self, image: np.ndarray, contrast_factor: float, resolution_scale: float = 1.0) -> np.ndarray:
        """
        マイクロコントラスト（立体感）の調整
        
        コーティングによる内部反射の抑制具合を、ローカルコントラストで表現。
        
        Args:
            image: float32, RGB, HDR
            contrast_factor: 1.0(標準), >1.0(鋭い), <1.0(柔らかい)
        
        Returns:
            コントラスト調整済み画像
        """
        if abs(contrast_factor - 1.0) < 0.01:
            return image
        
        # 輝度チャンネルを分離
        luminance = np.mean(image, axis=2, keepdims=True, dtype=np.float32)
        
        # アンシャープマスク的な手法でローカルコントラストを強調/抑制
        # ぼかし画像を作成(縮尺対応)
        blurred = cv2.GaussianBlur(luminance, (0, 0), sigmaX=max(1.0, 10.0 * float(resolution_scale)))
        if blurred.ndim == 2:
            blurred = blurred[:, :, np.newaxis]
        blurred = np.asarray(blurred, dtype=np.float32)
        
        # 詳細成分（ハイパス）
        detail = luminance - blurred
        
        # 詳細成分をスケールして戻す
        enhanced_luminance = blurred + detail * np.float32(contrast_factor)
        
        # 元の画像に比率を適用（色相は保つ）
        ratio = enhanced_luminance / (luminance + 1e-6)
        result = image * ratio
        
        return result
    
    def apply_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """彩度調整"""
        if abs(factor - 1.0) < 0.01:
            return image
        
        # 輝度を計算
        luminance = np.mean(image, axis=2, keepdims=True, dtype=np.float32)
        
        # 彩度補間
        result = luminance + (image - luminance) * np.float32(factor)
        return result
    
    def apply_preset(self, image: np.ndarray, preset_name: str,
                     light_source_intensity: float = 1.0,
                     resolution_scale: float = 1.0) -> np.ndarray:
        """
        定義されたプリセットを一键適用
        
        Args:
            image: float32, RGB, HDR
            preset_name: プリセット名（例：'VINTAGE_NO_COAT'）
            light_source_intensity: 光源の強さ（フレアに影響）
        
        Returns:
            変換された画像
        """
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        preset = self.presets[preset_name]
        result = image.copy()
        
        # 1. 透過色キャスト（コーティングの色乗り）
        result = self.apply_color_matrix(result, preset['color_matrix'])
        
        # 2. フレア耐性（光の散乱）
        # 光源が強いほどフレア効果が増すように調整
        effective_flare = float(preset['flare_factor'] * light_source_intensity)
        result = self.apply_veiling_glare(result, effective_flare, resolution_scale=resolution_scale)

        # 3. マイクロコントラスト（内部反射の抑制）
        result = self.apply_micro_contrast(result, preset['contrast_factor'], resolution_scale=resolution_scale)
        
        # 4. 彩度（発色の傾向）
        result = self.apply_saturation(result, preset['saturation_factor'])
        
        return result


# ============================================================================
# 使用例
# ============================================================================

if __name__ == "__main__":
    import imageio
    
    # 1. 画像読み込み（HDR 推奨）
    # image = imageio.imread("input.exr").astype(np.float32) / 65535.0 
    # 簡易テスト用ダミー画像作成
    h, w = 1080, 1920
    image = np.random.rand(h, w, 3).astype(np.float32) * 0.5
    # 光源っぽい明るい部分を作る
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    mask = np.exp(-((x-cx)**2 + (y-cy)**2) / (200**2))
    image += mask[:,:,np.newaxis] * 2.0  # HDR 値
    
    simulator = CoatingSimulator()
    
    # 2. 異なるコーティング特性を適用
    vintage_img = simulator.apply_preset(image, 'VINTAGE_NO_COAT', light_source_intensity=1.5)
    leica_img = simulator.apply_preset(image, 'LEICA_CLASSIC', light_source_intensity=1.0)
    modern_img = simulator.apply_preset(image, 'MODERN_MULTI_COAT', light_source_intensity=1.0)
    
    # 3. 保存（例）
    # imageio.imwrite("vintage.exr", (vintage_img * 65535).astype(np.uint16))
    # imageio.imwrite("leica.exr", (leica_img * 65535).astype(np.uint16))
    
    logging.info("コーティングシミュレーション完了！")
    logging.info("利用可能なプリセット：%s", list(simulator.presets.keys()))
