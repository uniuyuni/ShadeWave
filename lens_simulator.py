import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class LensCharacteristics:
    name: str
    temperature: float  # 色温度 (-100 to 100)
    saturation: float  # 彩度 (-100 to 100)
    contrast: float    # コントラスト (-100 to 100)
    vignette: float   # ケラレ強度 (0 to 1)
    sharpness: float  # シャープネス (-100 to 100)
    red_balance: float  # 赤の強さ (-100 to 100)
    green_balance: float  # 緑の強さ (-100 to 100)
    blue_balance: float   # 青の強さ (-100 to 100)
    color_cast: Tuple[float, float, float]  # 色かぶり (B, G, R)
    cast_strength: float  # 色かぶりの強さ (0 to 1)

class LensSimulator:
    LENS_PRESETS = {
        # ロシア/ソビエトレンズ
        "Helios 44-2 58mm f/2": LensCharacteristics(
            name="Helios 44-2 58mm f/2",
            temperature=30,
            saturation=20,
            contrast=30,
            vignette=0.4,
            sharpness=-10,
            red_balance=15,
            green_balance=-5,
            blue_balance=-10,
            color_cast=(20, 30, 40),
            cast_strength=0.15
        ),
        "Jupiter-9 85mm f/2": LensCharacteristics(
            name="Jupiter-9 85mm f/2",
            temperature=15,
            saturation=-10,
            contrast=20,
            vignette=0.3,
            sharpness=-20,
            red_balance=10,
            green_balance=0,
            blue_balance=-5,
            color_cast=(10, 20, 30),
            cast_strength=0.1
        ),
        "Mir-1B 37mm f/2.8": LensCharacteristics(
            name="Mir-1B 37mm f/2.8",
            temperature=20,
            saturation=15,
            contrast=25,
            vignette=0.35,
            sharpness=-5,
            red_balance=10,
            green_balance=-3,
            blue_balance=-7,
            color_cast=(15, 25, 35),
            cast_strength=0.12
        ),

        # Leicaレンズ
        "Leica Summicron 50mm f/2": LensCharacteristics(
            name="Leica Summicron 50mm f/2",
            temperature=0,
            saturation=10,
            contrast=40,
            vignette=0.2,
            sharpness=30,
            red_balance=5,
            green_balance=0,
            blue_balance=5,
            color_cast=(0, 0, 0),
            cast_strength=0
        ),
        "Leica Noctilux 50mm f/0.95": LensCharacteristics(
            name="Leica Noctilux 50mm f/0.95",
            temperature=-5,
            saturation=15,
            contrast=35,
            vignette=0.3,
            sharpness=20,
            red_balance=8,
            green_balance=2,
            blue_balance=3,
            color_cast=(0, 0, 5),
            cast_strength=0.05
        ),
        "Leica Summilux 35mm f/1.4": LensCharacteristics(
            name="Leica Summilux 35mm f/1.4",
            temperature=-3,
            saturation=12,
            contrast=38,
            vignette=0.25,
            sharpness=25,
            red_balance=6,
            green_balance=1,
            blue_balance=4,
            color_cast=(0, 0, 3),
            cast_strength=0.03
        ),

        # Nikonレンズ
        "Nikkor 58mm f/1.4 G": LensCharacteristics(
            name="Nikkor 58mm f/1.4 G",
            temperature=-5,
            saturation=15,
            contrast=25,
            vignette=0.25,
            sharpness=20,
            red_balance=0,
            green_balance=5,
            blue_balance=10,
            color_cast=(5, 0, 0),
            cast_strength=0.05
        ),
        "Noct-Nikkor 58mm f/1.2": LensCharacteristics(
            name="Noct-Nikkor 58mm f/1.2",
            temperature=5,
            saturation=20,
            contrast=30,
            vignette=0.35,
            sharpness=15,
            red_balance=8,
            green_balance=3,
            blue_balance=0,
            color_cast=(0, 0, 10),
            cast_strength=0.08
        ),
        "Nikkor 105mm f/1.4E": LensCharacteristics(
            name="Nikkor 105mm f/1.4E",
            temperature=-2,
            saturation=18,
            contrast=28,
            vignette=0.28,
            sharpness=25,
            red_balance=3,
            green_balance=4,
            blue_balance=5,
            color_cast=(2, 2, 2),
            cast_strength=0.03
        ),

        # Canonレンズ
        "Canon 85mm f/1.2L": LensCharacteristics(
            name="Canon 85mm f/1.2L",
            temperature=0,
            saturation=20,
            contrast=35,
            vignette=0.3,
            sharpness=22,
            red_balance=7,
            green_balance=2,
            blue_balance=3,
            color_cast=(0, 0, 5),
            cast_strength=0.04
        ),
        "Canon 50mm f/1.2L": LensCharacteristics(
            name="Canon 50mm f/1.2L",
            temperature=-2,
            saturation=18,
            contrast=32,
            vignette=0.28,
            sharpness=24,
            red_balance=5,
            green_balance=3,
            blue_balance=4,
            color_cast=(0, 0, 3),
            cast_strength=0.03
        ),

        # Zeissレンズ
        "Zeiss Otus 55mm f/1.4": LensCharacteristics(
            name="Zeiss Otus 55mm f/1.4",
            temperature=-3,
            saturation=25,
            contrast=45,
            vignette=0.2,
            sharpness=35,
            red_balance=4,
            green_balance=4,
            blue_balance=4,
            color_cast=(0, 0, 0),
            cast_strength=0
        ),
        "Zeiss Planar 50mm f/1.4": LensCharacteristics(
            name="Zeiss Planar 50mm f/1.4",
            temperature=0,
            saturation=22,
            contrast=40,
            vignette=0.25,
            sharpness=30,
            red_balance=5,
            green_balance=3,
            blue_balance=5,
            color_cast=(2, 2, 2),
            cast_strength=0.02
        ),

        # ビンテージレンズ
        "Super-Takumar 50mm f/1.4": LensCharacteristics(
            name="Super-Takumar 50mm f/1.4",
            temperature=35,
            saturation=5,
            contrast=15,
            vignette=0.4,
            sharpness=-15,
            red_balance=20,
            green_balance=-8,
            blue_balance=-12,
            color_cast=(25, 35, 45),
            cast_strength=0.2
        ),
        "Olympus Zuiko 50mm f/1.4": LensCharacteristics(
            name="Olympus Zuiko 50mm f/1.4",
            temperature=25,
            saturation=0,
            contrast=20,
            vignette=0.35,
            sharpness=-10,
            red_balance=15,
            green_balance=-5,
            blue_balance=-8,
            color_cast=(20, 30, 35),
            cast_strength=0.15
        ),

        # 現代のミラーレスレンズ
        "Sony 85mm f/1.4 GM": LensCharacteristics(
            name="Sony 85mm f/1.4 GM",
            temperature=-2,
            saturation=22,
            contrast=35,
            vignette=0.22,
            sharpness=32,
            red_balance=3,
            green_balance=3,
            blue_balance=5,
            color_cast=(1, 1, 1),
            cast_strength=0.02
        ),
        "Sigma 35mm f/1.4 Art": LensCharacteristics(
            name="Sigma 35mm f/1.4 Art",
            temperature=-1,
            saturation=20,
            contrast=38,
            vignette=0.24,
            sharpness=33,
            red_balance=4,
            green_balance=4,
            blue_balance=4,
            color_cast=(0, 0, 0),
            cast_strength=0.01
        ),

        # シネマレンズ
        "Cooke S4/i 50mm T2.0": LensCharacteristics(
            name="Cooke S4/i 50mm T2.0",
            temperature=5,
            saturation=15,
            contrast=25,
            vignette=0.3,
            sharpness=15,
            red_balance=8,
            green_balance=2,
            blue_balance=0,
            color_cast=(5, 10, 15),
            cast_strength=0.1
        ),
        "ARRI Master Prime 50mm T1.3": LensCharacteristics(
            name="ARRI Master Prime 50mm T1.3",
            temperature=0,
            saturation=18,
            contrast=30,
            vignette=0.25,
            sharpness=28,
            red_balance=5,
            green_balance=5,
            blue_balance=5,
            color_cast=(0, 0, 0),
            cast_strength=0.02
        ),

        # アナモフィックレンズ
        "Lomo Round-Front Anamorphic 50mm": LensCharacteristics(
            name="Lomo Round-Front Anamorphic 50mm",
            temperature=20,
            saturation=10,
            contrast=20,
            vignette=0.45,
            sharpness=-10,
            red_balance=12,
            green_balance=-3,
            blue_balance=-8,
            color_cast=(30, 20, 40),
            cast_strength=0.18
        ),
        "Kowa Prominar Anamorphic 40mm": LensCharacteristics(
            name="Kowa Prominar Anamorphic 40mm",
            temperature=15,
            saturation=5,
            contrast=25,
            vignette=0.4,
            sharpness=-5,
            red_balance=10,
            green_balance=-2,
            blue_balance=-5,
            color_cast=(25, 15, 35),
            cast_strength=0.15
        )
    }

    def adjust_temperature(self, image: np.ndarray, temperature: float) -> np.ndarray:
        """色温度を調整"""
        if temperature == 0:
            return image
        
        # 色温度の変換行列を作成
        if temperature > 0:
            blue_factor = 1 - (temperature / 100) * 0.2
            red_factor = 1 + (temperature / 100) * 0.2
        else:
            temp = -temperature
            blue_factor = 1 + (temp / 100) * 0.2
            red_factor = 1 - (temp / 100) * 0.2

        matrix = np.array([
            [red_factor, 0, 0],
            [0, 1, 0],
            [0, 0, blue_factor]
        ], dtype=np.float32)

        return cv2.transform(image, matrix)

    def adjust_saturation(self, image: np.ndarray, saturation: float) -> np.ndarray:
        """彩度を調整"""
        if saturation == 0:
            return image

        # HSV色空間で彩度を調整
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        factor = 1 + (saturation / 100)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def adjust_contrast(self, image: np.ndarray, contrast: float) -> np.ndarray:
        """コントラストを調整（より控えめな調整）"""
        if contrast == 0:
            return image

        # コントラストの調整係数を小さくして、より穏やかな変化に
        factor = 1 + (contrast / 100)  # 係数を半分に
        
        # ガンマ補正のような形でコントラストを調整
        mean = np.mean(image)
        return mean + (image - mean) * factor

    def apply_vignette(self, image: np.ndarray, strength: float) -> np.ndarray:
        """ケラレ効果を適用"""
        if strength == 0:
            return image

        rows, cols = image.shape[:2]
        X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
        radius = np.sqrt(X**2 + Y**2)
        mask = 1 - radius * strength
        mask = np.dstack([mask] * 3)
        return image * mask

    def adjust_sharpness(self, image: np.ndarray, sharpness: float) -> np.ndarray:
        """シャープネスを調整"""
        if sharpness == 0:
            return image

        blur = cv2.GaussianBlur(image, (0, 0), 3)
        if sharpness > 0:
            return image + (image - blur) * (sharpness / 100)
        else:
            return image + (blur - image) * (-sharpness / 100)

    def adjust_color_balance(self, image: np.ndarray, red: float, green: float, blue: float) -> np.ndarray:
        """色バランスを調整"""
        if red == 0 and green == 0 and blue == 0:
            return image

        matrix = np.array([
            [(blue + 100) / 100, 0, 0],
            [0, (green + 100) / 100, 0],
            [0, 0, (red + 100) / 100]
        ], dtype=np.float32)

        return cv2.transform(image, matrix)

    def apply_color_cast(self, image: np.ndarray, color: Tuple[float, float, float], strength: float) -> np.ndarray:
        """色かぶりを適用"""
        if strength == 0:
            return image

        color_layer = np.full_like(image, [c/255 for c in color])
        return cv2.addWeighted(image, 1 - strength, color_layer, strength, 0)

    def apply_lens_characteristics(self, img: np.ndarray, lens_name: str) -> np.ndarray:
        """指定されたレンズの特性を適用"""
        if lens_name not in LensSimulator.LENS_PRESETS:
            raise ValueError(f"Unknown lens: {lens_name}")

        # 画像を正規化
        chars = LensSimulator.LENS_PRESETS[lens_name]

        # 各効果を適用
        img = self.adjust_temperature(img, chars.temperature)
        img = self.adjust_saturation(img, chars.saturation)
        img = self.adjust_contrast(img, chars.contrast)
        #img = self.apply_vignette(img, chars.vignette)
        img = self.adjust_sharpness(img, chars.sharpness)
        img = self.adjust_color_balance(img, chars.red_balance, chars.green_balance, chars.blue_balance)
        img = self.apply_color_cast(img, chars.color_cast, chars.cast_strength)

        return img.astype(np.float32)

    def get_available_lenses(self) -> Dict[str, str]:
        """利用可能なレンズのリストを返す"""
        return {key: preset.name for key, preset in LensSimulator.LENS_PRESETS.items()}

def process_image(image, lens_name: str) -> np.ndarray:

    # レンズシミュレータを作成
    simulator = LensSimulator()

    # レンズ特性を適用
    result = simulator.apply_lens_characteristics(image, lens_name)

    return result
