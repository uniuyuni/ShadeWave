import numpy as np
from scipy.ndimage import gaussian_filter, shift
from typing import Tuple, Optional
import cv2


class LensAberrationSimulator:
    """レンズ収差シミュレーター"""
    
    def __init__(self, image_shape: Tuple[int, int]):
        """
        Args:
            image_shape: (height, width) の画像サイズ
        """
        self.height, self.width = image_shape
        self._create_coordinate_maps()
    
    def _create_coordinate_maps(self):
        """画像中心からの距離マップを作成"""
        cy = np.float32(self.height * 0.5)
        cx = np.float32(self.width * 0.5)
        y, x = np.meshgrid(
            np.arange(self.height, dtype=np.float32),
            np.arange(self.width, dtype=np.float32),
            indexing='ij',
        )
        self.distance_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        self.distance_map_normalized = self.distance_map / np.maximum(cx, cy)
    
    def lateral_chromatic_aberration(
        self,
        image: np.ndarray,
        strength: float = 0.5,
        radial: bool = True
    ) -> np.ndarray:
        """
        倍率色収差（横色収差）のシミュレーション
        
        異なる波長で像の倍率が異なる現象を再現。
        画像周辺で色ずれが発生する。
        
        Args:
            image: float32, RGB, HDR画像 (H, W, 3)
            strength: 収差の強さ (0.0-2.0)
            radial: 放射状の色ずれにするか（真実に近い）
        
        Returns:
            収差を適用した画像
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        result = image.copy()
        
        # RGB各チャンネルのずれ量（青が最も大きく、赤が最小）
        base_shift = np.float32(strength * 2.0)  # ピクセル単位の基本ずれ量
        
        if radial:
            # 放射状のずれ（より現実的）
            # 画像中心からの方向ベクトルを計算
            cy = np.float32(self.height * 0.5)
            cx = np.float32(self.width * 0.5)
            y, x = np.meshgrid(
                np.arange(self.height, dtype=np.float32),
                np.arange(self.width, dtype=np.float32),
                indexing='ij',
            )
            dm = self.distance_map
            direction_x = np.nan_to_num((x - cx) / dm, nan=0.0, posinf=0.0, neginf=0.0)
            direction_y = np.nan_to_num((y - cy) / dm, nan=0.0, posinf=0.0, neginf=0.0)
            
            # チャンネルごとのずれ量（青 > 緑 > 赤）
            shift_amount = self.distance_map_normalized * base_shift
            
            # 赤チャンネル（最小のずれ）
            shift_r = shift_amount * 0.5
            # 緑チャンネル（中間）
            shift_g = shift_amount * 1.0
            # 青チャンネル（最大のずれ）
            shift_b = shift_amount * 1.5
            
            # 各チャンネルをシフト
            result[:, :, 0] = self._radial_shift(image[:, :, 0], direction_x, direction_y, shift_r)
            result[:, :, 1] = self._radial_shift(image[:, :, 1], direction_x, direction_y, shift_g)
            result[:, :, 2] = self._radial_shift(image[:, :, 2], direction_x, direction_y, shift_b)
        else:
            # 単純な水平方向のずれ（簡易版）
            shift(
                image[:, :, 0],
                shift=(0.0, float(-base_shift * 0.5)),
                output=result[:, :, 0],
                mode='nearest',
                order=1,
            )
            result[:, :, 1] = image[:, :, 1]
            shift(
                image[:, :, 2],
                shift=(0.0, float(base_shift * 0.5)),
                output=result[:, :, 2],
                mode='nearest',
                order=1,
            )
        
        return result
    
    def _radial_shift(self, channel: np.ndarray, dir_x: np.ndarray, dir_y: np.ndarray, 
                      shift_amount: np.ndarray) -> np.ndarray:
        """放射状シフト処理"""
        result = np.zeros_like(channel)
        
        # 整数ピクセル単位でシフト（簡易実装）
        shift_pixels = np.round(shift_amount).astype(int)
        max_shift = np.max(shift_pixels)
        
        if max_shift == 0:
            return channel
        
        for i in range(self.height):
            for j in range(self.width):
                s = min(shift_pixels[i, j], max_shift)
                if s > 0:
                    dy = int(dir_y[i, j] * s)
                    dx = int(dir_x[i, j] * s)
                    src_y, src_x = i - dy, j - dx
                    if 0 <= src_y < self.height and 0 <= src_x < self.width:
                        result[i, j] = channel[src_y, src_x]
                    else:
                        result[i, j] = channel[i, j]
                else:
                    result[i, j] = channel[i, j]
        
        return result
    
    def longitudinal_chromatic_aberration(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        strength: float = 0.5,
        focus_depth: float = 0.5
    ) -> np.ndarray:
        """
        軸上色収差（縦色収差）のシミュレーション
        
        ピント位置からの前後で異なる色のにじみを再現。
        前ボケ：緑系、後ボケ：赤紫系 が一般的。
        
        Args:
            image: float32, RGB, HDR画像 (H, W, 3)
            depth_map: float32, 深度マップ (H, W), 0.0(手前)-1.0(奥)
            strength: 収差の強さ (0.0-2.0)
            focus_depth: ピント位置の深度 (0.0-1.0)
        
        Returns:
            収差を適用した画像
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        result = image.copy()
        dm = np.asarray(depth_map, dtype=np.float32)
        fd = np.float32(focus_depth)
        
        # 深度マップをピント位置からの距離に変換
        depth_diff = dm - fd  # 正:奥、負:手前
        
        # 前ボケ領域（深度差が負）
        front_mask = np.clip(-depth_diff * np.float32(5), np.float32(0), np.float32(1))
        # 後ボケ領域（深度差が正）
        back_mask = np.clip(depth_diff * np.float32(5), np.float32(0), np.float32(1))
        
        # ガウシアンぼかしでマスクを滑らかに（出力を float32 に固定）
        gaussian_filter(front_mask, sigma=3, output=front_mask)
        gaussian_filter(back_mask, sigma=3, output=back_mask)
        
        # 色収差の強度
        aberration_strength = np.float32(strength * 0.3)
        
        # 各チャンネルに色にじみを追加
        # 前ボケ領域
        result[:, :, 0] = result[:, :, 0] - front_mask * aberration_strength * 0.3
        result[:, :, 1] = result[:, :, 1] + front_mask * aberration_strength * 0.5
        result[:, :, 2] = result[:, :, 2] + front_mask * aberration_strength * 0.3
        
        # 後ボケ領域
        result[:, :, 0] = result[:, :, 0] + back_mask * aberration_strength * 0.5
        result[:, :, 1] = result[:, :, 1] - back_mask * aberration_strength * 0.2
        result[:, :, 2] = result[:, :, 2] + back_mask * aberration_strength * 0.4
        
        return result
    
    def spherical_aberration(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        strength: float = 0.5,
        aperture: float = 1.4,
        highlight_threshold: float = 0.7
    ) -> np.ndarray:
        """
        球面収差のシミュレーション
        
        レンズ周辺部の光が中心部より強く屈折する現象を再現。
        絞り開放で像が「甘く」滲み、ハイライトに輝きが生まれる。
        
        Args:
            image: float32, RGB, HDR画像 (H, W, 3)
            depth_map: float32, 深度マップ (H, W), Noneの場合は全領域に適用
            strength: 収差の強さ (0.0-2.0)
            aperture: 絞り値（小さいほど効果が強い）
            highlight_threshold: ハイライト検出の閾値
        
        Returns:
            収差を適用した画像
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        result = image.copy()
        
        # 絞り値による効果の調整（F値が小さいほど効果が強い）
        aperture_factor = np.float32(2.8 / aperture)  # F2.8を基準
        
        # 1. ハイライト領域の検出（球面収差で「輝き」が出る部分）
        luminance = np.mean(image, axis=2, dtype=np.float32)
        ht = np.float32(highlight_threshold)
        denom = np.float32(1.0) - ht
        highlight_mask = np.clip((luminance - ht) / denom, np.float32(0), np.float32(1))
        gaussian_filter(highlight_mask, sigma=5, output=highlight_mask)
        
        # 2. 深度マップがある場合はピント位置からのぼかし量を制御
        if depth_map is not None:
            dm = np.asarray(depth_map, dtype=np.float32)
            focus_depth_s = np.float32(np.median(dm))
            depth_diff = np.abs(dm - focus_depth_s)
            depth_weight = np.float32(1.0) - np.clip(depth_diff * np.float32(3), np.float32(0), np.float32(1))
        else:
            depth_weight = np.ones((self.height, self.width), dtype=np.float32)
        
        # 3. 球面収差特有の「周辺部の甘さ」を再現
        edge_weight = self.distance_map_normalized ** 2
        gaussian_filter(edge_weight, sigma=10, output=edge_weight)
        
        # 4. 総合ぼかし強度
        total_blur_strength = np.float32(float(strength) * float(aperture_factor) * 1.5)
        blur_sigma = total_blur_strength * depth_weight * (np.float32(0.5) + np.float32(0.5) * edge_weight)
        
        # 5. 平均ぼかし値で画像全体をソフトに
        avg_blur_sigma = np.mean(blur_sigma, dtype=np.float32)
        if avg_blur_sigma > np.float32(0.1):
            blurred = np.empty_like(result, dtype=np.float32)
            gaussian_filter(result, sigma=float(avg_blur_sigma), output=blurred)
            
            glow_src = result * highlight_mask[:, :, np.newaxis]
            glow = np.empty_like(result, dtype=np.float32)
            gaussian_filter(glow_src, sigma=float(avg_blur_sigma) * 2.0, output=glow)
            
            glow_strength = np.float32(float(strength) * 0.3 * float(aperture_factor))
            one = np.float32(1.0)
            result = result * (one - highlight_mask[:, :, np.newaxis] * glow_strength) + glow * glow_strength
            
            blend_ratio = np.clip(
                blur_sigma / (total_blur_strength + np.float32(0.01)),
                np.float32(0),
                np.float32(0.8),
            )
            result = result * (one - blend_ratio[:, :, np.newaxis]) + blurred * blend_ratio[:, :, np.newaxis]
        
        # 8. コントラストを少し低下（球面収差の特徴）
        contrast_reduction = np.float32(1.0 - float(strength) * 0.1 * float(aperture_factor))
        half = np.float32(0.5)
        result = (result - half) * contrast_reduction + half
        
        return result
    
    def apply_all_aberrations(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        lateral_strength: float = 0.3,
        longitudinal_strength: float = 0.4,
        spherical_strength: float = 0.5,
        focus_depth: float = 0.5,
        aperture: float = 1.4
    ) -> np.ndarray:
        """
        3つの収差を全て適用する統合関数
        
        Args:
            image: float32, RGB, HDR画像
            depth_map: 深度マップ（Noneの場合は軸上色収差をスキップ）
            lateral_strength: 倍率色収差の強さ
            longitudinal_strength: 軸上色収差の強さ
            spherical_strength: 球面収差の強さ
            focus_depth: ピント位置の深度
            aperture: 絞り値
        
        Returns:
            全ての収差を適用した画像
        """
        result = image.copy()
        
        # 1. 球面収差（最初に適用：ぼかし効果のため）
        result = self.spherical_aberration(
            result, depth_map, spherical_strength, aperture
        )
        
        # 2. 軸上色収差（深度マップが必要）
        if depth_map is not None:
            result = self.longitudinal_chromatic_aberration(
                result, depth_map, longitudinal_strength, focus_depth
            )
        
        # 3. 倍率色収差（最後に適用：エッジの色ずれ）
        result = self.lateral_chromatic_aberration(
            result, lateral_strength, radial=True
        )
        
        return result


# ============================================================================
# 使用例
# ============================================================================

def load_hdr_image(path: str) -> np.ndarray:
    """HDR画像の読み込み"""
    import imageio
    image = imageio.imread(path)
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    return image


def load_depth_map(path: str) -> np.ndarray:
    """深度マップの読み込み"""
    import imageio
    depth = imageio.imread(path)
    if depth.ndim > 2:
        depth = np.mean(depth, axis=2)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32) / 255.0
    return depth


def save_hdr_image(image: np.ndarray, path: str):
    """HDR画像の保存"""
    import imageio
    image_uint16 = (image * 65535).astype(np.uint16)
    imageio.imwrite(path, image_uint16)


if __name__ == "__main__":
    # 使用例
    # 1. 画像と深度マップを読み込み
    image = load_hdr_image("input.exr")  # HDR形式（.exr, .hdr）
    depth_map = load_depth_map("depth.png")  # 深度マップ（グレースケール）
    
    # 2. シミュレーターを初期化
    simulator = LensAberrationSimulator(image.shape[:2])
    
    # 3. 個別の収差を適用
    lateral_result = simulator.lateral_chromatic_aberration(image, strength=0.5)
    longitudinal_result = simulator.longitudinal_chromatic_aberration(
        image, depth_map, strength=0.5, focus_depth=0.5
    )
    spherical_result = simulator.spherical_aberration(
        image, depth_map, strength=0.5, aperture=1.4
    )
    
    # 4. 全ての収差をまとめて適用
    full_result = simulator.apply_all_aberrations(
        image,
        depth_map,
        lateral_strength=0.3,
        longitudinal_strength=0.4,
        spherical_strength=0.5,
        focus_depth=0.5,
        aperture=1.4
    )
    
    # 5. 結果を保存
    save_hdr_image(full_result, "output_with_aberrations.exr")
    
    print("収差シミュレーション完了！")