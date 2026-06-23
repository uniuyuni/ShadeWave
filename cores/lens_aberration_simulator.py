import logging

import numpy as np
from scipy.ndimage import gaussian_filter, shift
from typing import Tuple, Optional
import cv2


class LensAberrationSimulator:
    """レンズ収差シミュレーター"""
    
    def __init__(self, image_shape: Tuple[int, int], resolution_scale: float = 1.0):
        """
        Args:
            image_shape: (height, width) の画像サイズ
            resolution_scale: プレビュー/等倍の縮尺。ピクセル単位のパラメータ(シフト量・
                ぼかし半径)に掛けて、プレビューと書き出しで見え方を一致させる。
        """
        self.height, self.width = image_shape
        self.res_scale = float(max(0.05, float(resolution_scale)))
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
        base_shift = np.float32(strength * 2.0) * np.float32(self.res_scale)  # px(縮尺対応)
        
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
            with np.errstate(divide='ignore', invalid='ignore'):
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
        """放射状シフト処理（中心方向へ shift_amount 画素だけサンプル位置をずらす）。

        cv2.remap によるベクトル化＋バイリニア補間。旧実装は全画素を Python 二重ループで
        回しており、プレビューサイズで十数秒かかっていた(整数シフトでブロックノイズも発生)。
        """
        if float(np.max(shift_amount)) <= 0.0:
            return channel

        # 各出力画素 (j, i) は、中心方向へ shift_amount だけ戻した位置からサンプルする。
        j_idx, i_idx = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
            indexing='xy',
        )
        map_x = j_idx - dir_x * shift_amount
        map_y = i_idx - dir_y * shift_amount
        return cv2.remap(
            channel,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    
    def longitudinal_chromatic_aberration(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        strength: float = 0.5,
        focus_depth: float = 0.5
    ) -> np.ndarray:
        """
        軸上色収差（縦色収差 / LoCA）のシミュレーション

        各波長(R/G/B)がわずかに異なる深度でピントを結ぶため、ピント面から外れた領域の
        エッジに色フリンジ(マゼンタ寄りの芯＋グリーン寄りの縁)が現れる。これを「平面的な
        色被り」ではなく、G を基準に R/B をデフォーカス量に応じて差分ぼかしすることで
        エッジ起因のフリンジとして再現する。ピント面では一切変化しない。

        Args:
            image: float32, RGB, HDR画像 (H, W, 3)
            depth_map: float32, 深度マップ (H, W), 0.0(手前)-1.0(奥)
            strength: 収差の強さ (0.0-2.0)。フリンジのぼかし半径と合成量を制御。
            focus_depth: ピント位置の深度 (0.0-1.0)

        Returns:
            収差を適用した画像
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if strength <= 0.0:
            return image

        dm = np.asarray(depth_map, dtype=np.float32)
        signed = dm - np.float32(focus_depth)            # +:奥(後ボケ) / -:手前(前ボケ)
        defocus = np.abs(signed)                          # ピント面で 0、離れるほど大
        defocus = gaussian_filter(defocus, sigma=max(0.5, 2.0 * self.res_scale)).astype(np.float32)

        s = float(np.clip(strength, 0.0, 2.0))
        # フリンジ用ぼかし半径(控えめ)と合成重み。重みは convex blend なので出力は
        # 元画像と僅かにぼけたチャンネルの範囲内に収まり、平面部ではほぼ無変化。
        fringe_sigma = (0.6 + 1.4 * s) * self.res_scale  # px (s=1 で約2px, 縮尺対応)
        weight = np.clip(
            defocus * np.float32(0.5 + 0.25 * s), np.float32(0.0), np.float32(1.0)
        ).astype(np.float32)

        r = image[:, :, 0]
        b = image[:, :, 2]
        blur_r = gaussian_filter(r, sigma=fringe_sigma).astype(np.float32)
        blur_b = gaussian_filter(b, sigma=fringe_sigma).astype(np.float32)

        result = image.copy()
        # G は基準のまま。R/B のみデフォーカス領域で僅かにぼかす → エッジに色フリンジ。
        result[:, :, 0] = r * (np.float32(1.0) - weight) + blur_r * weight
        result[:, :, 2] = b * (np.float32(1.0) - weight) + blur_b * weight
        return result
    
    def spherical_aberration(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        strength: float = 0.5,
        aperture: float = 1.4,
        focus_depth: float = 0.5,
        highlight_threshold: float = 0.7
    ) -> np.ndarray:
        """
        球面収差のシミュレーション

        レンズ周辺部の光が中心部より強く屈折する現象を再現。
        絞り開放で像が「甘く」滲み、ハイライトに輝きが生まれる。

        Args:
            image: float32, RGB, HDR画像 (H, W, 3)
            depth_map: float32, 深度マップ (H, W), 0.0(手前)-1.0(奥)。None なら全域一律。
            strength: 収差の強さ (0.0-2.0)
            aperture: 絞り値（小さいほど効果が強い）
            focus_depth: ピント位置の深度 (0.0-1.0)。ここから外れるほど甘くなる。
            highlight_threshold: ハイライト検出の閾値

        Returns:
            収差を適用した画像
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        result = image.copy()
        rs = np.float32(self.res_scale)

        # 絞り値による効果の調整（F値が小さいほど効果が強い）
        aperture_factor = np.float32(2.8 / aperture)  # F2.8を基準

        # 1. ハイライト領域の検出（球面収差で「輝き」が出る部分）
        luminance = np.mean(image, axis=2, dtype=np.float32)
        ht = np.float32(highlight_threshold)
        denom = np.float32(1.0) - ht
        highlight_mask = np.clip((luminance - ht) / denom, np.float32(0), np.float32(1))
        gaussian_filter(highlight_mask, sigma=max(0.5, 5.0 * float(rs)), output=highlight_mask)

        # 2. 深度マップがある場合はユーザー指定のピント位置から外れるほど甘くする
        #    (ピント面=シャープ、デフォーカス=ソフト)。わずかな常時甘さ(base)も加える。
        if depth_map is not None:
            dm = np.asarray(depth_map, dtype=np.float32)
            depth_diff = np.abs(dm - np.float32(focus_depth))
            defocus = np.clip(depth_diff * np.float32(3), np.float32(0), np.float32(1))
            depth_weight = np.clip(np.float32(0.2) + defocus, np.float32(0), np.float32(1))
        else:
            depth_weight = np.ones((self.height, self.width), dtype=np.float32)

        # 3. 球面収差特有の「周辺部の甘さ」を再現
        edge_weight = self.distance_map_normalized ** 2
        gaussian_filter(edge_weight, sigma=max(0.5, 10.0 * float(rs)), output=edge_weight)

        # 4. 総合ぼかし強度(ぼかし半径=px なので縮尺を掛ける)
        total_blur_strength = np.float32(float(strength) * float(aperture_factor) * 1.5) * rs
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

        # 8. コントラストを少し低下（球面収差の特徴）。HDR/linear でも破綻しないよう
        #    画像平均を pivot にし、反転しないよう係数をクランプする。
        contrast_reduction = np.float32(
            float(np.clip(1.0 - float(strength) * 0.1 * float(aperture_factor), 0.3, 1.0))
        )
        pivot = np.float32(np.mean(result, dtype=np.float32))
        result = (result - pivot) * contrast_reduction + pivot

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
            result, depth_map, spherical_strength, aperture, focus_depth=focus_depth
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
    
    logging.info("収差シミュレーション完了！")
