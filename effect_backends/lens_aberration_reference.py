"""Reference Lens Aberration implementations used as fallback and numeric baseline."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift


def _radial_shift(channel: np.ndarray, dir_x: np.ndarray, dir_y: np.ndarray, shift_amount: np.ndarray) -> np.ndarray:
    """Sample ``channel`` from positions shifted toward the optical center."""
    if float(np.max(shift_amount)) <= 0.0:
        return channel

    height, width = channel.shape[:2]
    j_idx, i_idx = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
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


def apply_lateral_chromatic_aberration(
    image: np.ndarray,
    strength: float = 0.5,
    resolution_scale: float = 1.0,
    radial: bool = True,
) -> np.ndarray:
    """倍率色収差（横色収差）。

    異なる波長で像の倍率が異なる現象を再現する。radial=True では画像周辺ほど
    放射方向の色ずれが大きくなり、radial=False では簡易的な水平ずれにする。
    """
    image32 = np.asarray(image, dtype=np.float32)
    height, width = image32.shape[:2]
    result = image32.copy()
    base_shift = np.float32(strength * 2.0) * np.float32(max(0.05, float(resolution_scale)))

    if radial:
        cy = np.float32(height * 0.5)
        cx = np.float32(width * 0.5)
        y, x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        distance_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        distance_map_normalized = distance_map / np.maximum(cx, cy)
        with np.errstate(divide="ignore", invalid="ignore"):
            direction_x = np.nan_to_num((x - cx) / distance_map, nan=0.0, posinf=0.0, neginf=0.0)
            direction_y = np.nan_to_num((y - cy) / distance_map, nan=0.0, posinf=0.0, neginf=0.0)

        # チャンネルごとのずれ量は青 > 緑 > 赤。中心から離れるほど強くする。
        shift_amount = distance_map_normalized * base_shift
        result[:, :, 0] = _radial_shift(image32[:, :, 0], direction_x, direction_y, shift_amount * np.float32(0.5))
        result[:, :, 1] = _radial_shift(image32[:, :, 1], direction_x, direction_y, shift_amount)
        result[:, :, 2] = _radial_shift(image32[:, :, 2], direction_x, direction_y, shift_amount * np.float32(1.5))
    else:
        shift(
            image32[:, :, 0],
            shift=(0.0, float(-base_shift * 0.5)),
            output=result[:, :, 0],
            mode="nearest",
            order=1,
        )
        result[:, :, 1] = image32[:, :, 1]
        shift(
            image32[:, :, 2],
            shift=(0.0, float(base_shift * 0.5)),
            output=result[:, :, 2],
            mode="nearest",
            order=1,
        )

    return result.astype(np.float32, copy=False)


def apply_longitudinal_chromatic_aberration(
    image: np.ndarray,
    depth_map: np.ndarray,
    strength: float = 0.5,
    focus_depth: float = 0.5,
    resolution_scale: float = 1.0,
) -> np.ndarray:
    """軸上色収差（縦色収差 / LoCA）。

    各波長(R/G/B)がわずかに異なる深度でピントを結ぶため、ピント面から外れた領域の
    エッジに色フリンジが現れる。平面的な色被りではなく、G を基準に R/B を
    デフォーカス量に応じて差分ぼかしすることで、エッジ起因のフリンジとして再現する。
    ピント面では一切変化しない。
    """
    image32 = np.asarray(image, dtype=np.float32)
    if strength <= 0.0:
        return image32

    rs = float(max(0.05, float(resolution_scale)))
    dm = np.asarray(depth_map, dtype=np.float32)
    signed = dm - np.float32(focus_depth)
    defocus = np.abs(signed)
    defocus = gaussian_filter(defocus, sigma=max(0.5, 2.0 * rs)).astype(np.float32)

    s = float(np.clip(strength, 0.0, 2.0))
    # フリンジ用ぼかし半径と合成重み。重みは convex blend なので出力は元画像と
    # 僅かにぼけたチャンネルの範囲内に収まり、平面部ではほぼ無変化。
    fringe_sigma = (0.6 + 1.4 * s) * rs
    weight = np.clip(
        defocus * np.float32(0.5 + 0.25 * s), np.float32(0.0), np.float32(1.0)
    ).astype(np.float32)

    r = image32[:, :, 0]
    b = image32[:, :, 2]
    blur_r = gaussian_filter(r, sigma=fringe_sigma).astype(np.float32)
    blur_b = gaussian_filter(b, sigma=fringe_sigma).astype(np.float32)

    result = image32.copy()
    # G は基準のまま。R/B のみデフォーカス領域で僅かにぼかす。
    result[:, :, 0] = r * (np.float32(1.0) - weight) + blur_r * weight
    result[:, :, 2] = b * (np.float32(1.0) - weight) + blur_b * weight
    return result.astype(np.float32, copy=False)


def _distance_map_normalized(height: int, width: int) -> np.ndarray:
    cy = np.float32(height * 0.5)
    cx = np.float32(width * 0.5)
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    distance_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return distance_map / np.maximum(cx, cy)


def apply_spherical_aberration(
    image: np.ndarray,
    depth_map: np.ndarray | None = None,
    strength: float = 0.5,
    aperture: float = 1.4,
    focus_depth: float = 0.5,
    highlight_threshold: float = 0.7,
    resolution_scale: float = 1.0,
) -> np.ndarray:
    """球面収差。

    レンズ周辺部の光が中心部より強く屈折する現象を再現する。絞り開放で像が甘く
    滲み、ハイライトに輝きが生まれる。depth があればピント面から外れた領域を強める。
    """
    image32 = np.asarray(image, dtype=np.float32)
    result = image32.copy()
    height, width = image32.shape[:2]
    rs = np.float32(max(0.05, float(resolution_scale)))

    # 絞り値による効果の調整（F値が小さいほど効果が強い）。
    aperture_factor = np.float32(2.8 / aperture)

    luminance = np.mean(image32, axis=2, dtype=np.float32)
    ht = np.float32(highlight_threshold)
    denom = np.float32(1.0) - ht
    highlight_mask = np.clip((luminance - ht) / denom, np.float32(0), np.float32(1))
    gaussian_filter(highlight_mask, sigma=max(0.5, 5.0 * float(rs)), output=highlight_mask)

    if depth_map is not None:
        # ピント面から外れるほどぼかしを強くする。
        dm = np.asarray(depth_map, dtype=np.float32)
        depth_diff = np.abs(dm - np.float32(focus_depth))
        defocus = np.clip(depth_diff * np.float32(3), np.float32(0), np.float32(1))
        depth_weight = np.clip(np.float32(0.2) + defocus, np.float32(0), np.float32(1))
    else:
        depth_weight = np.ones((height, width), dtype=np.float32)

    # 周辺部ほど球面収差が目立つ。
    edge_weight = _distance_map_normalized(height, width) ** 2
    gaussian_filter(edge_weight, sigma=max(0.5, 10.0 * float(rs)), output=edge_weight)

    total_blur_strength = np.float32(float(strength) * float(aperture_factor) * 1.5) * rs
    blur_sigma = total_blur_strength * depth_weight * (np.float32(0.5) + np.float32(0.5) * edge_weight)

    avg_blur_sigma = np.mean(blur_sigma, dtype=np.float32)
    if avg_blur_sigma > np.float32(0.1):
        blurred = np.empty_like(result, dtype=np.float32)
        gaussian_filter(result, sigma=float(avg_blur_sigma), output=blurred)

        # HDR ハイライト成分は glow として足し、clip は下流へ委ねる。
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

    contrast_reduction = np.float32(
        float(np.clip(1.0 - float(strength) * 0.1 * float(aperture_factor), 0.3, 1.0))
    )
    pivot = np.float32(np.mean(result, dtype=np.float32))
    result = (result - pivot) * contrast_reduction + pivot

    return result.astype(np.float32, copy=False)
