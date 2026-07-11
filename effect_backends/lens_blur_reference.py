"""Reference (CPU / OpenCV) implementation of the depth-of-field lens blur.

This mirrors the original ``cores.filters.apply_lensblur`` and is kept as the
canonical fallback for the Metal backend in :mod:`lens_blur_adapter`.
"""

import numpy as np
import cv2


def compute_coc_radius(
    image_shape,
    depth_map=None,
    focus_depth=0.8,
    max_coc_radius=25,
):
    """CoC(Circle of Confusion)半径マップと有効 focus_depth を計算する。

    Metal バックエンドとロジックを共有するため adapter からも呼ばれる。
    戻り値は ``(coc_radius (H, W) float32, focus_depth)``。
    """
    H, W = image_shape[:2]

    if max_coc_radius <= 0:
        max_coc_radius = 1

    # depth_map が None の場合は全体を背景ぼけとして扱う
    if depth_map is None:
        # 全体を最奥(0.0)、ピント面を手前(1.0)に置くことで全体が均一な背景ぼけになる
        depth_smooth = np.zeros((H, W), dtype=np.float32)
        focus_depth = 1.0
    else:
        # 深度マップを平滑化(エッジでのアーティファクト軽減)
        depth_smooth = cv2.GaussianBlur(depth_map, (0, 0), 2.0)

    # 前景と背景で特性を変える
    is_foreground = depth_smooth > focus_depth
    foreground_weight = np.where(is_foreground, 1.2, 1.0)
    background_weight = np.where(~is_foreground, 0.9, 1.0)

    coc_radius = np.abs(depth_smooth - focus_depth) * max_coc_radius * foreground_weight * background_weight
    coc_radius = np.clip(coc_radius, 0, max_coc_radius).astype(np.float32)
    return coc_radius, focus_depth


def apply_lensblur(
    image,
    depth_map=None,
    focus_depth=0.8,
    max_coc_radius=25,
    num_levels=25,
    chromatic_aberration=0.04,
    spherical_aberration=0.6,
):
    """
    高級レンズのボケ味をシミュレートする

    Parameters:
    -----------
    image : numpy.ndarray
        float32 RGB画像 (H, W, 3), 値域 [0, 1]
    depth_map : numpy.ndarray or None
        float32 深度マップ (H, W), 手前が1.0, 奥が0.0
        Noneの場合は全体を背景ぼけ(後ろボケ)として扱う
    focus_depth : float
        ピント面の深度値 (0.0-1.0)
        depth_map=Noneの場合は無視され、全体が背景ぼけになる
    max_coc_radius : int
        最大ボケ半径(ピクセル)
    num_levels : int
        ぼかしレベルの離散化数(多いほど滑らかだが遅い)
    chromatic_aberration : float
        色収差の強さ (0.0-0.1)
    spherical_aberration : float
        球面収差の強さ (0.0-1.0) - ボケ輪郭の滑らかさ

    Returns:
    --------
    numpy.ndarray
        ボケ処理済みfloat32 RGB画像
    """
    H, W, C = image.shape
    assert C == 3, "Image must be RGB"

    if max_coc_radius <= 0:
        max_coc_radius = 1

    coc_radius, focus_depth = compute_coc_radius(
        image.shape, depth_map, focus_depth, max_coc_radius
    )

    # 色収差:チャンネルごとにCoCスケールを変える
    coc_scales = [
        1.0 + chromatic_aberration,  # R: 少し大きい
        1.0,                          # G: 基準
        1.0 - chromatic_aberration   # B: 少し小さい
    ]

    # 各チャンネルごとに処理
    result = np.zeros_like(image)

    for c in range(C):
        # このチャンネルのCoC半径
        coc_radius_c = coc_radius * coc_scales[c]
        blur_level_float_c = coc_radius_c * (num_levels - 1) / max_coc_radius

        # 複数レベルのガウスぼかしを事前計算
        blurred_stack = []
        for level in range(num_levels):
            # sigmaはCoC半径の約半分(ガウス分布の特性)
            sigma = level * max_coc_radius / (num_levels - 1) / 2.0

            # 球面収差のシミュレーション:レベルが高い(ボケが大きい)ほどsigmaを補正
            if spherical_aberration > 0 and sigma > 1.0:
                sigma_adjusted = sigma * (1.0 + spherical_aberration * 0.2)
            else:
                sigma_adjusted = sigma

            if sigma_adjusted < 0.1:
                blurred_stack.append(image[:, :, c].copy())
            else:
                blurred = cv2.GaussianBlur(image[:, :, c], (0, 0), sigma_adjusted)
                blurred_stack.append(blurred)

        # 各ピクセルで適切なぼかしを線形補間で合成
        channel_result = np.zeros((H, W), dtype=np.float32)
        weight_sum = np.zeros((H, W), dtype=np.float32)

        for level in range(num_levels):
            # このレベルの重み:三角錐型の重み付け
            weight = np.maximum(0, 1 - np.abs(blur_level_float_c - level))
            weight_sum += weight
            channel_result += blurred_stack[level] * weight

        # 正規化
        weight_sum = np.maximum(weight_sum, 1e-6)
        channel_result /= weight_sum
        result[:, :, c] = channel_result

    return result
