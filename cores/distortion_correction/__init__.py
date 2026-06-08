"""
画像歪み補正モジュール

コアロジック用パッケージ
"""

from .lens_distortion import correct_lens_distortion, detect_lens_distortion

from .trapezoid_correction_3d import correct_trapezoid_3d as correct_trapezoid, calculate_trapezoid_homography

from .four_point_correction import correct_four_points, detect_rectangle, calculate_four_point_homography

from .warp_correction import (
    warp_mesh,
    warp_mesh_with_mapper,
    outer_ring_pins_tcg,
    correct_with_lines,
    calculate_lines_homography,
    get_mesh_coordinates
)

# 4点補正とrectangle検出は必要に応じて別途実装可能
# 現在は3D台形補正のみ提供

__all__ = [
    # レンズ歪み補正
    'correct_lens_distortion',
    'detect_lens_distortion',
    # 台形補正（3D回転ベース）
    'correct_trapezoid',
    'calculate_trapezoid_homography',
    # 4点補正
    'correct_four_points',
    'calculate_four_point_homography',
    'detect_rectangle',
    # ワープ補正
    'warp_mesh',
    'warp_mesh_with_mapper',
    'outer_ring_pins_tcg',
    'correct_with_lines',
    'calculate_lines_homography',
    'get_mesh_coordinates',
]
