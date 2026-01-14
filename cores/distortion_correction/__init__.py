"""
画像歪み補正モジュール

コアロジック用パッケージ
"""

from .coordinate_utils import (
    tcg_to_image_coords,
    image_to_tcg_coords,
    kivy_to_tcg_coords,
    tcg_to_kivy_coords,
    tcg_points_to_image,
    image_points_to_tcg,
    tcg_lines_to_image,
    image_lines_to_tcg
)

from .lens_distortion import correct_lens_distortion, detect_lens_distortion

from .trapezoid_correction_3d import correct_trapezoid_3d as correct_trapezoid

from .four_point_correction import correct_four_points, detect_rectangle

from .warp_correction import (
    warp_mesh,
    correct_with_lines,
    warp_points,
    get_mesh_coordinates
)

# 4点補正とrectangle検出は必要に応じて別途実装可能
# 現在は3D台形補正のみ提供

__all__ = [
    # 座標変換
    'tcg_to_image_coords',
    'image_to_tcg_coords',
    'kivy_to_tcg_coords',
    'tcg_to_kivy_coords',
    'tcg_points_to_image',
    'image_points_to_tcg',
    'tcg_lines_to_image',
    'image_lines_to_tcg',
    # レンズ歪み補正
    'correct_lens_distortion',
    'detect_lens_distortion',
    # 台形補正（3D回転ベース）
    'correct_trapezoid',
    # 4点補正
    'correct_four_points',
    'detect_rectangle',
    # ワープ補正
    'warp_mesh',
    'correct_with_lines',
    'warp_points',
    'get_mesh_coordinates',
]
