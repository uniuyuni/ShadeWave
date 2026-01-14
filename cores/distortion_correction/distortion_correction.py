"""
画像歪み補正API

メインAPIモジュール
"""

# 座標変換ユーティリティ
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

# レンズ歪み補正
from .lens_distortion import correct_lens_distortion, detect_lens_distortion

# 台形補正
from .trapezoid_correction import (
    correct_trapezoid,
    correct_four_points,
    detect_rectangle
)

# ワープ補正
from .warp_correction import (
    warp_mesh,
    correct_with_lines,
    warp_points,
    get_mesh_coordinates
)

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
    
    # 台形補正
    'correct_trapezoid',
    'correct_four_points',
    'detect_rectangle',
    
    # ワープ補正
    'warp_mesh',
    'correct_with_lines',
    'warp_points',
    'get_mesh_coordinates',
]
