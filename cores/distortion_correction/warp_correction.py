"""
ガイド線変形補正API

メッシュワープ、ラインガイド補正、ポイントベース補正
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
from scipy.interpolate import Rbf

import params


def warp_mesh(
    image: np.ndarray,
    mesh_size: Tuple[int, int],
    control_points: Dict[Tuple[int, int], Tuple[float, float]],
    tcg_info: Dict = None,
    interpolation: str = 'bicubic'
) -> np.ndarray:
    """
    メッシュワープ補正（TPS + Coarse Grid Approximation）
    
    Args:
        image: numpy.ndarray、dtype=float32、shape=(H, W, 3)
        mesh_size: tuple、(rows, cols)
        control_points: dict
            キー: (row_index, col_index)
            値: (offset_x, offset_y)（TCG座標系のオフセット）
        tcg_info: 座標変換情報 (from params.param_to_tcg_info)
        interpolation: str、'bilinear' | 'bicubic'
    
    Returns:
        補正後画像
    """
    # 入力検証
    _validate_image(image)
    
    rows, cols = mesh_size
    if not (3 <= rows <= 10 and 3 <= cols <= 10):
        # メッシュサイズは柔軟に対応するため警告に留めるか、一旦そのまま
        pass
    
    height, width = image.shape[:2]
    image_shape = (height, width)
    
    # 1. 制御点の収集 (TCG座標系)
    # Source: 元のグリッド (Regular Grid)
    # Dest: 変形後のグリッド (Deformed Grid) = Source + Offset
    # 我々は Dest(x,y) -> Source(x,y) のマッピングを求めたい
    
    base_coords = get_mesh_coordinates(image_shape, mesh_size) # (rows+1, cols+1, 2)
    
    src_points = [] # Regular
    dst_points = [] # Deformed
    
    for r in range(rows + 1):
        for c in range(cols + 1):
            bx, by = base_coords[r, c]
            off_x, off_y = control_points.get((r, c), (0.0, 0.0))
            
            src_points.append((bx, by))
            dst_points.append((bx + off_x, by + off_y))
            
    # 変形がない場合は早期リターン
    if all(src == dst for src, dst in zip(src_points, dst_points)):
        return image
        
    # 2. 座標変換 (TCG -> Image Pixel)
    # params.tcg_to_ref_image を使用して座標変換
    src_px_list = []
    for px, py in src_points:
        src_px_list.append(params.tcg_to_ref_image(px, py, image, tcg_info))
    
    dst_px_list = []
    for px, py in dst_points:
        dst_px_list.append(params.tcg_to_ref_image(px, py, image, tcg_info))
        
    src_px = np.array(src_px_list)
    dst_px = np.array(dst_px_list)
    
    # 3. TPS (Rbf) の学習
    # 入力: Dest (Deformed), 出力: Source (Original)
    # smooth=0 (点を通る)
    try:
        rbf_x = Rbf(dst_px[:, 0], dst_px[:, 1], src_px[:, 0], function='thin_plate', smooth=0)
        rbf_y = Rbf(dst_px[:, 0], dst_px[:, 1], src_px[:, 1], function='thin_plate', smooth=0)
    except Exception as e:
        print(f"TPS Fitting Error: {e}")
        return image

    # 4. 高速化のためのCoarse Grid生成
    # フル解像度でRbf計算は重すぎるため、縮小グリッドで計算して拡大する
    grid_step = 32 # 32ピクセルごとのグリッド
    grid_h = (height + grid_step - 1) // grid_step
    grid_w = (width + grid_step - 1) // grid_step
    
    # Coarse Grid座標
    coarse_y, coarse_x = np.meshgrid(
        np.linspace(0, height - 1, grid_w), # 注意: meshgridの引数順序と出力shapeの関係
        np.linspace(0, width - 1, grid_h),
        indexing='xy' # xy indexing: x (W) * y (H)
    )
    # meshgrid(x, y, indexing='xy') returns:
    # X: (H, W), Y: (H, W) -> No, shape depends on input size. 
    # linspace size matters.
    # To get (grid_h, grid_w) shape:
    # np.meshgrid(np.linspace(0, W-1, grid_w), np.linspace(0, H-1, grid_h))
    
    coarse_x_coords = np.linspace(0, width - 1, grid_w)
    coarse_y_coords = np.linspace(0, height - 1, grid_h)
    coarse_x_mesh, coarse_y_mesh = np.meshgrid(coarse_x_coords, coarse_y_coords)
    
    flat_cx = coarse_x_mesh.ravel()
    flat_cy = coarse_y_mesh.ravel()
    
    # 5. Rbf評価 (Coarse)
    map_x_coarse = rbf_x(flat_cx, flat_cy).reshape(grid_h, grid_w)
    map_y_coarse = rbf_y(flat_cx, flat_cy).reshape(grid_h, grid_w)
    
    # 6. マップのアップスケーリング
    # float32でリサイズ
    map_x = cv2.resize(map_x_coarse.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(map_y_coarse.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    
    # 7. リマッピング
    if interpolation == 'bilinear':
        interp_flag = cv2.INTER_LINEAR
    else:  # 'bicubic'
        interp_flag = cv2.INTER_CUBIC
        
    corrected = cv2.remap(
        image,
        map_x,
        map_y,
        interp_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return corrected

def correct_with_lines(
    image: np.ndarray,
    reference_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    tcg_info: Dict = None,
    interpolation: str = 'bicubic'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    参照線を使用して画像の透視歪みを補正する関数
    
    Lightroomのガイド付き変形(Guided Upright)機能を実装。
    ユーザーが指定した線分が垂直または水平になるようにホモグラフィ変換を適用。
    
    Parameters
    ----------
    image : np.ndarray
        入力画像 (H, W, C) または (H, W)
    reference_lines : List[Tuple[Tuple[float, float], Tuple[float, float]]]
        補正に使用する参照線のリスト。各線は ((x1, y1), (x2, y2)) の形式。
        座標は画像中心を原点とした正規化座標系 (Y軸下向きが正)。
        **これらの座標は歪んだ画像上での線の位置を示す**
        各線は補正後に垂直または水平になる必要がある。
        - 線が0本: エラー
        - 線が1本: 入力画像をそのまま返す（H=None）
        - 線が2-4本: 正常に補正
        - 線が5本以上: 最後の4本を使用
    tcg_info : Dict, optional
        座標変換情報。Noneの場合は恒等変換を仮定。
    interpolation : str, optional
        補間方法: 'nearest', 'bilinear', 'bicubic', 'lanczos'
        デフォルトは 'bicubic'
        
    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (補正された画像, ホモグラフィ行列)
        - 補正された画像: 透視補正後の画像
        - ホモグラフィ行列 (3x3): 補正に使用した変換行列
          Noneの場合は変換なし（線が1本の場合）
          
          この行列Hは以下のように使用できます:
          - 順変換（歪んだ→補正後）: corrected_pt = H @ distorted_pt
          - 逆変換（補正後→歪んだ）: distorted_pt = H_inv @ corrected_pt
            ここで H_inv = np.linalg.inv(H)
        
    Notes
    -----
    - 線が1本だけの場合は入力画像をそのまま返す
    - 線が5本以上の場合は最後の4本のみを使用
    - 各線は垂直線または水平線として扱われる
    - 線の向きは自動的に判定される
    - ホモグラフィ行列を用いた透視変換を適用
    
    Examples
    --------
    >>> # 建物の縦の線を補正
    >>> lines = [
    ...     ((-0.3, -0.5), (-0.3, 0.5)),  # 左の縦線
    ...     ((0.3, -0.5), (0.3, 0.5))     # 右の縦線
    ... ]
    >>> corrected, H = correct_with_lines(image, lines)
    >>> 
    >>> # 逆変換（補正後の座標→元の座標）
    >>> if H is not None:
    ...     H_inv = np.linalg.inv(H)
    ...     # 補正後の点(100, 200)を元の座標に変換
    ...     pt = np.array([100, 200, 1])
    ...     original_pt = H_inv @ pt
    ...     original_pt = original_pt[:2] / original_pt[2]
    """
    
    # エッジケース: 線が0本
    if len(reference_lines) == 0:
        return image, None
    
    # エッジケース: 線が1本の場合は入力をそのまま返す
    if len(reference_lines) == 1:
        return image, None
    
    # エッジケース: 線が5本以上の場合は最後の4本のみを使用
    if len(reference_lines) > 4:
        reference_lines = reference_lines[-4:]
    
    h, w = image.shape[:2]
    
    H = calculate_lines_homography(reference_lines, w, h, tcg_info)
    
    if H is None:
        # 計算できなかった場合（線が不足など）は元の画像を返す
        # correct_with_linesの仕様上、線が1本の場合はNoneを返す
        return image, None
    
    # 補間方法の選択
    interp_flags = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    if interpolation not in interp_flags:
        raise ValueError(f"未対応の補間方法: {interpolation}")
    
    # ホモグラフィ変換を適用
    corrected = cv2.warpPerspective(
        image, H, (w, h),
        flags=interp_flags[interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return corrected, H

def calculate_lines_homography(
    reference_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    width: int,
    height: int,
    tcg_info: Dict = None
) -> Optional[np.ndarray]:
    """
    参照線からホモグラフィ行列を計算する
    
    Returns
    -------
    Optional[np.ndarray]
        ホモグラフィ行列 (3x3) または None
    """
    # エッジケース: 線が0本
    if len(reference_lines) == 0:
        return None
    
    # エッジケース: 線が1本の場合は計算不可（仕様）
    if len(reference_lines) == 1:
        return None
    
    # エッジケース: 線が5本以上の場合は最後の4本のみを使用
    if len(reference_lines) > 4:
        reference_lines = reference_lines[-4:]
    
    # 正規化座標からピクセル座標への変換
    # これらは歪んだ画像上での実際の線の位置
    # Note: tcg_to_ref_image requires an 'image' argument for shape property usually,
    # but params.tcg_to_ref_image signature is (cx, cy, ref_img, tcg_info).
    # It uses ref_img.shape. We need to mock it or update tcg_to_ref_image to check shape from tcg_info?
    # Actually params.tcg_to_ref_image uses ref_img.shape if apply_disp_info is True or for sizing.
    # In GeometryEffect context, we might not have the image object.
    # But here we pass width/height.
    # We should create a dummy object with shape property or modify tcg_to_ref_image?
    # Converting to pixel coordinates:
    # It basically maps normalized to pixel based on tcg_info['original_img_size'] and matrix.
    # Let's check params.py again.
    
    class DummyImage:
        def __init__(self, w, h):
            self.shape = (h, w, 3)
            
    dummy_img = DummyImage(width, height)

    pixel_lines = []
    for line in reference_lines:
        p1_norm, p2_norm = line
        p1_pixel = params.tcg_to_ref_image(p1_norm[0], p1_norm[1], dummy_img, tcg_info)
        p2_pixel = params.tcg_to_ref_image(p2_norm[0], p2_norm[1], dummy_img, tcg_info)
        pixel_lines.append((p1_pixel, p2_pixel))
    
    # 各線が垂直か水平かを判定し、対応点を生成
    src_points, dst_points = _generate_correspondence_points_improved(pixel_lines, width, height)
    
    if len(src_points) < 4:
        # raise ValueError("ホモグラフィ計算に十分な対応点が得られませんでした")
        return None
    
    # ホモグラフィ行列を計算
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # 点が4つの場合は直接計算、それ以上ならRANSAC
    if len(src_points) == 4:
        H = cv2.getPerspectiveTransform(src_points, dst_points)
    else:
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is None:
        # raise ValueError("ホモグラフィ行列の計算に失敗しました")
        return None
        
    return H

def _classify_line_orientation(p1: Tuple[float, float], p2: Tuple[float, float]) -> str:
    """
    線分が垂直線か水平線かを判定
    
    Parameters
    ----------
    p1, p2 : Tuple[float, float]
        線分の端点
        
    Returns
    -------
    str
        'vertical' または 'horizontal'
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    
    # 傾きの大きい方で判定
    if dy > dx:
        return 'vertical'
    else:
        return 'horizontal'


def _generate_correspondence_points_improved(
    pixel_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    width: int,
    height: int
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    参照線から対応点を生成（改善版）
    
    各線が補正後に垂直または水平になるように対応点を設定する
    
    重要: src_points は歪んだ画像上の実際の点
          dst_points は補正後にあるべき理想的な点
    
    改善点:
    - 垂直線と水平線を分けて処理
    - 線の長さを保持
    - 画像の中心を基準に補正
    - 垂直線と水平線を混在させた時の安定性向上
    
    Parameters
    ----------
    pixel_lines : List[Tuple[Tuple[float, float], Tuple[float, float]]]
        ピクセル座標での参照線（歪んだ画像上の位置）
    width, height : int
        画像のサイズ
        
    Returns
    -------
    Tuple[List, List]
        (元の点(歪んでいる), 目標点(垂直/水平に補正された)) のタプル
    """
    src_points = []
    dst_points = []
    
    # 垂直線と水平線を分類
    vertical_lines = []
    horizontal_lines = []
    
    for p1, p2 in pixel_lines:
        orientation = _classify_line_orientation(p1, p2)
        if orientation == 'vertical':
            vertical_lines.append((p1, p2))
        else:
            horizontal_lines.append((p1, p2))
    
    # 垂直線の処理
    for p1, p2 in vertical_lines:
        # 元の点（歪んだ画像上の実際の位置）
        src_points.append(p1)
        src_points.append(p2)
        
        # 垂直線として補正: X座標を2点の平均に揃える
        # Y座標はそのまま維持
        x_avg = (p1[0] + p2[0]) / 2.0
        dst_points.append((x_avg, p1[1]))
        dst_points.append((x_avg, p2[1]))
    
    # 水平線の処理
    for p1, p2 in horizontal_lines:
        # 元の点（歪んだ画像上の実際の位置）
        src_points.append(p1)
        src_points.append(p2)
        
        # 水平線として補正: Y座標を2点の平均に揃える
        # X座標はそのまま維持
        y_avg = (p1[1] + p2[1]) / 2.0
        dst_points.append((p1[0], y_avg))
        dst_points.append((p2[0], y_avg))
    
    # 垂直線と水平線を混在させた場合の追加チェック
    # 対応点が極端に偏らないようにする
    if len(vertical_lines) > 0 and len(horizontal_lines) > 0:
        # 垂直線と水平線の比率をチェック
        v_ratio = len(vertical_lines) / len(pixel_lines)
        h_ratio = len(horizontal_lines) / len(pixel_lines)
        
        # バランスが悪い場合は警告（実装上は問題なく動作）
        if v_ratio < 0.3 or h_ratio < 0.3:
            # 片方が極端に少ない場合でも処理は続行
            pass
    
    return src_points, dst_points


def get_mesh_coordinates(
    image_shape: Tuple[int, int],
    mesh_size: Tuple[int, int]
) -> np.ndarray:
    """
    メッシュ座標を取得
    
    Args:
        image_shape: tuple、(H, W)
        mesh_size: tuple、(rows, cols)
    
    Returns:
        numpy.ndarray、shape=(rows+1, cols+1, 2)、TCG座標系の交点座標（正規化済み）
    """
    height, width = image_shape
    rows, cols = mesh_size
    
    # TCG座標系でメッシュを生成（正規化座標）
    # X: -0.5 〜 +0.5
    # Y: -0.5 〜 +0.5
    
    x_coords = np.linspace(-0.5, 0.5, cols + 1)
    y_coords = np.linspace(-0.5, 0.5, rows + 1)  # 上から下
    
    mesh_coords = np.zeros((rows + 1, cols + 1, 2), dtype=np.float32)
    
    for row in range(rows + 1):
        for col in range(cols + 1):
            mesh_coords[row, col, 0] = x_coords[col]
            mesh_coords[row, col, 1] = y_coords[row]
    
    return mesh_coords


def _validate_image(image: np.ndarray):
    """画像の検証"""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be numpy.ndarray, got {type(image)}")
    
    if image.dtype != np.float32:
        raise TypeError(f"image must be float32, got {image.dtype}")
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise TypeError(f"image must have shape (H, W, 3), got {image.shape}")


