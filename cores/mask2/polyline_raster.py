"""
PolylineMask の線/塗りつぶしラスタ化。

FreeDrawMask のブラシ点描とは異なり、頂点列を OpenCV の
cv2.polylines / cv2.fillPoly でアンチエイリアス描画する。
soft (= hardness, 0..100) でエッジに Gaussian ぼかしを足して柔らかくする。
"""
from __future__ import annotations

import cv2
import numpy as np

from cores.mask2.freedraw_raster import _Raster as _FreeDrawRaster


class Polyline:
    """1 本の折れ線。

    points  : 頂点の (x, y) リスト (テクスチャ座標)
    is_erasing: 消去ポリラインかどうか
    size    : 線幅 (テクスチャ座標)。fill 時は無視されない (輪郭にも適用)
    soft    : 0..100, 100=ハード, 0=最大限ぼかし
    is_closed: True なら最終辺で始点と結ぶ
    is_filled: True かつ is_closed=True のとき内部を塗りつぶす
    """

    def __init__(self, is_erasing: bool = False, size: float = 10.0, soft: float = 100.0,
                 is_closed: bool = False, is_filled: bool = True):
        self.is_erasing = bool(is_erasing)
        self.size = float(size)
        self.soft = float(soft)
        self.is_closed = bool(is_closed)
        self.is_filled = bool(is_filled)
        self.points: list[tuple[float, float]] = []

    def add_point(self, x: float, y: float):
        self.points.append((float(x), float(y)))


_FREE = _FreeDrawRaster()


def _stroke_polyline(stroke_buf: np.ndarray, polyline: Polyline) -> None:
    """stroke_buf (float32, HxW) に polyline 1 本を 1.0 値で焼き込む。is_erasing は呼び出し側で吸収。"""
    pts = np.array(polyline.points, dtype=np.int32).reshape(-1, 1, 2)
    if pts.shape[0] == 0:
        return
    thickness = max(1, int(round(polyline.size)))

    if polyline.is_filled and polyline.is_closed and pts.shape[0] >= 3:
        # 内部塗りつぶし。輪郭線でなじませる場合は別途 polylines を重ねるが、
        # fillPoly のアンチエイリアスは精度が落ちるので最終的にエッジ Gaussian でなじませる。
        cv2.fillPoly(stroke_buf, [pts], 1.0, lineType=cv2.LINE_AA)
    else:
        if pts.shape[0] == 1:
            # 1 点だけなら円で描画
            x, y = int(polyline.points[0][0]), int(polyline.points[0][1])
            radius = max(1, thickness // 2)
            cv2.circle(stroke_buf, (x, y), radius, 1.0, -1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(
                stroke_buf,
                [pts],
                isClosed=bool(polyline.is_closed),
                color=1.0,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    # ハードネスに応じたエッジぼかし
    hardness = max(0.0, min(100.0, polyline.soft))
    if hardness < 100.0:
        # thickness の半分くらいまでぼかす
        softness = (100.0 - hardness) / 100.0
        sigma = max(0.5, thickness * 0.5 * softness)
        # ksize は奇数で 2..6 sigma 程度
        kz = int(max(3, sigma * 4)) | 1
        cv2.GaussianBlur(stroke_buf, (kz, kz), sigma, dst=stroke_buf)


def draw_polyline_texture(
    image_size: tuple[int, int],
    polylines: list[Polyline],
    allow_over_one: bool = False,
    allow_under_zero: bool = False,
) -> np.ndarray:
    """polyline 群を 1 枚のマスク画像に焼き込む。

    image_size: (width, height)
    polylines : Polyline のリスト (先頭から順に積み上げる)
    戻り値    : float32 (H, W), 値域 [0,1] (allow_over/under で拡張可)
    """
    try:
        width, height = image_size
        if width <= 0 or height <= 0:
            return np.zeros((100, 100), dtype=np.float32)
        image = np.zeros((height, width), dtype=np.float32)

        for poly in polylines:
            if len(poly.points) == 0:
                continue
            # 個別に stroke_buf を作って後で加減算
            pts_arr = np.array(poly.points)
            margin = max(int(poly.size), 8)
            min_x = max(0, int(np.min(pts_arr[:, 0])) - margin)
            max_x = min(width, int(np.max(pts_arr[:, 0])) + margin)
            min_y = max(0, int(np.min(pts_arr[:, 1])) - margin)
            max_y = min(height, int(np.max(pts_arr[:, 1])) + margin)
            if min_x >= max_x or min_y >= max_y:
                continue
            stroke_buf = np.zeros((max_y - min_y, max_x - min_x), dtype=np.float32)
            local = Polyline(
                is_erasing=poly.is_erasing,
                size=poly.size,
                soft=poly.soft,
                is_closed=poly.is_closed,
                is_filled=poly.is_filled,
            )
            for px, py in poly.points:
                local.add_point(px - min_x, py - min_y)
            _stroke_polyline(stroke_buf, local)

            target = image[min_y:max_y, min_x:max_x]
            if poly.is_erasing:
                image[min_y:max_y, min_x:max_x] = _FREE.clip_mask_range(
                    target - stroke_buf, allow_over_one, allow_under_zero
                )
            else:
                image[min_y:max_y, min_x:max_x] = _FREE.clip_mask_range(
                    target + stroke_buf, allow_over_one, allow_under_zero
                )
        return image
    except Exception:
        return np.zeros((max(1, image_size[1]), max(1, image_size[0])), dtype=np.float32)
