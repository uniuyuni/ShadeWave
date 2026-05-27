
from __future__ import annotations

import math

import cv2
import numpy as np

"""
楕円グラデーションマスクのラスタ化（Kivy 非依存）。
widgets.mask_editor2.CircularGradientMask.draw_elliptical_gradient と同一アルゴリズム。
"""
def draw_elliptical_gradient(
    image_size,
    center,
    inner_axes,
    outer_axes,
    angle_rad,
    invert=False,
    smoothness=1,
):
    width, height = image_size

    if width <= 0 or height <= 0:
        return np.zeros((height, width), dtype=np.float32)

    angle_rad = -angle_rad

    rx_in, ry_in = inner_axes
    rx_out, ry_out = outer_axes

    rx_mid = (rx_in + rx_out) / 2.0
    ry_mid = (ry_in + ry_out) / 2.0

    sigma_x = abs(rx_out - rx_in) * 0.25 * smoothness
    sigma_y = abs(ry_out - ry_in) * 0.25 * smoothness

    if sigma_x < 0.1 and sigma_y < 0.1:
        sigma_x = 0.1
        sigma_y = 0.1

    target_sigma = 4.0
    min_sigma = min(sigma_x, sigma_y)
    dest_scale = target_sigma / min_sigma if min_sigma > target_sigma else 1.0
    dest_scale = min(dest_scale, 1.0)

    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )

    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)

    corners_centered = corners - center
    x_rot = corners_centered[:, 0] * cos_a - corners_centered[:, 1] * sin_a
    y_rot = corners_centered[:, 0] * sin_a + corners_centered[:, 1] * cos_a

    min_x = np.min(x_rot)
    max_x = np.max(x_rot)
    min_y = np.min(y_rot)
    max_y = np.max(y_rot)

    src_scale = dest_scale
    eff_sigma_x = sigma_x * src_scale
    eff_sigma_y = sigma_y * src_scale

    pad_x = int(math.ceil(3.0 * eff_sigma_x))
    pad_y = int(math.ceil(3.0 * eff_sigma_y))

    unrot_w = max_x - min_x
    unrot_h = max_y - min_y

    src_w = int(math.ceil(unrot_w * src_scale)) + 2 * pad_x
    src_h = int(math.ceil(unrot_h * src_scale)) + 2 * pad_y

    src_origin_x = min_x * src_scale - pad_x
    src_origin_y = min_y * src_scale - pad_y

    ell_cx = -src_origin_x
    ell_cy = -src_origin_y

    src_img = np.zeros((src_h, src_w), dtype=np.float32)

    if invert is False:
        bg_color = 1.0
        fg_color = 0.0
    else:
        bg_color = 0.0
        fg_color = 1.0

    src_img.fill(bg_color)

    cv2.ellipse(
        src_img,
        (int(ell_cx), int(ell_cy)),
        (int(rx_mid * src_scale), int(ry_mid * src_scale)),
        0,
        0,
        360,
        color=fg_color,
        thickness=-1,
    )

    src_img = cv2.GaussianBlur(src_img, (0, 0), sigmaX=eff_sigma_x, sigmaY=eff_sigma_y)

    dest_w = int(width * dest_scale)
    dest_h = int(height * dest_scale)

    if dest_w <= 0 or dest_h <= 0:
        return np.zeros((height, width), dtype=np.float32)

    cos_v = np.cos(angle_rad)
    sin_v = np.sin(angle_rad)

    a00 = cos_v
    a01 = -sin_v
    a10 = sin_v
    a11 = cos_v

    ox = src_origin_x
    oy = src_origin_y

    cx = center[0] * dest_scale
    cy = center[1] * dest_scale

    tx = ox * cos_v - oy * sin_v + cx
    ty = ox * sin_v + oy * cos_v + cy

    M = np.array([[a00, a01, tx], [a10, a11, ty]], dtype=np.float32)

    border_val = bg_color

    dst_small = cv2.warpAffine(
        src_img,
        M,
        (dest_w, dest_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_val),
    )

    if dest_scale < 1.0:
        dst_img = cv2.resize(dst_small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        dst_img = dst_small

    return dst_img

"""
線形グラデーションマスクのラスタ化（Kivy 非依存）。
widgets.mask_editor2.GradientMask.draw_gradient と同一アルゴリズム。
"""
def draw_linear_gradient(image_size, center, start_point, end_point, smoothness=1):
    width, height = image_size

    start_x, start_y = end_point
    end_x, end_y = start_point
    vec_start_end = np.array([end_x - start_x, end_y - start_y])
    length_start_end = np.linalg.norm(vec_start_end)

    if length_start_end == 0:
        return np.zeros((height, width), dtype=np.float32)

    sigma = (length_start_end * 0.25) * smoothness
    if sigma < 0.1:
        img = np.zeros((height, width), dtype=np.float32)
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        unit_vec = vec_start_end / length_start_end
        y_coords, x_coords = np.indices((height, width))
        projected = (x_coords - mid_x) * unit_vec[0] + (y_coords - mid_y) * unit_vec[1]
        img[projected >= 0] = 1.0
        return img

    target_sigma = 4.0
    scale = target_sigma / sigma if sigma > target_sigma else 1.0
    scale = min(scale, 1.0)

    small_w = int(math.ceil(width * scale))
    small_h = int(math.ceil(height * scale))

    if small_w <= 0 or small_h <= 0:
        return np.zeros((height, width), dtype=np.float32)

    img_small = np.zeros((small_h, small_w), dtype=np.float32)

    start_x_s = start_x * scale
    start_y_s = start_y * scale
    end_x_s = end_x * scale
    end_y_s = end_y * scale
    mid_x_s = (start_x_s + end_x_s) / 2
    mid_y_s = (start_y_s + end_y_s) / 2

    vec_s = np.array([end_x_s - start_x_s, end_y_s - start_y_s])
    len_s = np.linalg.norm(vec_s)
    if len_s == 0:
        return np.zeros((height, width), dtype=np.float32)
    unit_vec_s = vec_s / len_s

    y_coords_s, x_coords_s = np.indices((small_h, small_w))
    projected_s = (x_coords_s - mid_x_s) * unit_vec_s[0] + (y_coords_s - mid_y_s) * unit_vec_s[1]

    img_small[projected_s >= 0] = 1.0

    eff_sigma = sigma * scale
    img_small = cv2.GaussianBlur(img_small, (0, 0), sigmaX=eff_sigma, sigmaY=eff_sigma)

    if scale < 1.0:
        img = cv2.resize(img_small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        img = img_small

    return img


"""
FreeDrawMask の線ラスタ化（mask_editor2.FreeDrawMask と同一ロジック）。
"""
class Line:
    def __init__(self, is_erasing=False, size=10, soft=100):
        self.is_erasing = is_erasing
        self.size = size
        self.soft = soft
        self.points = []

    def add_point(self, x, y):
        self.points.append((x, y))


class _Raster:
    def create_natural_brush(self, size, hardness=100):
        brush_size = int(size)
        brush_radius = brush_size // 2
        kernel = np.zeros((brush_size, brush_size), np.float32)
        hardness = float(hardness)
        hard_ratio = np.clip(hardness / 100.0, 0.0, 1.0)
        inner_radius = int(max(1, brush_radius * hard_ratio))
        cv2.circle(kernel, (brush_radius, brush_radius), inner_radius, 1, -1)
        if hard_ratio < 1.0:
            softness = 1.0 - hard_ratio
            kz = int(max(1, brush_radius * softness * 2.0)) | 1
            kernel = cv2.GaussianBlur(kernel, (kz, kz), 0)
        return kernel

    def clip_mask_range(self, image, allow_over_one=False, allow_under_zero=False):
        min_value = None if allow_under_zero else 0
        max_value = None if allow_over_one else 1
        if min_value is None and max_value is None:
            return image
        return np.clip(image, min_value, max_value)

    def apply_brush_at_point(
        self,
        image,
        x,
        y,
        brush,
        is_erasing=False,
        opacity=1.0,
        blend_mode="max",
        allow_over_one=False,
        allow_under_zero=False,
    ):
        if brush.size == 0:
            return
        brush_h, brush_w = brush.shape
        brush_center_x, brush_center_y = brush_w // 2, brush_h // 2
        img_y_min = int(y - brush_center_y)
        img_y_max = int(y - brush_center_y + brush_h)
        img_x_min = int(x - brush_center_x)
        img_x_max = int(x - brush_center_x + brush_w)
        img_h, img_w = image.shape
        img_y_min_clipped = max(0, img_y_min)
        img_y_max_clipped = min(img_h, img_y_max)
        img_x_min_clipped = max(0, img_x_min)
        img_x_max_clipped = min(img_w, img_x_max)
        if img_y_min_clipped >= img_y_max_clipped or img_x_min_clipped >= img_x_max_clipped:
            return
        brush_y_min = img_y_min_clipped - img_y_min
        brush_y_max = brush_y_min + (img_y_max_clipped - img_y_min_clipped)
        brush_x_min = img_x_min_clipped - img_x_min
        brush_x_max = brush_x_min + (img_x_max_clipped - img_x_min_clipped)
        brush_h, brush_w = brush.shape
        brush_y_min = max(0, min(brush_h - 1, brush_y_min))
        brush_y_max = max(brush_y_min + 1, min(brush_h, brush_y_max))
        brush_x_min = max(0, min(brush_w - 1, brush_x_min))
        brush_x_max = max(brush_x_min + 1, min(brush_w, brush_x_max))
        try:
            brush_part = brush[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
            if brush_part.size == 0:
                return
            brush_part = brush_part * opacity
            target_region = image[img_y_min_clipped:img_y_max_clipped, img_x_min_clipped:img_x_max_clipped]
            if blend_mode == "max":
                image[img_y_min_clipped:img_y_max_clipped, img_x_min_clipped:img_x_max_clipped] = np.maximum(
                    target_region, brush_part
                )
            elif is_erasing:
                image[img_y_min_clipped:img_y_max_clipped, img_x_min_clipped:img_x_max_clipped] = self.clip_mask_range(
                    target_region - brush_part, allow_over_one, allow_under_zero
                )
            else:
                image[img_y_min_clipped:img_y_max_clipped, img_x_min_clipped:img_x_max_clipped] = self.clip_mask_range(
                    target_region + brush_part, allow_over_one, allow_under_zero
                )
        except (IndexError, ValueError):
            pass

    def draw_smooth_line(
        self,
        image,
        points,
        brush_size,
        softness,
        is_erasing=False,
        blend_mode="add",
        allow_over_one=False,
        allow_under_zero=False,
    ):
        if len(points) == 0:
            return
        brush = self.create_natural_brush(brush_size, softness)
        if len(points) == 1:
            p = points[0]
            self.apply_brush_at_point(
                image,
                int(p[0]),
                int(p[1]),
                brush,
                is_erasing,
                blend_mode=blend_mode,
                allow_over_one=allow_over_one,
                allow_under_zero=allow_under_zero,
            )
            return
        texture_points = points
        opacity = 1.0 if blend_mode == "max" else 0.5
        for i in range(len(texture_points) - 1):
            p1 = texture_points[i]
            p2 = texture_points[i + 1]
            distance = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            steps = max(1, int(distance / (brush_size * 0.05)))
            for j in range(steps + 1):
                t = j / max(1, steps)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                self.apply_brush_at_point(
                    image,
                    int(x),
                    int(y),
                    brush,
                    is_erasing,
                    opacity,
                    blend_mode=blend_mode,
                    allow_over_one=allow_over_one,
                    allow_under_zero=allow_under_zero,
                )

    def draw_line(self, image_size, lines, allow_over_one=False, allow_under_zero=False):
        try:
            width, height = image_size
            if width <= 0 or height <= 0:
                return np.zeros((100, 100), dtype=np.float32)
            image = np.zeros((height, width), dtype=np.float32)
            for line in lines:
                if len(line.points) == 0:
                    continue
                is_erasing = line.is_erasing
                brush_size = line.size
                brush_soft = line.soft
                pts = np.array(line.points)
                min_x = np.min(pts[:, 0]) - brush_size
                max_x = np.max(pts[:, 0]) + brush_size
                min_y = np.min(pts[:, 1]) - brush_size
                max_y = np.max(pts[:, 1]) + brush_size
                min_x = max(0, int(min_x))
                max_x = min(width, int(max_x))
                min_y = max(0, int(min_y))
                max_y = min(height, int(max_y))
                if min_x >= max_x or min_y >= max_y:
                    continue
                stroke_buffer = np.zeros((max_y - min_y, max_x - min_x), dtype=np.float32)
                local_points = []
                for p in line.points:
                    local_points.append((p[0] - min_x, p[1] - min_y))
                self.draw_smooth_line(stroke_buffer, local_points, brush_size, brush_soft, False, blend_mode="max")
                target_region = image[min_y:max_y, min_x:max_x]
                if is_erasing:
                    image[min_y:max_y, min_x:max_x] = self.clip_mask_range(
                        target_region - stroke_buffer, allow_over_one, allow_under_zero
                    )
                else:
                    image[min_y:max_y, min_x:max_x] = self.clip_mask_range(
                        target_region + stroke_buffer, allow_over_one, allow_under_zero
                    )
            return image
        except Exception:
            return np.zeros((max(1, image_size[1]), max(1, image_size[0])), dtype=np.float32)


_r = _Raster()


def draw_line_texture(image_size, lines, allow_over_one=False, allow_under_zero=False):
    return _r.draw_line(image_size, lines, allow_over_one, allow_under_zero)


"""
PolylineMask の線/塗りつぶしラスタ化。

FreeDrawMask のブラシ点描とは異なり、頂点列を OpenCV の
cv2.polylines / cv2.fillPoly でアンチエイリアス描画する。
soft (= hardness, 0..100) でエッジに Gaussian ぼかしを足して柔らかくする。
"""
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


_FREE = _Raster()


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
