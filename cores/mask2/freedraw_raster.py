"""
FreeDrawMask の線ラスタ化（mask_editor2.FreeDrawMask と同一ロジック）。
"""
from __future__ import annotations

import math

import cv2
import numpy as np


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
