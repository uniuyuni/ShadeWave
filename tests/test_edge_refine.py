import pathlib
import sys
import unittest

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cores.core as core
from cores.mask2.coordinate_context import Mask2CoordinateContext
from cores.mask2 import edge_refine, extended_params, mask_rasters
from cores.mask2.edge_refine import refine_mask_edge_aware
from cores.mask2.headless_masks import (
    HeadlessCircularGradientMask,
    HeadlessFreeDrawMask,
    HeadlessFullMask,
    HeadlessGradientMask,
)


def _snow_like_scene_and_u_stroke():
    h, w = 180, 240
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[:, :] = (0.12, 0.10, 0.28)

    xs = np.arange(w)
    edge_y = 58 + 48 * (1 - ((xs - 120) / 78) ** 2)
    edge_y = np.clip(edge_y, 40, 112).astype(np.int32)

    cloud = np.zeros((h, w), dtype=np.uint8)
    for x, y in enumerate(edge_y):
        if 40 <= x <= 200:
            cloud[:y, x] = 1

    rng = np.random.default_rng(3)
    cloud_noise = rng.normal(0.0, 0.03, (int(cloud.sum()), 3)).astype(np.float32)
    image[cloud.astype(bool)] = np.array([0.82, 0.82, 0.96], dtype=np.float32) + cloud_noise

    # Snow-covered tree-like texture below the cloud edge. This makes the edge
    # map busy like the user's debug image without changing the intended edge.
    for x in (58, 75, 94, 148, 166, 183):
        y = int(edge_y[x] + 10)
        tri = np.array([[x, y - 35], [x - 9, y + 18], [x + 9, y + 18]], dtype=np.int32)
        cv2.fillConvexPoly(image, tri, (0.72, 0.72, 0.90))

    stroke = []
    for x in np.linspace(58, 182, 44):
        xi = int(round(x))
        stroke.append([xi, int(edge_y[xi] - 3)])
    stroke = np.asarray(stroke, dtype=np.int32).reshape((-1, 1, 2))
    mask_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask_u8, [stroke], False, 255, 22, cv2.LINE_AA)
    return np.clip(image, 0.0, 1.0), mask_u8.astype(np.float32) / 255.0, edge_y


def _snow_like_u_stroke_line(edge_y, size=22, offset=-3):
    line = mask_rasters.Line(False, size, 100)
    for x in np.linspace(58, 182, 44):
        xi = int(round(x))
        line.add_point(xi, int(edge_y[xi] + offset))
    return line


def _photo_like_cloud_scene():
    h, w = 220, 320
    rng = np.random.default_rng(23)
    image = np.zeros((h, w, 3), dtype=np.float32)

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    sky_top = np.array([0.17, 0.16, 0.36], dtype=np.float32)
    sky_bottom = np.array([0.42, 0.36, 0.63], dtype=np.float32)
    image[:, :] = sky_top * (1.0 - yy[..., None]) + sky_bottom * yy[..., None]

    xs = np.arange(w, dtype=np.float32)
    rough = rng.normal(0.0, 1.0, w).astype(np.float32)
    rough = cv2.GaussianBlur(rough[None, :], (0, 0), 5.0).reshape(-1)
    edge_y = (
        78
        + 19 * np.sin((xs - 30) / 42.0)
        + 8 * np.sin(xs / 15.0)
        + rough * 10
    )
    edge_y = np.clip(edge_y, 48, 132).astype(np.int32)

    cloud = np.zeros((h, w), dtype=bool)
    for x, y in enumerate(edge_y):
        cloud[:y, x] = True

    cloud_noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    cloud_noise = cv2.GaussianBlur(cloud_noise, (0, 0), 6.0)
    cloud_noise = (cloud_noise - cloud_noise.min()) / max(float(cloud_noise.max() - cloud_noise.min()), 1e-6)
    cloud_base = np.array([0.72, 0.72, 0.88], dtype=np.float32)
    cloud_shadow = np.array([0.46, 0.45, 0.67], dtype=np.float32)
    cloud_color = cloud_shadow + (cloud_base - cloud_shadow) * cloud_noise[..., None]
    image[cloud] = cloud_color[cloud]

    # Busy snow/tree texture below the cloud boundary, close to the user's
    # troublesome image: many small high-contrast edges that should not become
    # the selected boundary.
    for x in np.linspace(35, w - 35, 15):
        xi = int(round(x + rng.normal(0.0, 5.0)))
        top = int(edge_y[np.clip(xi, 0, w - 1)] + rng.integers(8, 36))
        height = int(rng.integers(32, 76))
        half = int(rng.integers(5, 14))
        pts = np.array(
            [[xi, top], [xi - half, min(h - 1, top + height)], [xi + half, min(h - 1, top + height)]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, pts, (0.72, 0.72, 0.91))
        cv2.line(image, (xi, top + 4), (xi, min(h - 1, top + height)), (0.12, 0.12, 0.22), 1)

    image += rng.normal(0.0, 0.018, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0), edge_y


def _photo_like_cloud_line(edge_y, size=26, offset=-2, accidental_cross=False):
    line = mask_rasters.Line(False, size, 100)
    for x in np.linspace(36, len(edge_y) - 36, 74):
        xi = int(round(x))
        y = float(edge_y[xi] + offset)
        if accidental_cross:
            y += 18.0 * np.exp(-((x - 170.0) ** 2) / (2.0 * 16.0 ** 2))
        line.add_point(xi, int(round(y)))
    return line


def _curve_side_mask(edge_y, shape, side, margin):
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    for x, y in enumerate(edge_y):
        if not (40 <= x <= 200):
            continue
        if side == "below":
            out[min(h, y + margin):, x] = True
        else:
            out[:max(0, y - margin), x] = True
    return out


class EdgeRefineTest(unittest.TestCase):
    def test_legacy_modes_normalize_to_quick_select(self):
        self.assertEqual(edge_refine.normalize_mode("Grow"), "Quick Select")
        self.assertEqual(edge_refine.normalize_mode("Lock"), "Quick Select")
        self.assertEqual(edge_refine.normalize_mode("Refine"), "Quick Select")

    def test_quick_select_does_not_cross_different_colored_region(self):
        h, w = 80, 120
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :55] = (0.4, 0.7, 1.0)
        image[:, 55:] = (0.1, 0.5, 0.1)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[34:46, 35:50] = 1.0

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(42, 40),
            mode="Quick Select",
            radius=45,
            strength=80,
        )

        self.assertGreater(float(refined[36:44, 35:52].mean()), 0.5)
        self.assertLess(float(refined[:, 65:].max()), 0.01)

    def test_quick_select_expands_into_similar_color_inside_radius(self):
        h, w = 80, 120
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :] = (0.4, 0.7, 1.0)
        image[:, 90:] = (0.1, 0.5, 0.1)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[34:46, 20:30] = 1.0

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(25, 40),
            mode="Quick Select",
            radius=35,
            strength=80,
        )

        self.assertGreater(float(refined[36:44, 45:55].mean()), 0.5)
        self.assertLess(float(refined[:, 95:].max()), 0.01)

    def test_draw_edge_snap_can_reach_nearby_edge_but_does_not_cross_it(self):
        h, w = 64, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        line = mask_rasters.Line(False, 8, 100)
        line.add_point(20, 32)
        line.add_point(42, 32)
        mask = mask_rasters.draw_line_texture((w, h), [line])
        self.assertLess(float(mask[32, 55]), 0.01)

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(20, 32),
            mode="Quick Select",
            radius=24,
            strength=80,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertGreater(float(refined[32, 48]), 0.5)
        self.assertLess(float(refined[32, 55]), 0.01)
        self.assertLess(float(refined[:, 55:].max()), 0.01)

    def test_draw_edge_snap_radius_controls_edge_ribbon_reach(self):
        h, w = 64, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        line = mask_rasters.Line(False, 6, 100)
        line.add_point(22, 32)
        line.add_point(38, 32)
        mask = mask_rasters.draw_line_texture((w, h), [line])

        small_radius = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(22, 32),
            mode="Quick Select",
            radius=3,
            strength=80,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )
        large_radius = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(22, 32),
            mode="Quick Select",
            radius=16,
            strength=80,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertLess(float(mask[32, 49]), 0.01)
        self.assertLess(float(small_radius[32, 49]), 0.01)
        self.assertGreater(float(large_radius[32, 49]), 0.5)
        self.assertLess(float(large_radius[:, 55:].max()), 0.01)

    def test_draw_stroke_crossing_edge_can_select_both_sides(self):
        h, w = 64, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        line = mask_rasters.Line(False, 16, 100)
        line.add_point(20, 32)
        line.add_point(80, 32)
        mask = mask_rasters.draw_line_texture((w, h), [line])

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(20, 32),
            mode="Quick Select",
            radius=24,
            strength=80,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertGreater(float(refined[32, 35]), 0.5)
        self.assertGreater(float(refined[32, 65]), 0.5)

    def test_circular_like_quick_select_from_guide_does_not_disappear(self):
        h, w = 100, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        mask = mask_rasters.draw_elliptical_gradient(
            (w, h),
            (25, 50),
            (5, 5),
            (15, 15),
            0.0,
            invert=True,
            smoothness=1.5,
        )

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(25, 50),
            mode="Quick Select",
            radius=30,
            strength=80,
            seed_from_guide=True,
            fill_grown_region=False,
        )

        self.assertGreater(float(refined[50, 25]), 0.5)
        self.assertLess(float(refined[50, 55]), 0.01)

    def test_circular_hint_uses_shape_even_when_center_side_is_inverse(self):
        h, w = 100, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        mask = mask_rasters.draw_elliptical_gradient(
            (w, h),
            (25, 50),
            (5, 5),
            (15, 15),
            0.0,
            invert=False,
            smoothness=1.5,
        )
        self.assertLess(float(mask[50, 25]), 0.1)

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(25, 50),
            mode="Quick Select",
            radius=3,
            strength=80,
            seed_from_guide=True,
            fill_grown_region=False,
        )

        self.assertGreater(float(refined[50, 35]), 0.2)
        self.assertLess(float(refined[50, 35]), 0.8)
        self.assertLess(float(refined[50, 55]), 0.01)

    def test_circular_quick_select_preserves_gradient_alpha(self):
        h, w = 100, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        mask = mask_rasters.draw_elliptical_gradient(
            (w, h),
            (25, 50),
            (5, 5),
            (15, 15),
            0.0,
            invert=True,
            smoothness=1.5,
        )

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(25, 50),
            mode="Quick Select",
            radius=30,
            strength=80,
            seed_from_guide=True,
            fill_grown_region=False,
        )

        self.assertAlmostEqual(float(refined[50, 35]), float(mask[50, 35]), delta=0.02)
        self.assertLess(float(refined[50, 55]), 0.01)

    def test_circular_quick_select_ignores_texture_noise_inside_region(self):
        h, w = 160, 160
        rng = np.random.default_rng(1)
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :] = (0.2, 0.65, 0.25)
        image[:, 90:] = (0.45, 0.7, 1.0)
        image[:, :90] += rng.normal(0.0, 0.035, (h, 90, 3)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        mask = mask_rasters.draw_elliptical_gradient(
            (w, h),
            (50, 80),
            (10, 10),
            (55, 55),
            0.0,
            invert=True,
            smoothness=1.5,
        )

        refined, support = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(50, 80),
            mode="Quick Select",
            radius=60,
            strength=80,
            seed_from_guide=True,
            fill_grown_region=False,
            return_support=True,
        )

        self.assertGreater(int(support.sum()), 8000)
        self.assertGreater(float(refined[80, 80]), 0.2)
        self.assertLess(float(refined[:, 100:].max()), 0.01)

    def test_circular_quick_select_does_not_disappear_on_dense_texture_edges(self):
        h, w = 140, 140
        rng = np.random.default_rng(4)
        image = rng.random((h, w, 3), dtype=np.float32)
        mask = mask_rasters.draw_elliptical_gradient(
            (w, h),
            (70, 70),
            (10, 10),
            (35, 35),
            0.0,
            invert=False,
            smoothness=1.5,
        )

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(70, 70),
            mode="Quick Select",
            radius=40,
            strength=100,
            seed_from_guide=True,
            fill_grown_region=False,
        )

        self.assertGreater(float(refined.sum()), float(mask.sum()) * 0.05)

    def test_strength_tightens_edge_lock_without_changing_radius(self):
        h, w = 80, 120
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :55] = (0.40, 0.70, 1.00)
        image[:, 55:] = (0.44, 0.70, 0.96)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[34:46, 42:52] = 1.0

        loose = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(47, 40),
            mode="Quick Select",
            radius=35,
            strength=0,
        )
        locked = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(47, 40),
            mode="Quick Select",
            radius=35,
            strength=100,
        )

        self.assertLess(float(locked.sum()), float(loose.sum()))
        self.assertLess(float(locked[:, 65:].max()), 0.01)

    def test_edge_lock_zero_creates_no_edge_walls(self):
        rng = np.random.default_rng(7)
        image = rng.random((80, 80, 3), dtype=np.float32)

        stop = edge_refine._make_edge_stop_mask(image, 0)

        self.assertEqual(int(stop.sum()), 0)

    def test_edge_lock_does_not_punch_holes_on_barrier_pixels(self):
        h, w = 64, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.20, 0.70, 0.20)
        image[:, 50:] = (0.40, 0.70, 1.00)
        mask = np.ones((h, w), dtype=np.float32)
        seed_mask = np.zeros_like(mask, dtype=bool)
        seed_mask[28:36, 24:34] = True

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(28, 32),
            mode="Quick Select",
            radius=60,
            strength=100,
            seed_mask=seed_mask,
            fill_grown_region=False,
        )

        self.assertGreater(float(refined[28:36, 48:52].mean()), 0.1)
        self.assertLess(float(refined[:, 58:].max()), 0.01)

    def test_quick_select_blocks_similar_color_across_strong_edge(self):
        h, w = 80, 120
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :55] = (0.40, 0.70, 1.00)
        image[:, 55:] = (0.45, 0.72, 0.95)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[34:46, 42:52] = 1.0

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(47, 40),
            mode="Quick Select",
            radius=35,
            strength=80,
        )

        self.assertGreater(float(refined[36:44, 45:53].mean()), 0.5)
        self.assertLess(float(refined[:, 65:].max()), 0.01)

    def test_small_radius_strength_does_not_fall_back_to_raw_mask(self):
        h, w = 64, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        line = mask_rasters.Line(False, 16, 100)
        line.add_point(20, 32)
        line.add_point(46, 32)
        mask = mask_rasters.draw_line_texture((w, h), [line])

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(20, 32),
            mode="Quick Select",
            radius=1,
            strength=80,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertFalse(np.allclose(refined, mask))
        self.assertLess(float(refined[32, 55]), 0.01)

    def test_draw_component_snap_keeps_final_mask_when_it_crosses_edge(self):
        h, w = 64, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[24:40, 20:80] = 1.0
        seed_mask = np.zeros_like(mask, dtype=bool)
        seed_mask[28:36, 24:34] = True

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(28, 32),
            mode="Quick Select",
            radius=1,
            strength=80,
            seed_mask=seed_mask,
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertGreater(float(refined[28:36, 24:44].mean()), 0.5)
        self.assertGreater(float(refined[28:36, 56:76].mean()), 0.5)
        self.assertLess(float(refined[:, 86:].max()), 0.01)

    def test_draw_quick_select_zero_lock_still_respects_strong_edges(self):
        h, w = 64, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[24:40, 20:80] = 1.0
        seed_mask = np.zeros_like(mask, dtype=bool)
        seed_mask[28:36, 24:34] = True

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(28, 32),
            mode="Quick Select",
            radius=24,
            strength=0,
            seed_mask=seed_mask,
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertGreater(float(refined[28:36, 24:44].mean()), 0.5)
        self.assertGreater(float(refined[28:36, 56:76].mean()), 0.5)
        self.assertLess(float(refined[:, 86:].max()), 0.01)

    def test_draw_edge_snap_keeps_result_on_seed_side_of_strong_edge(self):
        h, w = 80, 120
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :60] = (0.2, 0.7, 0.2)
        image[:, 60:] = (0.4, 0.7, 1.0)
        line = mask_rasters.Line(False, 10, 100)
        line.add_point(25, 40)
        line.add_point(45, 40)
        mask = mask_rasters.draw_line_texture((w, h), [line])

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(35, 40),
            mode="Quick Select",
            radius=24,
            strength=100,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertGreater(float(refined[40, 45]), 0.5)
        self.assertGreater(float(refined[40, 55]), 0.5)
        self.assertLess(float(refined[40, 65]), 0.01)

    def test_draw_edge_snap_does_not_tunnel_through_edge(self):
        h, w = 64, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.40, 0.70, 1.00)
        image[:, 50:] = (0.45, 0.72, 0.95)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[24:40, 20:44] = 1.0

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(28, 32),
            mode="Quick Select",
            radius=24,
            strength=80,
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertGreater(float(refined[28:36, 24:44].mean()), 0.5)
        self.assertLess(float(refined[:, 56:].max()), 0.01)

    def test_draw_edge_snap_snow_u_stroke_radius_one_lock_zero_clips_inside_edge(self):
        image, mask, edge_y = _snow_like_scene_and_u_stroke()
        sky_side = _curve_side_mask(edge_y, mask.shape, "below", 7)

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(120, 92),
            mode="Quick Select",
            radius=1,
            strength=0,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        far_sky_side = _curve_side_mask(edge_y, mask.shape, "below", 20)
        self.assertLess(float(refined[far_sky_side].sum()), 1.0)
        self.assertLess(float(refined.sum()), float(mask.sum()) * 1.1)
        self.assertGreater(float(refined.sum()), float(mask.sum()) * 0.70)
        self.assertLess(float(refined[sky_side].sum()), float(mask[sky_side].sum()) * 0.70)

    def test_draw_edge_snap_snow_u_stroke_large_radius_does_not_inflate_outward(self):
        image, mask, edge_y = _snow_like_scene_and_u_stroke()
        sky_side = _curve_side_mask(edge_y, mask.shape, "below", 7)

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(120, 92),
            mode="Quick Select",
            radius=80,
            strength=0,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        far_sky_side = _curve_side_mask(edge_y, mask.shape, "below", 20)
        self.assertLess(float(refined[far_sky_side].sum()), 1.0)
        self.assertLess(float(refined[sky_side].sum()), float(mask[sky_side].sum()) * 0.10)
        self.assertGreater(float(refined.sum()), float(mask.sum()) * 0.50)

    def test_draw_component_snap_uses_final_mask_not_stroke_geometry(self):
        image, mask, edge_y = _snow_like_scene_and_u_stroke()
        sky_side = _curve_side_mask(edge_y, mask.shape, "below", 7)
        far_sky_side = _curve_side_mask(edge_y, mask.shape, "below", 20)

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(120, 92),
            mode="Quick Select",
            radius=1,
            strength=0,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertLess(float(refined[far_sky_side].sum()), 1.0)
        self.assertLess(float(refined.sum()), float(mask.sum()) * 1.1)
        self.assertGreater(float(refined.sum()), float(mask.sum()) * 0.70)
        self.assertLess(float(refined[sky_side].sum()), float(mask[sky_side].sum()) * 0.70)

    def test_draw_component_snap_can_look_outside_final_mask_to_reach_edge(self):
        image, _, edge_y = _snow_like_scene_and_u_stroke()
        line = _snow_like_u_stroke_line(edge_y, offset=-24)
        mask = mask_rasters.draw_line_texture((image.shape[1], image.shape[0]), [line])
        h, w = mask.shape
        edge_band = np.zeros_like(mask, dtype=bool)
        outside_edge = np.zeros_like(mask, dtype=bool)
        for x, y in enumerate(edge_y):
            if not (40 <= x <= 200):
                continue
            edge_band[max(0, int(y) - 3):min(h, int(y) + 3), x] = True
            outside_edge[min(h, int(y) + 8):, x] = True

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(120, 92),
            mode="Quick Select",
            radius=24,
            strength=0,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[line],
        )

        self.assertGreater(float(refined[edge_band].mean()), 0.15)
        self.assertGreater(float(refined[mask <= 0.01].sum()), 100.0)
        self.assertLess(float(refined[outside_edge].sum()), float(refined.sum()) * 0.08)

    def test_draw_edge_snap_photo_like_cloud_boundary_removes_far_sky_side(self):
        image, edge_y = _photo_like_cloud_scene()
        line = _photo_like_cloud_line(edge_y, size=28, offset=-2)
        mask = mask_rasters.draw_line_texture((image.shape[1], image.shape[0]), [line])
        h, w = mask.shape
        edge_band = np.zeros_like(mask, dtype=bool)
        far_sky = np.zeros_like(mask, dtype=bool)
        near_sky = np.zeros_like(mask, dtype=bool)
        for x, y in enumerate(edge_y):
            edge_band[max(0, y - 1):min(h, y + 2), x] = True
            near_sky[min(h, y + 8):, x] = True
            far_sky[min(h, y + 22):, x] = True

        for radius in (1, 8, 28, 72):
            refined = refine_mask_edge_aware(
                image,
                mask,
                guide_point=(160, int(edge_y[160])),
                mode="Quick Select",
                radius=radius,
                strength=70,
                selection_strategy=edge_refine.STRATEGY_DRAW,
                draw_strokes=[line],
            )

            self.assertGreater(float(refined[edge_band].mean()), 0.65)
            self.assertLess(float(refined[near_sky].sum()), float(mask[near_sky].sum()) * 0.45)
            self.assertLess(float(refined[far_sky].sum()), float(refined.sum()) * 0.05)

    def test_draw_edge_snap_photo_like_accidental_crossing_rejects_far_side(self):
        image, edge_y = _photo_like_cloud_scene()
        line = _photo_like_cloud_line(edge_y, size=26, offset=-4, accidental_cross=True)
        mask = mask_rasters.draw_line_texture((image.shape[1], image.shape[0]), [line])
        h, w = mask.shape
        far_sky = np.zeros_like(mask, dtype=bool)
        for x, y in enumerate(edge_y):
            far_sky[min(h, y + 24):, x] = True

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(160, int(edge_y[160])),
            mode="Quick Select",
            radius=30,
            strength=80,
            selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[line],
        )

        self.assertLess(float(refined[far_sky].sum()), float(refined.sum()) * 0.03)
        self.assertLess(float(refined[far_sky].sum()), float(mask[far_sky].sum()) * 2.0)

    def test_draw_edge_snap_ignores_tiny_target_fragment_on_long_stroke(self):
        h, w = 96, 160
        line = mask_rasters.Line(False, 18, 100)
        for x, y in ((24, 24), (30, 58), (72, 70), (116, 58), (132, 24)):
            line.add_point(x, y)
        component = mask_rasters.draw_line_texture((w, h), [line]) > 0.02

        fg_line = mask_rasters.Line(False, 2, 100)
        for x, y in line.points:
            fg_line.add_point(x, y)
        fg_seed = mask_rasters.draw_line_texture((w, h), [fg_line]) > 0.02

        target_edge_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.line(target_edge_u8, (25, 38), (27, 56), 1, 1, cv2.LINE_AA)
        target_edge = target_edge_u8 > 0

        self.assertFalse(
            edge_refine._draw_grabcut_band_target_edge_reliable(
                target_edge,
                fg_seed,
                component,
                half_width=9,
                has_strokes=True,
            )
        )

    def test_draw_target_edge_ignores_weak_boundary_surface_lines(self):
        h, w = 80, 140
        component = np.zeros((h, w), dtype=bool)
        component[28:54, 24:116] = True
        seed = np.zeros_like(component)
        seed[38:44, 42:98] = True

        edge_strength = np.zeros((h, w), dtype=np.float32)
        cv2.line(edge_strength, (28, 29), (112, 29), 0.18, 1, cv2.LINE_AA)
        hard_edge = edge_strength >= 0.25
        self.assertFalse(np.any(hard_edge))

        target = edge_refine._draw_component_target_edge(
            hard_edge,
            edge_strength,
            component,
            search_radius=24,
            half_width=13,
            seed=seed,
            seed_from_stroke=True,
            strength=90,
        )

        self.assertEqual(int(np.count_nonzero(target)), 0)

    def test_draw_component_snap_crop_matches_full_render_for_simple_edge(self):
        h, w = 120, 160
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :84] = (0.78, 0.78, 0.92)
        image[:, 84:] = (0.10, 0.10, 0.28)
        line = mask_rasters.Line(False, 14, 100)
        line.add_point(32, 60)
        line.add_point(76, 60)
        full_mask = mask_rasters.draw_line_texture((w, h), [line])
        full_refined = refine_mask_edge_aware(
            image,
            full_mask,
            guide_point=(56, 60),
            mode="Quick Select",
            radius=22,
            strength=80,
            seed_mask=edge_refine.make_confident_seed(full_mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        x0, y0, x1, y1 = 20, 30, 120, 90
        crop_image = image[y0:y1, x0:x1]
        crop_line = mask_rasters.Line(False, line.size, line.soft)
        for x, y in line.points:
            crop_line.add_point(x - x0, y - y0)
        crop_mask = mask_rasters.draw_line_texture((x1 - x0, y1 - y0), [crop_line])
        crop_refined = refine_mask_edge_aware(
            crop_image,
            crop_mask,
            guide_point=(56 - x0, 60 - y0),
            mode="Quick Select",
            radius=22,
            strength=80,
            seed_mask=edge_refine.make_confident_seed(crop_mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        expected = full_refined[y0:y1, x0:x1]
        core = np.s_[8:-8, 8:-8]
        self.assertLess(float(np.abs(expected[core] - crop_refined[core]).mean()), 0.08)
        self.assertGreater(float(crop_refined.max()), 0.5)

    def test_draw_component_snap_photo_crop_does_not_use_clipped_component_width(self):
        image, edge_y = _photo_like_cloud_scene()
        line = _photo_like_cloud_line(edge_y, size=28, offset=-2)
        x0, y0, x1, y1 = 30, 35, 250, 165
        crop_image = image[y0:y1, x0:x1]
        crop_line = mask_rasters.Line(False, line.size, line.soft)
        for x, y in line.points:
            crop_line.add_point(x - x0, y - y0)
        crop_mask = mask_rasters.draw_line_texture((x1 - x0, y1 - y0), [crop_line])

        refined = refine_mask_edge_aware(
            crop_image,
            crop_mask,
            guide_point=(160 - x0, int(edge_y[160]) - y0),
            mode="Quick Select",
            radius=72,
            strength=70,
            selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[crop_line],
        )

        h, _w = crop_mask.shape
        edge_band = np.zeros_like(crop_mask, dtype=bool)
        near_sky = np.zeros_like(crop_mask, dtype=bool)
        far_sky = np.zeros_like(crop_mask, dtype=bool)
        for x in range(x0, x1):
            y = int(edge_y[x] - y0)
            xx = x - x0
            edge_band[max(0, y - 1):min(h, y + 2), xx] = True
            near_sky[min(h, y + 8):, xx] = True
            far_sky[min(h, y + 22):, xx] = True

        self.assertGreater(float(refined[edge_band].mean()), 0.65)
        self.assertLess(float(refined[near_sky].sum()), float(crop_mask[near_sky].sum()) * 0.55)
        self.assertLess(float(refined[far_sky].sum()), float(refined.sum()) * 0.03)

    def test_draw_component_snap_uniform_image_does_not_expand_to_radius(self):
        h, w = 80, 120
        image = np.full((h, w, 3), (0.45, 0.45, 0.45), dtype=np.float32)
        line = mask_rasters.Line(False, 14, 100)
        line.add_point(25, 40)
        line.add_point(65, 40)
        mask = mask_rasters.draw_line_texture((w, h), [line])

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(45, 40),
            mode="Quick Select",
            radius=45,
            strength=100,
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertLess(float(refined[mask <= 0.01].sum()), 1.0)
        self.assertAlmostEqual(float(refined.sum()), float((mask > 0.02).sum()), delta=2.0)

    def test_draw_component_snap_respects_erased_split(self):
        h, w = 80, 140
        image = np.full((h, w, 3), (0.45, 0.45, 0.45), dtype=np.float32)
        add = mask_rasters.Line(False, 16, 100)
        add.add_point(20, 40)
        add.add_point(120, 40)
        erase = mask_rasters.Line(True, 28, 100)
        erase.add_point(70, 22)
        erase.add_point(70, 58)
        mask = mask_rasters.draw_line_texture((w, h), [add, erase])
        self.assertLess(float(mask[:, 66:75].max()), 0.01)

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(45, 40),
            mode="Quick Select",
            radius=40,
            strength=100,
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertLess(float(refined[:, 66:75].max()), 0.01)
        n_labels, labels = cv2.connectedComponents((refined > 0.5).astype(np.uint8), connectivity=8)
        touched = [label_id for label_id in range(1, n_labels) if np.count_nonzero(labels == label_id) > 8]
        self.assertGreaterEqual(len(touched), 2)

    def test_draw_component_snap_all_erased_returns_empty_mask(self):
        h, w = 64, 100
        image = np.full((h, w, 3), (0.3, 0.3, 0.3), dtype=np.float32)
        add = mask_rasters.Line(False, 18, 100)
        add.add_point(20, 32)
        add.add_point(80, 32)
        erase = mask_rasters.Line(True, 30, 100)
        erase.add_point(20, 32)
        erase.add_point(80, 32)
        mask = mask_rasters.draw_line_texture((w, h), [add, erase])

        refined = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(50, 32),
            mode="Quick Select",
            radius=40,
            strength=100,
            selection_strategy=edge_refine.STRATEGY_DRAW,
        )

        self.assertLess(float(mask.max()), 0.01)
        self.assertLess(float(refined.max()), 0.01)

    def test_headless_quick_select_is_stable_when_zoomed(self):
        image = np.zeros((100, 100, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)

        def render(disp_info):
            ctx = Mask2CoordinateContext()
            ctx.set_texture_size(100, 100)
            primary = {
                "original_img_size": (100, 100),
                "img_size": (100, 100),
                "disp_info": disp_info,
                "rotation": 0,
                "rotation2": 0,
                "flip_mode": 0,
                "matrix": np.eye(3),
            }
            ctx.set_primary_param(primary, disp_info)
            ctx.set_ref_image(image, image)
            mask = HeadlessFreeDrawMask(ctx)
            line = mask_rasters.Line(False, 16, 100)
            line.add_point(-30, 0)
            line.add_point(-4, 0)
            mask.lines = [line]
            mask.center = (-30, 0)
            mask.effects_param["switch_mask2_options"] = True
            mask.effects_param["mask2_edge_refine_mode"] = "Quick Select"
            mask.effects_param["mask2_edge_refine_radius"] = 24
            mask.effects_param["mask2_edge_refine_strength"] = 80
            return mask.get_mask_image()

        full = render((0, 0, 100, 100, 1.0))
        zoom = render((25, 25, 50, 50, 2.0))
        expected_zoom = full[25:75, 25:75].repeat(2, axis=0).repeat(2, axis=1)

        self.assertLess(float(np.abs(expected_zoom - zoom).mean()), 0.07)
        self.assertGreater(float(zoom.max()), 0.5)
        self.assertLess(float(zoom[:, 60:].max()), 0.01)

    def test_zoom_crop_rgb_survives_hls_cache_for_edge_refine(self):
        crop = np.zeros((100, 100, 3), dtype=np.float32)
        crop[:, :50] = (0.2, 0.7, 0.2)
        crop[:, 50:] = (0.4, 0.7, 1.0)
        original = np.full_like(crop, (0.25, 0.25, 0.25))

        ctx = Mask2CoordinateContext()
        ctx.set_texture_size(100, 100)
        disp_info = (25, 25, 50, 50, 2.0)
        primary = {
            "original_img_size": (100, 100),
            "img_size": (100, 100),
            "disp_info": disp_info,
            "rotation": 0,
            "rotation2": 0,
            "flip_mode": 0,
            "matrix": np.eye(3),
        }
        ctx.set_primary_param(primary, disp_info)
        ctx.set_ref_image(crop, original)

        self.assertIsNotNone(ctx.get_crop_image_hls())
        self.assertIs(ctx.crop_image_rgb, crop)

        guide = extended_params._get_edge_refine_guide_image(ctx, (100, 100))

        self.assertIs(guide, crop)

    def test_freedraw_full_view_roi_uses_padded_square_coordinates(self):
        image = np.zeros((60, 100, 3), dtype=np.float32)
        image[..., 0] = np.arange(100, dtype=np.float32)[None, :] / 100.0
        image[..., 1] = np.arange(60, dtype=np.float32)[:, None] / 60.0

        full = extended_params._crop_padded_image_region(image, (0, 20, 100, 80))
        top_pad = extended_params._crop_padded_image_region(image, (0, 0, 100, 20))

        self.assertEqual(full.shape, image.shape)
        self.assertTrue(np.allclose(full, image))
        self.assertEqual(top_pad.shape, (20, 100, 3))
        self.assertLess(float(np.max(top_pad)), 0.001)

    def test_freedraw_full_view_matches_local_preview_scale_for_full_display(self):
        original = np.zeros((100, 160, 3), dtype=np.float32)
        original[:, :80] = (0.2, 0.7, 0.2)
        original[:, 80:] = (0.4, 0.7, 1.0)
        preview = cv2.resize(original, (80, 50), interpolation=cv2.INTER_AREA)
        disp_info = core.convert_rect_to_info(core.get_initial_crop_rect(160, 100), 0.5)

        ctx = Mask2CoordinateContext()
        ctx.set_texture_size(80, 80)
        primary = {
            "original_img_size": (160, 100),
            "img_size": (160, 100),
            "disp_info": disp_info,
            "rotation": 0,
            "rotation2": 0,
            "flip_mode": 0,
            "matrix": np.eye(3),
        }
        ctx.set_primary_param(primary, disp_info)
        ctx.set_ref_image(preview, original)

        line = mask_rasters.Line(False, 20, 100)
        for x in (30, 60, 90, 130):
            line.add_point(x - 80, 50 - 80)
        texture_line = mask_rasters.Line(False, ctx.tcg_to_image_scale(line.size, 0)[0], 100)
        for point in line.points:
            texture_line.add_point(*ctx.tcg_to_texture(*point))
        mask = mask_rasters.draw_line_texture((80, 80), [texture_line])
        effects_param = {
            "switch_mask2_options": True,
            "mask2_edge_refine_mode": "Quick Select",
            "mask2_edge_refine_radius": 24,
            "mask2_edge_refine_strength": 80,
        }

        local = extended_params.apply_extended_params(
            ctx,
            effects_param,
            mask,
            line.points[1],
            fill_grown_region=True,
            seed_mask=edge_refine.make_confident_seed(mask),
            edge_refine_selection_strategy=edge_refine.STRATEGY_DRAW,
            edge_refine_draw_strokes=[texture_line],
        )
        full = extended_params.render_freedraw_edge_refine_full_view(
            ctx,
            effects_param,
            [line],
            line.points[1],
            mask.shape,
        )

        self.assertIsNotNone(full)
        self.assertLess(float(np.abs(local - full).mean()), 0.015)

    def test_headless_parametric_masks_ignore_quick_select(self):
        image = np.zeros((100, 100, 3), dtype=np.float32)
        image[:, :50] = (0.2, 0.7, 0.2)
        image[:, 50:] = (0.4, 0.7, 1.0)
        ctx = Mask2CoordinateContext()
        ctx.set_texture_size(100, 100)
        disp_info = (0, 0, 100, 100, 1.0)
        primary = {
            "original_img_size": (100, 100),
            "img_size": (100, 100),
            "disp_info": disp_info,
            "rotation": 0,
            "rotation2": 0,
            "flip_mode": 0,
            "matrix": np.eye(3),
        }
        ctx.set_primary_param(primary, disp_info)
        ctx.set_ref_image(image, image)

        masks = []
        circle = HeadlessCircularGradientMask(ctx)
        circle.center = (-25, 0)
        circle.inner_radius_x = circle.inner_radius_y = 5
        circle.outer_radius_x = circle.outer_radius_y = 15
        masks.append(circle)

        line = HeadlessGradientMask(ctx)
        line.center = [0, 0]
        line.start_point = [-25, 0]
        line.end_point = [25, 0]
        masks.append(line)

        full = HeadlessFullMask(ctx)
        full.center = (0, 0)
        masks.append(full)

        for mask in masks:
            base = mask.get_mask_image().copy()
            mask.image_mask_cache = None
            mask.image_mask_cache_hash = None
            mask.effects_param["switch_mask2_options"] = True
            mask.effects_param["mask2_edge_refine_mode"] = "Quick Select"
            mask.effects_param["mask2_edge_refine_radius"] = 1
            mask.effects_param["mask2_edge_refine_strength"] = 100
            refined = mask.get_mask_image()
            np.testing.assert_allclose(refined, base, atol=1e-6)

    def test_off_mode_returns_input_mask(self):
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[2:4, 2:4] = 0.75
        image = np.zeros((8, 8, 3), dtype=np.float32)

        refined = refine_mask_edge_aware(image, mask, mode="Off")

        np.testing.assert_allclose(refined, mask)


if __name__ == "__main__":
    unittest.main()
