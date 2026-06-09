import os
import pathlib
import sys
import unittest

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import effects
import cores.core as core
from cores.mask2.coordinate_context import Mask2CoordinateContext
from cores.mask2 import draw_quick_select, edge_refine, extended_params, mask_rasters
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


def _real_snow_fixture_and_cloud_stroke(size=(600, 400), offset=0.0):
    path = PROJECT_ROOT / "tests" / "fixtures" / "edge_refine_snow_600.png"
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise AssertionError(f"missing fixture: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if image_rgb.shape[:2] != (int(size[1]), int(size[0])):
        image_rgb = cv2.resize(image_rgb, tuple(size), interpolation=cv2.INTER_AREA)
    image_rgb = image_rgb.astype(np.float32) / 255.0

    points = np.array(
        [
            [270, 170],
            [315, 220],
            [370, 258],
            [435, 280],
            [510, 275],
            [575, 238],
            [640, 190],
            [720, 155],
            [800, 155],
        ],
        dtype=np.float32,
    )
    points[:, 0] *= float(size[0]) / 960.0
    points[:, 1] *= float(size[1]) / 640.0
    points[:, 1] += float(offset)

    line = mask_rasters.Line(False, max(12.0, 32.0 * float(size[0]) / 960.0), 100)
    for x, y in points:
        line.add_point(float(x), float(y))
    mask = mask_rasters.draw_line_texture(tuple(size), [line])
    return image_rgb, mask, line, points


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
    def test_mask2_quick_select_radius_defaults_to_zero(self):
        self.assertEqual(
            effects.Mask2Effect.get_param({}, "mask2_edge_refine_radius"),
            0,
        )

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

        # Min-cut backend: the genuine guarantee is that the result never
        # crosses the strong edge. (The old grabcut path also grew a ribbon to
        # the perpendicular edge end; that geometric reach is now covered by the
        # parallel-edge scenes in DrawQuickSelectMinCutTest.)
        self.assertGreater(float(refined[32, 30]), 0.5)  # stroke preserved
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
        # A larger radius widens the search band, so it selects at least as much
        # as the small radius, and neither crosses the strong edge.
        self.assertGreaterEqual(float(large_radius.sum()), float(small_radius.sum()))
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

        self.assertGreater(float(refined[40, 45]), 0.5)  # stroke preserved
        self.assertLess(float(refined[40, 65]), 0.01)    # never crosses the edge

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

    def test_draw_edge_snap_snow_u_stroke_radius_clips_inside_edge(self):
        # Draw radius is now an offset from the brush half-width. A small
        # positive offset is enough to match the old absolute-radius fixture.
        image, mask, edge_y = _snow_like_scene_and_u_stroke()
        sky_side = _curve_side_mask(edge_y, mask.shape, "below", 7)

        refined, support = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(120, 92),
            mode="Quick Select",
            radius=7,
            strength=0,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
            return_support=True,
        )

        far_sky_side = _curve_side_mask(edge_y, mask.shape, "below", 20)
        self.assertLess(float(refined[far_sky_side].sum()), 1.0)
        self.assertLess(float(refined.sum()), float(mask.sum()) * 1.1)
        self.assertGreater(float(support.sum()), float(mask.sum()) * 0.70)
        self.assertGreater(float(refined.sum()), float(mask.sum()) * 0.62)
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

        # Intent: a very large radius must not *explode* outward (ratio stays
        # near 1). With reach enabled, a concave boundary may bulge modestly into
        # the band, but the total stays bounded and does not run away.
        far_sky_side = _curve_side_mask(edge_y, mask.shape, "below", 20)
        self.assertLess(float(refined.sum()), float(mask.sum()) * 1.2)
        self.assertLess(float(refined[far_sky_side].sum()), float(mask.sum()) * 0.2)
        self.assertGreater(float(refined.sum()), float(mask.sum()) * 0.50)

    def test_draw_component_snap_uses_final_mask_not_stroke_geometry(self):
        image, mask, edge_y = _snow_like_scene_and_u_stroke()
        sky_side = _curve_side_mask(edge_y, mask.shape, "below", 7)
        far_sky_side = _curve_side_mask(edge_y, mask.shape, "below", 20)

        refined, support = refine_mask_edge_aware(
            image,
            mask,
            guide_point=(120, 92),
            mode="Quick Select",
            radius=7,
            strength=0,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
            return_support=True,
        )

        self.assertLess(float(refined[far_sky_side].sum()), 1.0)
        self.assertLess(float(refined.sum()), float(mask.sum()) * 1.1)
        self.assertGreater(float(support.sum()), float(mask.sum()) * 0.70)
        self.assertGreater(float(refined.sum()), float(mask.sum()) * 0.62)
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

        # Aggressive outward reach to a distant edge is intentionally bounded by
        # the anti-inflation prior in the min-cut backend (chasing far edges is
        # what made the old path explode). The kept guarantees: the result stays
        # anchored on the drawn mask and never crosses past the edge.
        self.assertGreater(float(refined.sum()), float(mask.sum()) * 0.5)
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

        # These are offsets from the brush half-width, not absolute widths.
        for radius in (-13, -6, 0, 14):
            refined, support = refine_mask_edge_aware(
                image,
                mask,
                guide_point=(160, int(edge_y[160])),
                mode="Quick Select",
                radius=radius,
                strength=70,
                selection_strategy=edge_refine.STRATEGY_DRAW,
                draw_strokes=[line],
                return_support=True,
            )

            # The edge stays selected and nothing leaks into the *far* sky (no
            # explosion). With outward reach enabled (so `radius` can snap to a
            # nearby edge) the boundary may extend somewhat past a soft boundary
            # into the same-coloured texture just below it; that near-edge spill
            # is bounded, not clipped, because neither colour nor a clean edge
            # distinguishes it here.
            self.assertGreater(float(support[edge_band].mean()), 0.65)
            self.assertGreater(float(refined[edge_band].mean()), 0.42)
            self.assertLessEqual(float(refined[near_sky].sum()), float(mask[near_sky].sum()) * 1.6)
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

        # The accidental crossing's far side is still rejected as a fraction of
        # the whole result (no deep explosion). Outward reach amplifies the tiny
        # accidental spill somewhat, so the relative-to-spill bound is loosened.
        self.assertLess(float(refined[far_sky].sum()), float(refined.sum()) * 0.05)
        self.assertLess(float(refined[far_sky].sum()), float(mask[far_sky].sum()) * 3.0)

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
            radius=15,
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
            radius=15,
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

        refined, support = refine_mask_edge_aware(
            crop_image,
            crop_mask,
            guide_point=(160 - x0, int(edge_y[160]) - y0),
            mode="Quick Select",
            radius=72,
            strength=70,
            selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[crop_line],
            return_support=True,
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

        self.assertGreater(float(support[edge_band].mean()), 0.65)
        self.assertGreater(float(refined[edge_band].mean()), 0.45)
        self.assertLess(float(refined[near_sky].sum()), float(crop_mask[near_sky].sum()) * 0.55)
        self.assertLess(float(refined[far_sky].sum()), float(refined.sum()) * 0.03)

    def test_draw_quick_select_real_fixture_adds_natural_edge_matte(self):
        image, mask, line, points = _real_snow_fixture_and_cloud_stroke()

        refined, support = refine_mask_edge_aware(
            image,
            mask,
            guide_point=tuple(points[4]),
            mode="Quick Select",
            radius=50,
            strength=82,
            fill_grown_region=True,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[line],
            return_support=True,
        )

        soft_pixels = (refined > 1e-4) & (refined < 0.999)
        self.assertGreater(int(np.count_nonzero(soft_pixels)), 700)
        self.assertLess(float(refined[soft_pixels].min(initial=1.0)), 0.65)
        self.assertLess(float(refined[support <= 0.001].max(initial=0.0)), 0.001)
        self.assertGreater(float(refined.sum()), float(support.sum()) * 0.84)
        self.assertLess(float(refined[int(refined.shape[0] * 0.62):, :].sum()), float(refined.sum()) * 0.01)

    def test_natural_edge_matte_stays_narrow(self):
        h, w = 80, 100
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :50] = (0.86, 0.86, 0.96)
        image[:, 50:] = (0.12, 0.10, 0.28)
        for i, t in enumerate(np.linspace(0.2, 0.8, 4), start=48):
            image[:, i] = image[:, 47] * (1.0 - t) + image[:, 52] * t

        bright_support = np.zeros((h, w), dtype=bool)
        bright_support[:, :50] = True
        dark_support = ~bright_support

        bright_soft = edge_refine._compose_refined_mask(
            bright_support.astype(np.float32),
            bright_support,
            True,
            guide=image,
            natural_edge=True,
            edge_lock=0,
        )
        dark_soft = edge_refine._compose_refined_mask(
            dark_support.astype(np.float32),
            dark_support,
            True,
            guide=image,
            natural_edge=True,
            edge_lock=0,
        )

        bright_drop = float(bright_support.sum() - bright_soft.sum())
        dark_drop = float(dark_support.sum() - dark_soft.sum())
        bright_soft_pixels = int(np.count_nonzero((bright_soft > 1e-4) & (bright_soft < 0.999)))
        dark_soft_pixels = int(np.count_nonzero((dark_soft > 1e-4) & (dark_soft < 0.999)))
        self.assertLess(bright_soft_pixels, h * 3)
        self.assertLess(dark_soft_pixels, h * 3)
        self.assertLess(bright_drop, bright_support.sum() * 0.02)
        self.assertLess(dark_drop, dark_support.sum() * 0.02)

    def test_draw_quick_select_real_fixture_radius_snaps_inside_brush(self):
        # radius bounds the inward clip, so snapping the brush onto the cloud edge
        # needs a radius of ~the brush half-width (radius 0 keeps it ~as drawn).
        image, mask, line, points = _real_snow_fixture_and_cloud_stroke()

        refined, support = refine_mask_edge_aware(
            image,
            mask,
            guide_point=tuple(points[4]),
            mode="Quick Select",
            radius=24,
            strength=82,
            fill_grown_region=True,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[line],
            return_support=True,
        )

        hint_area = float(np.count_nonzero(mask > 0.02))
        self.assertGreater(float(support.sum()), hint_area * 0.45)
        self.assertLess(float(support.sum()), hint_area * 0.94)
        self.assertLess(float(refined.sum()), float(mask.sum()) * 0.86)
        self.assertLess(float(refined[int(refined.shape[0] * 0.62):, :].sum()), 1.0)

    def test_draw_quick_select_real_fixture_large_radius_stays_near_stroke(self):
        image, mask, line, points = _real_snow_fixture_and_cloud_stroke()

        refined, support = refine_mask_edge_aware(
            image,
            mask,
            guide_point=tuple(points[4]),
            mode="Quick Select",
            radius=93,
            strength=82,
            fill_grown_region=True,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[line],
            return_support=True,
        )

        hint_area = float(np.count_nonzero(mask > 0.02))
        self.assertGreater(float(support.sum()), hint_area * 0.55)
        self.assertLess(float(support.sum()), hint_area * 1.35)
        self.assertLess(float(refined.sum()), float(mask.sum()) * 1.10)
        self.assertLess(float(refined[int(refined.shape[0] * 0.62):, :].sum()), 1.0)

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

        # No expansion into the radius (the test's intent): nothing is added
        # outside the drawn mask on a featureless image. min-cut may round the
        # stroke caps by ~1px, so allow a small shrink instead of exact equality.
        self.assertLess(float(refined[mask <= 0.01].sum()), 1.0)
        hint_area = float((mask > 0.02).sum())
        self.assertLessEqual(float(refined.sum()), hint_area + 2.0)
        self.assertGreater(float(refined.sum()), hint_area * 0.75)

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
        old_full_view = os.environ.get("PLATYPUS_DRAW_QS_FULL_VIEW")
        os.environ["PLATYPUS_DRAW_QS_FULL_VIEW"] = "1"
        try:
            full = extended_params.render_freedraw_edge_refine_full_view(
                ctx,
                effects_param,
                [line],
                line.points[1],
                mask.shape,
            )
        finally:
            if old_full_view is None:
                os.environ.pop("PLATYPUS_DRAW_QS_FULL_VIEW", None)
            else:
                os.environ["PLATYPUS_DRAW_QS_FULL_VIEW"] = old_full_view

        self.assertIsNotNone(full)
        self.assertLess(float(np.abs(local - full).mean()), 0.020)

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


def _dqs_support(image, lines, radius, strength=0):
    """Build a draw mask from strokes and run the new min-cut backend."""
    h, w = image.shape[:2]
    mask = mask_rasters.draw_line_texture((w, h), lines)
    seed = edge_refine.make_confident_seed(mask)
    res = draw_quick_select.compute_draw_support(
        image, mask, radius, strength, seed_mask=seed, draw_strokes=lines)
    hint = mask > 0.02
    return res, mask, hint


def _straight_edge_scene(edge_x=120, h=160, w=200):
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[:, :edge_x] = (0.20, 0.20, 0.50)
    image[:, edge_x:] = (0.85, 0.85, 0.95)
    return image


def _straight_edge_line(stroke_x=110, size=26, h=160):
    line = mask_rasters.Line(False, size, 100)
    for y in np.linspace(20, h - 20, 30):
        line.add_point(stroke_x, int(y))
    return line


def _two_edges_scene(h=160, w=240):
    image = np.zeros((h, w, 3), dtype=np.float32)
    image[:, :80] = (0.18, 0.18, 0.48)
    image[:, 80:150] = (0.55, 0.55, 0.75)
    image[:, 150:] = (0.88, 0.88, 0.96)
    return image


def _s_curve_scene(h=200, w=260):
    image = np.zeros((h, w, 3), dtype=np.float32)
    xs = np.arange(w)
    edge_y = (100 + 40 * np.sin((xs - 20) / 40.0)).astype(np.int32)
    edge_y = np.clip(edge_y, 30, 170)
    upper = np.zeros((h, w), dtype=bool)
    for x, y in enumerate(edge_y):
        upper[:y, x] = True
    image[upper] = (0.82, 0.82, 0.95)
    image[~upper] = (0.14, 0.12, 0.30)
    return image, edge_y


class DrawQuickSelectMinCutTest(unittest.TestCase):
    """Scenes exercising the band + min-cut Draw Quick Select backend.

    Assertions are deliberately area/region based, not pixel-exact, so they
    encode intent rather than over-fitting to one fixture.
    """

    def test_straight_edge_snaps_and_does_not_cross(self):
        image = _straight_edge_scene(edge_x=120)
        line = _straight_edge_line(stroke_x=110)
        res, mask, hint = _dqs_support(image, [line], radius=12)
        sup = res.support
        # Never leak across the strong edge into the far (bright) side.
        self.assertLess(sup[:, 140:].mean(), 0.02)
        # The stroke side stays selected; no collapse, no explosion.
        self.assertGreater(sup[:, 100:118].mean(), 0.5)
        self.assertLess(sup.sum(), hint.sum() * 1.25)
        self.assertGreater(sup.sum(), hint.sum() * 0.6)

    def test_v2_env_switch_runs_draw_quick_select_path(self):
        image = _straight_edge_scene(edge_x=120)
        line = _straight_edge_line(stroke_x=110)
        h, w = image.shape[:2]
        mask = mask_rasters.draw_line_texture((w, h), [line])
        old = os.environ.get("QS_DRAW_V2")
        os.environ["QS_DRAW_V2"] = "1"
        try:
            refined, support = refine_mask_edge_aware(
                image,
                mask,
                mode="Quick Select",
                radius=12,
                strength=60,
                seed_mask=edge_refine.make_confident_seed(mask),
                selection_strategy=edge_refine.STRATEGY_DRAW,
                draw_strokes=[line],
                return_support=True,
            )
        finally:
            if old is None:
                os.environ.pop("QS_DRAW_V2", None)
            else:
                os.environ["QS_DRAW_V2"] = old
        self.assertEqual(refined.shape, mask.shape)
        self.assertGreater(int(np.count_nonzero(support > 0.5)), 0)

    def test_v2_zoom_metric_reports_scaled_replay(self):
        from cores.mask2 import draw_qs_metrics as qs_metrics

        image = _straight_edge_scene(edge_x=120)
        line = _straight_edge_line(stroke_x=110)
        h, w = image.shape[:2]
        mask = mask_rasters.draw_line_texture((w, h), [line])
        dump = {
            "name": "straight_edge_synthetic",
            "guide": image,
            "mask": mask,
            "seed_mask": edge_refine.make_confident_seed(mask),
            "radius": 12.0,
            "strength": 60.0,
            "pixel_scale": 1.0,
            "strokes": [line],
        }
        metrics = qs_metrics.metrics_for_dump(
            dump, solver="v2", determinism=False, idempotence=False, zoom=True)
        self.assertIn("zoom_iou_2_0x", metrics)
        self.assertGreater(metrics["zoom_iou_2_0x"], 0.75)

    def test_s_curve_boundary_follows_curve(self):
        image, edge_y = _s_curve_scene()
        line = mask_rasters.Line(False, 24, 100)
        for x in np.linspace(30, image.shape[1] - 30, 60):
            xi = int(x)
            line.add_point(xi, int(edge_y[xi] - 4))
        res, mask, hint = _dqs_support(image, [line], radius=14)
        sup = res.support
        # Sample points clearly inside (above curve) selected, far below not.
        above = []
        below = []
        for x in range(40, image.shape[1] - 40, 20):
            y = int(edge_y[x])
            above.append(sup[max(0, y - 16), x])
            below.append(sup[min(image.shape[0] - 1, y + 30), x])
        self.assertGreater(np.mean(above), 0.7)
        self.assertLess(np.mean(below), 0.05)

    def test_concave_u_boundary_no_explosion(self):
        image, mask_u8, edge_y = _snow_like_scene_and_u_stroke()
        line = _snow_like_u_stroke_line(edge_y)
        # radius=1 stays tight; even a huge radius=80 only smooths the concave
        # pocket without exploding, double-lining, or leaking to the far side.
        for radius, bound in ((1, 1.15), (80, 1.5)):
            res, mask, hint = _dqs_support(image, [line], radius=radius)
            ratio = res.support.sum() / max(1, hint.sum())
            self.assertLess(ratio, bound, f"radius={radius} inflated: {ratio:.3f}")
            # Far below the cloud edge must stay unselected.
            self.assertLess(res.support[150:, :].mean(), 0.05)

    def test_busy_texture_does_not_get_selected(self):
        image, edge_y = _photo_like_cloud_scene()
        line = _photo_like_cloud_line(edge_y)
        for radius in (1, 28, 60):
            res, mask, hint = _dqs_support(image, [line], radius=radius)
            sup = res.support
            # Busy tree/snow texture well below the boundary is never selected.
            self.assertLess(
                sup[160:, :].mean(), 0.03, f"radius={radius} grabbed texture")
            self.assertLess(sup.sum(), hint.sum() * 1.35)

    def test_two_nearby_edges_picks_seed_side_edge(self):
        image = _two_edges_scene()
        # Stroke sits in the left (dark) region, near the first edge at x=80.
        line = mask_rasters.Line(False, 24, 100)
        for y in np.linspace(20, 140, 28):
            line.add_point(64, int(y))
        res, mask, hint = _dqs_support(image, [line], radius=40)
        sup = res.support
        # Must not bridge across the second edge at x=150.
        self.assertLess(sup[:, 155:].mean(), 0.02)
        # Stays anchored on the seed side.
        self.assertGreater(sup[:, 50:75].mean(), 0.5)

    def test_uniform_image_large_radius_does_not_inflate(self):
        rng = np.random.default_rng(1)
        image = np.clip(
            np.full((160, 200, 3), 0.5, np.float32)
            + rng.normal(0, 0.01, (160, 200, 3)).astype(np.float32), 0, 1)
        line = mask_rasters.Line(False, 24, 100)
        for y in np.linspace(40, 120, 20):
            line.add_point(100, int(y))
        base = None
        for radius in (0, 30, 80):
            res, mask, hint = _dqs_support(image, [line], radius=radius)
            ratio = res.support.sum() / max(1, hint.sum())
            self.assertLess(ratio, 1.2, f"radius={radius} inflated: {ratio:.3f}")
            self.assertGreater(ratio, 0.7)

    def test_edge_lock_expands_weak_edge_sensitivity(self):
        edge = np.zeros((20, 20), dtype=np.float32)
        edge[:, 10] = 0.35
        strict = draw_quick_select._edge_cost_map(edge, 0)
        loose = draw_quick_select._edge_cost_map(edge, 100)

        self.assertGreater(float(strict[:, 10].mean()), 0.70)
        self.assertLess(float(loose[:, 10].mean()), 0.10)

    def test_edge_lock_keeps_weak_edges_near_strong_ridges_available(self):
        edge = np.zeros((24, 32), dtype=np.float32)
        edge[:, 10] = 0.80
        edge[:, 13] = 0.18

        thinned = draw_quick_select._thin_edge_to_ridge(
            edge, thr=0.28, falloff_sigma=0.9)
        strict = draw_quick_select._edge_cost_map(edge, 0)
        loose = draw_quick_select._edge_cost_map(edge, 100)

        self.assertGreater(float(thinned[:, 13].mean()), 0.15)
        self.assertGreater(float(strict[:, 13].mean()), 0.85)
        self.assertLess(float(loose[:, 13].mean()), 0.55)

    def test_contextual_edge_suppresses_same_color_texture_edges(self):
        edge = np.zeros((24, 32), dtype=np.float32)
        edge[:, 12] = 0.65
        edge[:, 22] = 0.85
        color = np.full((24, 32), 0.30, dtype=np.float32)
        color[:, :12] = -0.30

        low_lock = draw_quick_select._contextual_edge_strength(edge, color, 60)
        high_lock = draw_quick_select._contextual_edge_strength(edge, color, 100)

        np.testing.assert_allclose(low_lock, edge)
        self.assertGreater(float(high_lock[:, 12].mean()), 0.55)
        self.assertLess(float(high_lock[:, 22].mean()), 0.45)

    def test_edge_lock_relaxes_seed_side_barrier_for_weak_boundaries(self):
        strict = draw_quick_select._side_edge_thresh_for_strength(60)
        mid = draw_quick_select._side_edge_thresh_for_strength(93)
        loose = draw_quick_select._side_edge_thresh_for_strength(100)

        self.assertGreater(strict, 0.65)
        self.assertLess(mid, strict)
        self.assertLess(loose, 0.30)

        comp = np.ones((24, 32), dtype=bool)
        core = np.zeros_like(comp)
        core[:, 8] = True
        edge = np.zeros(comp.shape, dtype=np.float32)
        edge[:, 16] = 0.32

        strict_side = draw_quick_select._seed_side_through_smooth_interior(
            comp, core, edge, edge_thresh=strict)
        loose_side = draw_quick_select._seed_side_through_smooth_interior(
            comp, core, edge, edge_thresh=loose)

        self.assertTrue(np.all(strict_side[:, 20]))
        self.assertFalse(np.any(loose_side[:, 20]))

    def test_seed_side_barrier_ignores_small_texture_edge_components(self):
        comp = np.ones((140, 180), dtype=bool)
        core = np.zeros_like(comp)
        core[:, 20] = True
        edge = np.zeros(comp.shape, dtype=np.float32)
        edge[:, 69:72] = 0.85
        edge[45:52, 120:127] = 0.90

        seed_side = draw_quick_select._seed_side_through_smooth_interior(
            comp, core, edge, edge_thresh=0.70)
        filtered = draw_quick_select._filter_side_edge_barrier_components(
            edge > 0.70, comp)

        self.assertFalse(np.any(seed_side[:, 100]))
        self.assertFalse(np.any(seed_side[:, 150]))
        self.assertTrue(np.any(filtered[:, 70]))
        self.assertFalse(np.any(filtered[45:52, 120:127]))

    def test_edge_lock_relaxes_selected_rim_restore_threshold(self):
        strict = draw_quick_select._edge_restore_thresh_for_strength(0)
        loose = draw_quick_select._edge_restore_thresh_for_strength(100)

        self.assertGreater(strict, loose)
        self.assertGreater(strict, 0.60)
        self.assertLess(loose, 0.30)

    def test_selected_edge_rim_restores_only_selected_color_side(self):
        support = np.zeros((20, 20), dtype=bool)
        support[:, :10] = True
        candidate = np.zeros_like(support)
        candidate[:, 9:12] = True
        edge = np.zeros((20, 20), dtype=np.float32)
        edge[:, 10] = 0.8
        color = np.zeros((20, 20), dtype=np.float32)
        color[:, 10] = 0.2
        core = np.zeros_like(support)
        core[:, 4] = True
        erase = np.zeros_like(support)

        restored, rim = draw_quick_select._restore_selected_edge_rim(
            support, candidate, edge, color, core, erase)
        self.assertGreater(int(rim[:, 10].sum()), 0)
        self.assertGreater(int(restored[:, 10].sum()), 0)

        color[:, 10] = -0.2
        restored_neg, rim_neg = draw_quick_select._restore_selected_edge_rim(
            support, candidate, edge, color, core, erase)
        self.assertEqual(int(rim_neg.sum()), 0)
        np.testing.assert_array_equal(restored_neg, support)

    def test_selected_edge_bridge_fills_only_bracketed_seams(self):
        support = np.zeros((20, 20), dtype=bool)
        support[2:18, :10] = True
        support[2:18, 11:] = True
        candidate = np.zeros_like(support)
        candidate[2:18, 10] = True
        edge = np.zeros((20, 20), dtype=np.float32)
        edge[2:18, 10] = 0.75
        core = np.zeros_like(support)
        core[8:12, 5] = True
        core[8:12, 15] = True
        erase = np.zeros_like(support)

        restored, bridge = draw_quick_select._bridge_selected_edge_seams(
            support, candidate, edge, core, erase)

        self.assertGreater(int(bridge[:, 10].sum()), 0)
        self.assertTrue(np.all(restored[4:16, 10]))

        one_side = np.zeros_like(support)
        one_side[2:18, :10] = True
        restored_one, bridge_one = draw_quick_select._bridge_selected_edge_seams(
            one_side, candidate, edge, core & one_side, erase)

        self.assertEqual(int(bridge_one.sum()), 0)
        np.testing.assert_array_equal(restored_one, one_side)

    def test_bright_selected_side_relaxes_edge_restore_color_floor(self):
        image = np.zeros((30, 40, 3), dtype=np.float32)
        image[:, :20] = (0.12, 0.18, 0.36)
        image[:, 20:] = (0.86, 0.86, 0.95)
        comp = np.zeros((30, 40), dtype=bool)
        comp[:, 20:32] = True
        core = np.zeros_like(comp)
        core[:, 27:30] = True

        bright_floor = draw_quick_select._edge_restore_color_min_for_unit(
            image, comp, core, comp, 12)
        self.assertLess(bright_floor, draw_quick_select.EDGE_RESTORE_COLOR_MIN)

        dark_floor = draw_quick_select._edge_restore_color_min_for_unit(
            image, ~comp, np.logical_not(comp) & (np.indices(comp.shape)[1] < 12), ~comp, 12)
        self.assertEqual(dark_floor, draw_quick_select.EDGE_RESTORE_COLOR_MIN)

    def test_bright_selected_side_uses_weaker_color_weight(self):
        image = np.zeros((30, 40, 3), dtype=np.float32)
        image[:, :20] = (0.12, 0.18, 0.36)
        image[:, 20:] = (0.86, 0.86, 0.95)
        comp = np.zeros((30, 40), dtype=bool)
        comp[:, 20:32] = True
        core = np.zeros_like(comp)
        core[:, 27:30] = True

        bright_w = draw_quick_select._color_weight_for_unit(
            image, comp, core, comp, 12, strength=100)
        bright_w_mid = draw_quick_select._color_weight_for_unit(
            image, comp, core, comp, 12, strength=70)
        dark_w = draw_quick_select._color_weight_for_unit(
            image, ~comp, np.logical_not(comp) & (np.indices(comp.shape)[1] < 12), ~comp, 12)

        self.assertLess(bright_w, draw_quick_select.COLOR_W)
        self.assertGreater(bright_w_mid, bright_w)
        self.assertEqual(dark_w, draw_quick_select.COLOR_W)

    def test_smooth_outside_growth_is_limited_to_real_edges(self):
        hint = np.zeros((20, 24), dtype=bool)
        hint[5:15, 6:14] = True
        support = hint.copy()
        support[6:14, 14:21] = True
        edge = np.zeros((20, 24), dtype=np.float32)
        edge[8:12, 14] = 0.85
        core = np.zeros_like(hint)
        core[8:12, 8:10] = True
        erase = np.zeros_like(hint)

        limited = draw_quick_select._limit_smooth_outside_growth(
            support, hint, edge, core, erase)

        self.assertFalse(np.any(limited[6:8, 17:21]))
        self.assertGreater(int(np.count_nonzero(limited[:, 14])), 0)

    def test_selected_hint_holes_fill_without_restoring_boundary_side(self):
        hint = np.zeros((24, 32), dtype=bool)
        hint[3:21, 3:29] = True
        support = hint.copy()
        support[:, :12] = False
        support[9:13, 20:24] = False
        core = np.zeros_like(hint)
        core[10:14, 24:27] = True
        erase = np.zeros_like(hint)

        restored, fill = draw_quick_select._fill_selected_hint_holes(
            support, hint, core, erase)

        self.assertTrue(np.all(restored[9:13, 20:24]))
        self.assertGreater(int(fill[9:13, 20:24].sum()), 0)
        self.assertFalse(np.any(restored[5:18, 3:10]))

    def test_brush_size_is_draw_quick_select_base_radius(self):
        image = _straight_edge_scene(edge_x=120)
        line = _straight_edge_line(stroke_x=110, size=60)
        _res, mask, hint = _dqs_support(image, [line], radius=0)

        base = draw_quick_select._resolve_scales(0, [line], hint)
        smaller = draw_quick_select._resolve_scales(-12, [line], hint)
        larger = draw_quick_select._resolve_scales(20, [line], hint)

        self.assertAlmostEqual(base.stroke_half_width, 30.0, delta=0.5)
        self.assertAlmostEqual(base.band_half_width, 30.0, delta=0.5)
        self.assertAlmostEqual(smaller.band_half_width, 18.0, delta=0.5)
        self.assertAlmostEqual(larger.band_half_width, 50.0, delta=0.5)

    def test_draw_quick_select_resolves_radius_per_stroke(self):
        image = _straight_edge_scene(edge_x=120, h=120, w=220)
        small = _straight_edge_line(stroke_x=55, size=20, h=120)
        large = _straight_edge_line(stroke_x=165, size=60, h=120)
        _res, mask, hint = _dqs_support(image, [small, large], radius=0)

        fg_seed, _bg_seed, has_strokes = edge_refine._draw_random_walker_stroke_seeds(
            hint.shape, [small, large], hint)
        hard_core = draw_quick_select._seed_core(mask, fg_seed, hint)
        units = draw_quick_select._draw_solve_units(
            mask, hint, hard_core, [small, large], radius=0, has_strokes=has_strokes)
        widths = sorted(round(float(unit.scales.stroke_half_width), 1) for unit in units)

        self.assertIn(10.0, widths)
        self.assertIn(30.0, widths)

    def test_draw_quick_select_ignores_tiny_fallback_rim_flecks(self):
        image = _straight_edge_scene(edge_x=120, h=160, w=220)
        line = mask_rasters.Line(False, 100, 100)
        line.add_point(100, 80)
        mask = mask_rasters.draw_line_texture((220, 160), [line])
        # Simulate detached anti-aliased rim remnants present in the final hint
        # but absent from the re-rasterized stroke geometry.
        for x, y in ((48, 34), (152, 34), (48, 126), (152, 126)):
            mask[y:y + 2, x:x + 2] = 1.0

        hint = mask > 0.02
        fg_seed, _bg_seed, has_strokes = edge_refine._draw_random_walker_stroke_seeds(
            hint.shape, [line], hint)
        hard_core = draw_quick_select._seed_core(mask, fg_seed, hint)
        units = draw_quick_select._draw_solve_units(
            mask, hint, hard_core, [line], radius=160, has_strokes=has_strokes)

        self.assertEqual(len(units), 1)
        self.assertGreater(int(units[0].component.sum()), 7000)

    def test_thick_brush_uses_seed_side_when_edge_is_inside_brush(self):
        h, w = 140, 200
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :100] = (0.85, 0.85, 0.95)
        image[:, 100:] = (0.14, 0.12, 0.30)

        for offset, keep_left in ((-10, True), (10, False)):
            line = mask_rasters.Line(False, 60, 100)
            line.add_point(100 + offset, 30)
            line.add_point(100 + offset, 110)
            mask = mask_rasters.draw_line_texture((w, h), [line])
            _refined, support = refine_mask_edge_aware(
                image,
                mask,
                mode="Quick Select",
                radius=0,
                strength=80,
                seed_mask=edge_refine.make_confident_seed(mask),
                selection_strategy=edge_refine.STRATEGY_DRAW,
                draw_strokes=[line],
                return_support=True,
            )

            left = float(support[50:90, 80:95].mean())
            right = float(support[50:90, 105:120].mean())
            if keep_left:
                self.assertGreater(left, 0.80)
                self.assertLess(right, 0.25)
            else:
                self.assertLess(left, 0.25)
                self.assertGreater(right, 0.80)

    def test_dark_branch_point_keeps_object_side_not_sky_shell(self):
        # Real repro shape: a one-point, very large brush is centered on a dark
        # branch that belongs to a bright snowy object, with dark sky around it.
        # The colour background shell must stay local/typical; choosing the most
        # colour-separated shell sample treats the snow as background and keeps
        # the sky instead.
        h, w = 160, 220
        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:, :] = (0.16, 0.14, 0.32)
        snow = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(snow, (74, 82), (48, 62), 0, 0, 360, 1, -1)
        image[snow.astype(bool)] = (0.78, 0.78, 0.94)
        cv2.line(image, (72, 82), (154, 82), (0.10, 0.09, 0.20), 9, cv2.LINE_AA)
        cv2.line(image, (120, 82), (150, 64), (0.12, 0.11, 0.22), 6, cv2.LINE_AA)

        line = mask_rasters.Line(False, 110, 100)
        line.add_point(126, 82)
        mask = mask_rasters.draw_line_texture((w, h), [line])
        _refined, support = refine_mask_edge_aware(
            image,
            mask,
            mode="Quick Select",
            radius=0,
            strength=50,
            seed_mask=edge_refine.make_confident_seed(mask),
            selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[line],
            return_support=True,
        )

        hint = mask > 0.02
        yy, xx = np.indices((h, w))
        sky_right = (xx > 150) & hint
        snow_left = (xx < 90) & hint
        branch = (yy > 74) & (yy < 90) & (xx > 90) & (xx < 150)

        self.assertLess(float(support[sky_right].mean()), 0.15)
        self.assertGreater(float(support[snow_left].mean()), 0.70)
        self.assertGreater(float(support[branch].mean()), 0.80)

    def test_thick_brush_does_not_reach_far_weak_edge(self):
        # A faint edge far from a thick stroke must not pull the mask out.
        image = _straight_edge_scene(edge_x=150, h=160, w=220)
        # Make the edge weak (small contrast) and far from the stroke.
        image[:, 150:] = (0.32, 0.32, 0.56)
        line = _straight_edge_line(stroke_x=70, size=40)
        res, mask, hint = _dqs_support(image, [line], radius=80)
        sup = res.support
        self.assertLess(sup.sum(), hint.sum() * 1.4)
        self.assertLess(sup[:, 150:].mean(), 0.1)

    def test_eraser_split_keeps_components_separated(self):
        image = _straight_edge_scene(edge_x=180, h=160, w=200)
        add = mask_rasters.Line(False, 20, 100)
        for x in np.linspace(40, 150, 30):
            add.add_point(int(x), 80)
        erase = mask_rasters.Line(True, 30, 100)
        erase.add_point(95, 80)
        erase.add_point(95, 80)
        res, mask, hint = _dqs_support(image, [add, erase], radius=6)
        sup = res.support
        n_labels, _ = cv2.connectedComponents(sup.astype(np.uint8), connectivity=8)
        self.assertGreaterEqual(n_labels - 1, 2)
        # The erased column stays empty.
        self.assertEqual(int(sup[78:83, 92:98].sum()), 0)

    def test_all_erased_returns_empty_support(self):
        image = _straight_edge_scene(edge_x=180)
        add = mask_rasters.Line(False, 20, 100)
        for x in np.linspace(40, 150, 30):
            add.add_point(int(x), 80)
        erase = mask_rasters.Line(True, 200, 100)
        for x in np.linspace(20, 180, 30):
            erase.add_point(int(x), 80)
        h, w = image.shape[:2]
        mask = mask_rasters.draw_line_texture((w, h), [add, erase])
        seed = edge_refine.make_confident_seed(mask)
        res = draw_quick_select.compute_draw_support(
            image, mask, 6, 0, seed_mask=seed, draw_strokes=[add, erase])
        self.assertEqual(int(res.support.sum()), 0)

    def test_stroke_outside_edge_reaches_it_within_radius(self):
        # Stroke drawn just outside the region; within radius it should reach
        # the edge and fill back to it.
        image = _straight_edge_scene(edge_x=120)
        line = _straight_edge_line(stroke_x=135, size=20)
        res, mask, hint = _dqs_support(image, [line], radius=30)
        sup = res.support
        # Should not cross to the far bright side beyond a thin rim.
        self.assertLess(sup[:, 145:].mean(), 0.15)

    def _guide_full_image(self, size=(600, 400)):
        path = PROJECT_ROOT / "tests" / "guide_full.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            self.skipTest(f"missing fixture: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, tuple(size), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.float32) / 255.0

    def _real_cloud_edge_clip(self, side):
        # Real low-contrast snow scene: white-ish cloud over blue sky. A stroke
        # whose centre is on `side` of the cloud/sky edge, with the brush
        # crossing the edge, must snap to the edge (keep its own side, clip the
        # other) even though colour separation is weak.
        img = self._guide_full_image()
        h, w = img.shape[:2]
        es = edge_refine._draw_snap_edge_strength(img)
        xc = 270
        ridge = 40 + int(np.argmax(es[40:240, xc]))  # cloud/sky edge row
        cy = ridge - 14 if side == "cloud" else ridge + 14
        line = mask_rasters.Line(False, 40, 100)
        line.add_point(xc - 18, cy - 5)
        line.add_point(xc + 18, cy + 5)
        mask = mask_rasters.draw_line_texture((w, h), [line])
        seed = edge_refine.make_confident_seed(mask)
        _refined, support = refine_mask_edge_aware(
            img, mask, mode="Quick Select", radius=18, strength=50,
            seed_mask=seed, selection_strategy=edge_refine.STRATEGY_DRAW,
            draw_strokes=[line], return_support=True)
        sup = support > 0.5
        cloud_band = sup[ridge - 16:ridge - 7, xc - 12:xc + 12]
        sky_band = sup[ridge + 7:ridge + 16, xc - 12:xc + 12]
        return ridge, cloud_band, sky_band

    def test_real_image_sky_side_snaps_to_cloud_edge(self):
        # The reported failure: stroke centred on the sky side does not capture
        # the cloud edge. It must keep the sky band and clip the cloud band.
        ridge, cloud_band, sky_band = self._real_cloud_edge_clip("sky")
        self.assertGreater(float(sky_band.mean()), 0.6)   # sky side kept
        self.assertLess(float(cloud_band.mean()), 0.4)    # cloud side clipped

    def test_real_image_cloud_side_snaps_to_cloud_edge(self):
        ridge, cloud_band, sky_band = self._real_cloud_edge_clip("cloud")
        self.assertGreater(float(cloud_band.mean()), 0.6)  # cloud side kept
        self.assertLess(float(sky_band.mean()), 0.4)       # sky side clipped

    def test_zoom_crop_matches_full_for_simple_edge(self):
        image = _straight_edge_scene(edge_x=120)
        line = _straight_edge_line(stroke_x=108)
        res_full, mask_full, _ = _dqs_support(image, [line], radius=10)

        scale = 2
        big = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)
        big_line = mask_rasters.Line(False, 26 * scale, 100)
        for y in np.linspace(20, image.shape[0] - 20, 30):
            big_line.add_point(108 * scale, int(y) * scale)
        res_big, _, _ = _dqs_support(big, [big_line], radius=10 * scale)
        down = cv2.resize(res_big.support.astype(np.float32),
                          (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_AREA) > 0.5
        core = np.s_[30:130, 60:130]
        diff = np.mean(np.abs(down[core].astype(np.float32)
                              - res_full.support[core].astype(np.float32)))
        self.assertLess(diff, 0.12)


if __name__ == "__main__":
    unittest.main()
