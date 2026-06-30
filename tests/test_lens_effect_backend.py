import os
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import lens_effect_adapter, lens_effect_reference


def _hdr_edge_image(height=48, width=64):
    img = np.full((height, width, 3), 0.08, dtype=np.float32)
    img[:, :width // 2, 0] = 4.0
    img[:, :width // 2, 1] = 0.2
    img[:, :width // 2, 2] = 3.5
    img[:, width // 2:, 0] = 0.2
    img[:, width // 2:, 1] = 4.5
    img[:, width // 2:, 2] = 0.3
    return img


def _front_back_depth(height=48, width=64):
    depth = np.empty((height, width), dtype=np.float32)
    depth[:, :width // 2] = 0.25
    depth[:, width // 2:] = 0.75
    return depth


def _point_source_image(size=129, value=8.0, bg=0.02):
    img = np.full((size, size, 3), bg, dtype=np.float32)
    c = size // 2
    img[c - 1:c + 2, c - 1:c + 2] = value
    return img


class LensEffectBackendTest(unittest.TestCase):
    def test_bokeh_fringe_without_depth_is_noop(self):
        img = _hdr_edge_image()
        out = lens_effect_adapter.apply_bokeh_color_fringe(
            img,
            depth_map=None,
            focus_depth=0.5,
            strength=100,
            resolution_scale=1.0,
        )
        self.assertIs(out, img)

    def test_forced_reference_matches_reference_module_and_preserves_hdr(self):
        img = _hdr_edge_image()
        depth = _front_back_depth(*img.shape[:2])
        old_backend = os.environ.get("PLATYPUS_LENS_EFFECT_BACKEND")
        os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "reference"
        try:
            out = lens_effect_adapter.apply_bokeh_color_fringe(
                img,
                depth,
                focus_depth=0.5,
                strength=90,
                resolution_scale=0.75,
            )
            expected = lens_effect_reference.apply_bokeh_color_fringe(
                img,
                depth,
                focus_depth=0.5,
                strength=90,
                resolution_scale=0.75,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_EFFECT_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = old_backend

        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), 1.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_shaped_bokeh_is_served_by_lens_effect_reference(self):
        img = _point_source_image()
        old_backend = os.environ.get("PLATYPUS_LENS_EFFECT_BACKEND")
        os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "reference"
        try:
            out = lens_effect_adapter.apply_shaped_bokeh(
                img,
                depth_map=None,
                focus_depth=0.5,
                strength=100,
                radius=16,
                shape="Hexagon",
                rim=25,
            )
            expected = lens_effect_reference.apply_shaped_bokeh(
                img,
                depth_map=None,
                focus_depth=0.5,
                strength=100,
                radius=16,
                shape="Hexagon",
                rim=25,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_EFFECT_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = old_backend

        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), float(img.max()))
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_swirl_bokeh_is_served_by_lens_effect_reference_and_preserves_hdr(self):
        img = _hdr_edge_image(80, 96)
        depth = _front_back_depth(*img.shape[:2])
        radial = np.ones(img.shape[:2], dtype=np.float32)
        center = ((img.shape[1] - 1) * 0.5, (img.shape[0] - 1) * 0.5)
        out = lens_effect_adapter.apply_swirl_bokeh(
            img,
            depth,
            focus_depth=0.5,
            strength=75,
            resolution_scale=1.0,
            center_xy=center,
            radial_norm=radial,
        )
        expected = lens_effect_reference.apply_swirl_bokeh(
            img,
            depth,
            focus_depth=0.5,
            strength=75,
            resolution_scale=1.0,
            center_xy=center,
            radial_norm=radial,
        )

        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), 1.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_sunstar_is_served_by_lens_effect_reference(self):
        img = _point_source_image(size=161, value=10.0)
        old_backend = os.environ.get("PLATYPUS_LENS_EFFECT_BACKEND")
        os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "reference"
        try:
            out = lens_effect_adapter.apply_sunstar(
                img,
                strength=90,
                length=50,
                threshold=50,
                blades="6",
                aperture=11.0,
                mag=1.0,
                orig_size=None,
            )
            expected = lens_effect_reference.apply_sunstar(
                img,
                strength=90,
                length=50,
                threshold=50,
                blades="6",
                aperture=11.0,
                mag=1.0,
                orig_size=None,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_EFFECT_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = old_backend

        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float((out - img).max()), 0.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    @unittest.skipUnless(lens_effect_adapter.native_available(), "lens effect Metal backend is not built")
    def test_metal_sunstar_emits_hdr_spikes(self):
        img = _point_source_image(size=193, value=10.0)
        old_backend = os.environ.get("PLATYPUS_LENS_EFFECT_BACKEND")
        try:
            os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "metal"
            status = lens_effect_adapter.backend_status()
            if not status.native:
                self.skipTest(status.detail or "Metal backend unavailable")
            actual = lens_effect_adapter.apply_sunstar(
                img,
                strength=90,
                length=50,
                threshold=50,
                blades="6",
                aperture=11.0,
                mag=1.0,
                orig_size=None,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_EFFECT_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = old_backend

        self.assertEqual(actual.shape, img.shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(float((actual - img).max()), 0.0)
        self.assertGreater(float(actual.max()), float(img.max()))

    @unittest.skipUnless(lens_effect_adapter.native_available(), "lens effect Metal backend is not built")
    def test_metal_swirl_bokeh_preserves_hdr_and_changes_image(self):
        img = _hdr_edge_image(80, 96)
        depth = _front_back_depth(*img.shape[:2])
        radial = np.ones(img.shape[:2], dtype=np.float32)
        center = ((img.shape[1] - 1) * 0.5, (img.shape[0] - 1) * 0.5)
        old_backend = os.environ.get("PLATYPUS_LENS_EFFECT_BACKEND")
        try:
            os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "metal"
            status = lens_effect_adapter.backend_status()
            if not status.native:
                self.skipTest(status.detail or "Metal backend unavailable")
            actual = lens_effect_adapter.apply_swirl_bokeh(
                img,
                depth,
                focus_depth=0.5,
                strength=75,
                resolution_scale=1.0,
                center_xy=center,
                radial_norm=radial,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_EFFECT_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = old_backend

        self.assertEqual(actual.shape, img.shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(float(actual.max()), 1.0)
        self.assertGreater(float(np.max(np.abs(actual - img))), 1e-3)

    @unittest.skipUnless(lens_effect_adapter.native_available(), "lens effect Metal backend is not built")
    def test_metal_shaped_bokeh_without_depth_matches_reference_closely(self):
        img = _point_source_image(size=97, value=8.0)
        old_backend = os.environ.get("PLATYPUS_LENS_EFFECT_BACKEND")
        try:
            os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "reference"
            reference = lens_effect_adapter.apply_shaped_bokeh(
                img,
                depth_map=None,
                focus_depth=0.5,
                strength=100,
                radius=12,
                shape="Hexagon",
                rim=0,
            )
            os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "metal"
            status = lens_effect_adapter.backend_status()
            if not status.native:
                self.skipTest(status.detail or "Metal backend unavailable")
            actual = lens_effect_adapter.apply_shaped_bokeh(
                img,
                depth_map=None,
                focus_depth=0.5,
                strength=100,
                radius=12,
                shape="Hexagon",
                rim=0,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_EFFECT_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = old_backend

        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, np.float32)
        self.assertGreater(float(actual.max()), float(img.max()))
        np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-3)

    @unittest.skipUnless(lens_effect_adapter.native_available(), "lens effect Metal backend is not built")
    def test_metal_bokeh_fringe_matches_reference_closely(self):
        img = _hdr_edge_image(36, 44)
        depth = _front_back_depth(*img.shape[:2])
        old_backend = os.environ.get("PLATYPUS_LENS_EFFECT_BACKEND")
        try:
            os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "reference"
            reference = lens_effect_adapter.apply_bokeh_color_fringe(
                img,
                depth,
                focus_depth=0.5,
                strength=85,
                resolution_scale=0.8,
            )
            os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = "metal"
            status = lens_effect_adapter.backend_status()
            if not status.native:
                self.skipTest(status.detail or "Metal backend unavailable")
            actual = lens_effect_adapter.apply_bokeh_color_fringe(
                img,
                depth,
                focus_depth=0.5,
                strength=85,
                resolution_scale=0.8,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_EFFECT_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_EFFECT_BACKEND"] = old_backend

        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, reference, rtol=4e-3, atol=4e-3)


if __name__ == "__main__":
    unittest.main()
