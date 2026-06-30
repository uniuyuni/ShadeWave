import os
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import lens_aberration_adapter, lens_aberration_reference


def _hdr_gradient_image(height=48, width=64):
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    img = np.empty((height, width, 3), dtype=np.float32)
    img[..., 0] = x * np.float32(2.5)
    img[..., 1] = y * np.float32(1.7)
    img[..., 2] = (x + y) * np.float32(1.2)
    return img


def _depth_gradient(height=48, width=64):
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    return np.repeat(x, height, axis=0)


class LensAberrationBackendTest(unittest.TestCase):
    def test_forced_reference_lateral_ca_matches_reference_module_and_preserves_hdr(self):
        old_backend = os.environ.get("PLATYPUS_LENS_ABERRATION_BACKEND")
        os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "reference"
        try:
            img = _hdr_gradient_image()
            out = lens_aberration_adapter.apply_lateral_chromatic_aberration(
                img,
                strength=1.5,
                resolution_scale=1.0,
            )
            expected = lens_aberration_reference.apply_lateral_chromatic_aberration(
                img,
                strength=1.5,
                resolution_scale=1.0,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_ABERRATION_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = old_backend

        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), 1.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_lateral_ca_non_radial_path_still_uses_reference_behavior(self):
        img = _hdr_gradient_image()
        out = lens_aberration_adapter.apply_lateral_chromatic_aberration(
            img,
            strength=1.0,
            resolution_scale=1.0,
            radial=False,
        )
        expected = lens_aberration_reference.apply_lateral_chromatic_aberration(
            img,
            strength=1.0,
            resolution_scale=1.0,
            radial=False,
        )

        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_longitudinal_ca_matches_reference_module_and_preserves_hdr(self):
        img = _hdr_gradient_image()
        depth = _depth_gradient(*img.shape[:2])
        old_backend = os.environ.get("PLATYPUS_LENS_ABERRATION_BACKEND")
        os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "reference"
        try:
            out = lens_aberration_adapter.apply_longitudinal_chromatic_aberration(
                img,
                depth,
                strength=1.3,
                focus_depth=0.45,
                resolution_scale=0.75,
            )
            expected = lens_aberration_reference.apply_longitudinal_chromatic_aberration(
                img,
                depth,
                strength=1.3,
                focus_depth=0.45,
                resolution_scale=0.75,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_ABERRATION_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = old_backend

        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), 1.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_spherical_ca_matches_reference_module_and_preserves_hdr(self):
        img = _hdr_gradient_image()
        depth = _depth_gradient(*img.shape[:2])
        old_backend = os.environ.get("PLATYPUS_LENS_ABERRATION_BACKEND")
        os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "reference"
        try:
            out = lens_aberration_adapter.apply_spherical_aberration(
                img,
                depth,
                strength=0.9,
                aperture=1.4,
                focus_depth=0.4,
                highlight_threshold=0.65,
                resolution_scale=0.5,
            )
            expected = lens_aberration_reference.apply_spherical_aberration(
                img,
                depth,
                strength=0.9,
                aperture=1.4,
                focus_depth=0.4,
                highlight_threshold=0.65,
                resolution_scale=0.5,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_ABERRATION_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = old_backend

        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), 1.0)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    @unittest.skipUnless(lens_aberration_adapter.native_available(), "lens aberration Metal backend is not built")
    def test_metal_lateral_ca_matches_reference_closely(self):
        img = _hdr_gradient_image(40, 52)
        old_backend = os.environ.get("PLATYPUS_LENS_ABERRATION_BACKEND")
        try:
            os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "reference"
            reference = lens_aberration_adapter.apply_lateral_chromatic_aberration(
                img,
                strength=1.2,
                resolution_scale=1.0,
            )
            os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "metal"
            status = lens_aberration_adapter.backend_status()
            if not status.native:
                self.skipTest(status.detail or "Metal backend unavailable")
            actual = lens_aberration_adapter.apply_lateral_chromatic_aberration(
                img,
                strength=1.2,
                resolution_scale=1.0,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_ABERRATION_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = old_backend

        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-3)

    @unittest.skipUnless(lens_aberration_adapter.native_available(), "lens aberration Metal backend is not built")
    def test_metal_longitudinal_ca_matches_reference_closely(self):
        img = _hdr_gradient_image(36, 44)
        depth = _depth_gradient(*img.shape[:2])
        old_backend = os.environ.get("PLATYPUS_LENS_ABERRATION_BACKEND")
        try:
            os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "reference"
            reference = lens_aberration_adapter.apply_longitudinal_chromatic_aberration(
                img,
                depth,
                strength=1.1,
                focus_depth=0.45,
                resolution_scale=1.0,
            )
            os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "metal"
            status = lens_aberration_adapter.backend_status()
            if not status.native:
                self.skipTest(status.detail or "Metal backend unavailable")
            actual = lens_aberration_adapter.apply_longitudinal_chromatic_aberration(
                img,
                depth,
                strength=1.1,
                focus_depth=0.45,
                resolution_scale=1.0,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_ABERRATION_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = old_backend

        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, reference, rtol=8e-3, atol=8e-3)

    @unittest.skipUnless(lens_aberration_adapter.native_available(), "lens aberration Metal backend is not built")
    def test_metal_spherical_ca_matches_reference_closely(self):
        img = _hdr_gradient_image(36, 44)
        depth = _depth_gradient(*img.shape[:2])
        old_backend = os.environ.get("PLATYPUS_LENS_ABERRATION_BACKEND")
        try:
            os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "reference"
            reference = lens_aberration_adapter.apply_spherical_aberration(
                img,
                depth,
                strength=0.9,
                aperture=1.4,
                focus_depth=0.4,
                highlight_threshold=0.65,
                resolution_scale=0.75,
            )
            os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = "metal"
            status = lens_aberration_adapter.backend_status()
            if not status.native:
                self.skipTest(status.detail or "Metal backend unavailable")
            actual = lens_aberration_adapter.apply_spherical_aberration(
                img,
                depth,
                strength=0.9,
                aperture=1.4,
                focus_depth=0.4,
                highlight_threshold=0.65,
                resolution_scale=0.75,
            )
        finally:
            if old_backend is None:
                os.environ.pop("PLATYPUS_LENS_ABERRATION_BACKEND", None)
            else:
                os.environ["PLATYPUS_LENS_ABERRATION_BACKEND"] = old_backend

        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
