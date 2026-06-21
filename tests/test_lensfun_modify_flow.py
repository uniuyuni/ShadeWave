import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cores import core


class _SubpixelOnlyModifier:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def apply_color_modification(self, img):
        return False

    def apply_subpixel_distortion(self):
        y, x = np.indices((self.height, self.width), dtype=np.float32)
        coords = np.empty((self.height, self.width, 3, 2), dtype=np.float32)
        coords[..., 0, 0] = np.clip(x + 1.0, 0.0, self.width - 1.0)
        coords[..., 0, 1] = y
        coords[..., 1, 0] = np.clip(x - 1.0, 0.0, self.width - 1.0)
        coords[..., 1, 1] = y
        coords[..., 2, 0] = x
        coords[..., 2, 1] = np.clip(y + 1.0, 0.0, self.height - 1.0)
        return coords

    def apply_geometry_distortion(self):
        return None


class _CombinedModifier(_SubpixelOnlyModifier):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.combined_calls = 0
        self.subpixel_calls = 0
        self.geometry_calls = 0

    def apply_subpixel_distortion(self):
        self.subpixel_calls += 1
        return super().apply_subpixel_distortion()

    def apply_subpixel_geometry_distortion(self):
        self.combined_calls += 1
        return super().apply_subpixel_distortion()

    def apply_geometry_distortion(self):
        self.geometry_calls += 1
        y, x = np.indices((self.height, self.width), dtype=np.float32)
        return np.stack([x, y], axis=-1)


class _FakeLensfunModifier:
    initialized = None

    def __init__(self, lens, crop_factor, width, height):
        self.lens = lens
        self.crop_factor = crop_factor
        self.width = width
        self.height = height

    def initialize(self, focal_length, aperture, distance, *, pixel_format):
        type(self).initialized = (focal_length, aperture, distance, pixel_format)


class _FakeLensfunDatabase:
    def find_cameras(self, make, model, loose_search=True):
        return [SimpleNamespace(crop_factor=1.5)]

    def find_lenses(self, camera, lensmake, lensmodel, loose_search=False):
        return [object()]


class LensfunModifyFlowTest(unittest.TestCase):
    def setUp(self):
        _FakeLensfunModifier.initialized = None
        core._lensfun_db_instance = None

    def test_setup_lensfun_normalizes_exiftool_display_values(self):
        fake_lensfunpy = SimpleNamespace(
            Database=_FakeLensfunDatabase,
            Modifier=_FakeLensfunModifier,
        )
        exif_data = {
            "Make": "FUJIFILM",
            "Model": "X-T5",
            "LensMake": "FUJIFILM",
            "LensModel": "XF33mmF1.4 R LM WR",
            "FocalLength": "33.5 mm",
            "ApertureValue": "2.97 (f/2.8)",
            "SubjectDistanceRange": "Unknown",
        }

        with patch.dict(sys.modules, {"lensfunpy": fake_lensfunpy}):
            mod = core.setup_lensfun((4000, 3000), exif_data)

        self.assertIsInstance(mod, _FakeLensfunModifier)
        focal_length, aperture, distance, pixel_format = _FakeLensfunModifier.initialized
        self.assertEqual(33.5, focal_length)
        self.assertEqual(2.8, aperture)
        self.assertEqual(100, distance)
        self.assertEqual(np.float32, pixel_format)

    def test_setup_lensfun_converts_aperture_value_apex_when_no_f_number_exists(self):
        fake_lensfunpy = SimpleNamespace(
            Database=_FakeLensfunDatabase,
            Modifier=_FakeLensfunModifier,
        )
        exif_data = {
            "Make": "FUJIFILM",
            "Model": "X-T5",
            "LensMake": "FUJIFILM",
            "LensModel": "XF33mmF1.4 R LM WR",
            "FocalLength": "33 mm",
            "ApertureValue": "3",
            "SubjectDistance": "1.25 m",
        }

        with patch.dict(sys.modules, {"lensfunpy": fake_lensfunpy}):
            core.setup_lensfun((4000, 3000), exif_data)

        focal_length, aperture, distance, _pixel_format = _FakeLensfunModifier.initialized
        self.assertEqual(33.0, focal_length)
        self.assertAlmostEqual(2 ** 1.5, aperture)
        self.assertEqual(1.25, distance)

    def test_subpixel_modifier_does_not_mutate_input_when_color_mod_is_off(self):
        height, width = 8, 9
        img = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
        original = img.copy()

        out, is_cm, is_sd, is_gd = core.modify_lensfun(
            _SubpixelOnlyModifier(height, width),
            img,
            is_cm=False,
            is_sd=True,
            is_gd=False,
        )

        self.assertFalse(is_cm)
        self.assertTrue(is_sd)
        self.assertFalse(is_gd)
        self.assertFalse(np.shares_memory(out, img))
        np.testing.assert_array_equal(img, original)
        self.assertFalse(np.array_equal(out, original))

    def test_repeated_subpixel_toggle_calls_are_not_cumulative(self):
        height, width = 8, 9
        img = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
        original = img.copy()
        modifier = _SubpixelOnlyModifier(height, width)

        first, _, _, _ = core.modify_lensfun(
            modifier,
            img,
            is_cm=False,
            is_sd=True,
            is_gd=False,
        )

        for _ in range(5):
            off, is_cm, is_sd, is_gd = core.modify_lensfun(
                modifier,
                img,
                is_cm=False,
                is_sd=False,
                is_gd=False,
            )
            self.assertFalse(is_cm)
            self.assertFalse(is_sd)
            self.assertFalse(is_gd)
            np.testing.assert_array_equal(off, original)
            np.testing.assert_array_equal(img, original)

            on, is_cm, is_sd, is_gd = core.modify_lensfun(
                modifier,
                img,
                is_cm=False,
                is_sd=True,
                is_gd=False,
            )
            self.assertFalse(is_cm)
            self.assertTrue(is_sd)
            self.assertFalse(is_gd)
            np.testing.assert_array_equal(img, original)
            np.testing.assert_allclose(on, first, rtol=0.0, atol=0.0)

    def test_combined_subpixel_geometry_map_is_used_without_second_geometry_pass(self):
        height, width = 8, 9
        img = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
        modifier = _CombinedModifier(height, width)

        out, is_cm, is_sd, is_gd = core.modify_lensfun(
            modifier,
            img,
            is_cm=False,
            is_sd=True,
            is_gd=True,
        )

        self.assertFalse(is_cm)
        self.assertTrue(is_sd)
        self.assertTrue(is_gd)
        self.assertEqual(modifier.combined_calls, 1)
        self.assertEqual(modifier.subpixel_calls, 0)
        self.assertEqual(modifier.geometry_calls, 0)
        self.assertFalse(np.shares_memory(out, img))
        self.assertFalse(np.array_equal(out, img))


if __name__ == "__main__":
    unittest.main()
