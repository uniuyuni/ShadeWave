import pathlib
import importlib.util
import os
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import export


class ExportTiffSharpenFlowTest(unittest.TestCase):
    def test_tiff_sharpen_keeps_16bit_values_from_turning_white(self):
        import numpy as np

        img = np.full((32, 32, 3), 0.25, dtype=np.float32)
        out = export._prepare_output_array(
            img,
            ".TIFF",
            resize_str="",
            sharpen=0.5,
            dithering=False,
        )

        self.assertEqual(out.dtype, np.uint16)
        self.assertLess(out.mean(), 20000)
        self.assertGreater(out.mean(), 15000)

    def test_tiff_save_options_do_not_request_invalid_bitdepth(self):
        with open(export.__file__, "r") as f:
            source = f.read()

        self.assertNotIn("save_options['bitdepth']", source)

    def test_exr_export_keeps_float32_values(self):
        import numpy as np

        img = np.full((32, 32, 3), 1.25, dtype=np.float32)
        out = export._prepare_output_array(
            img,
            ".EXR",
            resize_str="",
            sharpen=0.5,
            dithering=True,
        )

        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out.mean()), 1.25, places=6)

    def test_exr_color_conversion_stays_linear(self):
        import numpy as np

        img = np.full((1, 1, 3), 0.18, dtype=np.float32)
        exr = export._convert_export_color(img, ".EXR", "sRGB IEC61966-2.1")
        jpg = export._convert_export_color(img, ".JPG", "sRGB IEC61966-2.1")

        self.assertLess(float(abs(exr.mean() - 0.18)), 0.001)
        self.assertGreater(float(jpg.mean()), 0.4)

    def test_exr_writer_reports_missing_openexr_dependency(self):
        if importlib.util.find_spec("OpenEXR") is not None:
            self.skipTest("OpenEXR is installed in this environment")

        import numpy as np

        img = np.full((2, 2, 3), 0.25, dtype=np.float32)
        with self.assertRaises(export.ExportFormatError):
            export._write_openexr_file("/tmp/platypus-test-missing-openexr.exr", img)

    def test_exr_writer_uses_piz_compression_constant(self):
        with open(export.__file__, "r") as f:
            source = f.read()

        self.assertIn("OpenEXR.PIZ_COMPRESSION", source)
        self.assertNotIn("exrsave", source)

    def test_exr_writer_roundtrips_float32_when_openexr_is_installed(self):
        if importlib.util.find_spec("OpenEXR") is None:
            self.skipTest("OpenEXR is not installed in this environment")

        import numpy as np
        import OpenEXR

        img = np.full((8, 9, 3), 0.25, dtype=np.float32)
        path = tempfile.mktemp(suffix=".exr")
        try:
            export._write_openexr_file(path, img)
            with OpenEXR.File(path) as f:
                pixels = f.channels()["RGB"].pixels
            self.assertEqual(pixels.dtype, np.float32)
            self.assertEqual(pixels.shape, img.shape)
            self.assertAlmostEqual(float(pixels.mean()), 0.25, places=6)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_exr_writer_stores_profile_chromaticities(self):
        if importlib.util.find_spec("OpenEXR") is None:
            self.skipTest("OpenEXR is not installed in this environment")

        import numpy as np
        import OpenEXR

        img = np.full((2, 3, 3), 0.25, dtype=np.float32)
        chromaticities = export._openexr_chromaticities_for_profile("ProPhoto RGB")
        path = tempfile.mktemp(suffix=".exr")
        try:
            export._write_openexr_file(path, img, chromaticities=chromaticities)
            with OpenEXR.File(path) as f:
                header_chromaticities = f.header()["chromaticities"]
            self.assertEqual(len(header_chromaticities), 8)
            for actual, expected in zip(header_chromaticities, chromaticities):
                self.assertAlmostEqual(float(actual), float(expected), places=6)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_all_icc_profiles_convert_for_vips_and_exr_outputs(self):
        import numpy as np

        profiles = export.get_available_icc_profiles()
        self.assertIn("sRGB IEC61966-2.1", profiles)
        self.assertIn("ProPhoto RGB", profiles)
        img = np.full((4, 5, 3), 0.18, dtype=np.float32)

        for profile in profiles:
            with self.subTest(profile=profile):
                tiff = export._convert_export_color(img, ".TIFF", profile)
                exr = export._convert_export_color(img, ".EXR", profile)
                chromaticities = export._openexr_chromaticities_for_profile(profile)
                vips_image = export._prepare_output_vips_image(tiff, ".TIFF", "", 0, False)

                self.assertEqual(tiff.dtype, np.float32)
                self.assertEqual(exr.dtype, np.float32)
                self.assertEqual(len(chromaticities), 8)
                self.assertEqual(vips_image.format, "ushort")

    def test_aces2065_sets_aces_image_container_flag_only_for_ap0(self):
        self.assertEqual(export._openexr_aces_image_container_flag("ACES2065-1"), 1)
        self.assertIsNone(export._openexr_aces_image_container_flag("ACEScg"))

    def test_aces2065_exr_writer_stores_aces_image_container_flag(self):
        if importlib.util.find_spec("OpenEXR") is None:
            self.skipTest("OpenEXR is not installed in this environment")

        import numpy as np
        import OpenEXR

        img = np.full((2, 3, 3), 0.25, dtype=np.float32)
        path = tempfile.mktemp(suffix=".exr")
        try:
            export._write_openexr_file(
                path,
                img,
                chromaticities=export._openexr_chromaticities_for_profile("ACES2065-1"),
                aces_image_container_flag=export._openexr_aces_image_container_flag("ACES2065-1"),
            )
            with OpenEXR.File(path) as f:
                self.assertEqual(f.header()["acesImageContainerFlag"], 1)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
