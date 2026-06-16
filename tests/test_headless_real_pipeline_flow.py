import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import effects
import pipeline
from imageset import ImageSet
from utils.exiftool_safe import safe_get_metadata


ROOT = Path(__file__).resolve().parents[1]
TEST_PHOTOS = ROOT.parent / "test_photos"


class MaskEditorStub:
    def __init__(self):
        self.texture_size = None
        self.primary = None
        self.ref_image = None

    def set_texture_size(self, width, height):
        self.texture_size = (width, height)

    def set_primary_param(self, param, disp_info, redraw_mask=True):
        self.primary = (param, disp_info, redraw_mask)

    def set_ref_image(self, imgc, pre_rotation_img):
        self.ref_image = (imgc, pre_rotation_img)

    def get_mask_list(self):
        return []

    def update(self):
        pass


def install_headless_config(preview_size=384):
    config._config = {
        "import_path": os.getcwd(),
        "lut_path": os.getcwd() + "/lut",
        "preview_size": preview_size,
        "ai_demosaic": False,
        "raw_auto_exposure": False,
        "scale_threshold": 0.5,
        "inpaint_resize_limit": 1024,
        "inpaint_use_realesrgan": False,
        "display_color_gamut": "sRGB",
        "gpu_device": "mps",
        "cat": "cat16",
        "base_resolution_scale": [4096, 4096],
        "display_output_dither": False,
        "display_output_downscale": True,
        "debug_nan_inf_check": False,
        "mesh_rbf_function": "mls",
    }
    config._preview_texture_size = (preview_size, preview_size)


def read_metadata(path):
    metadata = safe_get_metadata(
        [str(path)],
        common_args=[
            "-b",
            "-s",
            "-a",
            "-G1",
            "-x",
            "IFD1:PreviewTIFF",
            "-x",
            "SubIFD1:PreviewTIFF",
        ],
    )[0]
    if not metadata.get("ImageSize") and not metadata.get("RawImageCroppedSize"):
        raise unittest.SkipTest(f"metadata unavailable for real pipeline test: {path}")
    return metadata


def assert_real_pipeline_output(testcase, imgset, param, preview_size):
    mask_editor = MaskEditorStub()
    primary_effects = effects.create_effects()
    out, crop = pipeline.process_pipeline(
        imgset.img,
        None,
        False,
        1.0,
        preview_size,
        preview_size,
        0,
        0,
        primary_effects,
        param,
        mask_editor,
        None,
        1,
        "Ed",
        loading_flag=-1,
        is_drag=False,
    )

    testcase.assertIsNotNone(out)
    testcase.assertIsNotNone(crop)
    testcase.assertEqual(out.shape[:2], (preview_size, preview_size))
    testcase.assertEqual(crop.shape[:2], (preview_size, preview_size))
    testcase.assertEqual(out.dtype, np.float32)
    testcase.assertTrue(np.isfinite(out).all())
    testcase.assertIsNotNone(mask_editor.texture_size)
    testcase.assertIsNotNone(mask_editor.primary)
    testcase.assertIsNotNone(mask_editor.ref_image)


class HeadlessRealPipelineFlowTest(unittest.TestCase):
    def setUp(self):
        self.old_config = config._config
        self.old_preview_texture_size = config._preview_texture_size
        self.preview_size = int(os.getenv("PLATYPUS_REAL_PIPELINE_PREVIEW_SIZE", "384"))
        install_headless_config(self.preview_size)

    def tearDown(self):
        config._config = self.old_config
        config._preview_texture_size = self.old_preview_texture_size

    def test_rgb_file_loads_and_runs_pipeline_without_gui(self):
        path = TEST_PHOTOS / "X-T5 ON2.jpg"
        if not path.exists():
            self.skipTest(f"test photo not available: {path}")

        param = {}
        imgset = ImageSet()
        file_path, loaded, _exif, param, stage = imgset._load_rgb(
            None,
            str(path),
            read_metadata(path),
            param,
        )

        self.assertEqual(file_path, str(path))
        self.assertIs(loaded, imgset)
        self.assertIsNotNone(stage)
        self.assertEqual(param.get("rgb_or_raw"), "rgb")
        self.assertIsNotNone(param.get("original_img_size"))
        self.assertEqual(imgset.img.dtype, np.float32)
        self.assertGreater(imgset.img.shape[0], self.preview_size)
        assert_real_pipeline_output(self, imgset, param, self.preview_size)

    def test_raw_file_demosaics_and_runs_pipeline_without_gui(self):
        path = TEST_PHOTOS / "GR DIGITAL 4 Twilight.DNG"
        if not path.exists():
            self.skipTest(f"test photo not available: {path}")

        param = {}
        imgset = ImageSet()
        file_path, loaded, _exif, param = imgset._load_raw_process(
            None,
            str(path),
            read_metadata(path),
            param,
        )

        self.assertEqual(file_path, str(path))
        self.assertIs(loaded, imgset)
        self.assertEqual(param.get("rgb_or_raw"), "raw")
        self.assertIsNotNone(param.get("original_img_size"))
        self.assertEqual(imgset.img.dtype, np.float32)
        self.assertGreater(imgset.img.shape[0], self.preview_size)
        assert_real_pipeline_output(self, imgset, param, self.preview_size)


if __name__ == "__main__":
    unittest.main()
