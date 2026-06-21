import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import effects
import params
import pipeline
from cores import core
from cores.ai_job_manager import AIJobStatus
from cores.ai_job_manager.ai_noise import (
    ai_noise_content_key,
    ai_noise_source_signature,
)
from cores.mask2.headless_pipeline import Mask2HeadlessPipeline
from cores.mask2.mask_types import MaskTypeStr
from enums import ImageFidelity


def install_headless_config(preview_size=64):
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


class PreviewMask2HeadlessPipeline(Mask2HeadlessPipeline):
    def set_primary_param(self, primary_param, disp_info, redraw_mask=True):
        super().set_primary_param(primary_param, disp_info)


class QueuedThenCompleteAIManager:
    """Small deterministic stand-in for AIJobManager without CoreML/SCUNet."""

    def __init__(self):
        self._queued_keys = set()
        self.request_log = []
        self.cancelled_paths = []

    def request_ai_noise(self, file_path, img, param):
        source_signature = ai_noise_source_signature(file_path, img, param)
        content_key = ai_noise_content_key(
            file_path,
            img,
            param,
            source_signature=source_signature,
        )
        self.request_log.append((file_path, content_key))

        if content_key not in self._queued_keys:
            self._queued_keys.add(content_key)
            return AIJobStatus.QUEUED, None, content_key, source_signature

        mean = np.mean(img, axis=(0, 1), keepdims=True, dtype=np.float32)
        raw = np.ascontiguousarray(np.clip(img * 0.86 + mean * 0.14, 0.0, 1.0), dtype=np.float32)
        return AIJobStatus.COMPLETE, raw, content_key, source_signature

    def cancel_path(self, file_path):
        self.cancelled_paths.append(file_path)


def make_image(seed, size=96):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = np.stack(
        (
            xx / max(size - 1, 1),
            yy / max(size - 1, 1),
            0.35 + 0.25 * np.sin((xx + yy) / 13.0),
        ),
        axis=-1,
    )
    noise = rng.normal(0.0, 0.015, base.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(base + noise, 0.0, 1.0), dtype=np.float32)


def make_param(img, file_path):
    param = {}
    params.set_image_param(param, img)
    params.set_temperature_to_param(param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0)))
    param["_source_file_path"] = file_path
    param["image_fidelity"] = ImageFidelity.FULL.value
    param["switch_ai_noise_reduction"] = True
    param["ai_noise_reduction"] = True
    param["ai_noise_reduction_intensity"] = 65
    param["switch_lens_modifier"] = False
    param["lens_modifier"] = False
    param["switch_exposure_contrast"] = True
    param["exposure"] = 0.0
    param["contrast"] = 0.0
    param["switch_global"] = True
    param["color_separation"] = 0.0
    return param


def make_mask_editor(param, img, preview_size):
    mask_editor = PreviewMask2HeadlessPipeline()
    mask_editor.set_texture_size(preview_size, preview_size)
    mask_editor.set_primary_param(param, params.get_disp_info(param))
    mask_editor.set_ref_image(img, img)
    return mask_editor


def add_full_frame_exposure_mask(mask_editor):
    mask_editor.deserialize(
        {
            "mask2": [
                {
                    "type": MaskTypeStr.COMPOSIT,
                    "name": "Stress Composit",
                    "effects_param": {
                        "switch_exposure_contrast": True,
                        "exposure": 0.12,
                        "ai_noise_reduction": False,
                    },
                    "mask_list": [
                        [
                            {
                                "type": MaskTypeStr.FULL,
                                "name": "Stress Full",
                                "center": [0.5, 0.5],
                                "effects_param": {},
                            },
                            "Add",
                        ]
                    ],
                }
            ]
        }
    )


def assert_float_image(testcase, img, shape=None):
    testcase.assertIsNotNone(img)
    if shape is not None:
        testcase.assertEqual(img.shape[:2], shape)
    testcase.assertEqual(img.dtype, np.float32)
    testcase.assertTrue(np.isfinite(img).all())


class AINRHeadlessStressFlowTest(unittest.TestCase):
    def setUp(self):
        self.old_config = config._config
        self.old_preview_texture_size = config._preview_texture_size
        self.preview_size = 64
        install_headless_config(self.preview_size)

    def tearDown(self):
        config._config = self.old_config
        config._preview_texture_size = self.old_preview_texture_size

    def _run_preview(self, img, crop_image, primary_effects, param, mask_editor, ai_manager, *, is_drag=False):
        out, crop = pipeline.process_pipeline(
            img,
            crop_image,
            False,
            1.0,
            self.preview_size,
            self.preview_size,
            0,
            0,
            primary_effects,
            param,
            mask_editor,
            None,
            1,
            "Ed",
            loading_flag=-1,
            is_drag=is_drag,
            ai_job_manager=ai_manager,
        )
        assert_float_image(self, out, (self.preview_size, self.preview_size))
        assert_float_image(self, crop, (self.preview_size, self.preview_size))
        return out, crop

    def test_mixed_ai_nr_selection_param_mask_and_export_flow_runs_headless(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [str(Path(tmpdir) / f"stress_{i}.jpg") for i in range(3)]
            for i, p in enumerate(paths):
                Path(p).write_bytes(bytes([i + 1]) * 128)

            images = [make_image(i) for i in range(len(paths))]
            params_by_path = [make_param(img, path) for img, path in zip(images, paths)]
            primary_effects = effects.create_effects()
            ai_manager = QueuedThenCompleteAIManager()

            for index, (img, param) in enumerate(zip(images, params_by_path)):
                effects.reeffect_all(primary_effects)
                mask_editor = make_mask_editor(param, img, self.preview_size)
                crop = None

                _out_preview, crop = self._run_preview(
                    img,
                    crop,
                    primary_effects,
                    param,
                    mask_editor,
                    ai_manager,
                )
                self.assertNotIn("ai_noise_reduction_result", param)

                param["ai_noise_reduction_intensity"] = 35 + index * 20
                param["exposure"] = 0.05 * (index + 1)
                effects.reeffect_all(primary_effects)
                _out_nr, crop = self._run_preview(
                    img,
                    crop,
                    primary_effects,
                    param,
                    mask_editor,
                    ai_manager,
                )
                self.assertIsInstance(param.get("ai_noise_reduction_result"), np.ndarray)

                if index == 1:
                    add_full_frame_exposure_mask(mask_editor)
                    param["color_separation"] = 20.0
                    effects.reeffect_all(primary_effects)
                    _out_masked, crop = self._run_preview(
                        img,
                        crop,
                        primary_effects,
                        param,
                        mask_editor,
                        ai_manager,
                    )
                    self.assertEqual(len(mask_editor.get_mask_list()), 1)

                export_effects = effects.create_effects()
                export_out = pipeline.export_pipeline(img, export_effects, param, mask_editor)
                assert_float_image(self, export_out)

            self.assertGreaterEqual(len(ai_manager.request_log), len(paths) * 2)
            self.assertEqual(ai_manager.cancelled_paths, [])


if __name__ == "__main__":
    unittest.main()
