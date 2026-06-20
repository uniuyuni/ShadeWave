import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import colour_functions_adapter as colour_functions
from utils import aiutils


class AiDisplayInputAdapterTest(unittest.TestCase):
    def test_to_ai_display_rgb_matches_colour_display_transform(self):
        image = np.array(
            [
                [[0.0, 0.2, 0.8], [1.2, 0.5, -0.1]],
                [[0.03, 0.04, 0.05], [0.8, 0.7, 0.6]],
            ],
            dtype=np.float32,
        )

        out = aiutils.to_ai_display_rgb(image, cat="CAT16")
        basis = colour_functions.display_color_transform_basis("ProPhoto RGB", "sRGB", "CAT16")
        expected = colour_functions.apply_display_color_transform(image, basis, "sRGB")
        expected = np.clip(expected, 0.0, 1.0).astype(np.float32)

        np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)
        self.assertEqual(out.dtype, np.float32)

    def test_to_ai_display_rgb_can_be_disabled(self):
        image = np.array([[[0.0, 1.2, -0.1]]], dtype=np.float32)
        old = os.environ.get("PLATYPUS_AI_DISPLAY_INPUT")
        try:
            os.environ["PLATYPUS_AI_DISPLAY_INPUT"] = "0"
            out = aiutils.to_ai_display_rgb(image)
        finally:
            if old is None:
                os.environ.pop("PLATYPUS_AI_DISPLAY_INPUT", None)
            else:
                os.environ["PLATYPUS_AI_DISPLAY_INPUT"] = old

        np.testing.assert_array_equal(out, image)


class Mask2InferenceRuntimeAiInputTest(unittest.TestCase):
    def setUp(self):
        from cores.mask2 import inference_runtime

        self.runtime = inference_runtime
        self.runtime._sam3_processor = None
        self.runtime._depth_model = None
        self.runtime._faces = None

    def tearDown(self):
        self.runtime._sam3_processor = None
        self.runtime._depth_model = None
        self.runtime._faces = None

    def test_depth_and_face_use_ai_display_adapter_but_sam3_uses_original_input(self):
        runtime = self.runtime
        original = np.zeros((8, 10, 3), dtype=np.float32)
        converted = np.ones_like(original, dtype=np.float32) * 0.42
        calls = {}

        depth_module = types.ModuleType("helpers.depth_pro_helper")
        depth_module.setup_model = mock.Mock(return_value="depth-model")
        def predict_depth(model, img):
            calls["depth_img"] = img
            return np.zeros((8, 10), dtype=np.float32)
        depth_module.predict_model = mock.Mock(side_effect=predict_depth)

        class Faces:
            def __eq__(self, other):
                return False

        facer_module = types.ModuleType("helpers.facer_helper")
        def create_faces(img, device="cpu"):
            calls["face_img"] = img
            return Faces()
        facer_module.create_faces = mock.Mock(side_effect=create_faces)
        facer_module.draw_face_mask = mock.Mock(return_value=np.ones((8, 10), dtype=np.float32))

        sam3_module = types.ModuleType("helpers.sam3_helper")
        sam3_module.setup_sam3 = mock.Mock(return_value="sam3")
        def predict_sam3(processor, img, bbox, image_key=None):
            calls["sam3_img"] = img
            calls["sam3_bbox"] = bbox
            calls["sam3_image_key"] = image_key
            return np.ones(img.shape[:2], dtype=np.float32)
        sam3_module.predict_sam3_for_bbox = mock.Mock(side_effect=predict_sam3)

        with mock.patch.dict(sys.modules, {
            "helpers.depth_pro_helper": depth_module,
            "helpers.facer_helper": facer_module,
            "helpers.sam3_helper": sam3_module,
        }):
            with mock.patch.object(runtime.config, "get_config", return_value="cpu"):
                with mock.patch.object(runtime.aiutils, "to_ai_display_rgb", return_value=converted) as adapter:
                    with mock.patch.object(
                            runtime.cutout_guided,
                            "create_cutout_mask_guided",
                            side_effect=lambda image, mask, radius, eps: mask,
                    ) as guided:
                        runtime.predict_depth_map(original)
                        runtime.predict_face_mask(original, [])
                        sam3_mask = runtime.predict_sam3_bbox(original, [1, 1, 4, 4], False)

        self.assertEqual(adapter.call_count, 2)
        self.assertIs(calls["depth_img"], converted)
        self.assertIs(calls["face_img"], converted)
        self.assertEqual(guided.call_count, 2)
        self.assertIs(guided.call_args_list[0].args[0], converted)
        self.assertTrue(np.shares_memory(guided.call_args_list[1].args[0], original))
        self.assertEqual(guided.call_args_list[0].kwargs["radius"], 20)
        self.assertEqual(guided.call_args_list[1].kwargs["radius"], 20)
        self.assertTrue(np.shares_memory(calls["sam3_img"], original))
        self.assertEqual(calls["sam3_bbox"], [1, 1, 4, 4])
        self.assertEqual(calls["sam3_image_key"][0], "roi")
        self.assertEqual(int(sam3_mask.sum()), 16)
        self.assertEqual(float(sam3_mask[:1, :].sum()), 0.0)
        self.assertEqual(float(sam3_mask[:, :1].sum()), 0.0)
        self.assertEqual(float(sam3_mask[5:, :].sum()), 0.0)
        self.assertEqual(float(sam3_mask[:, 5:].sum()), 0.0)

    def test_sam3_bbox_is_clamped_before_prediction(self):
        runtime = self.runtime
        original = np.zeros((6, 8, 3), dtype=np.float32)
        calls = {}

        sam3_module = types.ModuleType("helpers.sam3_helper")
        sam3_module.setup_sam3 = mock.Mock(return_value="sam3")
        def predict_sam3(processor, img, bbox, image_key=None):
            calls["bbox"] = bbox
            calls["image_key"] = image_key
            return np.ones(img.shape[:2], dtype=np.float32)
        sam3_module.predict_sam3_for_bbox = mock.Mock(side_effect=predict_sam3)

        with mock.patch.dict(sys.modules, {"helpers.sam3_helper": sam3_module}):
            with mock.patch.object(runtime.config, "get_config", return_value="cpu"):
                with mock.patch.object(runtime.aiutils, "to_ai_display_rgb", return_value=original):
                    with mock.patch.object(
                            runtime.cutout_guided,
                            "create_cutout_mask_guided",
                            side_effect=lambda image, mask, radius, eps: mask,
                    ):
                        mask = runtime.predict_sam3_bbox(original, [-3.2, 2.2, 7.6, 9.0], False)

        self.assertEqual(calls["bbox"], [0, 1, 5, 4])
        self.assertEqual(calls["image_key"][0], "roi")
        self.assertEqual(int(mask.sum()), 20)
        self.assertEqual(float(mask[:2, :].sum()), 0.0)
        self.assertEqual(float(mask[:, 5:].sum()), 0.0)

    def test_sam3_bbox_prediction_uses_roi_image_and_local_bbox(self):
        runtime = self.runtime
        original = np.zeros((120, 160, 3), dtype=np.float32)
        calls = {}

        sam3_module = types.ModuleType("helpers.sam3_helper")
        sam3_module.setup_sam3 = mock.Mock(return_value="sam3")
        def predict_sam3(processor, img, bbox, image_key=None):
            calls["image_shape"] = img.shape
            calls["bbox"] = bbox
            calls["image_key"] = image_key
            return np.ones(img.shape[:2], dtype=np.float32)
        sam3_module.predict_sam3_for_bbox = mock.Mock(side_effect=predict_sam3)

        old_scale = os.environ.get("PLATYPUS_SAM3_ROI_SCALE")
        try:
            os.environ["PLATYPUS_SAM3_ROI_SCALE"] = "1.5"
            with mock.patch.dict(sys.modules, {"helpers.sam3_helper": sam3_module}):
                with mock.patch.object(runtime.config, "get_config", return_value="cpu"):
                    with mock.patch.object(runtime.aiutils, "to_ai_display_rgb", return_value=original):
                        with mock.patch.object(
                                runtime.cutout_guided,
                                "create_cutout_mask_guided",
                                side_effect=lambda image, mask, radius, eps: mask,
                        ):
                            mask = runtime.predict_sam3_bbox(original, [70, 80, 10, 8], False)
        finally:
            if old_scale is None:
                os.environ.pop("PLATYPUS_SAM3_ROI_SCALE", None)
            else:
                os.environ["PLATYPUS_SAM3_ROI_SCALE"] = old_scale

        self.assertEqual(calls["image_shape"], (12, 16, 3))
        self.assertEqual(calls["bbox"], [3, 2, 10, 8])
        self.assertEqual(calls["image_key"][0], "roi")
        self.assertEqual(calls["image_key"][2], (67, 78, 83, 90))
        self.assertEqual(int(mask.sum()), 80)

    def test_sam3_bbox_guided_filter_uses_smaller_bbox_roi(self):
        runtime = self.runtime
        original = np.zeros((120, 160, 3), dtype=np.float32)
        calls = {}

        sam3_module = types.ModuleType("helpers.sam3_helper")
        sam3_module.setup_sam3 = mock.Mock(return_value="sam3")
        sam3_module.predict_sam3_for_bbox = mock.Mock(
            side_effect=lambda _processor, img, _bbox, image_key=None: np.full(img.shape[:2], 2.0, dtype=np.float32)
        )

        def fake_guided(image, src, radius, eps):
            calls["guide_shape"] = image.shape
            calls["src_shape"] = src.shape
            calls["radius"] = radius
            return src

        with mock.patch.dict(sys.modules, {"helpers.sam3_helper": sam3_module}):
            with mock.patch.object(runtime.config, "get_config", return_value="cpu"):
                with mock.patch.object(runtime.aiutils, "to_ai_display_rgb", return_value=original):
                    with mock.patch.object(
                            runtime.cutout_guided,
                            "create_cutout_mask_guided",
                            side_effect=fake_guided,
                    ):
                        mask = runtime.predict_sam3_bbox(original, [70, 80, 10, 8], False)

        self.assertEqual(calls["guide_shape"], (48, 50, 3))
        self.assertEqual(calls["src_shape"], (48, 50))
        self.assertEqual(calls["radius"], 20)
        self.assertEqual(float(mask[80:88, 70:80].sum()), 160.0)
        self.assertEqual(float(mask[:80, :].sum()), 0.0)
        self.assertEqual(float(mask[:, :70].sum()), 0.0)


class Sam3HelperImageKeyTest(unittest.TestCase):
    def test_bbox_prediction_reuses_set_image_for_same_image_key(self):
        from helpers import sam3_helper
        import torch

        class FakeProcessor:
            def __init__(self):
                self.set_image_calls = 0

            def set_image(self, image):
                self.set_image_calls += 1
                return {"image_shape": image.shape}

            def reset_all_prompts(self, inference_state):
                return None

            def add_geometric_prompt(self, state, box, label=True):
                h, w = state["image_shape"][:2]
                return {"masks": [torch.ones((1, h, w), dtype=torch.float32)]}

        processor = FakeProcessor()
        sam3_dict = {
            "processor": processor,
            "image": None,
            "image_key": None,
            "inference_state": None,
            "_device_logged": True,
        }
        image = np.zeros((6, 8, 3), dtype=np.float32)
        image_key = ("roi", "same-source", (0, 0, 8, 6), image.shape)

        sam3_helper.predict_sam3_for_bbox(sam3_dict, image, [1, 1, 3, 2], image_key=image_key)
        sam3_helper.predict_sam3_for_bbox(sam3_dict, image.copy(), [2, 1, 3, 2], image_key=image_key)

        self.assertEqual(processor.set_image_calls, 1)

    def test_bbox_prediction_selects_candidate_with_bbox_overlap(self):
        from helpers import sam3_helper
        import torch

        class FakeProcessor:
            def set_image(self, image):
                return {"image_shape": image.shape}

            def reset_all_prompts(self, inference_state):
                return None

            def add_geometric_prompt(self, state, box, label=True):
                mask_outside = torch.zeros((1, 6, 8), dtype=torch.float32)
                mask_outside[:, 0:2, 0:2] = 1.0
                mask_inside = torch.zeros((1, 6, 8), dtype=torch.float32)
                mask_inside[:, 2:4, 3:6] = 1.0
                return {"masks": [mask_outside, mask_inside]}

        sam3_dict = {
            "processor": FakeProcessor(),
            "image": None,
            "image_key": None,
            "inference_state": None,
            "_device_logged": True,
        }
        image = np.zeros((6, 8, 3), dtype=np.float32)

        mask = sam3_helper.predict_sam3_for_bbox(
            sam3_dict,
            image,
            [3, 2, 3, 2],
            image_key=("roi", "candidate-test"),
        )

        self.assertEqual(int(mask[2:4, 3:6].sum()), 6)
        self.assertEqual(int(mask[0:2, 0:2].sum()), 0)

    def test_bbox_candidate_log_includes_count_and_selected_index(self):
        from helpers import sam3_helper
        import torch

        class FakeProcessor:
            def set_image(self, image):
                return {"image_shape": image.shape}

            def reset_all_prompts(self, inference_state):
                return None

            def add_geometric_prompt(self, state, box, label=True):
                mask_a = torch.zeros((1, 6, 8), dtype=torch.float32)
                mask_b = torch.zeros((1, 6, 8), dtype=torch.float32)
                mask_b[:, 2:4, 3:6] = 1.0
                return {"masks": [mask_a, mask_b]}

        sam3_dict = {
            "processor": FakeProcessor(),
            "image": None,
            "image_key": None,
            "inference_state": None,
            "_device_logged": True,
        }
        image = np.zeros((6, 8, 3), dtype=np.float32)

        with self.assertLogs("helpers.sam3_helper", level="INFO") as logs:
            sam3_helper.predict_sam3_for_bbox(
                sam3_dict,
                image,
                [3, 2, 3, 2],
                image_key=("roi", "candidate-log-test"),
            )

        text = "\n".join(logs.output)
        self.assertIn("SAM3 bbox mask candidates count=2 selected_index=1", text)

    def test_bbox_prediction_prefers_tighter_candidate_over_huge_overlap(self):
        from helpers import sam3_helper
        import torch

        class FakeProcessor:
            def set_image(self, image):
                return {"image_shape": image.shape}

            def reset_all_prompts(self, inference_state):
                return None

            def add_geometric_prompt(self, state, box, label=True):
                huge = torch.ones((1, 10, 10), dtype=torch.float32)
                tight = torch.zeros((1, 10, 10), dtype=torch.float32)
                tight[:, 3:7, 3:7] = 1.0
                return {"masks": [huge, tight]}

        sam3_dict = {
            "processor": FakeProcessor(),
            "image": None,
            "image_key": None,
            "inference_state": None,
            "_device_logged": True,
        }
        image = np.zeros((10, 10, 3), dtype=np.float32)

        mask = sam3_helper.predict_sam3_for_bbox(
            sam3_dict,
            image,
            [3, 3, 4, 4],
            image_key=("roi", "tight-candidate-test"),
        )

        self.assertEqual(int(mask[3:7, 3:7].sum()), 16)
        self.assertEqual(int(mask[:3, :].sum()), 0)
        self.assertEqual(int(mask[:, :3].sum()), 0)


class Sam3CoreMLBackboneHelperTest(unittest.TestCase):
    def test_compiled_model_uses_package_metadata_fallback(self):
        from helpers import sam3_coreml_backbone_helper as helper

        old_model_path = os.environ.get("PLATYPUS_SAM3_COREML_BACKBONE_MODEL")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = pathlib.Path(tmp)
                compiled_path = tmp_path / "sam3_backbone.mlmodelc"
                compiled_path.mkdir()
                package_metadata_path = tmp_path / "sam3_backbone.mlpackage.json"
                package_metadata_path.write_text("{}", encoding="utf-8")
                os.environ["PLATYPUS_SAM3_COREML_BACKBONE_MODEL"] = str(compiled_path)

                self.assertEqual(helper._model_path(), compiled_path)
                self.assertEqual(helper._metadata_path(compiled_path), package_metadata_path)
        finally:
            if old_model_path is None:
                os.environ.pop("PLATYPUS_SAM3_COREML_BACKBONE_MODEL", None)
            else:
                os.environ["PLATYPUS_SAM3_COREML_BACKBONE_MODEL"] = old_model_path

    def test_tensor_tree_flatten_and_restore_preserves_backbone_shape(self):
        from helpers import sam3_coreml_backbone_helper as helper
        from sam3.model.data_misc import NestedTensor
        import torch

        output = {
            "vision_features": torch.ones((1, 2, 3, 4), dtype=torch.float32),
            "vision_pos_enc": [torch.ones((1, 2, 3, 4), dtype=torch.float32) * 2],
            "backbone_fpn": [torch.ones((1, 2, 3, 4), dtype=torch.float32) * 3],
            "sam2_backbone_out": {
                "vision_features": torch.ones((1, 2, 3, 4), dtype=torch.float32) * 4,
                "vision_pos_enc": [torch.ones((1, 2, 3, 4), dtype=torch.float32) * 5],
                "backbone_fpn": [
                    NestedTensor(
                        torch.ones((1, 2, 3, 4), dtype=torch.float32) * 6,
                        None,
                    )
                ],
            },
        }

        spec, flat = helper._flatten_tensor_tree(output)
        outputs = {name: tensor.numpy() for name, tensor in flat}
        restored = helper._restore_tensor_tree(spec, outputs, torch.device("cpu"))

        self.assertEqual(set(restored.keys()), set(output.keys()))
        np.testing.assert_array_equal(restored["vision_features"].numpy(), output["vision_features"].numpy())
        np.testing.assert_array_equal(
            restored["sam2_backbone_out"]["backbone_fpn"][0].tensors.numpy(),
            output["sam2_backbone_out"]["backbone_fpn"][0].tensors.numpy(),
        )
        self.assertIsNone(restored["sam2_backbone_out"]["backbone_fpn"][0].mask)

    def test_restore_uses_vision_features_when_compiled_fpn_output_is_folded(self):
        from helpers import sam3_coreml_backbone_helper as helper
        import torch

        spec = {
            "kind": "dict",
            "items": {
                "backbone_fpn": {
                    "kind": "list",
                    "items": [
                        {"kind": "tensor", "name": "out_backbone_fpn_2"},
                    ],
                },
            },
        }
        outputs = {
            "out_vision_features": np.ones((1, 2, 3, 4), dtype=np.float32),
        }

        restored = helper._restore_tensor_tree(spec, outputs, torch.device("cpu"))

        np.testing.assert_array_equal(restored["backbone_fpn"][0].numpy(), outputs["out_vision_features"])


if __name__ == "__main__":
    unittest.main()
