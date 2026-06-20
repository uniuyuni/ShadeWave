import pathlib
import os
import sys
import unittest

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import params
from cores import core
from effect_backends import image_transform_adapter, image_transform_reference


class ImageTransformBackendTest(unittest.TestCase):
    def test_backend_status_is_reported(self):
        status = image_transform_adapter.backend_status()

        self.assertEqual(status.effect, "image_transform")
        self.assertIn(status.backend, {"effect_backends._image_transform_metal", "effect_backends.image_transform_reference"})

    def test_fit_crop_to_canvas_matches_current_opencv_path(self):
        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "reference"
        rng = np.random.default_rng(123)
        image = rng.random((48, 64, 3), dtype=np.float32)
        source_rect = (7, 5, 41, 30)
        canvas_width = 80
        canvas_height = 52
        draw_width = 71
        draw_height = 52
        offset_x = 4
        offset_y = 0

        resized = cv2.resize(
            image[5:35, 7:48],
            (draw_width, draw_height),
            interpolation=cv2.INTER_AREA,
        )
        expected = np.pad(
            resized,
            ((offset_y, canvas_height - (offset_y + draw_height)), (offset_x, canvas_width - (offset_x + draw_width)), (0, 0)),
            mode="constant",
        )

        try:
            actual = image_transform_adapter.fit_crop_to_canvas(
                image,
                source_rect,
                canvas_width,
                canvas_height,
                draw_width,
                draw_height,
                offset_x,
                offset_y,
                "area",
            )
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_core_crop_image_uses_adapter_without_changing_output(self):
        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "reference"
        rng = np.random.default_rng(456)
        image = rng.random((60, 90, 3), dtype=np.float32)
        disp_info = (4, 8, 62, 38, 1.0)
        crop_rect = (0, 0, 90, 60)
        texture_width = 100
        texture_height = 70

        new_width, new_height, offset_x, offset_y = core.crop_size_and_offset_from_texture(
            texture_width,
            texture_height,
            disp_info,
        )
        expected_img = image_transform_reference.fit_crop_to_canvas(
            image,
            (4, 8, 62, 38),
            texture_width,
            texture_height,
            new_width,
            new_height,
            offset_x,
            offset_y,
            "area",
        )

        try:
            actual_img, actual_disp = core.crop_image(
                image,
                disp_info,
                crop_rect,
                texture_width,
                texture_height,
                0,
                0,
                False,
            )
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

        np.testing.assert_allclose(actual_img, expected_img, rtol=0.0, atol=0.0)
        self.assertEqual(actual_disp, (4, 8, 62, 38, texture_width / 62))

    def test_zoom_crop_source_info_matches_crop_image_zoom_path(self):
        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "reference"
        rng = np.random.default_rng(567)
        image = rng.random((90, 130, 3), dtype=np.float32)
        disp_info = (12, 9, 90, 54, 1.2)
        crop_rect = (0, 0, 130, 90)
        texture_width = 80
        texture_height = 48
        click_x = 37
        click_y = 24
        zoom_ratio = 2.0

        try:
            actual_img, actual_disp = core.crop_image(
                image,
                disp_info,
                crop_rect,
                texture_width,
                texture_height,
                click_x,
                click_y,
                True,
                center_pos=None,
                zoom_ratio=zoom_ratio,
            )
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

        clamped_crop = core._clamp_crop_rect_to_image(crop_rect, image.shape[1], image.shape[0])
        clamped_disp = core._clamp_disp_info_to_image(disp_info, image.shape[1], image.shape[0])
        new_width, new_height, offset_x, offset_y = core.crop_size_and_offset_from_texture(texture_width, texture_height, clamped_disp)
        scale = texture_width / clamped_disp[2] if clamped_disp[2] >= clamped_disp[3] else texture_height / clamped_disp[3]
        source_info, _ = core.zoom_crop_source_info(
            clamped_disp,
            clamped_crop,
            texture_width,
            texture_height,
            click_x,
            click_y,
            None,
            zoom_ratio,
            base_scale=scale,
            base_offset=(offset_x, offset_y),
        )
        cropped, expected_disp = core.crop_image_info(image, source_info, clamped_crop)
        expected_img = image_transform_reference.fit_crop_to_canvas(
            cropped,
            (0, 0, cropped.shape[1], cropped.shape[0]),
            texture_width,
            texture_height,
            texture_width,
            texture_height,
            0,
            0,
            "nearest",
        )
        expected_disp = (
            expected_disp[0],
            expected_disp[1],
            expected_disp[2],
            expected_disp[3],
            texture_width / max(1, expected_disp[2]),
        )

        np.testing.assert_allclose(actual_img, expected_img, rtol=0.0, atol=0.0)
        self.assertEqual(actual_disp, expected_disp)

    def test_transform_to_canvas_matches_current_opencv_affine_path(self):
        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "reference"
        rng = np.random.default_rng(789)
        image = rng.random((40, 56, 3), dtype=np.float32)
        matrix = cv2.getRotationMatrix2D((28, 20), 12.0, 1.0)
        try:
            actual = image_transform_adapter.transform_to_canvas(
                image,
                matrix,
                64,
                64,
                transform_type="affine",
                interpolation="linear",
                border_mode="reflect",
            )
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

        expected = cv2.warpAffine(
            image,
            matrix,
            (64, 64),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_core_rotation_matches_reference_when_backend_is_reference(self):
        previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "reference"
        rng = np.random.default_rng(901)
        image = rng.random((37, 53, 3), dtype=np.float32)
        height, width = image.shape[:2]
        size = max(width, height)
        center = (int(width / 2), int(height / 2))
        matrix = cv2.getRotationMatrix2D(center, -18.0, 1)
        matrix[0, 2] += (size / 2) - center[0]
        matrix[1, 2] += (size / 2) - center[1]
        expected = cv2.warpAffine(
            image,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        try:
            actual = core.rotation(image, -18.0, inter_mode="bilinear", border_mode="constant")
        finally:
            if previous is None:
                os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
            else:
                os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_geometry_effect_defers_trapezoid_as_perspective_preview(self):
        from effects import EffectConfig, GeometryEffect

        previous_native_available = image_transform_adapter.native_available
        image_transform_adapter.native_available = lambda: True
        try:
            image = np.zeros((48, 72, 3), dtype=np.float32)
            geometry = GeometryEffect()
            param = geometry.get_param_dict({"original_img_size": (72, 48)})
            param["original_img_size"] = (72, 48)
            param["correct_horizontal"] = 20
            param["correct_vertical"] = 8
            params.set_crop_rect(param, (0, 0, 72, 48))
            params.set_disp_info(param, (0, 0, 72, 72, 1.0))
            efconfig = EffectConfig()

            diff = geometry.make_diff(image, param, efconfig)

            self.assertIsNone(diff)
            self.assertIsNotNone(efconfig.deferred_geometry_transform)
            self.assertEqual(efconfig.deferred_geometry_transform["transform_type"], "perspective")
            self.assertEqual(efconfig.deferred_geometry_transform["matrix"].shape, (3, 3))
        finally:
            image_transform_adapter.native_available = previous_native_available

    def test_geometry_effect_defers_reference_lines_as_perspective_preview(self):
        from effects import EffectConfig, GeometryEffect

        previous_native_available = image_transform_adapter.native_available
        image_transform_adapter.native_available = lambda: True
        try:
            image = np.zeros((48, 72, 3), dtype=np.float32)
            geometry = GeometryEffect()
            param = geometry.get_param_dict({"original_img_size": (72, 48)})
            param["original_img_size"] = (72, 48)
            param["switch_distortion_correction"] = True
            param["reference_lines"] = [
                ((-0.35, -0.45), (-0.25, 0.45)),
                ((0.25, -0.45), (0.35, 0.45)),
            ]
            params.set_crop_rect(param, (0, 0, 72, 48))
            params.set_disp_info(param, (0, 0, 72, 72, 1.0))
            efconfig = EffectConfig()

            diff = geometry.make_diff(image, param, efconfig)

            self.assertIsNone(diff)
            self.assertIsNotNone(efconfig.deferred_geometry_transform)
            self.assertEqual(efconfig.deferred_geometry_transform["transform_type"], "perspective")
            self.assertEqual(efconfig.deferred_geometry_transform["matrix"].shape, (3, 3))
        finally:
            image_transform_adapter.native_available = previous_native_available

    def test_line_homography_tcg_info_drops_orientation_only(self):
        import effects

        matrix = np.array(
            [[1.0, 0.12, 4.0], [0.03, 0.96, -2.0], [0.0004, -0.0002, 1.0]],
            dtype=np.float64,
        )
        tcg_info = {
            "original_img_size": (72, 48),
            "disp_info": (0, 0, 72, 72, 1.0),
            "rotation": np.deg2rad(17.0),
            "rotation2": np.deg2rad(90.0),
            "flip_mode": 3,
            "matrix": matrix,
        }

        line_tcg_info = effects._line_homography_tcg_info(tcg_info)

        self.assertEqual(line_tcg_info["rotation"], 0.0)
        self.assertEqual(line_tcg_info["rotation2"], 0.0)
        self.assertEqual(line_tcg_info["flip_mode"], 0)
        self.assertIs(line_tcg_info["matrix"], matrix)
        self.assertEqual(tcg_info["flip_mode"], 3)

    def test_geometry_effect_defers_lens_strength_preview(self):
        from effects import EffectConfig, GeometryEffect

        previous_native_available = image_transform_adapter.native_available
        image_transform_adapter.native_available = lambda: True
        try:
            image = np.zeros((48, 72, 3), dtype=np.float32)
            geometry = GeometryEffect()
            param = geometry.get_param_dict({"original_img_size": (72, 48)})
            param["original_img_size"] = (72, 48)
            param["switch_distortion_correction"] = True
            param["lens_distortion_strength"] = 24
            param["lens_distortion_scale"] = 0
            params.set_crop_rect(param, (0, 0, 72, 48))
            params.set_disp_info(param, (0, 0, 72, 72, 1.0))
            efconfig = EffectConfig()

            diff = geometry.make_diff(image, param, efconfig)

            self.assertIsNone(diff)
            self.assertIsNotNone(efconfig.deferred_geometry_transform)
            self.assertEqual(efconfig.deferred_geometry_transform["lens_strength"], 24)
            self.assertEqual(efconfig.deferred_geometry_transform["lens_scale"], 1.0)
        finally:
            image_transform_adapter.native_available = previous_native_available

    def test_geometry_effect_defers_mesh_mls_preview(self):
        from effects import EffectConfig, GeometryEffect

        previous_native_available = image_transform_adapter.native_available
        image_transform_adapter.native_available = lambda: True
        try:
            image = np.zeros((48, 72, 3), dtype=np.float32)
            geometry = GeometryEffect()
            param = geometry.get_param_dict({"original_img_size": (72, 48)})
            param["original_img_size"] = (72, 48)
            param["switch_distortion_correction"] = True
            param["mesh_size"] = [4, 4]
            param["control_points"] = {
                (2, 2): (0.06, -0.04),
            }
            params.set_crop_rect(param, (0, 0, 72, 48))
            params.set_disp_info(param, (0, 0, 72, 72, 1.0))
            efconfig = EffectConfig()

            diff = geometry.make_diff(image, param, efconfig)

            self.assertIsNone(diff)
            self.assertIsNotNone(efconfig.deferred_geometry_transform)
            self.assertIsNotNone(efconfig.deferred_geometry_transform["mesh_map_x"])
            self.assertIsNotNone(efconfig.deferred_geometry_transform["mesh_map_y"])
            self.assertEqual(efconfig.deferred_geometry_transform["mesh_map_x"].dtype, np.float32)
        finally:
            image_transform_adapter.native_available = previous_native_available


if __name__ == "__main__":
    unittest.main()
