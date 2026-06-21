import msgpack
import numpy as np
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import params


class _EmptyMaskEditor:
    def serialize(self):
        return None

    def clear_mask(self):
        pass


class ParamsMsgpackSafeTest(unittest.TestCase):
    def test_serialize_converts_numpy_scalars_in_mesh_control_points(self):
        param = {
            "original_img_size": (100, 80),
            "crop_rect": (0, 0, 100, 80),
            "control_points": {
                "1,1": (np.float32(0.125), np.float32(-0.25)),
            },
        }

        ser = params.serialize(param, _EmptyMaskEditor())

        msgpack.packb(ser, use_bin_type=True)
        control_point = ser["primary_param"]["control_points"]["1,1"]
        self.assertEqual(control_point, [0.125, -0.25])
        self.assertTrue(all(type(v) is float for v in control_point))

    def test_msgpack_safe_value_converts_numpy_dict_keys_and_arrays(self):
        value = {
            (np.int32(1), np.int32(2)): np.array([np.float32(1.5), np.float32(2.5)]),
        }

        safe = params._msgpack_safe_value(value)

        msgpack.packb(safe, use_bin_type=True)
        self.assertEqual(safe, {"1,2": [1.5, 2.5]})

    def test_ai_noise_heavy_result_marks_deferred_when_not_loaded(self):
        ser = {
            "primary_param": {
                "crop_rect": (0, 0, 2, 2),
                "original_img_size": (2, 2),
                "ai_noise_reduction": True,
                "ai_noise_reduction_result": [[[0.0, 0.0, 0.0]]],
                "ai_noise_reduction_content_key": "key",
                "ai_noise_reduction_source_signature": "source",
                "heavy_saved_at_fidelity": "full",
            },
        }
        param = {}

        params.deserialize(ser, param, _EmptyMaskEditor(), load_heavy=False)

        self.assertTrue(param["_ai_noise_reduction_result_deferred"])
        self.assertNotIn("ai_noise_reduction_result", param)
        self.assertNotIn("ai_noise_reduction_content_key", param)

    def test_ai_noise_keys_without_result_are_not_serialized(self):
        param = {
            "original_img_size": (2, 2),
            "crop_rect": (0, 0, 2, 2),
            "ai_noise_reduction": True,
            "ai_noise_reduction_content_key": "stale-key",
            "ai_noise_reduction_source_signature": "stale-source",
        }

        ser = params.serialize(param, _EmptyMaskEditor())

        primary = ser["primary_param"]
        self.assertNotIn("ai_noise_reduction_content_key", primary)
        self.assertNotIn("ai_noise_reduction_source_signature", primary)


if __name__ == "__main__":
    unittest.main()
