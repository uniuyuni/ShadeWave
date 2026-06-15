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


if __name__ == "__main__":
    unittest.main()
