import os
import sys
import unittest

import msgpack
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cores.mask2 import cache_keys


class Mask2AICacheKeyFlowTest(unittest.TestCase):
    def test_target_text_key_is_stable_and_msgpack_safe(self):
        key1 = cache_keys.target_text_cache_key((4000, 3000), "person", True)
        key2 = cache_keys.target_text_cache_key([4000, 3000], "person", np.bool_(True))

        self.assertEqual(key1, key2)

        packed = msgpack.packb({"key": key1}, use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)
        self.assertEqual(key1, unpacked["key"])

    def test_face_key_uses_serializable_lists_not_tuples(self):
        key = cache_keys.face_cache_key((6000, 4000), ("rb", "lb", "nose"))

        self.assertEqual("mask2-ai-cache", key[0])
        self.assertEqual("face", key[2])
        self.assertIsInstance(key[3], list)
        self.assertIsInstance(key[4], list)
        self.assertEqual(["rb", "lb", "nose"], key[4])

    def test_numeric_keys_normalize_numpy_scalars(self):
        key = cache_keys.segment_cache_key(
            (np.int64(100), np.int64(50)),
            (np.float32(1.5), np.float64(2.5)),
            (np.float32(40.0), np.float64(20.0)),
            np.bool_(False),
        )

        packed = msgpack.packb({"key": key}, use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)
        self.assertEqual(key, unpacked["key"])


if __name__ == "__main__":
    unittest.main()
