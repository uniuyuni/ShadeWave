import os
import pathlib
import sys
import tempfile
import unittest

import msgpack


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores import pmck_store


class PMCKStoreTest(unittest.TestCase):
    def test_read_missing_can_return_empty_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "a.raw")

            data = pmck_store.read_image(image_path, default_empty=True)

        self.assertEqual("Platypus", data["make"])
        self.assertEqual({}, data["primary_param"])

    def test_write_and_read_image_pmck(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "a.raw")
            data = pmck_store.empty_pmck()
            data["primary_param"]["exposure"] = 1.25

            self.assertTrue(pmck_store.write_image(image_path, data))
            loaded = pmck_store.read_image(image_path)

        self.assertEqual(1.25, loaded["primary_param"]["exposure"])

    def test_expected_token_rejects_concurrent_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "a.raw")
            pmck_path = pmck_store.image_pmck_path(image_path)
            initial = pmck_store.empty_pmck()
            initial["primary_param"]["value"] = "initial"
            pmck_store.write_image(image_path, initial)
            _loaded, token = pmck_store.read_image_with_token(image_path)

            changed = pmck_store.empty_pmck()
            changed["primary_param"]["value"] = "changed"
            pmck_store.write_image(image_path, changed)

            stale = pmck_store.empty_pmck()
            stale["primary_param"]["value"] = "stale"
            self.assertFalse(pmck_store.write_image(image_path, stale, expected_token=token))
            raw = msgpack.unpackb(pathlib.Path(pmck_path).read_bytes(), raw=False)

        self.assertEqual("changed", raw["primary_param"]["value"])

    def test_delete_image_pmck(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "a.raw")
            pmck_store.write_image(image_path, pmck_store.empty_pmck())

            self.assertTrue(pmck_store.exists_image(image_path))
            self.assertTrue(pmck_store.delete_image(image_path))
            self.assertFalse(pmck_store.exists_image(image_path))


if __name__ == "__main__":
    unittest.main()
