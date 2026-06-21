import os
import pathlib
import sys
import tempfile
import threading
import unittest

import msgpack


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores import pmck_store
from utils import preset_utils, rating_io


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

    def test_update_image_serializes_concurrent_read_modify_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "a.raw")
            pmck_store.write_image(image_path, pmck_store.empty_pmck())
            ready = threading.Barrier(6)

            def add_value(index):
                ready.wait(timeout=2)

                def updater(data):
                    primary = data.setdefault("primary_param", {})
                    values = list(primary.get("values", []))
                    values.append(index)
                    primary["values"] = values
                    return data

                self.assertTrue(pmck_store.update_image(image_path, updater))

            threads = [threading.Thread(target=add_value, args=(i,)) for i in range(5)]
            for thread in threads:
                thread.start()
            ready.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())

            loaded = pmck_store.read_image(image_path)

        self.assertEqual([0, 1, 2, 3, 4], sorted(loaded["primary_param"]["values"]))

    def test_update_image_can_delete_under_store_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "a.raw")
            pmck_store.write_image(image_path, pmck_store.empty_pmck())

            self.assertTrue(pmck_store.update_image(image_path, lambda _data: pmck_store.DELETE))

            self.assertFalse(pmck_store.exists_image(image_path))

    def test_rating_and_preset_updates_share_transaction_gateway(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "a.raf")
            ready = threading.Barrier(3)

            def write_rating():
                ready.wait(timeout=2)
                self.assertTrue(rating_io.merge_raw_pmck_rating(image_path, 4))

            def write_preset():
                ready.wait(timeout=2)
                preset_utils.apply_partial_to_pmck_file(image_path, {"exposure": 1.25})

            threads = [threading.Thread(target=write_rating), threading.Thread(target=write_preset)]
            for thread in threads:
                thread.start()
            ready.wait(timeout=2)
            for thread in threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())

            loaded = pmck_store.read_image(image_path)

        self.assertEqual(4, loaded[rating_io.PMCK_RAW_RATING_KEY])
        self.assertEqual(1.25, loaded["primary_param"]["exposure"])


if __name__ == "__main__":
    unittest.main()
