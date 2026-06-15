import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import paths


class UserDataPathsTest(unittest.TestCase):
    def test_user_data_dir_uses_shade_wave_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with mock.patch("utils.paths.Path.home", return_value=home):
                self.assertEqual(paths.user_data_dir(), home / "Pictures" / "Shade Wave")

    def test_ensure_user_data_dir_migrates_legacy_platypus_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            legacy = home / "Pictures" / "Platypus"
            legacy.mkdir(parents=True)
            (legacy / "config.json").write_text('{"kept": true}', encoding="utf-8")

            with mock.patch("utils.paths.Path.home", return_value=home):
                folder = paths.ensure_user_data_dir()

            self.assertEqual(folder, home / "Pictures" / "Shade Wave")
            self.assertTrue((folder / "config.json").is_file())
            self.assertFalse(legacy.exists())

    def test_ensure_user_data_dir_leaves_legacy_folder_when_new_folder_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            pictures = home / "Pictures"
            legacy = pictures / "Platypus"
            current = pictures / "Shade Wave"
            legacy.mkdir(parents=True)
            current.mkdir()
            (legacy / "config.json").write_text('{"old": true}', encoding="utf-8")

            with mock.patch("utils.paths.Path.home", return_value=home):
                folder = paths.ensure_user_data_dir()

            self.assertEqual(folder, current)
            self.assertTrue(legacy.exists())


if __name__ == "__main__":
    unittest.main()
