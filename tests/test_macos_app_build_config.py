import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MacOSAppBuildConfigTests(unittest.TestCase):
    def test_libraw_enhanced_is_bundled_for_frozen_app(self):
        source = (ROOT / "scripts" / "build_macos_app_pyinstaller.py").read_text(encoding="utf-8")

        self.assertIn("external' / 'libraw_enhanced", source)
        self.assertIn('"libraw_enhanced"', source)
        self.assertIn("--collect-all", source)


if __name__ == "__main__":
    unittest.main()
