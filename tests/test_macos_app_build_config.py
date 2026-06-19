import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MacOSAppBuildConfigTests(unittest.TestCase):
    def test_libraw_enhanced_is_bundled_for_frozen_app(self):
        source = (ROOT / "scripts" / "build_macos_app_pyinstaller.py").read_text(encoding="utf-8")

        self.assertIn("external' / 'libraw_enhanced", source)
        self.assertIn('"libraw_enhanced"', source)
        self.assertIn("--collect-all", source)

    def test_build_repairs_broken_root_dylib_symlinks(self):
        source = (ROOT / "scripts" / "build_macos_app_pyinstaller.py").read_text(encoding="utf-8")

        self.assertIn("link.is_symlink() and not link.exists()", source)
        self.assertIn("Path(\"../Frameworks/lib\") / dylib.name", source)
        self.assertIn("bundle と扱うため、dangling symlink は必ず張り替える", source)

    def test_build_resigns_after_post_processing(self):
        source = (ROOT / "scripts" / "build_macos_app_pyinstaller.py").read_text(encoding="utf-8")

        self.assertIn("def _ad_hoc_codesign_app", source)
        self.assertIn('"--force", "--deep", "--sign", "-"', source)
        self.assertLess(source.index("_create_framework_lib_symlinks(app)"), source.index("_ad_hoc_codesign_app(app)"))

    def test_build_includes_torch_mps_fallback_runtime_hook(self):
        source = (ROOT / "scripts" / "build_macos_app_pyinstaller.py").read_text(encoding="utf-8")
        hook = (ROOT / "scripts" / "pyinstaller" / "rth_torch_mps_fallback.py").read_text(encoding="utf-8")

        self.assertIn("rth_torch_mps_fallback.py", source)
        self.assertIn("--runtime-hook", source)
        self.assertIn("PYTORCH_ENABLE_MPS_FALLBACK", hook)
        self.assertIn('setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")', hook)
        self.assertIn('setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")', hook)
        self.assertIn('setdefault("PYTORCH_MPS_FAST_MATH", "0")', hook)
        self.assertIn('setdefault("ENABLE_PJRT_COMPATIBILITY", "1")', hook)
        self.assertIn('setdefault("KMP_DUPLICATE_LIB_OK", "FALSE")', hook)
        self.assertIn('setdefault("OMP_DISPLAY_ENV", "FALSE")', hook)
        self.assertIn('setdefault("QS_DRAW_V4", "1")', hook)
        self.assertIn('setdefault("QS_V4_EDGE_SNAP", "1")', hook)
        self.assertNotIn('setdefault("CONDA_PREFIX"', hook)
        self.assertNotIn('setdefault("PYTHONPATH"', hook)


if __name__ == "__main__":
    unittest.main()
