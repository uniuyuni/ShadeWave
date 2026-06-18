import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class FrozenStartupLoggingTests(unittest.TestCase):
    def test_frozen_app_installs_file_logging_before_kivy_imports(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        logging_pos = source.index("_install_frozen_startup_logging()")
        kivy_pos = source.index("from kivy.config import Config")

        self.assertLess(logging_pos, kivy_pos)
        self.assertIn('"~/Library/Logs"', source)
        self.assertIn('"Shade Wave"', source)
        self.assertIn('"startup.log"', source)
        self.assertIn("_logging_early.FileHandler", source)
        self.assertIn("_sys_early.excepthook", source)

    def test_main_loop_logs_unhandled_errors_and_always_cleans_cache(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn('logging.exception("Unhandled error in app main loop")', source)
        self.assertIn("finally:", source)
        self.assertIn("cache_system.shutdown()", source)


if __name__ == "__main__":
    unittest.main()
