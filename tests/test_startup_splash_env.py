import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class StartupSplashEnvTests(unittest.TestCase):
    def test_splash_screen_is_opt_in_by_environment_variable(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn('"PLATYPUS_SPLASH_SCREEN"', source)
        self.assertIn("from utils.envutils import env_flag", source)
        self.assertIn("_splash_close_screen = _display_startup_splash()", source)
        self.assertIn("if not env_flag(_SPLASH_ENV):", source)
        self.assertIn("_close_startup_splash()", source)

    def test_old_commented_splash_hooks_are_removed(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertNotIn("#display_splash_screen", source)
        self.assertNotIn("#close_splash_screen", source)


if __name__ == "__main__":
    unittest.main()
