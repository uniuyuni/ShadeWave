import ast
import pathlib
import types
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def _source():
    return MAIN_PATH.read_text(encoding="utf-8")


def _find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


def _class_method_source(class_name, method_name):
    source = _source()
    tree = ast.parse(source)
    cls = _find_class(tree, class_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{class_name}.{method_name} was not found")


def _load_main_widget_methods(*names):
    source = _source()
    tree = ast.parse(source)
    cls = _find_class(tree, "MainWidget")
    found = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = node
    missing = [name for name in names if name not in found]
    if missing:
        raise AssertionError(f"missing MainWidget methods: {missing}")
    module = ast.Module(body=[found[name] for name in names], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {}
    exec(compile(module, str(MAIN_PATH), "exec"), ns)
    return ns


class FakeConfig:
    def __init__(self, display_color_gamut):
        self.display_color_gamut = display_color_gamut

    def get_config(self, key):
        if key != "display_color_gamut":
            raise AssertionError(key)
        return self.display_color_gamut


class FakeDevice:
    def __init__(self, detected):
        self.detected = detected
        self.calls = []

    def get_app_window_screen_color_space_name(self, default, normalize):
        self.calls.append((default, normalize))
        return self.detected


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, *args):
        self.messages.append(args)

    def exception(self, *args):
        self.messages.append(args)


class FakeSelf:
    def __init__(self, methods):
        for name, func in methods.items():
            if callable(func):
                setattr(self, name, types.MethodType(func, self))
        self._auto_display_color_gamut = "sRGB"
        self._auto_display_color_gamut_last_reason = None
        self._fast_display_transform_cache = {"stale": object()}
        self.imgset = None
        self.redraws = 0

    def start_draw_image(self, fast_display=False):
        self.redraws += 1


class AutoDisplayColorGamutFlowTest(unittest.TestCase):
    def _methods(self, configured="auto", detected="Display P3"):
        methods = _load_main_widget_methods(
            "_configured_display_color_gamut",
            "_is_display_color_gamut_auto",
            "_supported_display_color_gamut",
            "refresh_auto_display_color_gamut",
            "_effective_display_color_gamut",
        )
        methods["config"] = FakeConfig(configured)
        methods["device"] = FakeDevice(detected)
        methods["logging"] = FakeLogger()
        return methods

    def test_manual_display_gamut_config_is_used_without_auto_detection(self):
        methods = self._methods(configured="Adobe RGB (1998)", detected="Display P3")
        owner = FakeSelf(methods)
        owner._auto_display_color_gamut = "Display P3"

        self.assertFalse(owner.refresh_auto_display_color_gamut(reason="startup"))
        self.assertEqual(owner._effective_display_color_gamut(), "Adobe RGB (1998)")
        self.assertEqual(methods["device"].calls, [])

    def test_auto_display_gamut_detects_and_caches_supported_screen_profile(self):
        methods = self._methods(configured="auto", detected="Display P3")
        owner = FakeSelf(methods)

        changed = owner.refresh_auto_display_color_gamut(reason="startup", redraw=False)

        self.assertTrue(changed)
        self.assertEqual(owner._auto_display_color_gamut, "Display P3")
        self.assertEqual(owner._effective_display_color_gamut(), "Display P3")
        self.assertEqual(owner._fast_display_transform_cache, {})
        self.assertEqual(methods["device"].calls, [("sRGB", True)])

    def test_auto_display_gamut_keeps_previous_value_for_unsupported_custom_profile(self):
        methods = self._methods(configured="auto", detected="Custom Calibrated Display")
        owner = FakeSelf(methods)
        owner._auto_display_color_gamut = "Rec.2020"

        changed = owner.refresh_auto_display_color_gamut(reason="move", redraw=True)

        self.assertFalse(changed)
        self.assertEqual(owner._auto_display_color_gamut, "Rec.2020")
        self.assertEqual(owner.redraws, 0)

    def test_draw_path_uses_effective_display_gamut_for_auto_config(self):
        draw_source = _class_method_source("MainWidget", "draw_image_core")

        self.assertIn("dst_space = self._effective_display_color_gamut()", draw_source)
        self.assertNotIn("dst_space = config.get_config('display_color_gamut')", draw_source)

    def test_startup_resize_and_move_refresh_auto_display_gamut(self):
        build_source = _class_method_source("MainApp", "build")
        resize_source = _class_method_source("MainApp", "on_window_resize")
        move_source = _class_method_source("MainApp", "on_window_move")
        bind_source = _class_method_source("MainApp", "_bind_window_move_events")

        self.assertLess(
            build_source.index("config.load_config()"),
            build_source.index('refresh_auto_display_color_gamut(reason="startup"'),
        )
        self.assertIn('refresh_auto_display_color_gamut(reason="resize"', resize_source)
        self.assertIn('refresh_auto_display_color_gamut(reason="move"', move_source)
        self.assertIn('"on_move": self.on_window_move', bind_source)
        self.assertIn('"left": self.on_window_move', bind_source)
        self.assertIn('"top": self.on_window_move', bind_source)


if __name__ == "__main__":
    unittest.main()
