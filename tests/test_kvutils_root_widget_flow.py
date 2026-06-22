import pathlib
import sys
import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_kvutils_with_stubs():
    class FakeApp:
        @staticmethod
        def get_running_app():
            return None

    stubs = {
        "kivy": types.ModuleType("kivy"),
        "kivy.clock": types.ModuleType("kivy.clock"),
        "kivy.app": types.ModuleType("kivy.app"),
        "kivy.core": types.ModuleType("kivy.core"),
        "kivy.core.window": types.ModuleType("kivy.core.window"),
        "kivy.uix": types.ModuleType("kivy.uix"),
        "kivy.uix.widget": types.ModuleType("kivy.uix.widget"),
        "kivy.uix.scrollview": types.ModuleType("kivy.uix.scrollview"),
        "macos": types.ModuleType("macos"),
    }
    stubs["kivy.clock"].Clock = object()
    stubs["kivy.app"].App = FakeApp
    stubs["kivy.core.window"].Window = object()
    stubs["kivy.uix.widget"].Widget = object
    stubs["kivy.uix.scrollview"].ScrollView = type("ScrollView", (), {})
    stubs["macos"].dpi_scale = lambda: 1.0
    stubs["macos"].get_self_window_position = lambda: (0, 0, 0, 0, None)

    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "_kvutils_under_test",
            PROJECT_ROOT / "utils" / "kvutils.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class DummyWidget:
    def __init__(self, parent=None, children=None):
        self.parent = parent
        self.children = children or []


class KvutilsRootWidgetFlowTest(unittest.TestCase):
    def test_running_app_root_wins_over_window_overlay_children(self):
        kvutils = _load_kvutils_with_stubs()
        app_root = object()
        leaf = DummyWidget()

        with mock.patch.object(
            kvutils.KVApp,
            "get_running_app",
            return_value=SimpleNamespace(root=app_root),
        ):
            self.assertIs(kvutils.get_root_widget(leaf), app_root)

    def test_fallback_uses_original_window_root_after_overlay_is_added(self):
        kvutils = _load_kvutils_with_stubs()
        app_root = object()
        overlay = object()
        window = DummyWidget(children=[overlay, app_root])
        window.parent = window
        leaf = DummyWidget(parent=window)

        with mock.patch.object(kvutils.KVApp, "get_running_app", return_value=None):
            self.assertIs(kvutils.get_root_widget(leaf), app_root)


if __name__ == "__main__":
    unittest.main()
