import ast
import inspect
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MASK_EDITOR2_PATH = ROOT / "widgets" / "mask_editor2.py"
MAIN_KV_PATH = ROOT / "main.kv"

import effects


def _class_function_source(class_name, function_name):
    source = MASK_EDITOR2_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return ast.get_source_segment(source, item)
    raise AssertionError(f"{class_name}.{function_name} not found")


class Mask2DrawBrushFlowTest(unittest.TestCase):
    def test_mask2_options_include_brush_size_param(self):
        defaults = effects.Mask2Effect.get_param_dict({})
        options = effects.Mask2Effect.get_param_dict({}, "mask2_options")

        self.assertEqual(defaults["mask2_freedraw_brush_size"], 300)
        self.assertIn("mask2_freedraw_brush_size", options)

    def test_mask2_brush_size_slider_is_in_kv(self):
        kv = MAIN_KV_PATH.read_text(encoding="utf-8")

        self.assertIn("id: slider_mask2_freedraw_brush_size", kv)
        self.assertIn('text: "Brush Size"', kv)
        self.assertLess(
            kv.index("id: slider_mask2_freedraw_brush_size"),
            kv.index("id: slider_mask2_freedraw_brush_hardness"),
        )

    def test_effect_binding_transfers_brush_size_slider(self):
        source_set = inspect.getsource(effects.Mask2Effect.set2widget)
        source_get = inspect.getsource(effects.Mask2Effect.set2param)

        self.assertIn('slider_mask2_freedraw_brush_size', source_set)
        self.assertIn("mask2_freedraw_brush_size", source_set)
        self.assertIn('slider_mask2_freedraw_brush_size', source_get)
        self.assertIn("param['mask2_freedraw_brush_size']", source_get)

    def test_scroll_routes_size_and_command_scroll_routes_hardness(self):
        helper_source = _class_function_source("BaseMask", "_adjust_draw_brush_from_scroll")
        free_source = _class_function_source("FreeDrawMask", "on_touch_down")
        poly_source = _class_function_source("PolylineMask", "on_touch_down")

        self.assertIn("'meta' in modifiers or 'ctrl' in modifiers", helper_source)
        self.assertIn("slider_mask2_freedraw_brush_hardness", helper_source)
        self.assertIn("slider_mask2_freedraw_brush_size", helper_source)
        self.assertIn("_adjust_draw_brush_from_scroll(touch)", free_source)
        self.assertIn("_adjust_draw_brush_from_scroll(touch)", poly_source)

    def test_new_strokes_read_current_brush_params(self):
        free_source = _class_function_source("FreeDrawMask", "on_touch_down")
        poly_source = _class_function_source("PolylineMask", "_begin_new_polyline")

        self.assertIn("self.brush_size = self._draw_brush_size()", free_source)
        self.assertIn("hardness = self._draw_brush_hardness()", free_source)
        self.assertIn("self.brush_size = self._draw_brush_size()", poly_source)
        self.assertIn("hardness = self._draw_brush_hardness()", poly_source)


if __name__ == "__main__":
    unittest.main()
