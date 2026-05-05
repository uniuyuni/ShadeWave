import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"
SWITCH_RESET_MAP_PATH = PROJECT_ROOT / "widgets" / "switch_reset_map.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class RotationHeadResetFlowTest(unittest.TestCase):
    def test_rotation_head_label_has_reset_target_without_saved_switch_param(self):
        main_kv = MAIN_KV_PATH.read_text()
        reset_map = SWITCH_RESET_MAP_PATH.read_text()

        self.assertIn("id: switch_rotation", main_kv)
        self.assertIn('button_enabled: False', main_kv)
        self.assertIn('"switch_rotation": (0, "geometry", "rotation")', reset_map)

    def test_geometry_rotation_subreset_only_resets_rotation_params(self):
        node = _load_class_function(EFFECTS_PATH, "GeometryEffect", "get_param_dict")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), node)

        self.assertIn('if subname == "rotation":', source)
        self.assertIn("'rotation': 0", source)
        self.assertIn("'rotation2': 0", source)
        self.assertIn("'flip_mode': 0", source)
        rotation_block = source.split('if subname == "rotation":', 1)[1].split("param2 = param.copy()", 1)[0]
        self.assertNotIn("crop_rect", rotation_block)
        self.assertNotIn("disp_info", rotation_block)
        self.assertNotIn("switch_rotation", source)


if __name__ == "__main__":
    unittest.main()
