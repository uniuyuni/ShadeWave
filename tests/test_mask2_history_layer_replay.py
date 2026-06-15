import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MASK_EDITOR_PATH = PROJECT_ROOT / "widgets" / "mask_editor2.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class Mask2HistoryLayerReplayTest(unittest.TestCase):
    def test_layer_history_replay_uses_render_update_entrypoint(self):
        source_text = MASK_EDITOR_PATH.read_text()
        node = _load_class_function(MASK_EDITOR_PATH, "MaskEditor2", "update_layer")
        source = ast.get_source_segment(source_text, node)

        self.assertIn("reason=\"history.layer_update\"", source)
        self.assertIn("reason=\"history.layer_create\"", source)
        self.assertGreaterEqual(source.count("self.request_mask_render_update("), 2)
        self.assertNotIn("mask.update_mask()", source)
        self.assertNotIn("self.dispatch('on_structure_change')", source)


if __name__ == "__main__":
    unittest.main()
