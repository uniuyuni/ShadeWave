import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
MASK2_CONTENT_PATH = PROJECT_ROOT / "widgets" / "mask2_content.py"


def _load_class_function(path, class_name, function_name):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class Mask2CreatedMaskEditFlowTest(unittest.TestCase):
    def test_effect_edits_target_created_mask_before_first_stroke(self):
        source = _load_class_function(MAIN_PATH, "MainWidget", "_get_active_effects")

        self.assertIn("mask = editor.get_created_mask() or editor.get_active_mask()", source)
        self.assertIn("mask_index = editor.get_mask_list().index(mask)", source)
        self.assertIn("composit_mask = editor.find_composit_mask(mask, mask_index)", source)
        self.assertIn("effects_owner = composit_mask if composit_mask is not None else mask", source)
        self.assertIn("return (effects_owner.effects, mask.effects_param, mask.mask_id)", source)

    def test_mask2_list_marks_created_mask_as_selected(self):
        source = _load_class_function(MASK2_CONTENT_PATH, "Mask2ContentPanel", "refresh_list")

        self.assertIn("created_mask = self.editor.get_created_mask()", source)
        self.assertIn("'active': mask == active_mask or mask == created_mask", source)


if __name__ == "__main__":
    unittest.main()
