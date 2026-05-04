import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def _load_on_fcs_get_file_node():
    tree = ast.parse(MAIN_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "on_fcs_get_file":
            return node
    raise AssertionError("on_fcs_get_file was not found")


def _iter_for_tuple_names(node):
    for child in ast.walk(node):
        if isinstance(child, ast.For) and isinstance(child.iter, ast.Tuple):
            values = []
            for elt in child.iter.elts:
                if isinstance(elt, ast.Constant):
                    values.append(elt.value)
            yield child, tuple(values)


class FullDecodeCropLoadFlowTest(unittest.TestCase):
    def test_full_decode_does_not_overwrite_user_crop_rect(self):
        source = MAIN_PATH.read_text()

        self.assertNotIn("_loaded_pmck_has_crop_rect", source)
        self.assertIn("ユーザー編集値 crop_rect 自体は上書きしない", source)
        self.assertIn("core.convert_rect_to_info", source)
        self.assertIn("params.get_crop_rect(self.primary_param)", source)
        self.assertNotIn("self.primary_param[_k] = param[_k]\n                    self.ids['mask_editor2'].set_primary_param", source)

    def test_full_decode_always_copies_only_size_before_crop_decision(self):
        on_fcs_get_file = _load_on_fcs_get_file_node()
        tuple_loops = [values for _, values in _iter_for_tuple_names(on_fcs_get_file)]

        self.assertIn(("original_img_size", "img_size"), tuple_loops)
        self.assertNotIn(
            ("original_img_size", "img_size", "crop_rect", "disp_info"),
            tuple_loops,
            "FULL_DECODE must not overwrite crop_rect/disp_info outside history.",
        )


if __name__ == "__main__":
    unittest.main()
