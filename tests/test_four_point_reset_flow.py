import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
FOUR_POINT_WIDGET_PATH = PROJECT_ROOT / "widgets" / "distortion_correction" / "four_point_correction_widget.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class FourPointResetFlowTest(unittest.TestCase):
    def test_empty_four_points_syncs_editor_to_default_corners(self):
        node = _load_class_function(
            FOUR_POINT_WIDGET_PATH,
            "FourPointCorrectionWidget",
            "set_correction_params",
        )
        source = ast.get_source_segment(FOUR_POINT_WIDGET_PATH.read_text(), node)

        self.assertIn("if four_points != [] else self._default_corners()", source)
        self.assertIn("self.tcg_info = params.param_to_tcg_info(param)", source)
        self.assertIn("self._sync_tcg_to_kivy()", source)
        self.assertIn("self.update_preview()", source)

    def test_reset_button_and_param_sync_share_the_same_default_corners(self):
        reset_node = _load_class_function(
            FOUR_POINT_WIDGET_PATH,
            "FourPointCorrectionWidget",
            "_reset_corners",
        )
        default_node = _load_class_function(
            FOUR_POINT_WIDGET_PATH,
            "FourPointCorrectionWidget",
            "_default_corners",
        )
        source = FOUR_POINT_WIDGET_PATH.read_text()

        self.assertIn("self.corner_positions_tcg = self._default_corners()", ast.get_source_segment(source, reset_node))
        self.assertIn("(-0.5, -0.5)", ast.get_source_segment(source, default_node))
        self.assertIn("(0.5, 0.5)", ast.get_source_segment(source, default_node))

    def test_default_corner_markers_are_clamped_without_becoming_params(self):
        sync_node = _load_class_function(
            FOUR_POINT_WIDGET_PATH,
            "FourPointCorrectionWidget",
            "_sync_tcg_to_kivy",
        )
        get_node = _load_class_function(
            FOUR_POINT_WIDGET_PATH,
            "FourPointCorrectionWidget",
            "get_correction_params",
        )
        drag_node = _load_class_function(
            FOUR_POINT_WIDGET_PATH,
            "FourPointCorrectionWidget",
            "_on_handle_move",
        )
        source = FOUR_POINT_WIDGET_PATH.read_text()

        self.assertIn("if self._using_default_corners:", ast.get_source_segment(source, sync_node))
        self.assertIn("_clamp_handle_center_to_widget", ast.get_source_segment(source, sync_node))
        self.assertIn('return {"four_points": []}', ast.get_source_segment(source, get_node))
        self.assertIn("preserve_default_corners=preserve_default_corners", ast.get_source_segment(source, drag_node))


if __name__ == "__main__":
    unittest.main()
