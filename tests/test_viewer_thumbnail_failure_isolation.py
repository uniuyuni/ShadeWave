import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
VIEWER_PATH = ROOT / "widgets" / "viewer.py"


def _function_node(name):
    source = VIEWER_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


class ViewerThumbnailFailureIsolationTest(unittest.TestCase):
    def test_thumbnail_decode_failure_isolated_to_single_file(self):
        node, source = _function_node("process_exif_data")

        loop_nodes = [n for n in ast.walk(node) if isinstance(n, ast.For)]
        self.assertTrue(loop_nodes)
        self.assertTrue(any(isinstance(child, ast.Try) for child in ast.iter_child_nodes(loop_nodes[0])))
        self.assertIn("thumb_data_list.append(None)", source)
        self.assertIn("logging.exception", source)
        self.assertNotIn("return [None]*len(file_path_list)", source)
        self.assertNotIn("return [None] * len(file_path_list)", source)


if __name__ == "__main__":
    unittest.main()
