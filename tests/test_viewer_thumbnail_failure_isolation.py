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


def _calls_in(node, func_name):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == func_name
    ]


class ViewerThumbnailFailureIsolationTest(unittest.TestCase):
    def test_thumbnail_decode_failure_isolated_to_single_file(self):
        node, source = _function_node("_process_metadata_chunk")

        # The per-file loop must guard each item with its own try so one bad
        # file cannot abort the rest of the chunk.
        loop_nodes = [n for n in ast.walk(node) if isinstance(n, ast.For)]
        self.assertTrue(loop_nodes)
        self.assertTrue(any(isinstance(child, ast.Try) for child in ast.iter_child_nodes(loop_nodes[0])))

        # _build_thumbnail is called inside a try, failures are logged and the
        # item falls back to no-thumbnail instead of failing the whole chunk.
        try_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Try)]
        self.assertTrue(any(_calls_in(t, "_build_thumbnail") for t in try_nodes))
        self.assertIn("logging.exception", source)
        self.assertIn("thumb, deferred = None, False", source)

    def test_deferred_raw_failure_isolated_to_single_file(self):
        node, source = _function_node("_process_deferred_raw")

        # The deferred RAW demosaic path must also catch per-file failures,
        # log them, and still apply a None thumbnail to unblock the item.
        try_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Try)]
        self.assertTrue(any(_calls_in(t, "_build_thumbnail") for t in try_nodes))
        self.assertIn("logging.exception", source)
        self.assertIn("thumb = None", source)
        self.assertTrue(_calls_in(node, "_apply_thumbnail"))


if __name__ == "__main__":
    unittest.main()
