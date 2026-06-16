import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"
DEFINE_PATH = ROOT / "define.py"


def _function_source(name):
    source = MAIN_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


class ResizeDebugEnvFlowTest(unittest.TestCase):
    def test_resize_debug_uses_environment_variable(self):
        source = _function_source("_resize_debug_enabled")

        self.assertIn('"PLATYPUS_RESIZE_DEBUG"', source)
        self.assertIn('{"1", "true", "yes", "on"}', source)

    def test_resize_debug_no_longer_lives_in_define(self):
        define_source = DEFINE_PATH.read_text(encoding="utf-8")
        main_source = MAIN_PATH.read_text(encoding="utf-8")

        self.assertNotIn("RESIZE_DEBUG =", define_source)
        self.assertNotIn('getattr(define, "RESIZE_DEBUG"', main_source)


if __name__ == "__main__":
    unittest.main()
