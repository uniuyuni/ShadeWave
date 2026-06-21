import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXR_IO_PATH = PROJECT_ROOT / "cores" / "exr_io.py"
EXPORT_PATH = PROJECT_ROOT / "export.py"


def _load_function(path, function_name):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{function_name} was not found in {path}")


def _load_class_function(path, class_name, function_name):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found in {path}")


class ExportAtomicOutputFlowTest(unittest.TestCase):
    def test_export_writes_temporary_file_before_replacing_final_path(self):
        source = _load_class_function(EXPORT_PATH, "ExportFile", "write_to_file")

        self.assertIn("tmp_ex_path = _make_temp_output_path(self.ex_path)", source)
        self.assertIn("_write_openexr_file(\n                    tmp_ex_path,", source)
        self.assertIn("vips_image.write_to_file(tmp_ex_path", source)
        self.assertGreaterEqual(source.count("os.replace(tmp_ex_path, self.ex_path)"), 2)
        self.assertIn("_cleanup_temp_output(tmp_ex_path)", source)

    def test_exr_reader_rejects_empty_parts_with_value_error(self):
        source = _load_function(EXR_IO_PATH, "read_exr")

        self.assertIn("if not f.parts:", source)
        self.assertIn("raise ValueError", source)
        self.assertIn("if not f2.parts:", source)


if __name__ == "__main__":
    unittest.main()
