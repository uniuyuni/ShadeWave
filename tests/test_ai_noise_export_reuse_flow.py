import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
EXPORT_PATH = PROJECT_ROOT / "export.py"
PIPELINE_PATH = PROJECT_ROOT / "pipeline.py"


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


class AINoiseExportReuseFlowTest(unittest.TestCase):
    def test_export_pipeline_passes_source_file_path_to_effect_config(self):
        source = _load_function(PIPELINE_PATH, "export_pipeline")

        self.assertIn("efconfig.file_path = primary_param.get('_source_file_path')", source)

    def test_export_pipeline_initializes_stable_upstream_hash_for_lv0_effects(self):
        source = _load_function(PIPELINE_PATH, "export_pipeline")

        self.assertIn("efconfig.upstream_hash = hash(id(img))", source)
        self.assertIn("efconfig.stable_upstream_hash = hash((", source)
        self.assertLess(
            source.index("efconfig.stable_upstream_hash = hash(("),
            source.index("pipeline_lv0("),
        )

    def test_export_file_sets_source_file_path_before_pipeline(self):
        source = _load_class_function(EXPORT_PATH, "ExportFile", "write_to_file")

        self.assertIn("self.param['_source_file_path'] = self.file_path", source)
        self.assertLess(
            source.index("self.param['_source_file_path'] = self.file_path"),
            source.index("pipeline.export_pipeline("),
        )

    def test_ai_noise_uses_file_key_when_file_path_exists_even_during_export(self):
        source = _load_class_function(EFFECTS_PATH, "AINoiseReductonEffect", "make_diff")

        self.assertIn("if file_path:", source)
        self.assertLess(
            source.index("if file_path:"),
            source.index("else:\n                content_key = _ai_noise_content_key"),
        )
        self.assertIn("ai_job_manager = self.ai_job_manager", source)
        self.assertIn("if ai_job_manager is not None and file_path and efconfig.mode != EffectMode.EXPORT:", source)


if __name__ == "__main__":
    unittest.main()
