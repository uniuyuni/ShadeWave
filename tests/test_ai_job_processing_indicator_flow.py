import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
CONFIG_PATH = PROJECT_ROOT / "config.py"


def _load_class_function(class_name, function_name):
    source = MAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{function_name} was not found")


class AIJobProcessingIndicatorFlowTest(unittest.TestCase):
    def test_top_processing_indicator_only_tracks_current_ai_job(self):
        source = _load_class_function("MainWidget", "update_async_results")

        self.assertIn("current_path = self.imgset.file_path if self.imgset is not None else None", source)
        self.assertIn("has_pending_job_for_path(current_path)", source)
        self.assertNotIn("has_tasks = has_tasks or self.ai_job_manager.has_pending_jobs()", source)

    def test_import_path_applies_ai_job_resume_hook(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("_main_widget.ids['viewer'].set_path(import_path)", source)
        self.assertIn("_main_widget.on_import_path_applied(import_path)", source)

    def test_folder_resume_only_requeues_unfinished_ai_noise_jobs(self):
        source = _load_class_function("MainWidget", "_enqueue_resumable_ai_noise_jobs_for_folder")

        self.assertIn("ai_noise_enabled(primary)", source)
        self.assertIn('primary.get("ai_noise_reduction_result") is not None', source)
        self.assertIn("enqueue_ai_noise_file(image_path, primary)", source)


if __name__ == "__main__":
    unittest.main()
