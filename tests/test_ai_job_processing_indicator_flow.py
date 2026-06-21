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
        self.assertIn("current_ai_status in (AIJobStatus.QUEUED, AIJobStatus.RUNNING)", source)
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

    def test_ai_noise_controls_are_disabled_until_full_decode(self):
        source = _load_class_function("MainWidget", "update_load_dependent_panels_enabled")

        self.assertIn("disabled = bool(self.mask2_wait_full_load) or not bool(self.image_loaded)", source)
        self.assertIn('"switch_ai_noise_reduction"', source)
        self.assertIn('"chip_ai_noise_reduction"', source)
        self.assertIn('"slider_ai_noise_reduction_intensity"', source)

    def test_selection_change_queues_unfinished_ai_noise_job(self):
        source = _load_class_function("MainWidget", "on_select")

        self.assertIn("_enqueue_unfinished_ai_noise_for_path", source)
        self.assertIn('reason="selection_changed"', source)

    def test_ai_job_viewer_state_accepts_progress_text(self):
        source = _load_class_function("MainWidget", "_set_ai_job_viewer_state")

        self.assertIn('progress_text=""', source)
        self.assertIn("viewer.set_ai_job_state_for_path(file_path, state, progress_text)", source)


if __name__ == "__main__":
    unittest.main()
