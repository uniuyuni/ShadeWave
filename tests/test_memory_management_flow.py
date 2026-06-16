import ast
import os
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
FCS_PATH = ROOT / "file_cache_system.py"
MAIN_PATH = ROOT / "main.py"
PIPELINE_PATH = ROOT / "pipeline.py"
MEMORY_MANAGER_PATH = ROOT / "memory_manager.py"


def _source(path):
    return path.read_text(encoding="utf-8")


def _function_source(path, name):
    source = _source(path)
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function not found: {name}")


def _class_source(path, name):
    source = _source(path)
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"class not found: {name}")


class MemoryManagementFlowTest(unittest.TestCase):
    def test_file_cache_system_owns_memory_management_entrypoints(self):
        source = _class_source(FCS_PATH, "FileCacheSystem")

        self.assertIn("def release_pmck_payload", source)
        self.assertIn("owner._last_pmck_dict = None", source)
        self.assertIn("def on_image_selection_changed", source)
        self.assertIn("previous_file_path != current_file_path", source)
        self.assertIn("def enforce_memory_policy", source)
        self.assertIn("memory_manager.enforce_memory_policy", source)
        self.assertIn("def log_display_ready_memory", source)
        self.assertIn("memory_manager.log_memory_report", source)
        self.assertIn("def cache_memory_bytes", source)
        self.assertIn("self.final_display_cache = OrderedDict()", source)
        self.assertIn("def remember_final_display_image", source)
        self.assertIn("memory_manager.copy_image_for_cache(image)", source)
        self.assertIn("def get_final_display_image", source)
        self.assertIn("def evict_final_display_cache_for_memory", source)
        self.assertIn("final_display_cache_memory_bytes", source)

    def test_main_releases_pmck_after_saving_previous_selection(self):
        source = _function_source(MAIN_PATH, "on_select")

        save_pos = source.index("self.save_current_sidecar()")
        release_pos = source.index("self.cache_system.on_image_selection_changed(")
        self.assertLess(save_pos, release_pos)
        self.assertIn("previous_file_path = self.imgset.file_path", source)
        self.assertIn("current_file_path = card.file_path if card is not None else None", source)

    def test_main_defers_selection_when_draw_thread_holds_param_lock(self):
        source = _function_source(MAIN_PATH, "on_select")

        self.assertIn("threads.primary_param_lock.acquire(blocking=False)", source)
        self.assertIn("on_select deferred while draw/update owns primary_param_lock", source)
        self.assertIn("KVClock.schedule_once(_retry_select, 0.05)", source)
        self.assertIn("threads.primary_param_lock.release()", source)

    def test_display_ready_memory_log_and_pressure_check_are_hooked_after_blit(self):
        source = _function_source(MAIN_PATH, "draw_image_core")

        self.assertIn("self.cache_system.remember_final_display_image(", source)
        self.assertIn("self._log_display_ready_memory(", source)
        self.assertIn("self.cache_system.enforce_memory_policy(owner=self, reason=\"display_ready\")", source)
        self.assertLess(source.index("self.blit_image("), source.index("self.cache_system.remember_final_display_image("))
        self.assertLess(source.index("self.blit_image("), source.index("self._log_display_ready_memory("))

    def test_reselect_can_show_cached_final_image_before_loading_finishes(self):
        source = _function_source(MAIN_PATH, "on_select")
        helper = _function_source(MAIN_PATH, "_show_cached_final_display_image")

        self.assertIn("self._show_cached_final_display_image(card.file_path)", source)
        self.assertLess(
            source.index("self._show_cached_final_display_image(card.file_path)"),
            source.index("self.cache_system.register_for_preload("),
        )
        self.assertIn("self.cache_system.get_final_display_image(file_path)", helper)
        self.assertIn("self.blit_image(cached", helper)

    def test_async_processor_can_drop_completed_cache_without_canceling_running_tasks(self):
        source = _class_source(PIPELINE_PATH, "AsyncPipelineManager")

        self.assertIn("def clear_completed_cache", source)
        self.assertIn('value.get("status") == "COMPLETE"', source)
        self.assertNotIn("self.worker.cancel_all()", _function_source(PIPELINE_PATH, "clear_completed_cache"))

    def test_memory_manager_uses_env_thresholds_and_clears_effect_intermediates(self):
        source = _source(MEMORY_MANAGER_PATH)

        self.assertIn("PLATYPUS_MEMORY_DEBUG", source)
        self.assertIn("PLATYPUS_MEMORY_AVAILABLE_MIN_MB", source)
        self.assertIn("PLATYPUS_MEMORY_RSS_LIMIT_MB", source)
        self.assertIn("/usr/bin/vm_stat", source)
        self.assertIn("def copy_image_for_cache", source)
        self.assertIn("final_display_cache_bytes", source)
        self.assertIn("def clear_effect_intermediate_caches", source)
        self.assertIn("effect.reeffect()", source)
        self.assertIn('"clear_completed_cache"', source)

    def test_file_cache_system_logs_stalled_loads_and_uses_valid_start_method(self):
        source = _source(FCS_PATH)
        process_queue = _function_source(FCS_PATH, "process_preload_queue")

        self.assertIn("PLATYPUS_LOAD_STALL_WARN_SECONDS", source)
        self.assertIn("FCS load still waiting", source)
        self.assertIn("concurrent.futures.wait(", source)
        self.assertIn("self._start_loading_thread(file_path, exif_data, param, imgset)", process_queue)
        self.assertNotIn("_start_loading_process", process_queue)


if __name__ == "__main__":
    unittest.main()
