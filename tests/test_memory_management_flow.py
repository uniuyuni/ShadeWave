import ast
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
FCS_PATH = ROOT / "file_cache_system.py"
MAIN_PATH = ROOT / "main.py"
PIPELINE_PATH = ROOT / "pipeline.py"
MEMORY_MANAGER_PATH = ROOT / "memory_manager.py"
MASK2_INFERENCE_RUNTIME_PATH = ROOT / "cores" / "mask2" / "inference_runtime.py"
SAM3_HELPER_PATH = ROOT / "helpers" / "sam3_helper.py"
AIUTILS_PATH = ROOT / "utils" / "aiutils.py"


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
        cache_helper = _function_source(MAIN_PATH, "_remember_final_display_image_or_defer")
        pressure_helper = _function_source(MAIN_PATH, "_maybe_enforce_display_memory_policy")

        self.assertIn("self._remember_final_display_image_or_defer(", source)
        self.assertIn("self._log_display_ready_memory(", source)
        self.assertIn("self.cache_system.remember_final_display_image(", cache_helper)
        self.assertIn("self.current_op is not None", cache_helper)
        self.assertIn("self._pending_final_display_cache", cache_helper)
        self.assertIn("self.cache_system.enforce_memory_policy(owner=self, reason=reason)", pressure_helper)
        self.assertIn("now - self._last_display_memory_policy_at < 1.0", pressure_helper)
        self.assertLess(source.index("self.blit_image("), source.index("self._remember_final_display_image_or_defer("))
        self.assertLess(source.index("self.blit_image("), source.index("self._log_display_ready_memory("))

    def test_display_pixel_sampling_does_not_copy_full_clipped_preview(self):
        blit_source = _function_source(MAIN_PATH, "blit_image")
        pixel_source = _function_source(MAIN_PATH, "update_preview_pixel_info")
        draw_source = _function_source(MAIN_PATH, "draw_image_core")

        self.assertIn("self.preview_sample_image = img", blit_source)
        self.assertNotIn("self.preview_sample_image = np.clip(img, 0, 1)", blit_source)
        self.assertIn("np.clip(v, 0, 1)", pixel_source)
        self.assertIn("img = np.asarray(img)", draw_source)
        self.assertNotIn("img = np.array(img)", draw_source)

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

    def test_async_processor_can_clear_cache_without_restarting_idle_worker(self):
        source = _function_source(PIPELINE_PATH, "cancel_all")

        self.assertIn("restart_idle=True", source)
        self.assertIn("restart_idle or self.worker.has_pending_tasks()", source)

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
        self.assertIn("def clear_primary_param_ai_caches", source)
        self.assertIn('"ai_noise_reduction_result"', source)
        self.assertIn('"ai_noise_reduction_content_key"', source)
        self.assertIn("def clear_mask2_ai_caches", source)
        self.assertIn("clear_ai_intermediate_caches", source)
        self.assertIn("def release_ai_model_runtimes", source)
        self.assertIn("clear_mask2_results: bool = False", source)
        self.assertIn("release_ai_models: bool = True", source)
        self.assertIn("depth_model_released", source)
        self.assertIn("face_runtime_released", source)

    def test_memory_manager_clears_ai_noise_raw_result_from_primary_param(self):
        import memory_manager

        param = {
            "ai_noise_reduction_result": np.zeros((8, 8, 3), dtype=np.float32),
            "ai_noise_reduction_content_key": "raw-key",
            "unrelated": "keep",
        }

        result = memory_manager.clear_primary_param_ai_caches(param)

        self.assertEqual(result["primary_param_entries"], 1)
        self.assertEqual(result["primary_param_bytes"], 8 * 8 * 3 * 4)
        self.assertNotIn("ai_noise_reduction_result", param)
        self.assertNotIn("ai_noise_reduction_content_key", param)
        self.assertEqual(param["unrelated"], "keep")

    def test_memory_manager_clears_mask2_ai_intermediate_caches(self):
        import memory_manager

        class MaskEditor:
            def clear_ai_intermediate_caches(self):
                return {"mask2_entries": 2, "mask2_bytes": 128}

        result = memory_manager.clear_effect_intermediate_caches(
            mask_editor2=MaskEditor(),
            clear_mask2_results=True,
            release_ai_models=False,
        )

        self.assertEqual(result["mask2_entries"], 2)
        self.assertEqual(result["mask2_bytes"], 128)

    def test_ai_model_runtimes_have_explicit_release_hooks(self):
        runtime_source = _source(MASK2_INFERENCE_RUNTIME_PATH)
        helper_source = _source(SAM3_HELPER_PATH)
        aiutils_source = _source(AIUTILS_PATH)

        self.assertIn("def release_sam3_runtime", runtime_source)
        self.assertIn("_sam3_processor.clear()", runtime_source)
        self.assertIn("_sam3_processor = None", runtime_source)
        self.assertIn('sys.modules.get("helpers.sam3_helper")', runtime_source)
        self.assertIn("release_model()", runtime_source)
        self.assertIn("def release_depth_runtime", runtime_source)
        self.assertIn("_depth_model = None", runtime_source)
        self.assertIn("def release_face_runtime", runtime_source)
        self.assertIn("_faces.clear()", runtime_source)
        self.assertIn("def release_ai_model_runtimes", runtime_source)
        self.assertIn("release_depth_runtime()", runtime_source)
        self.assertIn("release_face_runtime()", runtime_source)
        self.assertIn("aiutils.empty_cache()", runtime_source)
        self.assertIn("def release_sam3_model", helper_source)
        self.assertIn("__model = None", helper_source)
        self.assertIn("aiutils.empty_cache()", helper_source)
        self.assertIn("torch.mps.empty_cache", aiutils_source)

    def test_file_cache_system_passes_primary_param_to_memory_manager(self):
        source = _function_source(FCS_PATH, "enforce_memory_policy")

        self.assertIn('getattr(owner, "primary_param", None)', source)
        self.assertIn('ids.get("mask_editor2")', source)
        self.assertIn("memory_manager.enforce_memory_policy(effects, processor, primary_param, mask_editor2, reason=reason)", source)

    def test_main_has_manual_memory_release_shortcut(self):
        release_source = _function_source(MAIN_PATH, "force_release_memory_caches")
        shortcut_source = _function_source(MAIN_PATH, "_is_force_memory_release_shortcut")
        key_source = _function_source(MAIN_PATH, "on_key_down")

        self.assertIn("memory_manager.clear_effect_intermediate_caches", release_source)
        self.assertIn("clear_final_display_cache", release_source)
        self.assertIn("self._pending_final_display_cache = None", release_source)
        self.assertIn("self.crop_image = None", release_source)
        self.assertIn("gc.collect()", release_source)
        self.assertIn("threads.primary_param_lock.acquire(blocking=False)", release_source)
        self.assertIn("primary_param_lock_busy", release_source)
        self.assertIn("threads.primary_param_lock.release()", release_source)
        self.assertIn("clear_mask2_results=False", release_source)
        self.assertIn("release_ai_models=True", release_source)
        self.assertIn('"shift"', shortcut_source)
        self.assertIn('"meta"', shortcut_source)
        self.assertIn('"ctrl"', shortcut_source)
        self.assertIn("(8, 127, 266)", shortcut_source)
        self.assertIn('self.force_release_memory_caches(reason="manual_shortcut")', key_source)
        self.assertLess(
            key_source.index("self._is_force_memory_release_shortcut"),
            key_source.index("(codepoint or \"\").lower() == 'm'"),
        )

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
