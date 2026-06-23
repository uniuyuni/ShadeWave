import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MASK_EDITOR_PATH = ROOT / "widgets" / "mask_editor2.py"
HEADLESS_PIPELINE_PATH = ROOT / "cores" / "mask2" / "headless_pipeline.py"
HEADLESS_MASKS_PATH = ROOT / "cores" / "mask2" / "headless_masks.py"
PARAMS_PATH = ROOT / "params.py"
MAIN_PATH = ROOT / "main.py"
PIPELINE_PATH = ROOT / "pipeline.py"


def _class_method_source(path, class_name, method_name):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(source, child)
    raise AssertionError(f"{class_name}.{method_name} not found")


def _function_source(path, name):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found")


class AIImageCacheFlowTest(unittest.TestCase):
    def test_depth_map_mask_uses_editor_ai_image_cache(self):
        source = _class_method_source(MASK_EDITOR_PATH, "DepthMapMask", "get_mask_image")

        self.assertIn("self.editor.get_ai_depth_map", source)
        self.assertNotIn("self._get_or_compute_image_mask_cache", source)
        self.assertNotIn("self.image_mask_cache", source)

    def test_depth_map_mask_does_not_serialize_raw_depth_per_mask(self):
        serialize_source = _class_method_source(MASK_EDITOR_PATH, "DepthMapMask", "serialize")
        deserialize_source = _class_method_source(MASK_EDITOR_PATH, "DepthMapMask", "deserialize")

        self.assertNotIn("image_mask_cache", serialize_source)
        self.assertNotIn("image_mask_cache_key", serialize_source)
        self.assertNotIn("image_mask_cache", deserialize_source)
        self.assertNotIn("image_mask_cache_key", deserialize_source)

    def test_params_persist_ai_image_cache_as_top_level_pmck_entry(self):
        serialize_source = _function_source(PARAMS_PATH, "serialize")
        deserialize_source = _function_source(PARAMS_PATH, "deserialize")
        load_source = _function_source(PARAMS_PATH, "load_json")

        self.assertIn('ser["ai_image_cache"] = ai_image_cache', serialize_source)
        self.assertIn('ser.get("ai_image_cache")', deserialize_source)
        self.assertIn("set_ai_image_cache(None)", load_source)
        self.assertIn('set_ai_image_cache(dict_.get("ai_image_cache"))', load_source)

    def test_headless_pipeline_exposes_ai_image_cache_to_depth_mask(self):
        pipeline_source = HEADLESS_PIPELINE_PATH.read_text(encoding="utf-8")
        depth_source = _class_method_source(HEADLESS_MASKS_PATH, "HeadlessDepthMapMask", "get_mask_image")

        self.assertIn("self.ai_image_cache = AIImageCache()", pipeline_source)
        self.assertIn("def get_ai_depth_map", pipeline_source)
        self.assertIn("self.pipeline.get_ai_depth_map", depth_source)
        self.assertNotIn("self.image_mask_cache", depth_source)

    def test_main_widget_owns_and_passes_ai_image_cache(self):
        source = MAIN_PATH.read_text(encoding="utf-8")

        self.assertIn("self.ai_image_cache = AIImageCache()", source)
        self.assertIn("self.ids['mask_editor2'].set_ai_image_cache(self.ai_image_cache)", source)
        self.assertIn("self.ai_image_cache.clear()", source)
        self.assertIn("ai_image_cache=self.ai_image_cache", source)

    def test_pipeline_exposes_depth_getter_on_effect_config(self):
        source = PIPELINE_PATH.read_text(encoding="utf-8")
        process_source = _function_source(PIPELINE_PATH, "process_pipeline")
        export_source = _function_source(PIPELINE_PATH, "export_pipeline")

        self.assertIn("def _install_ai_depth_map_getter", source)
        self.assertIn("efconfig.get_ai_depth_map = get_ai_depth_map", source)
        self.assertIn("def _set_ai_depth_map_current_context", source)
        self.assertIn("get_derived_depth_map", source)
        self.assertIn("ai_image_cache=None", process_source)
        self.assertIn("_install_ai_depth_map_getter(efconfig, img, primary_param", process_source)
        self.assertIn("_set_ai_depth_map_current_context(", process_source)
        self.assertIn("ai_image_cache=ai_image_cache", process_source)
        self.assertIn("_install_ai_depth_map_getter(efconfig, img, primary_param", export_source)
        self.assertIn("_set_ai_depth_map_current_context(", export_source)


if __name__ == "__main__":
    unittest.main()
