import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC_PATH = PROJECT_ROOT / "docs" / "environment-variables.md"


class EnvironmentVariablesDocTest(unittest.TestCase):
    def test_runtime_environment_variables_are_documented(self):
        text = DOC_PATH.read_text(encoding="utf-8")
        required = [
            "PLATYPUS_AI_DISPLAY_INPUT",
            "PLATYPUS_AI_COMPLETED_CACHE_MAX_MB",
            "PLATYPUS_AI_JOB_NICE",
            "PLATYPUS_AI_SIDECAR_MERGE_MAX_PENDING",
            "PLATYPUS_SAM3_BBOX_CLIP",
            "PLATYPUS_SAM3_ROI_INPUT",
            "PLATYPUS_SAM3_ROI_SCALE",
            "PLATYPUS_IMAGE_TRANSFORM_BACKEND",
            "PLATYPUS_CROSS_FILTER_BACKEND",
            "PLATYPUS_LOW_FREQUENCY_TRANSFER_BACKEND",
            "PLATYPUS_DEBUG_EDGE_REFINE",
            "PLATYPUS_DEBUG_PIPELINE_STATS",
            "PLATYPUS_LOAD_STALL_WARN_SECONDS",
            "PLATYPUS_MEMORY_DEBUG",
            "RUNWARE_API_KEY",
            "DASHSCOPE_API_KEY",
        ]
        for name in required:
            with self.subTest(name=name):
                self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
