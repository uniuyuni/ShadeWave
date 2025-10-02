
import os
from typing import List

from iopaint.schema import ModelType, ModelInfo
import logging
from pathlib import Path

from iopaint.const import (
    DEFAULT_MODEL_DIR,
)

def folder_name_to_show_name(name: str) -> str:
    return name.replace("models--", "").replace("--", "/")


def scan_inpaint_models(model_dir: Path) -> List[ModelInfo]:
    res = []
    from iopaint.model import models

    # logging.info(f"Scanning inpaint models in {model_dir}")

    for name, m in models.items():
        if m.is_erase_model and m.is_downloaded():
            res.append(
                ModelInfo(
                    name=name,
                    path=name,
                    model_type=ModelType.INPAINT,
                )
            )
    return res

def scan_models() -> List[ModelInfo]:
    model_dir = os.getenv("XDG_CACHE_HOME", DEFAULT_MODEL_DIR)
    available_models = []
    available_models.extend(scan_inpaint_models(model_dir))
    return available_models
