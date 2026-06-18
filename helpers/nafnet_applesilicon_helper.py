"""Platypus helper for NAFNet AppleSilicon Core ML denoising."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from effect_backends import low_frequency_transfer_adapter


helpers_dir = Path(__file__).resolve().parent
platypus_dir = helpers_dir.parent


def _first_existing_project_root() -> Path:
    env_root = os.environ.get("NAFNET_APPLESILICON_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            platypus_dir / "NAFNet_AppleSilicon",
            platypus_dir / "nafnet_applesilicon",
            platypus_dir / "NAFNetAppleSilicon",
            platypus_dir.parent / "nagi" / "nagi_denoise" / "packages" / "nafnet_applesilicon",
        ]
    )
    for candidate in candidates:
        if (candidate / "src" / "nafnet_applesilicon").exists():
            return candidate.resolve()
    return candidates[0].resolve()


project_root = _first_existing_project_root()
package_src = project_root / "src"
if package_src.exists() and str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))

from nafnet_applesilicon import DenoiseConfig, NAFNetAppleSilicon

try:
    import waitinfo
except Exception:  # pragma: no cover - helper still works outside Platypus.
    waitinfo = None


DEFAULT_FAST_MODEL = project_root / "models" / "nafnet_width64_neuralnetwork_b1_256.mlmodel"
DEFAULT_PATCH_MODEL = project_root / "models" / "nafnet_width64_fp16_512.mlpackage"
DEFAULT_SAFE_MODEL = project_root / "models" / "nafnet_width64_fp16_b4_256.mlpackage"


def _set_wait_text(text: str) -> None:
    if waitinfo is not None:
        waitinfo.set_text("ai_noise_reduction", text)


def _set_progress(done: int, total: int, phase: str) -> None:
    label = "Patch" if phase else "NAFNet"
    _set_wait_text(f"{label} {done} / {total}")


def _restore_low_frequency_enabled(default: bool = True) -> bool:
    value = os.environ.get("PLATYPUS_NAFNET_RESTORE_LOW_FREQUENCY", "").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def setup(
    mode: str = "safe",
    input_space: str = "linear",
    output_space: str = "linear",
    overlap: int = 32,
    compute_units: str | None = None,
):
    """Create a reusable NAFNet AppleSilicon engine.

    mode:
      safe    -> MLProgram CPU/GPU model, artifact patch enabled.
      fast    -> experimental neuralnetwork CPU/NE-capable model.
    """
    mode = str(mode).lower()
    if mode == "fast":
        model_path = DEFAULT_FAST_MODEL
        units = compute_units or "all"
        tile = 256
        batch = 1
    elif mode == "safe":
        model_path = DEFAULT_SAFE_MODEL
        units = compute_units or "cpu_and_gpu"
        tile = 256
        batch = 4
    else:
        raise ValueError(f"unknown NAFNet AppleSilicon mode: {mode}")

    cfg = DenoiseConfig(
        model_path=model_path,
        compute_units=units,
        tile=tile,
        overlap=overlap,
        batch=batch,
        input_space=input_space,
        output_space=output_space,
        artifact_detection=True,
        artifact_patch=True,
        patch_model_path=DEFAULT_PATCH_MODEL,
        patch_compute_units="cpu_and_gpu",
        patch_tile=512,
        hf_threshold=0.060,
        hf_gain=2.0,
        chroma_threshold=0.050,
        chroma_gain=2.0,
        fine_hf_threshold=0.014,
        fine_hf_gain=1.35,
        fine_chroma_threshold=0.008,
        checker_threshold=0.002,
        checker_gain=1.25,
        progress_every=8,
        progress_callback=_set_progress,
        metadata={"mode": mode, "project_root": str(project_root)},
    )
    logging.info("Loading NAFNet AppleSilicon model: %s", model_path)
    return NAFNetAppleSilicon(cfg)


def predict(engine: NAFNetAppleSilicon, np_image: np.ndarray, restore_low_frequency: bool = True) -> np.ndarray:
    """Denoise a Platypus float32 image."""
    org_image = np.ascontiguousarray(np.asarray(np_image, dtype=np.float32))
    org_image = np.nan_to_num(org_image, nan=0.0, posinf=1.0, neginf=0.0)
    logging.info("NAFNet AppleSilicon Predicting...")
    _set_wait_text("NAFNet AppleSilicon...")
    t0 = time.time()
    result, meta = engine.denoise(org_image)

    if restore_low_frequency and _restore_low_frequency_enabled(True):
        _set_wait_text("Finalizing...")
        result = low_frequency_transfer_adapter.apply_low_frequency_transfer(
            result,
            org_image,
            sigma=75,
            highlight_threshold=None,
            luminance_transfer_strength=0.0,
        )

    artifact_count = int(meta.get("artifact_count", 0))
    logging.info(
        "NAFNet AppleSilicon completed in %.2f seconds, artifact_count=%d",
        time.time() - t0,
        artifact_count,
    )
    _set_wait_text("")
    return np.asarray(result, dtype=np.float32)


def predict_helper(engine: NAFNetAppleSilicon, np_image: np.ndarray) -> np.ndarray:
    """Compatibility entry point matching the other Platypus AI helpers."""
    return predict(engine, np_image)


if __name__ == "__main__":
    logging.info("NAFNet AppleSilicon helper")
    logging.info("project_root=%s", project_root)
    logging.info("fast_model=%s %s", DEFAULT_FAST_MODEL.exists(), DEFAULT_FAST_MODEL)
    logging.info("patch_model=%s %s", DEFAULT_PATCH_MODEL.exists(), DEFAULT_PATCH_MODEL)
