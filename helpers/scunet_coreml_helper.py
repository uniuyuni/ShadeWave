"""Platypus helper for SCUNet CoreML denoising."""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from effect_backends import low_frequency_transfer_adapter
from utils.external_paths import external_path


helpers_dir = Path(__file__).resolve().parent
platypus_dir = helpers_dir.parent


def _first_existing_project_root() -> Path:
    env_root = os.environ.get("SCUNET_APPLESILICON_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            external_path("SCUNet_CoreML"),
            platypus_dir / "SCUNet_CoreML",
            platypus_dir / "scunet_coreml",
            platypus_dir / "SCUNetCoreML",
        ]
    )
    for candidate in candidates:
        if (candidate / "src" / "scunet_coreml").exists():
            return candidate.resolve()
    return candidates[0].resolve()


project_root = _first_existing_project_root()
package_src = project_root / "src"
if package_src.exists() and str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))

import utils.aiutils as aiutils
from scunet_coreml import DenoiseConfig, SCUNetCoreML
try:
    from scunet_coreml import CancelledError as SCUNetCancelledError
except ImportError:  # pragma: no cover - older helper package
    class SCUNetCancelledError(RuntimeError):
        pass

try:
    import waitinfo
except Exception:  # pragma: no cover
    waitinfo = None


DEFAULT_MODEL = project_root / "models" / "scunet_color_real_psnr_448_fp16.mlpackage"
_progress_callback = None
_cancel_callback = None


def set_progress_callback(callback) -> None:
    global _progress_callback
    _progress_callback = callback


def set_cancel_callback(callback) -> None:
    global _cancel_callback
    _cancel_callback = callback


def is_cancelled_error(exc: Exception) -> bool:
    return isinstance(exc, SCUNetCancelledError)


def _set_wait_text(text: str) -> None:
    if waitinfo is not None:
        waitinfo.set_text("ai_noise_reduction", text)


def _cancel_requested() -> bool:
    if _cancel_callback is None:
        return False
    try:
        return bool(_cancel_callback())
    except Exception:
        logging.exception("SCUNet cancel callback failed")
        return False


@contextmanager
def _cancelable_engine_predict(engine: SCUNetCoreML):
    model = getattr(engine, "model", None)
    original_predict = getattr(model, "predict", None)
    if model is None or original_predict is None:
        yield
        return

    def predict_with_cancel(*args, **kwargs):
        if _cancel_requested():
            raise SCUNetCancelledError("SCUNet CoreML cancelled before tile prediction")
        return original_predict(*args, **kwargs)

    replaced_model = False
    try:
        model.predict = predict_with_cancel
    except Exception:
        class _PredictProxy:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

            def predict(self, *args, **kwargs):
                return predict_with_cancel(*args, **kwargs)

        engine.model = _PredictProxy(model)
        replaced_model = True
    try:
        yield
    finally:
        if replaced_model:
            engine.model = model
        else:
            model.predict = original_predict


def _set_progress(done: int, total: int) -> None:
    _set_wait_text(f"SCUNet {done} / {total}")
    if _progress_callback is not None:
        try:
            _progress_callback(int(done), int(total))
        except Exception:
            logging.exception("SCUNet progress callback failed")


def setup(compute_units: str = "cpu_and_gpu", tile: int = 448, overlap: int = 64):
    cfg = DenoiseConfig(
        model_path=DEFAULT_MODEL,
        compute_units=compute_units,
        tile=tile,
        overlap=overlap,
        progress_every=4,
        progress_callback=_set_progress,
        metadata={"project_root": str(project_root)},
    )
    logging.info("Loading SCUNet CoreML model: %s", DEFAULT_MODEL)
    return SCUNetCoreML(cfg)


def predict(engine: SCUNetCoreML, np_image: np.ndarray, restore_low_frequency: bool = True) -> np.ndarray:
    """Denoise a Platypus linear-light float32 image.

    This intentionally mirrors helpers/scunet_helper.py:
    linear RGB -> log1p tonemap -> SCUNet -> inverse log1p -> color-only low-frequency transfer.
    """
    org_image = np.ascontiguousarray(np.asarray(np_image, dtype=np.float32))
    org_image = np.nan_to_num(org_image, nan=0.0, posinf=1.0, neginf=0.0)

    logging.info("SCUNet CoreML Predicting...")
    _set_wait_text("SCUNet CoreML...")
    t0 = time.time()
    k = aiutils.LOG1P_TONEMAP_K_DEFAULT
    work, hdr_white = aiutils.log1p_tonemap_forward_hdr(org_image, k=k, clip_nonnegative=True)
    with _cancelable_engine_predict(engine):
        result, meta = engine.denoise(work)
    if _cancel_requested():
        raise SCUNetCancelledError("SCUNet CoreML cancelled after denoise")
    result = aiutils.log1p_tonemap_inverse_hdr(result, hdr_white, k=k)

    if restore_low_frequency:
        if _cancel_requested():
            raise SCUNetCancelledError("SCUNet CoreML cancelled before finalizing")
        _set_wait_text("Finalizing...")
        result = low_frequency_transfer_adapter.apply_low_frequency_transfer(
            result,
            org_image,
            sigma=75,
            highlight_threshold=0.7,
            highlight_transition=0.3,
            highlight_detail_strength=0.0,
            luminance_transfer_strength=0.35,
            downsample="auto",
        )

    logging.info(
        "SCUNet CoreML completed in %.2f seconds, tiles=%s",
        time.time() - t0,
        meta.get("tiles"),
    )
    _set_wait_text("")
    return np.asarray(result, dtype=np.float32)


def predict_helper(engine: SCUNetCoreML, np_image: np.ndarray) -> np.ndarray:
    return predict(engine, np_image)


if __name__ == "__main__":
    logging.info("SCUNet CoreML helper")
    logging.info("project_root=%s", project_root)
    logging.info("model=%s %s", DEFAULT_MODEL.exists(), DEFAULT_MODEL)
