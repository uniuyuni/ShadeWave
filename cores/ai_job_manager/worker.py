from __future__ import annotations

import copy
import logging
import os
from multiprocessing import shared_memory
from queue import Empty
import sys
import threading
import traceback

import numpy as np

import config
from .types import AIJobStatus
from .ai_noise import ai_noise_content_key, ai_noise_enabled, ai_noise_source_signature


_EXIF_SIZE_KEYS = ("RawImageCroppedSize", "FullImageSize", "RawImageSize", "ImageSize")


class AIJobCancelled(RuntimeError):
    pass


def ai_job_nice_increment() -> int:
    raw = os.getenv("PLATYPUS_AI_JOB_NICE", "10").strip()
    try:
        value = int(raw)
    except ValueError:
        logging.warning("Invalid PLATYPUS_AI_JOB_NICE=%r; using 10", raw)
        value = 10
    return max(0, min(20, value))


def apply_ai_job_process_priority() -> int:
    increment = ai_job_nice_increment()
    if increment <= 0:
        logging.info("AIJob worker nice adjustment disabled.")
        return 0
    try:
        applied = os.nice(increment)
        logging.info("AIJob worker nice adjusted by +%s; current nice=%s", increment, applied)
        return applied
    except Exception:
        logging.exception("AIJob worker nice adjustment failed")
        return 0


def run_ai_noise(image: np.ndarray, progress_callback=None, cancel_callback=None) -> np.ndarray:
    import helpers.scunet_coreml_helper as scunet_helper

    if not hasattr(run_ai_noise, "_engine"):
        run_ai_noise._engine = scunet_helper.setup()
    scunet_helper.set_progress_callback(progress_callback)
    if hasattr(scunet_helper, "set_cancel_callback"):
        scunet_helper.set_cancel_callback(cancel_callback)
    try:
        return scunet_helper.predict_helper(run_ai_noise._engine, image)
    except Exception as exc:
        if getattr(scunet_helper, "is_cancelled_error", lambda _exc: False)(exc):
            raise AIJobCancelled("AI-NR cancelled") from exc
        raise
    finally:
        scunet_helper.set_progress_callback(None)
        if hasattr(scunet_helper, "set_cancel_callback"):
            scunet_helper.set_cancel_callback(None)


def _read_input_shm(shm_name, shape, dtype_str):
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        return arr.copy()
    finally:
        shm.close()


def _write_result_shm(result: np.ndarray):
    result = np.ascontiguousarray(result, dtype=np.float32)
    shm = shared_memory.SharedMemory(create=True, size=result.nbytes)
    arr = np.ndarray(result.shape, dtype=result.dtype, buffer=shm.buf)
    arr[:] = result
    shm.close()
    return shm.name, result.shape, str(result.dtype)


def _snapshot_exif_data(param_snapshot):
    exif_data = (param_snapshot or {}).get("exif_data")
    if isinstance(exif_data, dict):
        return copy.deepcopy(exif_data)
    return {}


def _has_decode_size_hint(exif_data):
    return isinstance(exif_data, dict) and any(exif_data.get(key) for key in _EXIF_SIZE_KEYS)


def _load_exif_for_file(file_path, param_snapshot):
    exif_data = _snapshot_exif_data(param_snapshot)
    if _has_decode_size_hint(exif_data):
        logging.info("AIJob worker using snapshot EXIF for file decode: file=%s keys=%s", file_path, sorted(exif_data.keys()))
        return exif_data
    try:
        from utils.exiftool_safe import safe_get_metadata

        rows = safe_get_metadata(
            [file_path],
            common_args=["-s", "-a", "-G1"],
            timeout=60,
        )
        loaded = rows[0] if rows else {}
        if isinstance(loaded, dict):
            logging.info(
                "AIJob worker loaded EXIF for file decode: file=%s has_size_hint=%s keys=%s",
                file_path,
                _has_decode_size_hint(loaded),
                sorted(k for k in loaded.keys() if k in _EXIF_SIZE_KEYS or k in ("Orientation", "WB_GRBLevels")),
            )
            return loaded
    except Exception:
        logging.exception("AIJob worker failed to load EXIF for file decode: %s", file_path)
    return exif_data


def _load_file_image(file_path, param_snapshot):
    import file_cache_system
    import imageset

    exif_data = _load_exif_for_file(file_path, param_snapshot)
    param = dict(param_snapshot or {})
    if exif_data:
        param["exif_data"] = copy.deepcopy(exif_data)
    imgset = imageset.ImageSet()
    tasks = imgset.preload(file_path, exif_data, param)
    if not tasks:
        raise ValueError(f"unsupported file for AI job: {file_path}")
    result = file_cache_system.run_method(
        imgset,
        tasks[-1].worker,
        config._config,
        None,
        file_path,
        exif_data,
        param,
    )
    if len(result) >= 5:
        _, payload, _exif, loaded_param, _stage = result[:5]
    else:
        _, payload, _exif, loaded_param = result[:4]
    if isinstance(payload, tuple):
        loaded = imageset.shared_memory_to_imageset(*payload)
    else:
        loaded = payload
    if loaded is None or getattr(loaded, "img", None) is None:
        raise ValueError(f"failed to load file for AI job: {file_path}")
    return loaded.img, loaded_param


def ai_job_worker(input_queue, result_queue, stop_event, config_dict, apply_process_nice=True):
    config._config = config_dict
    logging.basicConfig(level=logging.INFO, format="[%(levelname)-7s] %(message)s")
    if apply_process_nice:
        apply_ai_job_process_priority()

    while not stop_event.is_set():
        try:
            task = input_queue.get(timeout=0.1)
        except Empty:
            continue
        except Exception:
            logging.exception("AIJob worker queue error")
            continue

        if task is None:
            continue

        job_id = task["job_id"]
        try:
            cancel_event = task.get("cancel_event")

            def _cancel_requested():
                return stop_event.is_set() or (cancel_event is not None and cancel_event.is_set())

            result_queue.put({"job_id": job_id, "status": AIJobStatus.RUNNING.value})
            if _cancel_requested():
                raise AIJobCancelled("AI job cancelled before input load")
            if task.get("shm_name"):
                image = _read_input_shm(task["shm_name"], task["shape"], task["dtype"])
                content_key = task.get("content_key")
                source_signature = task.get("source_signature")
                loaded_param = task.get("param_snapshot") or {}
            else:
                image, loaded_param = _load_file_image(task["file_path"], task.get("param_snapshot") or {})
                if not ai_noise_enabled(loaded_param):
                    result_queue.put(
                        {
                            "job_id": job_id,
                            "status": AIJobStatus.STALE.value,
                            "error": "AI-NR disabled before inference",
                        }
                    )
                    continue
                source_signature = ai_noise_source_signature(task["file_path"], image, loaded_param)
                content_key = ai_noise_content_key(
                    task["file_path"],
                    image,
                    loaded_param,
                    source_signature=source_signature,
                )
            kind = task["kind"]
            if kind == "ai_noise_reduction":
                def _progress(done, total):
                    if _cancel_requested():
                        raise AIJobCancelled("AI job cancelled during progress update")
                    result_queue.put(
                        {
                            "job_id": job_id,
                            "status": "progress",
                            "done": int(done),
                            "total": int(total),
                        }
                    )

                result = run_ai_noise(image, progress_callback=_progress, cancel_callback=_cancel_requested)
            else:
                raise ValueError(f"unknown AI job kind: {kind}")
            if _cancel_requested():
                raise AIJobCancelled("AI job cancelled before result write")
            shm_name, shape, dtype = _write_result_shm(result)
            result_queue.put(
                {
                    "job_id": job_id,
                    "status": AIJobStatus.COMPLETE.value,
                    "shm_name": shm_name,
                    "shape": shape,
                    "dtype": dtype,
                    "content_key": content_key,
                    "source_signature": source_signature,
                }
            )
        except AIJobCancelled as exc:
            logging.info("AIJob worker cancelled job_id=%s: %s", job_id, exc)
            result_queue.put({"job_id": job_id, "status": AIJobStatus.CANCELLED.value, "error": str(exc)})
        except Exception as exc:
            logging.error("AIJob worker failed: %s", exc)
            traceback.print_exc()
            result_queue.put({"job_id": job_id, "status": AIJobStatus.ERROR.value, "error": str(exc)})


def should_use_thread_mode() -> bool:
    return bool(getattr(sys, "frozen", False))


def start_thread_worker(input_queue, result_queue, stop_event, config_dict):
    worker = threading.Thread(
        target=ai_job_worker,
        name="AIJobWorkerThread",
        args=(input_queue, result_queue, stop_event, config_dict, False),
        daemon=True,
    )
    worker.start()
    return worker
