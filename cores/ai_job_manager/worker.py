from __future__ import annotations

import logging
from multiprocessing import shared_memory
from queue import Empty
import sys
import threading
import traceback

import numpy as np

import config
from .types import AIJobStatus
from .ai_noise import ai_noise_content_key, ai_noise_source_signature


def run_ai_noise(image: np.ndarray) -> np.ndarray:
    import helpers.scunet_coreml_helper as scunet_helper

    if not hasattr(run_ai_noise, "_engine"):
        run_ai_noise._engine = scunet_helper.setup()
    return scunet_helper.predict_helper(run_ai_noise._engine, image)


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


def _load_file_image(file_path, param_snapshot):
    import file_cache_system
    import imageset

    exif_data = {}
    param = dict(param_snapshot or {})
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


def ai_job_worker(input_queue, result_queue, stop_event, config_dict):
    config._config = config_dict
    logging.basicConfig(level=logging.INFO, format="[%(levelname)-7s] %(message)s")

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
            result_queue.put({"job_id": job_id, "status": AIJobStatus.RUNNING.value})
            if task.get("shm_name"):
                image = _read_input_shm(task["shm_name"], task["shape"], task["dtype"])
                content_key = task.get("content_key")
                source_signature = task.get("source_signature")
            else:
                image, loaded_param = _load_file_image(task["file_path"], task.get("param_snapshot") or {})
                source_signature = ai_noise_source_signature(task["file_path"], image, loaded_param)
                content_key = ai_noise_content_key(
                    task["file_path"],
                    image,
                    loaded_param,
                    source_signature=source_signature,
                )
            kind = task["kind"]
            if kind == "ai_noise_reduction":
                result = run_ai_noise(image)
            else:
                raise ValueError(f"unknown AI job kind: {kind}")
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
        args=(input_queue, result_queue, stop_event, config_dict),
        daemon=True,
    )
    worker.start()
    return worker
