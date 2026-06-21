from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from multiprocessing import Event, Process, Queue, shared_memory
import time
from dataclasses import replace
from queue import Empty, Queue as ThreadQueue
from typing import Any

import numpy as np

import config
import threads
from .ai_noise import (
    AI_NOISE_KIND,
    ai_noise_content_key,
    ai_noise_source_signature,
    clone_ai_noise_param_snapshot,
)
from .types import AIJob, AIJobResult, AIJobStatus, viewer_state_for_status
from .worker import ai_job_worker, should_use_thread_mode, start_thread_worker


FOREGROUND_PRIORITY = 0
BACKGROUND_PRIORITY = 100
FINISHED_JOB_RETENTION = 64
MAX_WORKER_RESTARTS = 2
DEFAULT_COMPLETED_RESULTS_CACHE_MB = 512


def completed_results_cache_limit_bytes() -> int:
    raw = os.getenv("PLATYPUS_AI_COMPLETED_CACHE_MAX_MB", str(DEFAULT_COMPLETED_RESULTS_CACHE_MB)).strip()
    try:
        value = float(raw)
    except ValueError:
        logging.warning("Invalid PLATYPUS_AI_COMPLETED_CACHE_MAX_MB=%r; using %s", raw, DEFAULT_COMPLETED_RESULTS_CACHE_MB)
        value = float(DEFAULT_COMPLETED_RESULTS_CACHE_MB)
    if value <= 0:
        return 0
    return int(value * 1024 * 1024)


class AIJobManager:
    def __init__(self, *, viewer_state_callback=None):
        self._lock = threads.ai_job_manager_lock
        self.thread_mode = should_use_thread_mode()
        self.input_queue = ThreadQueue() if self.thread_mode else Queue()
        self.result_queue = ThreadQueue() if self.thread_mode else Queue()
        self.stop_event = threading.Event() if self.thread_mode else Event()
        self.process = None
        self.viewer_state_callback = viewer_state_callback

        self._job_counter = 0
        self.jobs: dict[int, AIJob] = {}
        self.status_by_job: dict[int, AIJobStatus] = {}
        self.job_by_target: dict[tuple[str, str], int] = {}
        self.completed_results: OrderedDict[tuple[str, str, str], np.ndarray] = OrderedDict()
        self.completed_results_total_bytes = 0
        self.active_input_shms: dict[int, shared_memory.SharedMemory] = {}
        self.active_started_at: dict[int, float] = {}
        self.pending_payloads: dict[int, dict[str, Any]] = {}
        self.cancel_events: dict[int, Any] = {}
        self.priority_by_job: dict[int, int] = {}
        self.dispatched_job_id: int | None = None
        self.worker_restart_count = 0

    def start(self):
        with self._lock:
            if self.process is not None and self.process.is_alive():
                return
            self.stop_event.clear()
            if self.thread_mode:
                self.process = start_thread_worker(self.input_queue, self.result_queue, self.stop_event, config._config)
            else:
                self.process = Process(
                    target=ai_job_worker,
                    name="AIJobWorker",
                    args=(self.input_queue, self.result_queue, self.stop_event, config._config, True),
                )
                self.process.daemon = True
                self.process.start()
            logging.info("AIJobManager worker started.")

    def stop(self):
        with self._lock:
            if self.process is None:
                return
            self._terminate_worker_process()
            self._cleanup_all_input_shms()

    def _restart_worker(self, *, count_restart: bool = True, mark_pending_on_limit: bool = True):
        if self.thread_mode:
            logging.warning("AI job worker restart skipped in thread mode to avoid overlapping CoreML inference.")
            return False
        if count_restart and self.worker_restart_count >= MAX_WORKER_RESTARTS:
            logging.error("AI job worker restart limit reached; pending jobs will be marked error.")
            if mark_pending_on_limit:
                self._mark_pending_jobs_error("AI job worker restart limit reached")
            return False
        if count_restart:
            self.worker_restart_count += 1
        pending = list(self.pending_payloads.values())
        self._drain_result_queue_discard_shm()
        self._terminate_worker_process()
        self.input_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = Event()
        self.dispatched_job_id = None
        self.start()
        self.pending_payloads = {
            int(payload["job_id"]): payload
            for payload in pending
        }
        self._dispatch_next_payload()
        return True

    def _terminate_worker_process(self):
        if self.process is None:
            return
        self.stop_event.set()
        self.process.join(timeout=0.2)
        if not self.thread_mode and self.process.is_alive():
            logging.warning("Terminating AI job worker process %s", self.process.pid)
            self.process.terminate()
            self.process.join(timeout=0.2)
        if not self.thread_mode and self.process.is_alive():
            try:
                self.process.kill()
                self.process.join()
            except Exception:
                logging.exception("failed to kill AI job worker")
        self.process = None

    def _next_job_id(self) -> int:
        self._job_counter += 1
        return self._job_counter

    def _target_key(self, kind: str, file_path: str) -> tuple[str, str]:
        return (kind, file_path)

    def _result_key(self, job: AIJob) -> tuple[str, str, str]:
        return (job.kind, job.file_path, job.content_key)

    def _emit_viewer_state(self, file_path: str, state: str | None, progress_text: str = ""):
        if self.viewer_state_callback is None:
            return
        try:
            self.viewer_state_callback(file_path, state, progress_text)
        except TypeError:
            try:
                self.viewer_state_callback(file_path, state)
            except Exception:
                logging.exception("AI job viewer state callback failed")
        except Exception:
            logging.exception("AI job viewer state callback failed")

    def _set_status(self, job: AIJob, status: AIJobStatus):
        self.status_by_job[job.job_id] = status
        self._emit_viewer_state(job.file_path, viewer_state_for_status(status), "")

    def _set_progress(self, job: AIJob, done: int, total: int):
        if total <= 0:
            return
        done = max(0, min(int(done), int(total)))
        self._emit_viewer_state(job.file_path, AIJobStatus.RUNNING.value, f"{done}/{int(total)}")

    def get_status_for_path(self, file_path: str, kind: str = AI_NOISE_KIND) -> AIJobStatus | None:
        with self._lock:
            self._mark_dead_worker_jobs_error()
            job_id = self.job_by_target.get(self._target_key(kind, file_path))
            if job_id is None:
                return None
            return self.status_by_job.get(job_id)

    def has_pending_job_for_path(self, file_path: str, kind: str = AI_NOISE_KIND) -> bool:
        status = self.get_status_for_path(file_path, kind=kind)
        return status in (AIJobStatus.QUEUED, AIJobStatus.RUNNING)

    def viewer_states(self) -> dict[str, str]:
        with self._lock:
            self._mark_dead_worker_jobs_error()
            states: dict[str, str] = {}
            for job_id, job in self.jobs.items():
                state = viewer_state_for_status(self.status_by_job.get(job_id))
                if state:
                    states[job.file_path] = state
            return states

    def pending_count(self) -> int:
        with self._lock:
            self._mark_dead_worker_jobs_error()
            return sum(
                1
                for status in self.status_by_job.values()
                if status in (AIJobStatus.QUEUED, AIJobStatus.RUNNING)
            )

    def has_pending_jobs(self) -> bool:
        with self._lock:
            self._mark_dead_worker_jobs_error()
            return any(
                status in (AIJobStatus.QUEUED, AIJobStatus.RUNNING)
                for status in self.status_by_job.values()
            )

    def get_completed_result(self, kind: str, file_path: str, content_key: str):
        with self._lock:
            key = (kind, file_path, content_key)
            result = self.completed_results.get(key)
            if result is not None:
                self.completed_results.move_to_end(key)
                logging.info("AIJobManager completed cache hit: kind=%s file=%s content_key=%s", kind, file_path, content_key)
            return result

    def discard_completed_result(self, job: AIJob) -> None:
        with self._lock:
            removed = self._pop_completed_result(self._result_key(job))
            if removed is not None:
                logging.info(
                    "AIJobManager discarded completed cache: file=%s content_key=%s bytes=%s remaining_bytes=%s",
                    job.file_path,
                    job.content_key,
                    int(getattr(removed, "nbytes", 0) or 0),
                    self.completed_results_bytes(),
                )

    def has_completed_result(self, job: AIJob) -> bool:
        with self._lock:
            return self._result_key(job) in self.completed_results

    def completed_results_bytes(self) -> int:
        with self._lock:
            return self.completed_results_total_bytes

    def _result_nbytes(self, result: np.ndarray | None) -> int:
        return int(getattr(result, "nbytes", 0) or 0)

    def _pop_completed_result(self, key: tuple[str, str, str]) -> np.ndarray | None:
        removed = self.completed_results.pop(key, None)
        if removed is not None:
            self.completed_results_total_bytes = max(0, self.completed_results_total_bytes - self._result_nbytes(removed))
        return removed

    def _store_completed_result(self, job: AIJob, raw: np.ndarray) -> bool:
        limit = completed_results_cache_limit_bytes()
        key = self._result_key(job)
        self._pop_completed_result(key)
        result_bytes = int(getattr(raw, "nbytes", 0) or 0)
        if limit <= 0:
            logging.info(
                "AIJobManager completed cache disabled; result is returned without caching: file=%s bytes=%s",
                job.file_path,
                result_bytes,
            )
            return False
        if result_bytes > limit:
            logging.warning(
                "AIJobManager completed result exceeds cache limit; result is returned without caching: "
                "file=%s bytes=%s limit=%s",
                job.file_path,
                result_bytes,
                limit,
            )
            return False
        self.completed_results[key] = raw
        self.completed_results_total_bytes += result_bytes
        self._enforce_completed_results_budget(limit)
        logging.info(
            "AIJobManager stored completed cache: file=%s content_key=%s bytes=%s total_bytes=%s limit=%s",
            job.file_path,
            job.content_key,
            result_bytes,
            self.completed_results_total_bytes,
            limit,
        )
        return True

    def _enforce_completed_results_budget(self, limit: int | None = None) -> None:
        if limit is None:
            limit = completed_results_cache_limit_bytes()
        while self.completed_results and self.completed_results_total_bytes > limit:
            key, removed = self.completed_results.popitem(last=False)
            self.completed_results_total_bytes = max(0, self.completed_results_total_bytes - self._result_nbytes(removed))
            logging.warning(
                "AIJobManager evicted completed cache to stay within memory budget: "
                "kind=%s file=%s content_key=%s bytes=%s remaining_bytes=%s limit=%s",
                key[0],
                key[1],
                key[2],
                int(getattr(removed, "nbytes", 0) or 0),
                self.completed_results_total_bytes,
                limit,
            )

    def request_ai_noise(self, file_path: str, image: np.ndarray, param: dict[str, Any]):
        source_signature = ai_noise_source_signature(file_path, image, param)
        content_key = ai_noise_content_key(
            file_path,
            image,
            param,
            source_signature=source_signature,
        )
        with self._lock:
            result_key = (AI_NOISE_KIND, file_path, content_key)
            result = self.completed_results.get(result_key)
            if result is not None:
                self.completed_results.move_to_end(result_key)
                logging.info("AIJobManager reused completed AI-NR result: file=%s content_key=%s", file_path, content_key)
                return AIJobStatus.COMPLETE, result, content_key, source_signature

            job = self.enqueue_image_job(
                kind=AI_NOISE_KIND,
                file_path=file_path,
                image=image,
                param_snapshot=clone_ai_noise_param_snapshot(param),
                content_key=content_key,
                source_signature=source_signature,
                replace_queued=True,
                priority=FOREGROUND_PRIORITY,
            )
            logging.info(
                "AIJobManager requested AI-NR job: file=%s job_id=%s status=%s content_key=%s",
                file_path,
                job.job_id,
                getattr(self.status_by_job.get(job.job_id), "value", self.status_by_job.get(job.job_id)),
                content_key,
            )
            return self.status_by_job.get(job.job_id, AIJobStatus.QUEUED), None, content_key, source_signature

    def enqueue_image_job(
        self,
        *,
        kind: str,
        file_path: str,
        image: np.ndarray,
        param_snapshot: dict[str, Any],
        content_key: str,
        source_signature: str,
        replace_queued: bool = True,
        priority: int = FOREGROUND_PRIORITY,
    ) -> AIJob:
        with self._lock:
            self.start()
            target_key = self._target_key(kind, file_path)
            existing_id = self.job_by_target.get(target_key)
            if existing_id is not None:
                existing_status = self.status_by_job.get(existing_id)
                existing_job = self.jobs.get(existing_id)
                if existing_job is not None and existing_job.content_key == content_key:
                    if existing_status is not None:
                        self._set_status(existing_job, existing_status)
                    return existing_job
                if replace_queued and existing_status == AIJobStatus.QUEUED:
                    self._cancel_job(existing_id, restart_running=False)
                elif (
                    priority <= FOREGROUND_PRIORITY
                    and existing_status == AIJobStatus.RUNNING
                    and existing_job is not None
                    and str(existing_job.content_key).startswith("pending:")
                ):
                    logging.info("AIJobManager preempting background AI-NR for foreground request: %s", file_path)
                    self._cancel_job(existing_id, restart_running=True)

            job = AIJob(
                job_id=self._next_job_id(),
                kind=kind,
                file_path=file_path,
                content_key=content_key,
                source_signature=source_signature,
                param_snapshot=dict(param_snapshot or {}),
            )
            self.jobs[job.job_id] = job
            self.job_by_target[target_key] = job.job_id
            self._set_status(job, AIJobStatus.QUEUED)
            self.priority_by_job[job.job_id] = int(priority)

            image32 = np.ascontiguousarray(image, dtype=np.float32)
            shm = shared_memory.SharedMemory(create=True, size=image32.nbytes)
            arr = np.ndarray(image32.shape, dtype=image32.dtype, buffer=shm.buf)
            arr[:] = image32
            self.active_input_shms[job.job_id] = shm
            self.active_started_at[job.job_id] = time.monotonic()
            payload = {
                "job_id": job.job_id,
                "kind": job.kind,
                "shm_name": shm.name,
                "shape": image32.shape,
                "dtype": str(image32.dtype),
                "content_key": content_key,
                "source_signature": source_signature,
                "priority": int(priority),
                "queued_at": time.monotonic(),
            }
            if self.thread_mode:
                cancel_event = threading.Event()
                self.cancel_events[job.job_id] = cancel_event
                payload["cancel_event"] = cancel_event
            self.pending_payloads[job.job_id] = payload
            if priority <= FOREGROUND_PRIORITY:
                self._preempt_running_background_for_foreground()
            self._dispatch_next_payload()
            self._prune_finished_jobs()
            return job

    def enqueue_ai_noise_file(self, file_path: str, param_snapshot: dict[str, Any]) -> AIJob:
        with self._lock:
            self.start()
            target_key = self._target_key(AI_NOISE_KIND, file_path)
            existing_id = self.job_by_target.get(target_key)
            if existing_id is not None:
                existing_status = self.status_by_job.get(existing_id)
                existing_job = self.jobs.get(existing_id)
                if existing_status in (AIJobStatus.QUEUED, AIJobStatus.RUNNING) and existing_job is not None:
                    self._set_status(existing_job, existing_status)
                    return existing_job

            job_id = self._next_job_id()
            job = AIJob(
                job_id=job_id,
                kind=AI_NOISE_KIND,
                file_path=file_path,
                content_key=f"pending:{job_id}",
                source_signature=f"pending:{job_id}",
                param_snapshot=dict(param_snapshot or {}),
            )
            self.jobs[job.job_id] = job
            self.job_by_target[target_key] = job.job_id
            self._set_status(job, AIJobStatus.QUEUED)
            self.priority_by_job[job.job_id] = BACKGROUND_PRIORITY
            payload = {
                "job_id": job.job_id,
                "kind": job.kind,
                "file_path": file_path,
                "param_snapshot": dict(param_snapshot or {}),
                "priority": BACKGROUND_PRIORITY,
                "queued_at": time.monotonic(),
            }
            if self.thread_mode:
                cancel_event = threading.Event()
                self.cancel_events[job.job_id] = cancel_event
                payload["cancel_event"] = cancel_event
            self.pending_payloads[job.job_id] = payload
            self._dispatch_next_payload()
            self._prune_finished_jobs()
            return job

    def _preempt_running_background_for_foreground(self):
        job_id = self.dispatched_job_id
        if job_id is None:
            return
        if self.status_by_job.get(job_id) not in (AIJobStatus.QUEUED, AIJobStatus.RUNNING):
            return
        if self.priority_by_job.get(job_id, BACKGROUND_PRIORITY) <= FOREGROUND_PRIORITY:
            return
        job = self.jobs.get(job_id)
        if job is None:
            return
        logging.info("AIJobManager preempting running background AI job for foreground request: %s", job.file_path)
        should_requeue = str(job.content_key).startswith("pending:")
        self._cancel_job(job_id, restart_running=True)
        if should_requeue:
            self._requeue_background_file_job(job)

    def _requeue_background_file_job(self, job: AIJob):
        new_job = AIJob(
            job_id=self._next_job_id(),
            kind=job.kind,
            file_path=job.file_path,
            content_key=f"pending:{self._job_counter}",
            source_signature=f"pending:{self._job_counter}",
            param_snapshot=dict(job.param_snapshot or {}),
        )
        self.jobs[new_job.job_id] = new_job
        self.job_by_target[self._target_key(new_job.kind, new_job.file_path)] = new_job.job_id
        self.priority_by_job[new_job.job_id] = BACKGROUND_PRIORITY
        self._set_status(new_job, AIJobStatus.QUEUED)
        payload = {
            "job_id": new_job.job_id,
            "kind": new_job.kind,
            "file_path": new_job.file_path,
            "param_snapshot": dict(new_job.param_snapshot or {}),
            "priority": BACKGROUND_PRIORITY,
            "queued_at": time.monotonic(),
        }
        if self.thread_mode:
            cancel_event = threading.Event()
            self.cancel_events[new_job.job_id] = cancel_event
            payload["cancel_event"] = cancel_event
        self.pending_payloads[new_job.job_id] = payload
        self._dispatch_next_payload()

    def _payload_sort_key(self, payload: dict[str, Any]):
        return (
            int(payload.get("priority", BACKGROUND_PRIORITY)),
            float(payload.get("queued_at", 0.0) or 0.0),
            int(payload.get("job_id", 0)),
        )

    def _dispatch_next_payload(self):
        if self.dispatched_job_id is not None:
            return
        if not self.pending_payloads:
            return
        payload = min(self.pending_payloads.values(), key=self._payload_sort_key)
        self.dispatched_job_id = int(payload["job_id"])
        self.input_queue.put(payload)

    def cancel_path(self, file_path: str, kind: str = AI_NOISE_KIND):
        with self._lock:
            job_id = self.job_by_target.get(self._target_key(kind, file_path))
            if job_id is not None:
                self._cancel_job(job_id, restart_running=True)

    def _cancel_job(self, job_id: int, *, restart_running: bool):
        job = self.jobs.get(job_id)
        if job is None:
            return
        status = self.status_by_job.get(job_id)
        self._set_status(job, AIJobStatus.CANCELLED)
        self.job_by_target.pop(self._target_key(job.kind, job.file_path), None)
        self.pending_payloads.pop(job_id, None)
        cancel_event = self.cancel_events.pop(job_id, None)
        if cancel_event is not None:
            try:
                cancel_event.set()
            except Exception:
                logging.exception("failed to signal AI job cancel event")
        if status == AIJobStatus.QUEUED:
            self._rebuild_input_queue_without(job_id)
            if self.dispatched_job_id == job_id:
                self.dispatched_job_id = None
        self._cleanup_input_shm(job_id)
        if restart_running and status == AIJobStatus.RUNNING:
            if self.thread_mode:
                if self.dispatched_job_id == job_id:
                    self.dispatched_job_id = None
                logging.info("AI job running task cancelled in thread mode; stale result will be discarded: %s", job.file_path)
                self._dispatch_next_payload()
            else:
                if self.dispatched_job_id == job_id:
                    self.dispatched_job_id = None
                restarted = self._restart_worker(count_restart=False, mark_pending_on_limit=False)
                if not restarted:
                    self._dispatch_next_payload()
        else:
            self._dispatch_next_payload()

    def _rebuild_input_queue_without(self, cancelled_job_id: int):
        kept = []
        while True:
            try:
                payload = self.input_queue.get_nowait()
            except Empty:
                break
            if int(payload.get("job_id", -1)) != cancelled_job_id:
                kept.append(payload)
        for payload in kept:
            self.input_queue.put(payload)

    def poll_results(self) -> list[AIJobResult]:
        completed_reads: list[tuple[int, AIJob, dict[str, Any]]] = []
        results: list[AIJobResult] = []
        with self._lock:
            self._mark_dead_worker_jobs_error()
            while True:
                try:
                    res = self.result_queue.get_nowait()
                except Empty:
                    break

                job_id = int(res.get("job_id"))
                job = self.jobs.get(job_id)
                self._cleanup_input_shm(job_id)
                if job is None:
                    self._discard_result_shm(res)
                    continue
                if self.status_by_job.get(job_id) == AIJobStatus.CANCELLED:
                    self._discard_result_shm(res)
                    self.pending_payloads.pop(job_id, None)
                    if self.dispatched_job_id == job_id:
                        self.dispatched_job_id = None
                    self._dispatch_next_payload()
                    continue

                raw_status = res.get("status", AIJobStatus.ERROR.value)
                if raw_status == "progress":
                    self._set_progress(job, int(res.get("done", 0) or 0), int(res.get("total", 0) or 0))
                    continue

                status = AIJobStatus(raw_status)
                if status == AIJobStatus.RUNNING:
                    self.pending_payloads.pop(job_id, None)
                    self._set_status(job, AIJobStatus.RUNNING)
                    continue
                if status == AIJobStatus.COMPLETE:
                    if res.get("content_key") and res.get("source_signature"):
                        job = replace(
                            job,
                            content_key=res["content_key"],
                            source_signature=res["source_signature"],
                        )
                        self.jobs[job_id] = job
                    completed_reads.append((job_id, job, res))
                elif status == AIJobStatus.STALE:
                    self.pending_payloads.pop(job_id, None)
                    self._set_status(job, AIJobStatus.STALE)
                    self.job_by_target.pop(self._target_key(job.kind, job.file_path), None)
                    if self.dispatched_job_id == job_id:
                        self.dispatched_job_id = None
                    results.append(AIJobResult(job=job, status=AIJobStatus.STALE, error=res.get("error")))
                else:
                    self.pending_payloads.pop(job_id, None)
                    self._set_status(job, AIJobStatus.ERROR)
                    if self.dispatched_job_id == job_id:
                        self.dispatched_job_id = None
                    results.append(AIJobResult(job=job, status=AIJobStatus.ERROR, error=res.get("error")))
                self._dispatch_next_payload()
                self._prune_finished_jobs()
            if not completed_reads:
                self._dispatch_next_payload()
                return results

        for job_id, job, res in completed_reads:
            try:
                raw = self._read_result_shm(res["shm_name"], res["shape"], res["dtype"])
            except Exception as exc:
                logging.exception("failed to read AI job result")
                with self._lock:
                    if self.status_by_job.get(job_id) != AIJobStatus.CANCELLED:
                        self._set_status(job, AIJobStatus.ERROR)
                        if self.dispatched_job_id == job_id:
                            self.dispatched_job_id = None
                        results.append(AIJobResult(job=job, status=AIJobStatus.ERROR, error=str(exc)))
                        self._dispatch_next_payload()
                        self._prune_finished_jobs()
                continue

            with self._lock:
                if self.status_by_job.get(job_id) == AIJobStatus.CANCELLED:
                    if self.dispatched_job_id == job_id:
                        self.dispatched_job_id = None
                    self._dispatch_next_payload()
                    continue
                self._store_completed_result(job, raw)
                self.pending_payloads.pop(job_id, None)
                self._set_status(job, AIJobStatus.COMPLETE)
                self.job_by_target.pop(self._target_key(job.kind, job.file_path), None)
                if self.dispatched_job_id == job_id:
                    self.dispatched_job_id = None
                self.worker_restart_count = 0
                results.append(AIJobResult(job=job, status=AIJobStatus.COMPLETE, result=raw))
                self._dispatch_next_payload()
                self._prune_finished_jobs()

        with self._lock:
            self._dispatch_next_payload()
        return results

    def _worker_is_alive(self) -> bool:
        proc = self.process
        return proc is not None and proc.is_alive()

    def _mark_dead_worker_jobs_error(self):
        if self.process is None or self._worker_is_alive():
            return
        if self.worker_restart_count < MAX_WORKER_RESTARTS:
            logging.error("AI job worker exited while jobs were pending; restarting worker.")
            self._prepare_recoverable_dead_worker_jobs()
            self._restart_worker()
            return
        self._mark_pending_jobs_error("AI job worker exited while job was pending")
        self._dispatch_next_payload()

    def _prepare_recoverable_dead_worker_jobs(self):
        for job_id, status in list(self.status_by_job.items()):
            if status not in (AIJobStatus.QUEUED, AIJobStatus.RUNNING):
                continue
            if job_id in self.pending_payloads:
                continue
            job = self.jobs.get(job_id)
            if job is None:
                continue
            if str(job.content_key).startswith("pending:"):
                self.pending_payloads[job_id] = {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "file_path": job.file_path,
                    "param_snapshot": dict(job.param_snapshot or {}),
                    "priority": BACKGROUND_PRIORITY,
                    "queued_at": time.monotonic(),
                }
                self._set_status(job, AIJobStatus.QUEUED)
                continue
            logging.error("AI job cannot be recovered after worker exit: %s", job.file_path)
            self._set_status(job, AIJobStatus.ERROR)
            self.job_by_target.pop(self._target_key(job.kind, job.file_path), None)
            if self.dispatched_job_id == job_id:
                self.dispatched_job_id = None

    def _mark_pending_jobs_error(self, message: str):
        for job_id, status in list(self.status_by_job.items()):
            if status not in (AIJobStatus.QUEUED, AIJobStatus.RUNNING):
                continue
            job = self.jobs.get(job_id)
            if job is None:
                continue
            logging.error("%s: %s", message, job.file_path)
            self.pending_payloads.pop(job_id, None)
            self._cleanup_input_shm(job_id)
            self._set_status(job, AIJobStatus.ERROR)
            self.job_by_target.pop(self._target_key(job.kind, job.file_path), None)
            if self.dispatched_job_id == job_id:
                self.dispatched_job_id = None

    def _read_result_shm(self, shm_name, shape, dtype_str):
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
            return arr.copy()
        finally:
            shm.close()
            shm.unlink()

    def _discard_result_shm(self, res: dict[str, Any]) -> None:
        shm_name = res.get("shm_name")
        if not shm_name:
            return
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            logging.exception("failed to discard AI job result shm")

    def _drain_result_queue_discard_shm(self) -> None:
        while True:
            try:
                res = self.result_queue.get_nowait()
            except Empty:
                return
            self._discard_result_shm(res)

    def _cleanup_input_shm(self, job_id: int):
        shm = self.active_input_shms.pop(job_id, None)
        self.active_started_at.pop(job_id, None)
        if shm is None:
            return
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            logging.exception("failed to cleanup AI job input shm")

    def _cleanup_all_input_shms(self):
        for job_id in list(self.active_input_shms):
            self._cleanup_input_shm(job_id)

    def _prune_finished_jobs(self):
        finished = [
            (job_id, self.jobs.get(job_id))
            for job_id, status in self.status_by_job.items()
            if status not in (AIJobStatus.QUEUED, AIJobStatus.RUNNING)
        ]
        if len(finished) <= FINISHED_JOB_RETENTION:
            return
        finished.sort(key=lambda item: getattr(item[1], "created_at", 0.0) if item[1] is not None else 0.0)
        for job_id, job in finished[: len(finished) - FINISHED_JOB_RETENTION]:
            if job is not None:
                self.job_by_target.pop(self._target_key(job.kind, job.file_path), None)
                self._pop_completed_result(self._result_key(job))
            self.jobs.pop(job_id, None)
            self.status_by_job.pop(job_id, None)
            self.pending_payloads.pop(job_id, None)
            self.cancel_events.pop(job_id, None)
            self.priority_by_job.pop(job_id, None)
            self._cleanup_input_shm(job_id)
