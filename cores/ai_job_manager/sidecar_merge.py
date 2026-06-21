from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from typing import Callable

import numpy as np

import threads
from .ai_noise import merge_ai_noise_result_into_pmck
from .types import AIJob


@dataclass(frozen=True)
class AISidecarMergeResult:
    job: AIJob
    result: np.ndarray | None
    merged: bool = False
    error: str | None = None


class AISidecarMergeQueue:
    """Serializes large .pmck merges away from the UI polling path."""

    def __init__(self, *, merge_ai_noise_fn: Callable[..., bool] | None = None, max_pending: int | None = None):
        self._lock = threads.ai_sidecar_merge_lock
        self._merge_ai_noise_fn = merge_ai_noise_fn or merge_ai_noise_result_into_pmck
        pending_limit = max_pending if max_pending is not None else os.getenv("PLATYPUS_AI_SIDECAR_MERGE_MAX_PENDING", "2")
        try:
            self.max_pending = max(0, int(pending_limit))
        except (TypeError, ValueError):
            logging.warning("Invalid PLATYPUS_AI_SIDECAR_MERGE_MAX_PENDING=%r; using 2", pending_limit)
            self.max_pending = 2
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AISidecarMerge")
        self._pending: dict[Future, tuple[AIJob, np.ndarray]] = {}

    def submit_ai_noise_result(self, job: AIJob, raw_result: np.ndarray) -> Future | None:
        with self._lock:
            if self.max_pending > 0 and len(self._pending) >= self.max_pending:
                logging.warning(
                    "AI sidecar merge queue is full; dropping pmck merge: file=%s pending=%s max=%s",
                    job.file_path,
                    len(self._pending),
                    self.max_pending,
                )
                return None
        future = self._executor.submit(
            self._merge_ai_noise_fn,
            job.file_path,
            raw_result,
            content_key=job.content_key,
            source_signature=job.source_signature,
        )
        with self._lock:
            self._pending[future] = (job, raw_result)
            pending_count = len(self._pending)
        logging.info(
            "AI sidecar merge queued: file=%s content_key=%s pending=%s max=%s",
            job.file_path,
            job.content_key,
            pending_count,
            self.max_pending,
        )
        return future

    def poll_results(self) -> list[AISidecarMergeResult]:
        results: list[AISidecarMergeResult] = []
        with self._lock:
            ready = [
                (future, self._pending.pop(future))
                for future in list(self._pending)
                if future.done()
            ]
        for future, (job, raw) in ready:
            try:
                merged = bool(future.result())
                logging.info("AI sidecar merge finished: file=%s content_key=%s merged=%s", job.file_path, job.content_key, merged)
                results.append(AISidecarMergeResult(job=job, result=raw, merged=merged))
            except Exception as exc:
                logging.error("AI sidecar merge errored: file=%s content_key=%s error=%s", job.file_path, job.content_key, exc)
                results.append(AISidecarMergeResult(job=job, result=raw, error=str(exc)))
        return results

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def has_pending_job(self, job: AIJob) -> bool:
        with self._lock:
            return any(pending_job == job for pending_job, _raw in self._pending.values())

    def shutdown(self) -> None:
        with self._lock:
            self._pending.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
