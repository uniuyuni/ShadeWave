from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class AIJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"
    STALE = "stale"


@dataclass(frozen=True)
class AIJob:
    job_id: int
    kind: str
    file_path: str
    content_key: str
    source_signature: str
    param_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class AIJobResult:
    job: AIJob
    status: AIJobStatus
    result: Any = None
    error: str | None = None


def viewer_state_for_status(status: AIJobStatus | str | None) -> str | None:
    if status is None:
        return None
    value = status.value if isinstance(status, AIJobStatus) else str(status)
    if value in {AIJobStatus.QUEUED.value, AIJobStatus.RUNNING.value, AIJobStatus.ERROR.value}:
        return value
    return None
