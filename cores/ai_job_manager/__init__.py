"""Background AI job management."""

from .ai_noise import (
    AI_NOISE_KIND,
    ai_noise_content_key,
    ai_noise_enabled,
    ai_noise_source_signature,
    ai_noise_valid_content_keys,
    ai_noise_valid_source_signatures,
    current_param_accepts_ai_noise_result,
    merge_ai_noise_result_into_param,
    merge_ai_noise_result_into_pmck,
)
from .manager import AIJobManager
from .sidecar_merge import AISidecarMergeQueue, AISidecarMergeResult
from .types import AIJob, AIJobResult, AIJobStatus

__all__ = [
    "AIJob",
    "AIJobManager",
    "AIJobResult",
    "AIJobStatus",
    "AISidecarMergeQueue",
    "AISidecarMergeResult",
    "AI_NOISE_KIND",
    "ai_noise_content_key",
    "ai_noise_enabled",
    "ai_noise_source_signature",
    "ai_noise_valid_content_keys",
    "ai_noise_valid_source_signatures",
    "current_param_accepts_ai_noise_result",
    "merge_ai_noise_result_into_param",
    "merge_ai_noise_result_into_pmck",
]
