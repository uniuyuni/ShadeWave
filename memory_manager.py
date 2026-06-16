import gc
import logging
import os
import resource
import subprocess
from typing import Any

import numpy as np

try:
    import psutil
except Exception:
    psutil = None


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def debug_enabled() -> bool:
    return _truthy_env("PLATYPUS_MEMORY_DEBUG")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip())
    except ValueError:
        return default


def current_rss_bytes() -> int | None:
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if os.uname().sysname == "Darwin":
        return int(usage)
    return int(usage) * 1024


def available_memory_bytes() -> int | None:
    if psutil is None:
        if os.uname().sysname != "Darwin":
            return None
        try:
            output = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=1.0)
        except Exception:
            return None
        page_size = 16384
        first = output.splitlines()[0] if output else ""
        if "page size of" in first:
            try:
                page_size = int(first.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                page_size = 16384
        pages = 0
        for line in output.splitlines():
            name, _, value = line.partition(":")
            if name.strip() not in {"Pages free", "Pages inactive", "Pages speculative"}:
                continue
            try:
                pages += int(value.strip().rstrip(".").replace(".", ""))
            except ValueError:
                pass
        return pages * page_size if pages else None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _bytes_of(obj: Any, seen: set[int]) -> int:
    if obj is None:
        return 0
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_bytes_of(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(_bytes_of(v, seen) for v in obj)
    return 0


def bytes_of(obj: Any) -> int:
    return _bytes_of(obj, set())


def copy_image_for_cache(image):
    arr = np.asarray(image)
    return np.ascontiguousarray(arr).copy()


def format_bytes(num: int | None) -> str:
    if num is None:
        return "unknown"
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def effect_cache_bytes(effects) -> int:
    total = 0
    seen: set[int] = set()
    for layer in effects or []:
        for effect in getattr(layer, "values", lambda: [])():
            total += _bytes_of(getattr(effect, "diff", None), seen)
            total += _bytes_of(getattr(effect, "_cached_predict", None), seen)
    return total


def processor_cache_bytes(processor) -> int:
    total = 0
    seen: set[int] = set()
    cache = getattr(processor, "cache", None)
    if not isinstance(cache, dict):
        return 0
    for entry in cache.values():
        if isinstance(entry, dict):
            total += _bytes_of(entry.get("result"), seen)
    return total


def clear_effect_intermediate_caches(effects, processor=None, *, reason: str = "memory_pressure") -> dict:
    effect_count = 0
    for layer in effects or []:
        for effect in getattr(layer, "values", lambda: [])():
            try:
                effect.reeffect()
                effect_count += 1
            except Exception:
                try:
                    effect.diff = None
                    effect.hash = None
                    effect_count += 1
                except Exception:
                    logging.exception("memory_manager: failed to clear effect cache")
            if hasattr(effect, "_cached_predict"):
                effect._cached_predict = None
            if hasattr(effect, "_cached_predict_key"):
                effect._cached_predict_key = None

    processor_entries = 0
    if processor is not None:
        clear = getattr(processor, "clear_completed_cache", None)
        if clear is not None:
            try:
                processor_entries = int(clear())
            except Exception:
                logging.exception("memory_manager: failed to clear processor cache")

    gc.collect()
    logging.info(
        "memory_manager cleared effect intermediates reason=%s effects=%d processor_entries=%d",
        reason,
        effect_count,
        processor_entries,
    )
    return {"effects": effect_count, "processor_entries": processor_entries}


def memory_pressure() -> tuple[bool, str]:
    available_min_mb = _env_float("PLATYPUS_MEMORY_AVAILABLE_MIN_MB", 1024.0)
    rss_limit_mb = _env_float("PLATYPUS_MEMORY_RSS_LIMIT_MB", 0.0)

    available = available_memory_bytes()
    if available is not None and available < available_min_mb * 1024 * 1024:
        return True, f"available<{available_min_mb:g}MB"

    rss = current_rss_bytes()
    if rss_limit_mb > 0 and rss is not None and rss > rss_limit_mb * 1024 * 1024:
        return True, f"rss>{rss_limit_mb:g}MB"

    return False, "ok"


def enforce_memory_policy(effects=None, processor=None, *, reason: str = "check") -> dict:
    pressured, pressure_reason = memory_pressure()
    if not pressured:
        return {"cleared": False, "reason": pressure_reason}
    cleared = clear_effect_intermediate_caches(
        effects,
        processor,
        reason=f"{reason}:{pressure_reason}",
    )
    cleared.update({"cleared": True, "reason": pressure_reason})
    return cleared


def build_memory_report(file_path=None, stage=None, cache_system=None, effects=None, processor=None, extra=None) -> dict:
    cache_bytes = 0
    final_display_cache_bytes = 0
    if cache_system is not None:
        cache_bytes = cache_system.cache_memory_bytes()
        final_display_cache_bytes = cache_system.final_display_cache_memory_bytes()
    report = {
        "file_path": file_path,
        "stage": str(stage) if stage is not None else None,
        "rss_bytes": current_rss_bytes(),
        "available_bytes": available_memory_bytes(),
        "fcs_cache_bytes": cache_bytes,
        "final_display_cache_bytes": final_display_cache_bytes,
        "effect_cache_bytes": effect_cache_bytes(effects),
        "processor_cache_bytes": processor_cache_bytes(processor),
    }
    if extra:
        report.update(extra)
    return report


def log_memory_report(label: str, report: dict, *, force: bool = False) -> None:
    if not force and not debug_enabled():
        return
    logging.debug(
        "[MEMORY] %s file=%s stage=%s rss=%s available=%s fcs_cache=%s final_display_cache=%s effect_cache=%s processor_cache=%s extra=%s",
        label,
        report.get("file_path"),
        report.get("stage"),
        format_bytes(report.get("rss_bytes")),
        format_bytes(report.get("available_bytes")),
        format_bytes(report.get("fcs_cache_bytes")),
        format_bytes(report.get("final_display_cache_bytes")),
        format_bytes(report.get("effect_cache_bytes")),
        format_bytes(report.get("processor_cache_bytes")),
        {
            k: v for k, v in report.items()
            if k not in {
                "file_path",
                "stage",
                "rss_bytes",
                "available_bytes",
                "fcs_cache_bytes",
                "final_display_cache_bytes",
                "effect_cache_bytes",
                "processor_cache_bytes",
            }
        },
    )
