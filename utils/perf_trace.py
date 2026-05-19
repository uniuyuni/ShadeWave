"""画像選択 → 初回表示までの主要イベント時刻を JSON Lines で記録する軽量トレーサ。

環境変数 PLATYPUS_PERF_TRACE=1 のときだけ動作する。それ以外では全 API が早期 return する。
ログ出力先: $PIXI_PROJECT_ROOT/.profiles/perf_trace-<起動時刻>.jsonl

タイムラインの流れ（メインプロセス完結）:
    select_start(file_path)
        → event("preview.decode_done"), event("preview.gamma_done") 等
        → event("fcs.callback", stage=...)
        → event("draw.pipeline_done"), event("draw.blit_done")
    flush() で 1 トレースを 1 行 JSON にして書き出す。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone


_ENABLED = os.getenv("PLATYPUS_PERF_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"}
_LOG_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
_LOCK = threading.Lock()
# トレースはファイル選択ごとに 1 本。並行選択は無いので単一の current で良い。
_current: dict | None = None


def enabled() -> bool:
    return _ENABLED


def _log_path() -> str:
    root = os.getenv("PIXI_PROJECT_ROOT") or os.getcwd()
    out_dir = os.path.join(root, ".profiles")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"perf_trace-{_LOG_STAMP}.jsonl")


def select_start(file_path: str | None) -> None:
    """ユーザのファイル選択開始時刻を記録して新しいトレースを始める。"""
    if not _ENABLED:
        return
    global _current
    with _LOCK:
        _current = {
            "file_path": file_path,
            "t0_perf": time.perf_counter(),
            "t0_wall_ms": int(time.time() * 1000),
            "t0_iso": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "events": [],
        }


def event(name: str, **extra) -> None:
    """イベントを記録。t0 からのミリ秒オフセットと追加情報を持つ。"""
    if not _ENABLED:
        return
    with _LOCK:
        if _current is None:
            return
        offset_ms = (time.perf_counter() - _current["t0_perf"]) * 1000.0
        entry = {"name": name, "ms": round(offset_ms, 3)}
        if extra:
            # JSON シリアライズ可能な値だけ受け入れる（万一の例外を握り潰す）
            for k, v in extra.items():
                try:
                    json.dumps(v)
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = repr(v)
        _current["events"].append(entry)


def flush(reason: str = "") -> None:
    """現在のトレースを 1 行 JSON にして書き出し、リセットする。"""
    if not _ENABLED:
        return
    global _current
    with _LOCK:
        if _current is None:
            return
        total_ms = (time.perf_counter() - _current["t0_perf"]) * 1000.0
        payload = {
            "file_path": _current["file_path"],
            "t0_wall_ms": _current["t0_wall_ms"],
            "t0_iso": _current["t0_iso"],
            "total_ms": round(total_ms, 3),
            "reason": reason,
            "events": _current["events"],
        }
        _current = None
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as e:
        logging.warning("perf_trace flush failed: %s", e)
