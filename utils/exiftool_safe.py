"""
exiftool 呼び出しの堅牢化ヘルパー。

設計の要点:
- 「ファイルが存在しない／壊れている／途中で消えた」はリトライしない（無限に失敗するため）
  → 事前に os.path.isfile で除外、または分割して失敗ファイルだけスキップ。
- 「タイムアウト・一時的 I/O エラー」はリトライ対象
  → 指数バックオフで小回数だけ再試行。
- 結果リストは必ず入力 files と同じ長さ・順序を返す（取れなかったファイルは空 dict）。
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Iterable, Sequence

import exiftool
from exiftool.exceptions import (
    ExifToolException,
    ExifToolExecuteError,
    ExifToolOutputEmptyError,
)


def safe_run_exiftool(
    argv: Sequence[str],
    *,
    timeout: float = 120,
    retries: int = 2,
    backoff: float = 0.3,
) -> subprocess.CompletedProcess:
    """
    subprocess.run のリトライ付きラッパー。

    再試行する例外: TimeoutExpired, OSError（一時的な FD 枯渇など）。
    再試行しない: FileNotFoundError（バイナリ自体がない → 設定ミス、即失敗）。
    returncode != 0 はそのまま返す（呼び出し側で判定する）。
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            raise
        except (subprocess.TimeoutExpired, OSError) as e:
            last_exc = e
            if attempt >= retries:
                break
            time.sleep(backoff * (2 ** attempt))
            logging.warning("safe_run_exiftool: 一時エラー %r。再試行 %d/%d", e, attempt + 1, retries)
    assert last_exc is not None
    raise last_exc


def safe_get_metadata(
    files: Iterable[str],
    *,
    common_args: Sequence[str] | None = None,
) -> list[dict]:
    """
    PyExifTool ExifToolHelper().get_metadata の堅牢版。

    - 存在しないファイルは事前に除外し、その位置には空 dict を入れる。
    - ExifToolExecuteError / ExifToolOutputEmptyError が発生したら chunk を二分して
      再帰し、最終的に 1 件単位で試行する。1 件で失敗したファイルは空 dict。
    - ExifToolHelper 自体の起動失敗（exiftool バイナリなし等）は全件空 dict で返し、
      警告ログのみ出して呼び出し側（スピナー解除等）を止めない。
    - 戻り値は入力 files と同じ長さ・順序。
    """
    file_list = list(files)
    n = len(file_list)
    results: list[dict] = [{} for _ in range(n)]
    if n == 0:
        return results

    alive_indices: list[int] = []
    alive_paths: list[str] = []
    for i, p in enumerate(file_list):
        try:
            if p and os.path.isfile(p):
                alive_indices.append(i)
                alive_paths.append(p)
        except OSError:
            pass
    if not alive_paths:
        return results

    common = list(common_args) if common_args else []

    try:
        with exiftool.ExifToolHelper(common_args=common) as et:
            _collect_metadata(et, alive_paths, alive_indices, results)
    except FileNotFoundError as e:
        logging.warning("safe_get_metadata: exiftool バイナリが見つかりません: %s", e)
    except ExifToolException as e:
        logging.warning("safe_get_metadata: ExifToolHelper 起動失敗: %s", e)
    except Exception as e:
        logging.exception("safe_get_metadata: 予期しない失敗: %s", e)

    return results


def _collect_metadata(
    et: "exiftool.ExifToolHelper",
    paths: list[str],
    indices: list[int],
    results: list[dict],
) -> None:
    """ヘルパー: paths を取り、失敗したら二分して再試行、最終的に 1 件単位。"""
    if not paths:
        return
    try:
        rows = et.get_metadata(paths)
    except (ExifToolExecuteError, ExifToolOutputEmptyError) as e:
        if len(paths) == 1:
            logging.warning("safe_get_metadata: 取得失敗（スキップ）: %s (%s)", paths[0], e)
            return
        mid = len(paths) // 2
        _collect_metadata(et, paths[:mid], indices[:mid], results)
        _collect_metadata(et, paths[mid:], indices[mid:], results)
        return

    if not isinstance(rows, list):
        return
    for row, idx in zip(rows, indices):
        if isinstance(row, dict):
            results[idx] = row
