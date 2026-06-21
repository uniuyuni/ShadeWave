from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime as dt
import logging
import os
import shutil
import tempfile
import threading
from typing import Any

import msgpack

import define
import threads

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


PMCK_SUFFIX = ".pmck"
StatToken = tuple[int, int, int]
NO_CHANGE = object()
DELETE = object()
_lock_state = threading.local()


def image_pmck_path(image_path: str | None) -> str:
    return (image_path or "") + PMCK_SUFFIX


def empty_pmck() -> dict[str, Any]:
    return {
        "make": "Platypus",
        "date": dt.now().strftime("%Y/%m/%d"),
        "version": define.VERSION,
        "primary_param": {},
    }


def ensure_primary_param(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = empty_pmck()
    if not isinstance(data.get("primary_param"), dict):
        data["primary_param"] = {}
    return data


def _global_lock_path() -> str:
    return os.path.join(tempfile.gettempdir(), "platypus_pmck_store.lock")


@contextmanager
def _serialized_pmck_io():
    with threads.pmck_store_lock:
        depth = int(getattr(_lock_state, "depth", 0) or 0)
        if depth > 0:
            _lock_state.depth = depth + 1
            try:
                yield
            finally:
                _lock_state.depth = depth
            return

        _lock_state.depth = 1
        lock_fd = None
        try:
            if fcntl is not None:
                lock_fd = os.open(_global_lock_path(), os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            _lock_state.depth = 0


def stat_token_path(pmck_path: str) -> StatToken | None:
    try:
        st = os.stat(pmck_path)
        return (int(st.st_mtime_ns), int(st.st_size), int(getattr(st, "st_ino", 0)))
    except FileNotFoundError:
        return None


def stat_token(image_path: str | None) -> StatToken | None:
    return stat_token_path(image_pmck_path(image_path))


def _read_path_unlocked(pmck_path: str, *, default_empty: bool = False) -> dict[str, Any] | None:
    try:
        with open(pmck_path, "rb") as f:
            data = msgpack.unpackb(f.read(), raw=False)
    except FileNotFoundError:
        return empty_pmck() if default_empty else None
    except Exception as e:
        logging.exception("pmck_store.read_path failed: %s", e)
        return empty_pmck() if default_empty else None
    if not isinstance(data, dict):
        return empty_pmck() if default_empty else None
    return ensure_primary_param(data)


def read_path(pmck_path: str, *, default_empty: bool = False) -> dict[str, Any] | None:
    with _serialized_pmck_io():
        return _read_path_unlocked(pmck_path, default_empty=default_empty)


def read_image(image_path: str | None, *, default_empty: bool = False) -> dict[str, Any] | None:
    return read_path(image_pmck_path(image_path), default_empty=default_empty)


def read_image_with_token(
    image_path: str | None,
    *,
    default_empty: bool = True,
) -> tuple[dict[str, Any] | None, StatToken | None]:
    pmck_path = image_pmck_path(image_path)
    with _serialized_pmck_io():
        token = stat_token_path(pmck_path)
        return _read_path_unlocked(pmck_path, default_empty=default_empty), token


def _write_path_unlocked(pmck_path: str, data: dict[str, Any], *, expected_token: StatToken | None = None) -> bool:
    directory = os.path.dirname(pmck_path) or "."
    if directory:
        os.makedirs(directory, exist_ok=True)
    if expected_token is not None and stat_token_path(pmck_path) != expected_token:
        return False

    fd, tmp_path = tempfile.mkstemp(prefix=".pmck.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(msgpack.packb(data, use_bin_type=True))
        if expected_token is not None and stat_token_path(pmck_path) != expected_token:
            return False
        os.replace(tmp_path, pmck_path)
        tmp_path = None
        return True
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass


def write_path(pmck_path: str, data: dict[str, Any], *, expected_token: StatToken | None = None) -> bool:
    with _serialized_pmck_io():
        return _write_path_unlocked(pmck_path, data, expected_token=expected_token)


def write_image(
    image_path: str | None,
    data: dict[str, Any],
    *,
    expected_token: StatToken | None = None,
) -> bool:
    return write_path(image_pmck_path(image_path), data, expected_token=expected_token)


def _delete_path_unlocked(pmck_path: str) -> bool:
    try:
        os.remove(pmck_path)
        return True
    except FileNotFoundError:
        return False


def delete_path(pmck_path: str) -> bool:
    with _serialized_pmck_io():
        return _delete_path_unlocked(pmck_path)


def delete_image(image_path: str | None) -> bool:
    return delete_path(image_pmck_path(image_path))


def exists_image(image_path: str | None) -> bool:
    with _serialized_pmck_io():
        return os.path.exists(image_pmck_path(image_path))


def update_path(
    pmck_path: str,
    updater,
    *,
    default_empty: bool = True,
) -> bool:
    """
    Serialize a full read-modify-write/delete operation through the .pmck gateway.

    updater receives the current dict (or empty shell) and may return:
    - a dict to write,
    - DELETE to remove the pmck,
    - NO_CHANGE to leave the file untouched,
    - None to write the mutated input dict.
    """
    with _serialized_pmck_io():
        data = _read_path_unlocked(pmck_path, default_empty=default_empty)
        result = updater(data)
        if result is NO_CHANGE:
            return True
        if result is DELETE:
            _delete_path_unlocked(pmck_path)
            return True
        if result is None:
            result = data
        if not isinstance(result, dict):
            logging.error("pmck_store.update_path updater returned non-dict: %r", type(result))
            return False
        return _write_path_unlocked(pmck_path, ensure_primary_param(result))


def update_image(image_path: str | None, updater, *, default_empty: bool = True) -> bool:
    return update_path(image_pmck_path(image_path), updater, default_empty=default_empty)


def copy_path_to_path(src_path: str, dst_path: str) -> bool:
    with _serialized_pmck_io():
        if not os.path.exists(src_path):
            return False
        directory = os.path.dirname(dst_path) or "."
        if directory:
            os.makedirs(directory, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True


def move_path_to_path(src_path: str, dst_path: str) -> bool:
    with _serialized_pmck_io():
        if not os.path.exists(src_path):
            return False
        directory = os.path.dirname(dst_path) or "."
        if directory:
            os.makedirs(directory, exist_ok=True)
        os.replace(src_path, dst_path)
        return True


def swap_paths(path_a: str, path_b: str, tmp_path: str) -> bool:
    with _serialized_pmck_io():
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            return False
        os.replace(path_a, tmp_path)
        os.replace(path_b, path_a)
        os.replace(tmp_path, path_b)
        return True
