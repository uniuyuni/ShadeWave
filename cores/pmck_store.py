from __future__ import annotations

from datetime import datetime as dt
import logging
import os
import tempfile
from typing import Any

import msgpack

import define


PMCK_SUFFIX = ".pmck"
StatToken = tuple[int, int, int]


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


def stat_token_path(pmck_path: str) -> StatToken | None:
    try:
        st = os.stat(pmck_path)
        return (int(st.st_mtime_ns), int(st.st_size), int(getattr(st, "st_ino", 0)))
    except FileNotFoundError:
        return None


def stat_token(image_path: str | None) -> StatToken | None:
    return stat_token_path(image_pmck_path(image_path))


def read_path(pmck_path: str, *, default_empty: bool = False) -> dict[str, Any] | None:
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


def read_image(image_path: str | None, *, default_empty: bool = False) -> dict[str, Any] | None:
    return read_path(image_pmck_path(image_path), default_empty=default_empty)


def read_image_with_token(
    image_path: str | None,
    *,
    default_empty: bool = True,
) -> tuple[dict[str, Any] | None, StatToken | None]:
    pmck_path = image_pmck_path(image_path)
    token = stat_token_path(pmck_path)
    return read_path(pmck_path, default_empty=default_empty), token


def write_path(pmck_path: str, data: dict[str, Any], *, expected_token: StatToken | None = None) -> bool:
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


def write_image(
    image_path: str | None,
    data: dict[str, Any],
    *,
    expected_token: StatToken | None = None,
) -> bool:
    return write_path(image_pmck_path(image_path), data, expected_token=expected_token)


def delete_path(pmck_path: str) -> bool:
    try:
        os.remove(pmck_path)
        return True
    except FileNotFoundError:
        return False


def delete_image(image_path: str | None) -> bool:
    return delete_path(image_pmck_path(image_path))


def exists_image(image_path: str | None) -> bool:
    return os.path.exists(image_pmck_path(image_path))
