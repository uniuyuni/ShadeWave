"""Shared helpers for small backend-adapter decisions."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType

from utils.envutils import env_flag


REFERENCE_BACKEND_VALUES = frozenset({"reference", "python", "off", "0", "false", "no"})


@dataclass(frozen=True)
class BackendStatus:
    effect: str
    backend: str
    native: bool
    detail: str = ""


def backend_preference(env_name: str, default: str = "") -> str:
    from os import environ

    return str(environ.get(env_name, default)).strip().lower()


def strict_enabled(env_name: str) -> bool:
    return env_flag(env_name)


def optional_backend(package: str, module_name: str) -> tuple[ModuleType | None, Exception | None]:
    try:
        return importlib.import_module(f"{package}.{module_name}"), None
    except Exception as exc:  # pragma: no cover - depends on local build state.
        return None, exc


def import_error_detail(exc: Exception | None) -> str:
    return "" if exc is None else str(exc)


def native_backend_enabled(native_backend, preference: str, *, disabled_values=REFERENCE_BACKEND_VALUES) -> bool:
    return native_backend is not None and preference not in disabled_values
