"""Shared helpers for small backend-adapter decisions."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Callable, Iterable

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


METAL_ENABLED_VALUES = frozenset({"", "auto", "metal"})
METAL_UNAVAILABLE_DETAIL = "Metal backend is built, but no Metal device is available"


class BackendSelector:
    """Parameterized backend-selection and status logic shared by the adapters.

    The selector reads ``_metal_backend`` / ``_cpu_backend`` (and the matching
    ``_METAL_IMPORT_ERROR`` / ``_CPU_IMPORT_ERROR``) from the owning adapter's
    module namespace at call time, so tests that monkeypatch those module
    globals keep working. Adapters keep thin module-level wrappers around the
    selector methods so their public/private call surface stays unchanged.
    """

    def __init__(
        self,
        effect: str,
        namespace: dict,
        *,
        env: str | None = None,
        metal_strict_env: str | None = None,
        native_strict_env: str | None = None,
        metal_name: str | None = None,
        cpu_name: str | None = None,
        reference_name: str = "",
        metal_enabled_values: Iterable[str] | None = METAL_ENABLED_VALUES,
        metal_disabled_values: Iterable[str] = frozenset(),
        cpu_disabled_values: Iterable[str] = REFERENCE_BACKEND_VALUES,
        metal_forced_values: Iterable[str] = frozenset(),
        available_requires_device: bool = False,
        reference_requested_detail: bool = False,
        fallback_import_detail: bool = True,
        device_available: Callable[[], bool] | None = None,
    ) -> None:
        self.effect = effect
        self._namespace = namespace
        self._env = env
        self._metal_strict_env = metal_strict_env
        self._native_strict_env = native_strict_env
        self.metal_name = metal_name
        self.cpu_name = cpu_name
        self.reference_name = reference_name
        # None means "any value that is not explicitly disabled enables Metal".
        self._metal_enabled_values = None if metal_enabled_values is None else frozenset(metal_enabled_values)
        self._metal_disabled_values = frozenset(metal_disabled_values)
        self._cpu_disabled_values = frozenset(cpu_disabled_values)
        self._metal_forced_values = frozenset(metal_forced_values)
        self._available_requires_device = available_requires_device
        self._reference_requested_detail = reference_requested_detail
        self._fallback_import_detail = fallback_import_detail
        self._device_available = device_available

    # -- live adapter globals ------------------------------------------------

    def _metal(self):
        return self._namespace.get("_metal_backend")

    def _cpu(self):
        return self._namespace.get("_cpu_backend")

    def _metal_error(self):
        return self._namespace.get("_METAL_IMPORT_ERROR")

    def _cpu_error(self):
        return self._namespace.get("_CPU_IMPORT_ERROR")

    # -- environment decisions ----------------------------------------------

    def preference(self) -> str:
        return backend_preference(self._env) if self._env else ""

    def metal_strict(self) -> bool:
        return strict_enabled(self._metal_strict_env)

    def native_strict(self) -> bool:
        return strict_enabled(self._native_strict_env)

    def metal_enabled(self) -> bool:
        value = self.preference()
        if value in self._metal_disabled_values:
            return False
        if self._metal_enabled_values is None:
            return True
        return value in self._metal_enabled_values

    def metal_device_available(self) -> bool:
        if self._device_available is not None:
            return self._device_available()
        metal = self._metal()
        if metal is None:
            return False
        try:
            return bool(metal.metal_available())
        except Exception:
            return False

    def metal_ready(self) -> bool:
        return self._metal() is not None and self.metal_enabled() and self.metal_device_available()

    def native_enabled(self) -> bool:
        return native_backend_enabled(self._cpu(), self.preference(), disabled_values=self._cpu_disabled_values)

    def native_available(self) -> bool:
        metal = self._metal()
        if metal is not None and (not self._available_requires_device or self.metal_device_available()):
            return True
        return self._cpu() is not None

    # -- status --------------------------------------------------------------

    def status(self) -> BackendStatus:
        metal = self._metal()
        if metal is not None and self.metal_enabled() and self.metal_device_available():
            return BackendStatus(self.effect, self.metal_name, True)
        if self._metal_forced_values and self.preference() in self._metal_forced_values:
            if metal is not None:
                detail = METAL_UNAVAILABLE_DETAIL
            else:
                detail = import_error_detail(self._metal_error())
            return BackendStatus(self.effect, self.reference_name, False, detail)
        if self.cpu_name is not None:
            cpu = self._cpu()
            if cpu is not None and self.native_enabled():
                return BackendStatus(self.effect, self.cpu_name, True)
            if cpu is not None:
                return BackendStatus(
                    self.effect,
                    self.reference_name,
                    False,
                    f"cpu backend available; {self._env} requested reference",
                )
            return BackendStatus(self.effect, self.reference_name, False, import_error_detail(self._cpu_error()))
        if self._reference_requested_detail and metal is not None:
            return BackendStatus(
                self.effect,
                self.reference_name,
                False,
                f"Metal backend available; {self._env} requested reference",
            )
        if self._fallback_import_detail:
            return BackendStatus(self.effect, self.reference_name, False, import_error_detail(self._metal_error()))
        return BackendStatus(self.effect, self.reference_name, False)
