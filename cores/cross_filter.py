"""Compatibility shim for the CrossFilter backend."""

from effect_backends.cross_filter_adapter import apply_cross_filter, backend_status, native_available

__all__ = ["apply_cross_filter", "backend_status", "native_available"]
