"""
PyInstaller ランタイムフック（macOS）。

同梱 conda の lib/ を DYLD_LIBRARY_PATH に通し、gettext/glib を先に RTLD_GLOBAL 読み込みする。
（ビルド後に lensfunpy の重複 dylib を削除する想定）
"""
from __future__ import annotations

import ctypes
import os
import sys

if getattr(sys, "frozen", False) and sys.platform == "darwin":
    _base = getattr(sys, "_MEIPASS", "") or ""
    _lib = os.path.join(_base, "lib")
    if os.path.isdir(_lib):
        os.environ["DYLD_LIBRARY_PATH"] = _lib + os.pathsep + os.environ.get("DYLD_LIBRARY_PATH", "")

    for name in (
        "libintl.8.dylib",
        "libglib-2.0.0.dylib",
        "libgmodule-2.0.0.dylib",
        "libgobject-2.0.0.dylib",
        "libgio-2.0.0.dylib",
    ):
        p = os.path.join(_lib, name)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
