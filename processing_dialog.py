
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from kivy.clock import Clock as KVClock

import define


def _asset_spinner_path():
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", "spinner.gif")


class _NullProcessingDialog:
    """PyObjC なし・非 macOS などのスタブ"""

    def show(self):
        pass

    def update(self):
        pass

    def hide(self):
        pass

    def set_text(self, text):
        pass


def _create_dialog_impl():
    if sys.platform == "darwin":
        try:
            import macos

            if macos.HAS_PYOBJC:
                return macos.MacOSProcessingOverlay(
                    _asset_spinner_path(),
                    define.APPNAME,
                )
        except Exception as e:
            print(f"processing_dialog: ネイティブ HUD の初期化に失敗しました: {e}")
    return _NullProcessingDialog()


__dialog = None
# Reentrant "is a processing dialog currently showing" counter. The native macOS
# HUD (MacOSProcessingOverlay) lives outside Kivy's widget tree, and pumping the
# AppKit run loop to keep it animated (see macos.py:_pump_runloop) proved
# unreliable for reliably swallowing clicks on the main window. So input is
# instead blocked at the Kivy level: MainWidget.on_touch_*/on_key_* check
# is_active() and swallow all input while this is > 0.
__active_count = 0


def create_processing_dialog():
    global __dialog
    __dialog = _create_dialog_impl()


def is_active():
    return __active_count > 0


def show_processing_dialog(dt=0):
    global __dialog, __active_count
    __active_count += 1
    if __dialog:
        __dialog.show()


def update_processing_dialog(dt=0):
    global __dialog
    if __dialog:
        __dialog.update()


def hide_processing_dialog(dt=0):
    global __dialog, __active_count
    __active_count = max(0, __active_count - 1)
    if __dialog:
        __dialog.hide()


def set_processing_text(text):
    global __dialog
    if __dialog:
        __dialog.set_text(text)


def wait_threading(process, *args, **kwargs):
    show_processing_dialog()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process, *args, **kwargs)
            while not future.done():
                update_processing_dialog()
                time.sleep(0.04)
            return future.result()
    finally:
        hide_processing_dialog()


def wait_processing(process, *args, **kwargs):
    if threading.current_thread() is threading.main_thread():
        return wait_threading(process, *args, **kwargs)

    KVClock.schedule_once(show_processing_dialog)
    event = KVClock.schedule_interval(update_processing_dialog, 0.04)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process, *args, **kwargs)
            while not future.done():
                time.sleep(0.5)
            return future.result()
    finally:
        KVClock.unschedule(event)
        KVClock.schedule_once(hide_processing_dialog)


def wait_prosessing(process, *args, **kwargs):
    """Backward-compatible alias for the old misspelled API."""
    return wait_processing(process, *args, **kwargs)
