
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


def create_processing_dialog():
    global __dialog
    __dialog = _create_dialog_impl()


def show_processing_dialog(dt=0):
    global __dialog
    if __dialog:
        __dialog.show()


def update_processing_dialog(dt=0):
    global __dialog
    if __dialog:
        __dialog.update()


def hide_processing_dialog(dt=0):
    global __dialog
    if __dialog:
        __dialog.hide()


def set_processing_text(text):
    global __dialog
    if __dialog:
        __dialog.set_text(text)


def wait_threading(process, *args, **kwargs):
    show_processing_dialog()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process, *args, **kwargs)
        while not future.done():
            update_processing_dialog()
            time.sleep(0.04)
        result = future.result()
    hide_processing_dialog()
    return result


def wait_prosessing(process, *args, **kwargs):
    if threading.current_thread() is threading.main_thread():
        return wait_threading(process, *args, **kwargs)

    KVClock.schedule_once(show_processing_dialog)
    event = KVClock.schedule_interval(update_processing_dialog, 0.04)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process, *args, **kwargs)
        while not future.done():
            time.sleep(0.5)
        result = future.result()
    KVClock.unschedule(event)
    KVClock.schedule_once(hide_processing_dialog)
    return result
