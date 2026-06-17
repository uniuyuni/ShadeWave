from kivy.clock import Clock as KVClock
from kivy.core.window import Window as KVWindow

from utils import kvutils


def apply_ref_scaling(dialog, center=True):
    kvutils.traverse_widget(dialog)
    content = getattr(dialog, "content", None)
    if content is not None:
        kvutils.traverse_widget(content)
        if hasattr(content, "do_layout"):
            content.do_layout()
    if hasattr(dialog, "do_layout"):
        dialog.do_layout()
    if center:
        dialog.center = KVWindow.center


def install_ref_scaling(dialog, center=True, poll_interval=0.25, on_rescale=None):
    state = {
        "last": None,
        "event": None,
    }

    def _scale(force=False):
        current = kvutils.get_window_state()
        changed = force or current != state["last"]
        if not changed:
            return
        state["last"] = current
        if on_rescale is not None:
            on_rescale()
        apply_ref_scaling(dialog, center=center)

    def _on_open(*_args):
        KVClock.schedule_once(lambda _dt: _scale(force=True), 0)
        if state["event"] is None:
            state["event"] = KVClock.schedule_interval(
                lambda _dt: _scale(force=False), poll_interval
            )

    def _on_dismiss(*_args):
        if state["event"] is not None:
            state["event"].cancel()
            state["event"] = None

    dialog.bind(on_open=_on_open)
    dialog.bind(on_dismiss=_on_dismiss)
    KVClock.schedule_once(lambda _dt: _scale(force=True), 0)
    if state["event"] is None:
        state["event"] = KVClock.schedule_interval(
            lambda _dt: _scale(force=False), poll_interval
        )
    return dialog
