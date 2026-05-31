"""Shared ref-based metrics for image-geometry overlay controls."""

from kivy.clock import Clock as KVClock

import macos as device
from utils import kvutils


def geom_scaled_width(value):
    return kvutils.dpi_scale_width(value)


def geom_scaled_height(value):
    return kvutils.dpi_scale_height(value)


def geom_scaled(value):
    return geom_scaled_width(value)


def geom_button_size(width_ref=50, height_ref=22):
    return (geom_scaled_width(width_ref), geom_scaled_height(height_ref))


def _set_ref_attr(widget, attr, value):
    if value is not None:
        setattr(widget, attr, value)


def apply_geom_layout_metrics(
        widget,
        width_ref=None,
        height_ref=None,
        spacing_ref=None,
        padding_ref=None):
    _set_ref_attr(widget, "ref_width", width_ref)
    _set_ref_attr(widget, "ref_height", height_ref)
    _set_ref_attr(widget, "ref_spacing", spacing_ref)
    _set_ref_attr(widget, "ref_padding", padding_ref)
    _apply_ref_widget_metrics(widget)
    return widget


def apply_geom_button_metrics(
        button,
        width_ref=50,
        height_ref=22,
        font_ref=11,
        padding_x_ref=8,
        padding_y_ref=2):
    button._geom_button_metrics = (
        width_ref,
        height_ref,
        font_ref,
        padding_x_ref,
        padding_y_ref,
    )
    button.size_hint = (None, None)
    button.ref_width = width_ref
    button.ref_height = height_ref
    _apply_geom_button_metrics_now(button)
    KVClock.schedule_once(lambda _dt: _apply_geom_button_metrics_now(button), 0)
    return button


def apply_geom_ref_metrics(root):
    kvutils.traverse_widget(root)
    for child in kvutils.get_entire_widget_tree(root):
        if hasattr(child, "_geom_button_metrics"):
            _apply_geom_button_metrics_now(child)


def install_geom_ref_scaling(root, poll_interval=0.25):
    event = getattr(root, "_geom_ref_scaling_event", None)
    if event is not None:
        return root

    state = {"last": None, "event": None, "attached": False}

    def _scale(force=False):
        if getattr(root, "parent", None) is not None:
            state["attached"] = True
        elif state["attached"]:
            if state["event"] is not None:
                state["event"].cancel()
                state["event"] = None
                root._geom_ref_scaling_event = None
            return
        else:
            return
        current = _window_state()
        if force or current != state["last"]:
            state["last"] = current
            apply_geom_ref_metrics(root)

    KVClock.schedule_once(lambda _dt: _scale(force=True), 0)
    state["event"] = KVClock.schedule_interval(
        lambda _dt: _scale(force=False),
        poll_interval,
    )
    root._geom_ref_scaling_event = state["event"]
    return root


def _apply_ref_widget_metrics(widget):
    if hasattr(widget, "ref_width") and widget.ref_width:
        widget.width = geom_scaled_width(widget.ref_width)
    if hasattr(widget, "ref_height") and widget.ref_height:
        widget.height = geom_scaled_height(widget.ref_height)
    if hasattr(widget, "ref_spacing") and widget.ref_spacing:
        widget.spacing = geom_scaled_width(widget.ref_spacing)
    if hasattr(widget, "ref_padding") and widget.ref_padding:
        widget.padding = geom_scaled_width(widget.ref_padding)


def _apply_geom_button_metrics_now(button):
    width_ref, height_ref, font_ref, padding_x_ref, padding_y_ref = (
        button._geom_button_metrics
    )
    width, height = geom_button_size(width_ref, height_ref)
    padding_x = geom_scaled_width(padding_x_ref)
    padding_y = geom_scaled_height(padding_y_ref)

    # KivyMD buttons recompute width/height from private min values in
    # button.kv, so setting only size is overwritten on the next layout pass.
    if hasattr(button, "_min_width"):
        button._min_width = width
    if hasattr(button, "_min_height"):
        button._min_height = height
    if hasattr(button, "padding"):
        button.padding = [padding_x, padding_y, padding_x, padding_y]
    if hasattr(button, "font_size"):
        button.font_size = geom_scaled_height(font_ref)

    button.size_hint = (None, None)
    button.width = width
    button.height = height
    button.size = (width, height)


def _window_state():
    try:
        scale = float(device.dpi_scale())
    except Exception:
        scale = 1.0
    try:
        from kivy.core.window import Window as KVWindow
        window_state = (
            float(KVWindow.width or 0),
            float(KVWindow.height or 0),
        )
    except Exception:
        window_state = (0.0, 0.0)
    try:
        x, y, w, h, display = device.get_self_window_position()
    except Exception:
        x, y, w, h, display = (0, 0, 0, 0, None)
    return (
        scale,
        *window_state,
        float(x or 0),
        float(y or 0),
        float(w or 0),
        float(h or 0),
        display,
    )
