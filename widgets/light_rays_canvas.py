"""Light-ray guide editor overlay.

This editor stores guides in normalized TCG coordinates:

``line`` guides use p1/p2, while ``point`` guides use p and optionally p2 as a
direction/projection handle for directional or projected radial emission.
"""

from __future__ import annotations

import time

from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, Line, PopMatrix, PushMatrix
from kivy.graphics.scissor_instructions import ScissorPop, ScissorPush
from kivy.uix.floatlayout import FloatLayout

import config
import params


class LightRaysCanvas(FloatLayout):
    GRAB_RADIUS = 22
    POINT_RADIUS = 6

    def __init__(self, image_widget=None, guides=None, callback=None, **kwargs):
        super().__init__(**kwargs)
        self.image_widget = image_widget
        self.guides = [self._copy_guide(g) for g in (guides or [])]
        self.callback = callback
        self.creation_type = "line"
        self.creation_mode = "parallel"
        self.creation_params = {}
        self.selected = -1
        self.selected_part = None
        self._dragging = False
        self._creating = None
        self._last_apply = 0.0
        self.tcg_info = params.param_to_tcg_info({})
        self._marker_trigger = Clock.create_trigger(self._refresh_markers, 0)
        self.bind(parent=self.on_parent_changed)

    def on_parent_changed(self, instance, parent):
        if parent:
            if self.image_widget is None:
                self.image_widget = parent
            self._sync_bounds()
            parent.bind(pos=self._sync_bounds, size=self._sync_bounds)

    def _sync_bounds(self, *args):
        if self.image_widget is not None:
            self.pos = self.image_widget.pos
            self.size = self.image_widget.size
        self._refresh_markers()

    def set_primary_param(self, primary_param):
        self.tcg_info = params.param_to_tcg_info(primary_param)
        self._marker_trigger()

    def set_creation(self, guide_type, mode, apply_to_selected=False):
        guide_type = str(guide_type or "line").strip().lower()
        mode = str(mode or "parallel").strip().lower()
        self.creation_type = "point" if guide_type == "point" else "line"
        self.creation_mode = self._normalize_mode_for_type(self.creation_type, mode)
        changed = False
        if apply_to_selected:
            target = self.selected if 0 <= self.selected < len(self.guides) else (0 if len(self.guides) == 1 else -1)
            if 0 <= target < len(self.guides):
                guide = self.guides[target]
                gtype = str(guide.get("type", "")).lower()
                new_mode = self._normalize_mode_for_type(gtype, mode)
                if guide.get("mode") != new_mode:
                    guide["mode"] = new_mode
                    changed = True
                if gtype == "point" and new_mode == "directional" and not self._is_point(guide.get("p2")):
                    p = guide.get("p")
                    if self._is_point(p):
                        guide["p2"] = (float(p[0]) + 0.18, float(p[1]))
                        changed = True
        if changed:
            self._refresh_markers()
        return changed

    def set_creation_params(self, guide_params):
        self.creation_params = self._copy_params(guide_params)

    def selected_index(self):
        return self.selected if 0 <= self.selected < len(self.guides) else -1

    def select_index(self, index):
        try:
            index = int(index)
        except Exception:
            index = -1
        if 0 <= index < len(self.guides):
            self.selected = index
            self.selected_part = "p1" if self.guides[index].get("type") == "line" else "p"
        else:
            self.selected = -1
            self.selected_part = None
        self._refresh_markers()

    def selected_guide_params(self):
        if self.selected_index() < 0:
            return {}
        return self._copy_params(self.guides[self.selected].get("params"))

    def set_active_params(self, guide_params):
        params = self._copy_params(guide_params)
        target = self.selected_index()
        if target < 0 and self.guides:
            target = 0
            self.selected = 0
            self.selected_part = "p1" if self.guides[0].get("type") == "line" else "p"
        if target >= 0:
            old_params = self.guides[target].get("params")
            if (
                isinstance(old_params, dict)
                and "light_ray_projection_length" in old_params
                and "light_ray_projection_length" not in params
            ):
                params["light_ray_projection_length"] = old_params["light_ray_projection_length"]
            if self.guides[target].get("params") != params:
                self.guides[target]["params"] = params
                self._refresh_markers()
                return True
            return False
        self.creation_params = params
        return False

    def set_guides(self, guides):
        self.guides = [self._copy_guide(g) for g in (guides or [])]
        if self.selected >= len(self.guides):
            self.selected = -1
            self.selected_part = None
        self._refresh_markers()

    def get_guides(self):
        return [self._copy_guide(g) for g in self.guides]

    def _window_to_tcg(self, x, y):
        tx, ty = params.window_to_tcg(x, y, self, config.get_preview_texture_size(), self.tcg_info)
        # Allow a small outside-image margin for edge-entering beams while still
        # keeping handles recoverable on screen.
        return (max(-0.65, min(0.65, tx)), max(-0.65, min(0.65, ty)))

    def _tcg_to_window(self, x, y):
        return params.tcg_to_window(x, y, self, config.get_preview_texture_size(), self.tcg_info)

    def on_touch_down(self, touch):
        if self.image_widget is None or not self.image_widget.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if self.callback is not None:
            self.callback("focus", self)
        if getattr(touch, "is_mouse_scrolling", False):
            return super().on_touch_down(touch)

        right = getattr(touch, "button", "left") == "right"
        hit = self._nearest(touch.x, touch.y)
        if right:
            if hit is not None:
                touch.grab(self)
                self._emit("start")
                index, _part = hit
                del self.guides[index]
                self.selected = -1
                self.selected_part = None
                self._refresh_markers()
                self._emit("apply")
                self._emit("end")
                return True
            return super().on_touch_down(touch)

        if hit is not None:
            touch.grab(self)
            self.selected, self.selected_part = hit
            self._dragging = True
            self._emit("select")
            self._emit("start")
            self._refresh_markers()
            return True

        p = self._window_to_tcg(touch.x, touch.y)
        self._emit("start")
        touch.grab(self)
        self._creating = {
            "type": self.creation_type,
            "mode": self.creation_mode,
            "start": p,
            "end": p,
        }
        self._dragging = True
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_move(touch)
        if self._creating is not None:
            self._creating["end"] = self._window_to_tcg(touch.x, touch.y)
            self._refresh_markers()
            return True
        if self._dragging and 0 <= self.selected < len(self.guides):
            self._move_selected(self._window_to_tcg(touch.x, touch.y))
            self._refresh_markers()
            now = time.monotonic()
            if now - self._last_apply >= 0.05:
                self._last_apply = now
                self._emit("apply")
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_up(touch)
        touch.ungrab(self)
        if self._creating is not None:
            end = self._window_to_tcg(touch.x, touch.y)
            start = self._creating["start"]
            dragged = self._dist2_window(start, end) > 8.0 ** 2
            if self._creating["type"] == "line":
                if dragged:
                    guide = {
                        "type": "line",
                        "mode": self._creating["mode"],
                        "p1": start,
                        "p2": end,
                        "params": self._copy_params(self.creation_params),
                    }
                else:
                    guide = None
            elif self._creating["mode"] == "radial":
                guide = {
                    "type": "point",
                    "mode": "radial",
                    "p": start,
                    "params": self._copy_params(self.creation_params),
                }
                if dragged:
                    guide["p2"] = end
            else:
                guide = {
                    "type": "point",
                    "mode": "directional",
                    "p": start,
                    "p2": end,
                    "params": self._copy_params(self.creation_params),
                } if dragged else None
            if guide is not None:
                self.guides.append(guide)
                self.selected = len(self.guides) - 1
                self.selected_part = "p2" if self._is_point(guide.get("p2")) else "p"
                self._emit("apply")
            self._creating = None
            self._dragging = False
            self._refresh_markers()
            self._emit("end")
            return True
        if self._dragging:
            self._dragging = False
            self._emit("apply")
            self._emit("end")
            return True
        return super().on_touch_up(touch)

    def _move_selected(self, p):
        guide = self.guides[self.selected]
        part = self.selected_part
        if guide.get("type") == "line":
            if part == "p1":
                guide["p1"] = p
            elif part == "p2":
                guide["p2"] = p
        else:
            if part == "p2":
                guide["p2"] = p
            else:
                guide["p"] = p

    def _nearest(self, wx, wy):
        best = None
        best_d = self.GRAB_RADIUS ** 2
        for index, guide in enumerate(self.guides):
            for part, point in self._guide_points(guide):
                x, y = self._tcg_to_window(point[0], point[1])
                d = (x - wx) ** 2 + (y - wy) ** 2
                if d <= best_d:
                    best = (index, part)
                    best_d = d
        return best

    def _guide_points(self, guide):
        if guide.get("type") == "line":
            return (("p1", guide.get("p1")), ("p2", guide.get("p2")))
        points = [("p", guide.get("p"))]
        if guide.get("p2") is not None:
            points.append(("p2", guide.get("p2")))
        return tuple((part, point) for part, point in points if self._is_point(point))

    def _refresh_markers(self, *args):
        self.canvas.after.clear()
        with self.canvas.after:
            PushMatrix()
            ScissorPush(x=int(self.pos[0]), y=int(self.pos[1]), width=int(self.size[0]), height=int(self.size[1]))
            for index, guide in enumerate(self.guides):
                self._draw_guide(index, guide)
            if self._creating is not None:
                self._draw_temp(self._creating)
            ScissorPop()
            PopMatrix()

    def _draw_guide(self, index, guide):
        selected = index == self.selected
        color = (1.0, 0.78, 0.22, 1.0) if selected else (0.92, 0.95, 1.0, 0.82)
        if guide.get("type") == "line":
            p1, p2 = guide.get("p1"), guide.get("p2")
            if not self._is_point(p1) or not self._is_point(p2):
                return
            x1, y1 = self._tcg_to_window(p1[0], p1[1])
            x2, y2 = self._tcg_to_window(p2[0], p2[1])
            Color(*color)
            Line(points=[x1, y1, x2, y2], width=1.6)
            self._draw_handle(x1, y1, selected and self.selected_part == "p1")
            self._draw_handle(x2, y2, selected and self.selected_part == "p2")
        else:
            p = guide.get("p")
            if not self._is_point(p):
                return
            x, y = self._tcg_to_window(p[0], p[1])
            if self._is_point(guide.get("p2")):
                p2 = guide.get("p2")
                x2, y2 = self._tcg_to_window(p2[0], p2[1])
                Color(*color)
                Line(points=[x, y, x2, y2], width=1.4, dash_length=6, dash_offset=2)
                self._draw_handle(x2, y2, selected and self.selected_part == "p2")
            self._draw_handle(x, y, selected and self.selected_part == "p", filled=True)

    def _draw_temp(self, temp):
        start = temp["start"]
        end = temp["end"]
        x1, y1 = self._tcg_to_window(start[0], start[1])
        x2, y2 = self._tcg_to_window(end[0], end[1])
        Color(1.0, 0.78, 0.22, 0.95)
        Line(points=[x1, y1, x2, y2], width=1.5, dash_length=5, dash_offset=5)
        self._draw_handle(x1, y1, True, filled=temp["type"] == "point")
        self._draw_handle(x2, y2, True)

    def _draw_handle(self, x, y, active, filled=False):
        r = self.POINT_RADIUS
        if active:
            Color(0.06, 0.16, 0.75, 1.0)
            Ellipse(pos=(x - r, y - r), size=(2 * r, 2 * r))
            Color(1.0, 1.0, 1.0, 1.0)
        else:
            if filled:
                Color(1.0, 1.0, 1.0, 0.95)
                Ellipse(pos=(x - r, y - r), size=(2 * r, 2 * r))
            Color(0.08, 0.09, 0.12, 0.85)
        Line(circle=(x, y, r), width=1.3)

    def _emit(self, proc):
        if self.callback is not None:
            self.callback(proc, self)

    def _dist2_window(self, a, b):
        ax, ay = self._tcg_to_window(a[0], a[1])
        bx, by = self._tcg_to_window(b[0], b[1])
        return (ax - bx) ** 2 + (ay - by) ** 2

    @staticmethod
    def _is_point(value):
        try:
            float(value[0])
            float(value[1])
            return True
        except Exception:
            return False

    @staticmethod
    def _normalize_mode_for_type(guide_type, mode):
        guide_type = str(guide_type or "").lower()
        mode = str(mode or "").lower()
        if guide_type == "point":
            if mode == "parallel":
                return "radial"
            return mode if mode in {"radial", "directional"} else "radial"
        if mode == "radial":
            return "parallel"
        return mode if mode in {"parallel", "directional"} else "parallel"

    @classmethod
    def _copy_guide(cls, guide):
        if not isinstance(guide, dict):
            return {}
        out = {"type": str(guide.get("type", "")), "mode": str(guide.get("mode", ""))}
        for key in ("p", "p1", "p2"):
            value = guide.get(key)
            if cls._is_point(value):
                out[key] = (float(value[0]), float(value[1]))
        if "angle" in guide:
            out["angle"] = float(guide["angle"])
        params = cls._copy_params(guide.get("params"))
        if params:
            out["params"] = params
        return out

    @staticmethod
    def _copy_params(params):
        if not isinstance(params, dict):
            return {}
        out = {}
        for key, value in params.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
        return out
