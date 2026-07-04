import pathlib
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import effects
from cores import light_rays
from widgets.light_rays_canvas import LightRaysCanvas


def _base(h=160, w=220, value=0.08):
    return np.full((h, w, 3), value, dtype=np.float32)


class _DummySwitch:
    def __init__(self, active=True):
        self.active = active


class _DummySpinner:
    def __init__(self, text=""):
        self.text = text
        self.values = []
        self.set_calls = []

    def set_text(self, value):
        self.text = value
        self.set_calls.append(value)


class _DummySlider:
    def __init__(self, value=0):
        self.value = value
        self.set_calls = []

    def set_slider_value(self, value):
        self.value = value
        self.set_calls.append(value)


class _DummyColorPicker:
    def __init__(self, hue=42, lum=58, sat=45):
        self.ids = {
            "slider_hue": _DummySlider(hue),
            "slider_lum": _DummySlider(lum),
            "slider_sat": _DummySlider(sat),
        }
        self.set_calls = []

    def set_slider_value(self, value):
        self.set_calls.append(value)


class _DummyWidget:
    def __init__(self, editor_type="Line", editor_mode="Parallel"):
        self.ids = {
            "switch_light_rays": _DummySwitch(True),
            "spinner_light_ray_editor_type": _DummySpinner(editor_type),
            "spinner_light_ray_editor_mode": _DummySpinner(editor_mode),
            "slider_light_ray_intensity": _DummySlider(60),
            "slider_light_ray_length": _DummySlider(30),
            "slider_light_ray_decay": _DummySlider(50),
            "slider_light_ray_width": _DummySlider(65),
            "slider_light_ray_softness": _DummySlider(45),
            "slider_light_ray_edge_bias": _DummySlider(0),
            "slider_light_ray_spread": _DummySlider(35),
            "slider_light_ray_count": _DummySlider(8),
            "slider_light_ray_density": _DummySlider(10),
            "slider_light_ray_variation": _DummySlider(45),
            "slider_light_ray_fog": _DummySlider(25),
            "slider_light_ray_occlusion": _DummySlider(30),
            "slider_light_ray_seed": _DummySlider(0),
            "cp_light_ray_color": _DummyColorPicker(),
        }


class LightRaysCoreTest(unittest.TestCase):
    def test_zero_intensity_no_change(self):
        img = _base()
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (0, 80), "p2": (220, 80)}],
            intensity=0,
        )
        self.assertTrue(np.array_equal(out, img))

    def test_line_parallel_creates_broad_volumetric_band(self):
        img = _base()
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (0, 80), "p2": (220, 80)}],
            intensity=80,
            width=20,
            density=30,
            occlusion=0,
            seed=2,
        )
        diff = (out - img).mean(axis=2)
        center_band = diff[74:86, :].mean()
        wide_band = diff[48:112, :].mean()
        edge_band = diff[0:12, :].mean()
        self.assertGreater(center_band, 0.05)
        # The effect should read as a thick shaft of lit air, not a hairline.
        self.assertGreater(wide_band, center_band * 0.45)
        self.assertGreater(wide_band, edge_band * 1.25)
        self.assertGreater(diff[78:82, 180:210].mean(), 0.002)

    def test_point_radial_emits_around_source(self):
        img = _base()
        out = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (110, 80)}],
            intensity=90,
            length=80,
            occlusion=0,
            seed=4,
        )
        diff = (out - img).mean(axis=2)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        r = np.sqrt((xx - 110) ** 2 + (yy - 80) ** 2)
        ring = diff[(r > 30) & (r < 70)].mean()
        far = diff[r > 100].mean()
        self.assertGreater(ring, far * 2.0)

    def test_point_radial_same_p2_uses_classic_radial_fallback(self):
        img = _base(h=180, w=240)
        kwargs = dict(intensity=90, length=100, decay=100, width=55, softness=45,
                      count=8, density=0, variation=20, occlusion=0, seed=5)
        classic = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (90, 90)}],
            **kwargs,
        )
        projected = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (90, 90), "p2": (90, 90)}],
            **kwargs,
        )
        self.assertLess(float(np.max(np.abs(classic - projected))), 1e-6)

    def test_point_radial_p2_projects_rays_toward_front_endpoint(self):
        img = _base(h=240, w=280)
        kwargs = dict(intensity=90, length=100, decay=100, width=55, softness=45,
                      count=8, density=0, variation=0, occlusion=0, seed=5)
        classic = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (90, 120)}],
            **kwargs,
        )
        projected = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (90, 120), "p2": (180, 120)}],
            **kwargs,
        )

        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]

        def centroid_x(out):
            diff = (out - img).mean(axis=2)
            return float((xx * diff).sum() / max(float(diff.sum()), 1e-6))

        projected_diff = (projected - img).mean(axis=2)
        self.assertGreater(centroid_x(projected), centroid_x(classic) + 10.0)
        self.assertGreater(projected_diff[:, 160:220].mean(), projected_diff[:, 20:80].mean() * 1.20)

    def test_point_radial_p2_short_guide_can_reach_full_tilt(self):
        img = _base(h=240, w=320)
        kwargs = dict(intensity=90, length=100, decay=80, width=65, softness=55,
                      spread=45, count=8, density=0, variation=20, occlusion=0, seed=8)
        classic = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (80, 120)}],
            **kwargs,
        )
        short = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (80, 120), "p2": (160, 120), "projection_length": 100}],
            **kwargs,
        )
        long = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (80, 120), "p2": (240, 120), "projection_length": 100}],
            **kwargs,
        )

        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]

        def centroid_x(out):
            diff = (out - img).mean(axis=2)
            return float((xx * diff).sum() / max(float(diff.sum()), 1e-6))

        classic_x = centroid_x(classic)
        short_shift = centroid_x(short) - classic_x
        long_shift = centroid_x(long) - classic_x
        self.assertGreater(short_shift, 20.0)
        self.assertGreater(short_shift, long_shift * 0.85)

    def test_point_radial_p2_reach_controls_projected_ray_distance(self):
        img = _base(h=240, w=320)
        guide = [{"type": "point", "mode": "radial", "p": (70, 120), "p2": (190, 120)}]
        kwargs = dict(intensity=90, decay=100, width=55, softness=45, spread=35,
                      count=8, density=0, variation=0, occlusion=0, seed=5)
        short = light_rays.apply_light_rays(img, guide, length=30, **kwargs)
        long = light_rays.apply_light_rays(img, guide, length=100, **kwargs)

        short_far = (short - img).mean(axis=2)[95:145, 230:300].mean()
        long_far = (long - img).mean(axis=2)[95:145, 230:300].mean()
        self.assertGreater(long_far, 0.005)
        self.assertGreater(long_far, short_far * 100.0)

    def test_point_radial_p2_fixed_projection_keeps_start_when_reach_changes(self):
        img = _base(h=240, w=320)
        guide = [{
            "type": "point",
            "mode": "radial",
            "p": (70, 120),
            "p2": (190, 120),
            "projection_length": 100,
        }]
        kwargs = dict(intensity=90, decay=100, width=55, softness=45, spread=35,
                      count=1, density=0, variation=0, occlusion=0, seed=5)
        short = light_rays.apply_light_rays(img, guide, length=30, **kwargs)
        long = light_rays.apply_light_rays(img, guide, length=100, **kwargs)

        short_diff = (short - img).mean(axis=2)
        long_diff = (long - img).mean(axis=2)
        root = np.s_[108:132, 70:120]
        far = np.s_[95:145, 230:300]
        self.assertLess(float(np.max(np.abs(short_diff[root] - long_diff[root]))), 1e-6)
        self.assertGreater(float(long_diff[far].mean()), float(short_diff[far].mean()) * 100.0)

    def test_point_radial_p2_source_does_not_read_as_dark_hole(self):
        img = _base(h=240, w=320)
        out = light_rays.apply_light_rays(
            img,
            [{
                "type": "point",
                "mode": "radial",
                "p": (90, 120),
                "p2": (190, 100),
                "projection_length": 100,
            }],
            intensity=90,
            length=80,
            decay=80,
            width=65,
            softness=55,
            spread=45,
            count=8,
            density=0,
            variation=45,
            occlusion=0,
            seed=8,
        )
        diff = (out - img).mean(axis=2)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        r = np.sqrt((xx - 90) ** 2 + (yy - 120) ** 2)
        source = diff[r < 12].mean()
        near_ring = diff[(r > 24) & (r < 48)].mean()

        self.assertGreater(source, 0.05)
        self.assertGreater(source, near_ring * 0.72)

    def test_point_radial_p2_width_thickens_shafts_not_source_viewpoint(self):
        img = _base(h=260, w=320)
        guide = [{"type": "point", "mode": "radial", "p": (100, 130), "p2": (210, 130)}]
        kwargs = dict(intensity=90, length=100, decay=100, softness=45,
                      spread=35, count=8, density=0, variation=0, occlusion=0, seed=5)
        narrow = light_rays.apply_light_rays(img, guide, width=20, **kwargs)
        wide = light_rays.apply_light_rays(img, guide, width=160, **kwargs)

        narrow_diff = (narrow - img).mean(axis=2)
        wide_diff = (wide - img).mean(axis=2)

        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        source_disk = ((xx - 100) ** 2 + (yy - 130) ** 2) < 20 ** 2
        far_air = np.s_[:, 250:300]
        self.assertGreater(wide_diff[far_air].mean(), narrow_diff[far_air].mean() * 2.0)
        self.assertLess(wide_diff[source_disk].mean(), narrow_diff[source_disk].mean() * 1.55 + 0.005)

    def test_point_radial_p2_cone_spread_controls_projected_fan(self):
        img = _base(h=260, w=320)
        guide = [{"type": "point", "mode": "radial", "p": (100, 130), "p2": (210, 130)}]
        kwargs = dict(intensity=90, length=100, decay=100, width=55, softness=45,
                      count=8, density=0, variation=0, occlusion=0, seed=5)
        narrow = light_rays.apply_light_rays(img, guide, spread=5, **kwargs)
        wide = light_rays.apply_light_rays(img, guide, spread=90, **kwargs)

        def active_cross_section(out):
            diff = (out - img).mean(axis=2)
            profile = diff[:, 130:170].mean(axis=1)
            active = np.where(profile > float(profile.max()) * 0.20)[0]
            return int(active[-1] - active[0] + 1) if len(active) else 0

        self.assertGreater(active_cross_section(wide), active_cross_section(narrow) * 1.45)

    def test_point_radial_one_point_has_no_left_angle_seam(self):
        img = _base(h=260, w=260)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (130, 130)}],
            intensity=90,
            length=100,
            decay=100,
            width=65,
            softness=45,
            count=8,
            density=30,
            variation=100,
            occlusion=0,
            seed=4,
        )
        diff = (out - img).mean(axis=2)
        seam = diff[130, 30:105].mean()
        neighbors = (diff[128, 30:105].mean() + diff[132, 30:105].mean()) * 0.5
        self.assertGreater(seam, neighbors * 0.72)

    def test_line_directional_keeps_user_start_inside_frame(self):
        img = _base()
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "directional", "p1": (40, 80), "p2": (180, 80)}],
            intensity=90,
            width=18,
            spread=20,
            occlusion=0,
            seed=1,
        )
        diff = (out - img).mean(axis=2)
        forward = diff[74:86, 100:150].mean()
        screen_entry = diff[74:86, 0:30].mean()
        beyond_end = diff[74:86, 205:219].mean()
        self.assertLess(screen_entry, forward * 0.35)
        self.assertGreater(forward, beyond_end * 3.0)

    def test_line_parallel_still_extends_visible_start_outside_frame(self):
        img = _base()
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (40, 80), "p2": (180, 80)}],
            intensity=90,
            width=18,
            occlusion=0,
            seed=1,
        )
        diff = (out - img).mean(axis=2)
        screen_entry = diff[74:86, 0:30].mean()
        middle = diff[74:86, 80:130].mean()
        self.assertGreater(screen_entry, 0.01)
        self.assertGreater(middle, 0.01)

    def test_line_directional_inside_p1_fades_in_without_backside_light(self):
        img = _base(h=140, w=220)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "directional", "p1": (90, 70), "p2": (190, 70)}],
            intensity=90,
            width=90,
            spread=5,
            count=5,
            occlusion=0,
            seed=31,
        )
        diff = (out - img).mean(axis=2)
        far_behind = diff[62:78, 12:32].mean()
        near_behind = diff[62:78, 62:82].mean()
        just_ahead = diff[62:78, 92:112].mean()
        later_ahead = diff[62:78, 132:158].mean()
        self.assertLess(far_behind, later_ahead * 0.03)
        self.assertLess(near_behind, later_ahead * 0.06)
        self.assertGreater(just_ahead, 0.002)
        self.assertGreater(later_ahead, just_ahead * 1.25)

    def test_line_directional_spread_does_not_reverse_flare_behind_start(self):
        img = _base(h=180, w=260)
        guide = [{"type": "line", "mode": "directional", "p1": (110, 90), "p2": (220, 90)}]
        narrow = light_rays.apply_light_rays(
            img,
            guide,
            intensity=90,
            width=24,
            softness=60,
            spread=5,
            count=1,
            density=0,
            variation=0,
            occlusion=0,
            seed=3,
        )
        wide = light_rays.apply_light_rays(
            img,
            guide,
            intensity=90,
            width=24,
            softness=60,
            spread=80,
            count=1,
            density=0,
            variation=0,
            occlusion=0,
            seed=3,
        )
        narrow_diff = (narrow - img).mean(axis=2)
        wide_diff = (wide - img).mean(axis=2)
        behind_off_axis_narrow = narrow_diff[48:70, 92:108].mean() + narrow_diff[110:132, 92:108].mean()
        behind_off_axis_wide = wide_diff[48:70, 92:108].mean() + wide_diff[110:132, 92:108].mean()
        forward_off_axis_narrow = narrow_diff[48:70, 138:154].mean() + narrow_diff[110:132, 138:154].mean()
        forward_off_axis_wide = wide_diff[48:70, 138:154].mean() + wide_diff[110:132, 138:154].mean()
        self.assertLess(behind_off_axis_wide, behind_off_axis_narrow * 1.20 + 0.002)
        self.assertGreater(forward_off_axis_wide, forward_off_axis_narrow * 1.25)

    def test_line_directional_start_brightness_rises_smoothly_forward_only(self):
        img = _base(h=180, w=260)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "directional", "p1": (110, 90), "p2": (220, 90)}],
            intensity=90,
            width=28,
            softness=55,
            spread=35,
            count=1,
            density=0,
            variation=0,
            occlusion=0,
            seed=5,
        )
        diff = (out - img).mean(axis=2)
        just_behind = diff[84:96, 86:104].mean()
        early_forward = diff[84:96, 112:128].mean()
        later_forward = diff[84:96, 150:176].mean()
        far_behind = diff[84:96, 22:44].mean()
        self.assertLess(just_behind, later_forward * 0.08)
        self.assertLess(far_behind, later_forward * 0.03)
        self.assertGreater(early_forward, 0.002)
        self.assertGreater(later_forward, early_forward * 1.20)

    def test_line_directional_variation_staggers_shaft_starts(self):
        img = _base(h=180, w=280)
        guide = [{"type": "line", "mode": "directional", "p1": (70, 90), "p2": (240, 90)}]

        def start_span(variation):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                width=40,
                softness=45,
                spread=30,
                count=5,
                density=0,
                variation=variation,
                occlusion=0,
                seed=11,
            )
            diff = (out - img).mean(axis=2)
            starts = []
            for y0, y1 in ((48, 64), (72, 84), (86, 98), (108, 120), (132, 148)):
                profile = diff[y0:y1, :].mean(axis=0)
                active = np.where(profile > float(profile.max()) * 0.65)[0]
                starts.append(int(active[0]) if len(active) else 999)
            return max(starts) - min(starts)

        self.assertGreater(start_span(100), start_span(0) + 7)

    def test_point_directional_uses_handle_direction(self):
        img = _base()
        out = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "directional", "p": (70, 80), "p2": (180, 80)}],
            intensity=90,
            width=18,
            spread=18,
            occlusion=0,
            seed=3,
        )
        diff = (out - img).mean(axis=2)
        right = diff[74:86, 115:170].mean()
        left = diff[74:86, 15:55].mean()
        self.assertGreater(right, left * 8.0)

    def test_point_directional_uses_point_apex_unlike_line_directional(self):
        img = _base(h=180, w=260)
        point = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "directional", "p": (70, 90), "p2": (190, 90)}],
            intensity=80,
            width=35,
            spread=22,
            count=5,
            density=20,
            variation=20,
            occlusion=0,
            seed=12,
        )
        line = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "directional", "p1": (70, 90), "p2": (190, 90)}],
            intensity=80,
            width=35,
            spread=22,
            count=5,
            density=20,
            variation=20,
            occlusion=0,
            seed=12,
        )
        point_diff = (point - img).mean(axis=2)
        line_diff = (line - img).mean(axis=2)
        point_center = point_diff[84:96, 72:88].mean()
        point_off_axis = point_diff[60:78, 72:88].mean() + point_diff[102:120, 72:88].mean()
        line_center = line_diff[84:96, 72:88].mean()
        line_off_axis = line_diff[60:78, 72:88].mean() + line_diff[102:120, 72:88].mean()
        self.assertGreater(point_center, point_off_axis * 10.0)
        # Goal: right at the start, a point source concentrates light at its
        # apex while a line source spreads it across the full band width.
        # Compare centre/off-axis concentration instead of absolute off-axis
        # brightness so the jagged/soft start tuning cannot break the pin.
        point_conc = point_center / max(point_off_axis, 1e-6)
        line_conc = line_center / max(line_off_axis, 1e-6)
        self.assertGreater(point_conc, line_conc * 4.0)
        self.assertGreater(float(np.max(np.abs(point - line))), 0.02)

    def test_point_directional_width_does_not_stretch_behind_source(self):
        img = _base(h=180, w=260)
        guide = [{"type": "point", "mode": "directional", "p": (110, 90), "p2": (220, 90)}]
        narrow = light_rays.apply_light_rays(
            img,
            guide,
            intensity=90,
            width=20,
            spread=35,
            softness=55,
            count=4,
            density=0,
            variation=0,
            occlusion=0,
            seed=17,
        )
        wide = light_rays.apply_light_rays(
            img,
            guide,
            intensity=90,
            width=160,
            spread=35,
            softness=55,
            count=4,
            density=0,
            variation=0,
            occlusion=0,
            seed=17,
        )
        narrow_diff = (narrow - img).mean(axis=2)
        wide_diff = (wide - img).mean(axis=2)
        narrow_back = narrow_diff[84:96, 58:82].mean()
        wide_back = wide_diff[84:96, 58:82].mean()
        wide_forward = wide_diff[84:96, 130:170].mean()
        self.assertLess(wide_back, narrow_back * 1.6 + 0.01)
        self.assertLess(wide_back, wide_forward * 0.18)

    def test_point_directional_has_no_separate_source_bloom(self):
        img = _base(h=180, w=260)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "directional", "p": (110, 90), "p2": (220, 90)}],
            intensity=90,
            width=180,
            spread=35,
            softness=55,
            count=4,
            density=0,
            variation=0,
            occlusion=0,
            seed=18,
        )
        diff = (out - img).mean(axis=2)
        behind_center = diff[84:96, 82:98].mean()
        behind_off_axis = diff[48:64, 82:98].mean() + diff[116:132, 82:98].mean()
        just_forward = diff[84:96, 104:116].mean()
        forward = diff[84:96, 130:170].mean()
        self.assertLess(behind_center, forward * 0.02)
        self.assertLess(behind_off_axis, forward * 0.02)
        self.assertLess(just_forward, forward * 0.85)

    def test_point_directional_variation_staggers_shaft_starts(self):
        img = _base(h=200, w=280)
        guide = [{"type": "point", "mode": "directional", "p": (70, 100), "p2": (240, 100)}]

        def off_axis_starts(variation):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                width=60,
                softness=45,
                spread=28,
                count=5,
                density=0,
                variation=variation,
                occlusion=0,
                seed=12,
            )
            diff = (out - img).mean(axis=2)
            starts = []
            for y0, y1 in ((74, 88), (112, 126)):
                profile = diff[y0:y1, :].mean(axis=0)
                active = np.where(profile > float(profile.max()) * 0.65)[0]
                starts.append(int(active[0]) if len(active) else 999)
            return starts

        low = off_axis_starts(0)
        high = off_axis_starts(100)
        self.assertGreater(min(h - l for h, l in zip(high, low)), 8)

    def test_line_segment_length_controls_parallel_reach(self):
        img = _base(h=140, w=240)
        short = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (20, 70), "p2": (90, 70)}],
            intensity=90,
            width=22,
            count=3,
            density=30,
            occlusion=0,
            seed=13,
        )
        long = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (20, 70), "p2": (210, 70)}],
            intensity=90,
            width=22,
            count=3,
            density=30,
            occlusion=0,
            seed=13,
        )
        short_diff = (short - img).mean(axis=2)
        long_diff = (long - img).mean(axis=2)
        near_short = short_diff[62:78, 52:82].mean()
        beyond_short = short_diff[62:78, 165:195].mean()
        beyond_long = long_diff[62:78, 165:195].mean()
        self.assertLess(beyond_short, near_short * 0.35)
        self.assertGreater(beyond_long, beyond_short * 3.0)

    def test_line_fade_uses_inverse_square_like_distance_falloff(self):
        img = _base(h=160, w=260)
        guide = [{"type": "line", "mode": "parallel", "p1": (20, 80), "p2": (180, 80)}]
        out = light_rays.apply_light_rays(
            img,
            guide,
            intensity=90,
            width=20,
            count=1,
            occlusion=0,
            decay=45,
            seed=1,
        )
        diff = (out - img).mean(axis=2)
        near = diff[74:86, 50:80].mean()
        mid = diff[74:86, 105:135].mean()
        end = diff[74:86, 170:190].mean()
        self.assertLess(mid, near * 0.65)
        self.assertLess(end, mid * 0.70)

    def test_line_end_has_soft_tail_instead_of_hard_cutoff(self):
        img = _base(h=160, w=260)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (20, 80), "p2": (180, 80)}],
            intensity=90,
            width=20,
            count=1,
            occlusion=0,
            decay=45,
            seed=1,
        )
        diff = (out - img).mean(axis=2)
        end = diff[74:86, 170:190].mean()
        tail = diff[74:86, 205:235].mean()
        self.assertGreater(tail, 0.005)
        self.assertLess(tail, end * 0.40)

    def test_dark_occluder_reduces_rays(self):
        img = np.full((150, 220, 3), 0.75, dtype=np.float32)
        img[:, 104:124, :] = 0.02
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (0, 75), "p2": (220, 75)}],
            intensity=80,
            width=25,
            occlusion=100,
            seed=5,
        )
        diff = (out - img).mean(axis=2)
        blocked = diff[68:82, 108:120].mean()
        clear = diff[68:82, 150:175].mean()
        self.assertLess(blocked, clear * 0.55)

    def test_occluder_casts_shadow_along_light_direction(self):
        img = np.full((150, 240, 3), 0.72, dtype=np.float32)
        img[45:105, 70:86, :] = 0.02
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (-80, 75), "p2": (240, 75)}],
            intensity=80,
            width=45,
            occlusion=100,
            seed=9,
        )
        diff = (out - img).mean(axis=2)
        upstream = diff[60:90, 28:58].mean()
        downstream_shadow = diff[60:90, 125:170].mean()
        self.assertLess(downstream_shadow, upstream * 0.85)

    def test_tree_shadow_reacts_to_moderately_dark_texture(self):
        img = np.full((150, 260, 3), 0.68, dtype=np.float32)
        img[35:120, 78:88, :] = 0.28
        img[20:138, 126:137, :] = 0.34
        guide = [{"type": "line", "mode": "parallel", "p1": (-50, 75), "p2": (255, 75)}]
        clear = light_rays.apply_light_rays(
            img, guide, intensity=80, width=38, count=4, density=20, occlusion=0, seed=22)
        shaded = light_rays.apply_light_rays(
            img, guide, intensity=80, width=38, count=4, density=20, occlusion=100, seed=22)
        clear_diff = (clear - img).mean(axis=2)
        shaded_diff = (shaded - img).mean(axis=2)
        clear_path = clear_diff[60:90, 155:215].mean()
        shaded_path = shaded_diff[60:90, 155:215].mean()
        self.assertLess(shaded_path, clear_path * 0.62)

    def test_texture_density_adds_visible_internal_variation(self):
        img = _base(h=180, w=280)
        guide = [{"type": "line", "mode": "parallel", "p1": (-30, 90), "p2": (270, 90)}]
        smooth = light_rays.apply_light_rays(
            img, guide, intensity=80, width=35, count=5, density=0, variation=20, occlusion=0, seed=30)
        textured = light_rays.apply_light_rays(
            img, guide, intensity=80, width=35, count=5, density=100, variation=20, occlusion=0, seed=30)
        smooth_profile = (smooth - img)[68:112, :, :].mean(axis=(0, 2))
        textured_profile = (textured - img)[68:112, :, :].mean(axis=(0, 2))
        smooth_detail = np.std(np.diff(smooth_profile))
        textured_detail = np.std(np.diff(textured_profile))
        self.assertGreater(textured_detail, smooth_detail * 1.45)

    def test_count_increases_visible_shaft_count(self):
        img = _base(h=180, w=260)
        guide = [{"type": "line", "mode": "parallel", "p1": (0, 90), "p2": (260, 90)}]
        low = light_rays.apply_light_rays(
            img, guide, intensity=80, width=45, count=2, density=45, variation=20, occlusion=0, seed=11)
        high = light_rays.apply_light_rays(
            img, guide, intensity=80, width=45, count=12, density=45, variation=20, occlusion=0, seed=11)

        def peak_count(arr):
            profile = (arr - img).mean(axis=(1, 2))
            profile = (profile - profile.min()) / max(float(profile.max() - profile.min()), 1e-6)
            return sum(
                1 for i in range(1, len(profile) - 1)
                if profile[i] > 0.45 and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]
            )

        self.assertGreater(peak_count(high), peak_count(low))

    def test_default_forest_beam_preset_draws_broad_visible_light(self):
        img = _base(h=240, w=360)
        defaults = effects.LightRaysEffect().get_param_dict({})
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (-20, 120), "p2": (380, 120)}],
            intensity=defaults["light_ray_intensity"],
            decay=defaults["light_ray_decay"],
            width=defaults["light_ray_width"],
            softness=defaults["light_ray_softness"],
            edge_bias=defaults["light_ray_edge_bias"],
            count=defaults["light_ray_count"],
            density=defaults["light_ray_density"],
            variation=defaults["light_ray_variation"],
            fog=defaults["light_ray_fog"],
            occlusion=0,
            seed=0,
        )
        diff = (out - img).mean(axis=2)
        profile = diff.mean(axis=1)
        norm = (profile - profile.min()) / max(float(profile.max() - profile.min()), 1e-6)
        active_fraction = np.count_nonzero(norm > 0.25) / norm.size
        strong_fraction = np.count_nonzero(norm > 0.50) / norm.size
        self.assertGreater(float(diff.mean()), 0.08)
        self.assertGreater(active_fraction, 0.50)
        self.assertGreater(strong_fraction, 0.15)

    def test_high_count_parallel_can_fill_the_frame_with_multiple_shafts(self):
        img = _base(h=240, w=360)
        guide = [{"type": "line", "mode": "parallel", "p1": (0, 120), "p2": (360, 120)}]
        out = light_rays.apply_light_rays(
            img,
            guide,
            intensity=80,
            width=18,
            count=28,
            density=45,
            variation=35,
            occlusion=0,
            seed=17,
        )
        diff = (out - img).mean(axis=2)
        profile = diff.mean(axis=1)
        norm = (profile - profile.min()) / max(float(profile.max() - profile.min()), 1e-6)
        peaks = [
            i for i in range(1, len(norm) - 1)
            if norm[i] > 0.25 and norm[i] >= norm[i - 1] and norm[i] >= norm[i + 1]
        ]
        self.assertGreaterEqual(len(peaks), 8)
        self.assertGreater(diff[:36, :].mean(), 0.02)
        self.assertGreater(diff[-36:, :].mean(), 0.02)

    def test_width_above_100_can_make_parallel_light_fill_frame_height(self):
        img = _base(h=240, w=360)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (-20, 120), "p2": (380, 120)}],
            intensity=70,
            width=200,
            softness=100,
            count=1,
            occlusion=0,
            seed=19,
        )
        diff = (out - img).mean(axis=2)
        center = diff[108:132, 80:280].mean()
        top = diff[:32, 80:280].mean()
        bottom = diff[-32:, 80:280].mean()
        self.assertGreater(top, center * 0.45)
        self.assertGreater(bottom, center * 0.45)

    def test_resolution_scale_does_not_change_current_pixel_space_width(self):
        img = _base(h=180, w=260)
        guide = [{"type": "line", "mode": "parallel", "p1": (-20, 90), "p2": (240, 90)}]
        base = light_rays.apply_light_rays(
            img,
            guide,
            intensity=80,
            width=42,
            softness=50,
            count=1,
            density=0,
            variation=0,
            occlusion=0,
            seed=21,
            resolution_scale=1.0,
        )
        exported = light_rays.apply_light_rays(
            img,
            guide,
            intensity=80,
            width=42,
            softness=50,
            count=1,
            density=0,
            variation=0,
            occlusion=0,
            seed=21,
            resolution_scale=3.0,
        )
        self.assertLess(float(np.max(np.abs(base - exported))), 1e-6)

    def test_export_sized_render_keeps_preview_relative_shaft_width(self):
        preview = _base(h=180, w=260)
        full = _base(h=360, w=520)
        preview_out = light_rays.apply_light_rays(
            preview,
            [{"type": "line", "mode": "parallel", "p1": (-20, 90), "p2": (240, 90)}],
            intensity=80,
            width=42,
            softness=50,
            count=1,
            density=0,
            variation=0,
            occlusion=0,
            seed=21,
            resolution_scale=1.0,
        )
        full_out = light_rays.apply_light_rays(
            full,
            [{"type": "line", "mode": "parallel", "p1": (-40, 180), "p2": (480, 180)}],
            intensity=80,
            width=42,
            softness=50,
            count=1,
            density=0,
            variation=0,
            occlusion=0,
            seed=21,
            resolution_scale=2.0,
        )
        preview_profile = (preview_out - preview).mean(axis=(1, 2))
        full_profile = (full_out - full).reshape(180, 2, 520, 3).mean(axis=(1, 2, 3))

        def half_width(profile):
            active = np.where(profile > float(profile.max()) * 0.5)[0]
            return int(active[-1] - active[0] + 1)

        self.assertLessEqual(abs(half_width(preview_profile) - half_width(full_profile)), 3)

    def test_small_preview_is_unchanged_by_resolution_scale(self):
        # Previews small enough to render quickly must stay byte-identical to
        # the native render regardless of resolution_scale; only large previews
        # are allowed to use the reduced-resolution path.
        img = _base(h=200, w=260)
        guide = [{"type": "line", "mode": "parallel", "p1": (-20, 100), "p2": (240, 100)}]
        kw = dict(intensity=70, width=45, softness=55, count=8, density=40,
                  variation=30, fog=30, occlusion=55, seed=7)
        full = light_rays.apply_light_rays(img, guide, resolution_scale=1.0, **kw)
        preview = light_rays.apply_light_rays(img, guide, resolution_scale=0.5, **kw)
        self.assertTrue(np.array_equal(full, preview))

    def test_large_preview_uses_reduced_resolution_but_keeps_geometry(self):
        # A large preview (resolution_scale < 1.0) renders the overlay smaller
        # and upsamples it, so the result differs slightly from the native
        # render but the shaft geometry (half-width) stays consistent.
        img = np.full((1600, 2200, 3), 0.12, dtype=np.float32)
        guide = [{"type": "line", "mode": "parallel", "p1": (-120, 800), "p2": (2320, 800)}]
        kw = dict(intensity=70, width=45, softness=55, count=1, density=0,
                  variation=0, fog=30, occlusion=0, seed=7)
        full = light_rays.apply_light_rays(img, guide, resolution_scale=1.0, **kw)
        preview = light_rays.apply_light_rays(img, guide, resolution_scale=0.5, **kw)
        self.assertFalse(np.array_equal(full, preview))

        def half_width(out):
            profile = (out - img).mean(axis=(1, 2))
            active = np.where(profile > float(profile.max()) * 0.5)[0]
            return int(active[-1] - active[0] + 1)

        self.assertLessEqual(abs(half_width(full) - half_width(preview)),
                             max(4, int(img.shape[0] * 0.01)))

    def test_large_preview_limits_light_ray_render_work(self):
        img = np.full((1600, 2200, 3), 0.12, dtype=np.float32)
        guide = [{"type": "point", "mode": "radial", "p": (1100, 800)}]
        seen_shapes = []
        original_compute = light_rays._compute_additive

        def spy_compute(image, *args, **kwargs):
            seen_shapes.append(image.shape[:2])
            return original_compute(image, *args, **kwargs)

        with mock.patch.object(light_rays, "_compute_additive", side_effect=spy_compute):
            light_rays.apply_light_rays(
                img,
                guide,
                intensity=60,
                length=100,
                decay=90,
                width=35,
                softness=45,
                count=8,
                density=10,
                variation=10,
                fog=25,
                occlusion=0,
                seed=0,
                resolution_scale=0.5,
            )

        self.assertEqual(len(seen_shapes), 1)
        self.assertLessEqual(max(seen_shapes[0]), 960)

    def test_edge_softness_zero_makes_shaft_edges_much_harder(self):
        img = _base(h=180, w=260)
        guide = [{"type": "line", "mode": "parallel", "p1": (-20, 90), "p2": (240, 90)}]

        def profile(softness):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                width=25,
                softness=softness,
                count=1,
                density=0,
                variation=0,
                occlusion=0,
                seed=1,
            )
            return (out - img).mean(axis=(1, 2))

        def half_width(p):
            active = np.where(p > float(p.max()) * 0.5)[0]
            return int(active[-1] - active[0] + 1)

        def edge_rise(p):
            # Distance (px) between the 10% and 90% crossings on the rising side.
            # Smaller = a crisper, harder edge.
            pk = float(p.max())
            c = int(p.argmax())
            up = p[: c + 1]
            d10 = np.where(up >= pk * 0.1)[0]
            d90 = np.where(up >= pk * 0.9)[0]
            return (int(d90[0]) - int(d10[0])) if len(d10) and len(d90) else c

        low = profile(0)
        high = profile(80)
        # Low Edge Softness must produce a markedly crisper boundary (steep rise),
        # not merely a thin line: the edge is much harder than a soft beam while
        # the shaft stays substantially thick.
        self.assertLess(edge_rise(low), edge_rise(high) * 0.4)
        self.assertGreater(half_width(low), half_width(high) * 0.3)

    def test_point_reach_100_uses_image_diagonal(self):
        img = _base(h=120, w=160)
        guide = [{"type": "point", "mode": "radial", "p": (0, 0)}]
        short = light_rays.apply_light_rays(
            img, guide, intensity=90, length=50, decay=100, width=80, count=4, occlusion=0, seed=2)
        full = light_rays.apply_light_rays(
            img, guide, intensity=90, length=100, decay=100, width=80, count=4, occlusion=0, seed=2)
        short_far = (short - img)[-12:, -12:, :].mean()
        full_far = (full - img)[-12:, -12:, :].mean()
        self.assertGreater(full_far, 0.005)
        self.assertGreater(full_far, short_far * 3.0)

    def test_point_radial_shaft_width_controls_angular_thickness(self):
        img = _base(h=220, w=220)
        guide = [{"type": "point", "mode": "radial", "p": (110, 110)}]
        narrow = light_rays.apply_light_rays(
            img, guide, intensity=90, length=100, decay=100, width=10, count=6, density=0, variation=0, occlusion=0, seed=4)
        wide = light_rays.apply_light_rays(
            img, guide, intensity=90, length=100, decay=100, width=180, count=6, density=0, variation=0, occlusion=0, seed=4)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        r = np.sqrt((xx - 110) ** 2 + (yy - 110) ** 2)
        ring = (r > 58) & (r < 82)
        narrow_values = (narrow - img).mean(axis=2)[ring]
        wide_values = (wide - img).mean(axis=2)[ring]
        narrow_fraction = np.count_nonzero(narrow_values > float(narrow_values.max()) * 0.70) / narrow_values.size
        wide_fraction = np.count_nonzero(wide_values > float(wide_values.max()) * 0.70) / wide_values.size
        self.assertGreater(wide_fraction, narrow_fraction * 1.55)

    def test_point_radial_width_expands_shafts_from_the_root(self):
        img = _base(h=220, w=220)
        guide = [{"type": "point", "mode": "radial", "p": (110, 110)}]
        narrow = light_rays.apply_light_rays(
            img, guide, intensity=90, length=100, decay=100, width=10, softness=50,
            count=6, density=0, variation=0, occlusion=0, seed=4)
        wide = light_rays.apply_light_rays(
            img, guide, intensity=90, length=100, decay=100, width=180, softness=50,
            count=6, density=0, variation=0, occlusion=0, seed=4)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        r = np.sqrt((xx - 110) ** 2 + (yy - 110) ** 2)
        root_ring = (r > 12) & (r < 28)
        narrow_values = (narrow - img).mean(axis=2)[root_ring]
        wide_values = (wide - img).mean(axis=2)[root_ring]
        narrow_fraction = np.count_nonzero(narrow_values > float(narrow_values.max()) * 0.45) / narrow_values.size
        wide_fraction = np.count_nonzero(wide_values > float(wide_values.max()) * 0.45) / wide_values.size
        self.assertGreater(wide_fraction, narrow_fraction * 2.1)
        self.assertLess(float(wide_values.max()), float(narrow_values.max()) * 1.35)

    def test_point_radial_width_broadens_shaft_starts_near_center(self):
        img = _base(h=240, w=240)
        guide = [{"type": "point", "mode": "radial", "p": (120, 120)}]

        def active_fraction(width):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                length=100,
                decay=100,
                width=width,
                softness=45,
                count=8,
                density=0,
                variation=0,
                occlusion=0,
                seed=8,
            )
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            r = np.sqrt((xx - 120) ** 2 + (yy - 120) ** 2)
            ring = (r > 14) & (r < 26)
            values = (out - img).mean(axis=2)[ring]
            return np.count_nonzero(values > float(values.max()) * 0.42) / max(1, values.size)

        self.assertGreater(active_fraction(180), active_fraction(18) * 1.45)

    def test_point_radial_width_strongly_affects_inner_root_width(self):
        img = _base(h=240, w=240)
        guide = [{"type": "point", "mode": "radial", "p": (120, 120)}]

        def active_fraction(width):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                length=100,
                decay=100,
                width=width,
                softness=45,
                count=8,
                density=0,
                variation=0,
                occlusion=0,
                seed=8,
            )
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            r = np.sqrt((xx - 120) ** 2 + (yy - 120) ** 2)
            ring = (r > 8) & (r < 16)
            values = (out - img).mean(axis=2)[ring]
            return np.count_nonzero(values > float(values.max()) * 0.40) / max(1, values.size)

        self.assertGreater(active_fraction(180), active_fraction(18) * 1.60)

    def test_point_radial_center_is_not_a_dark_hole(self):
        img = _base(h=220, w=220)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (110, 110)}],
            intensity=90,
            length=100,
            decay=100,
            width=80,
            softness=50,
            count=8,
            density=0,
            variation=0,
            occlusion=0,
            seed=4,
        )
        diff = (out - img).mean(axis=2)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        r = np.sqrt((xx - 110) ** 2 + (yy - 110) ** 2)
        center = diff[r < 5].mean()
        ring = diff[(r > 30) & (r < 50)].mean()
        self.assertGreater(center, ring * 0.85)

    def test_point_radial_width_controls_source_area_size(self):
        img = _base(h=240, w=240)
        guide = [{"type": "point", "mode": "radial", "p": (120, 120)}]

        def source_area_fraction(width):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                length=95,
                decay=100,
                width=width,
                softness=45,
                count=8,
                density=0,
                variation=0,
                occlusion=0,
                seed=14,
            )
            diff = (out - img).mean(axis=2)
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            r = np.sqrt((xx - 120) ** 2 + (yy - 120) ** 2)
            disk = r < 24
            values = diff[disk]
            return np.count_nonzero(values > float(values.max()) * 0.55) / max(1, values.size)

        self.assertGreater(source_area_fraction(180), source_area_fraction(18) * 1.70)

    def test_point_radial_variation_keeps_shaft_angles_from_bunching(self):
        img = _base(h=260, w=260)
        guide = [{"type": "point", "mode": "radial", "p": (130, 130)}]

        out = light_rays.apply_light_rays(
            img,
            guide,
            intensity=90,
            length=100,
            decay=100,
            width=32,
            softness=40,
            count=8,
            density=0,
            variation=100,
            occlusion=0,
            seed=21,
        )
        diff = (out - img).mean(axis=2)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        r = np.sqrt((xx - 130) ** 2 + (yy - 130) ** 2)
        theta = np.arctan2(yy - 130, xx - 130)
        ring = (r > 78) & (r < 98)
        values = diff[ring]
        angles = theta[ring]
        bins = np.linspace(-np.pi, np.pi, 721)
        centers = (bins[:-1] + bins[1:]) * 0.5
        profile = np.array([
            values[(angles >= bins[i]) & (angles < bins[i + 1])].mean()
            if np.any((angles >= bins[i]) & (angles < bins[i + 1])) else 0.0
            for i in range(len(bins) - 1)
        ])
        profile = (
            np.roll(profile, 2)
            + np.roll(profile, 1)
            + profile
            + np.roll(profile, -1)
            + np.roll(profile, -2)
        ) / 5.0

        phase = 2.0 * np.pi * ((21 % 997) / 997.0)
        offsets = []
        strengths = []
        for i in range(8):
            expected = ((phase + i * 2.0 * np.pi / 8.0 + np.pi) % (2.0 * np.pi)) - np.pi
            delta = np.angle(np.exp(1j * (centers - expected)))
            window = np.abs(delta) < np.deg2rad(16.0)
            peak = np.where(window)[0][np.argmax(profile[window])]
            offsets.append(abs(np.rad2deg(np.angle(np.exp(1j * (centers[peak] - expected))))))
            strengths.append(float(profile[peak]))

        self.assertLess(max(offsets), 5.0)
        self.assertGreater(min(strengths), float(profile.max()) * 0.04)

    def test_point_radial_irregularity_uses_rounded_starts_not_flat_cuts(self):
        img = _base(h=200, w=280)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "point", "mode": "radial", "p": (55, 100)}],
            intensity=90,
            length=100,
            decay=100,
            width=70,
            softness=35,
            count=1,
            density=0,
            variation=100,
            occlusion=0,
            seed=0,
        )
        diff = (out - img).mean(axis=2)

        starts = []
        for y in (112, 120, 128, 136):
            profile = diff[y, 65:240]
            active = np.where(profile > max(0.004, float(profile.max()) * 0.55))[0]
            starts.append(int(active[0]) + 65 if len(active) else 999)

        self.assertTrue(all(a < b for a, b in zip(starts, starts[1:])))
        self.assertGreater(starts[-1], starts[0] + 14)

    def test_point_radial_edge_bias_makes_one_side_of_rays_harder(self):
        img = _base(h=240, w=240)
        guide = [{"type": "point", "mode": "radial", "p": (120, 120)}]

        def side_widths(edge_bias):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                length=100,
                decay=100,
                width=34,
                softness=40,
                edge_bias=edge_bias,
                count=2,
                density=0,
                variation=0,
                occlusion=0,
                seed=0,
            )
            profile = (out - img).mean(axis=2)[86:155, 196]
            center = int(profile.argmax())
            half = float(profile.max()) * 0.5
            upper = np.where(profile[:center] < half)[0]
            lower = np.where(profile[center:] < half)[0]
            upper_w = center - (int(upper[-1]) if len(upper) else 0)
            lower_w = int(lower[0]) if len(lower) else len(profile) - center - 1
            return upper_w, lower_w

        pos_bias = side_widths(85)
        neg_bias = side_widths(-85)
        neutral = side_widths(0)
        self.assertLess(pos_bias[1], neutral[1])
        self.assertLess(neg_bias[0], neutral[0])

    def test_point_directional_edge_bias_makes_one_side_of_cone_harder(self):
        img = _base(h=200, w=280)
        guide = [{"type": "point", "mode": "directional", "p": (80, 100), "p2": (230, 100)}]

        def side_widths(edge_bias):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                length=100,
                decay=100,
                width=60,
                softness=35,
                spread=20,
                edge_bias=edge_bias,
                count=1,
                density=0,
                variation=0,
                occlusion=0,
                seed=6,
            )
            profile = (out - img).mean(axis=2)[55:145, 150]
            center = int(profile.argmax())
            half = float(profile.max()) * 0.5
            upper = np.where(profile[:center] < half)[0]
            lower = np.where(profile[center:] < half)[0]
            upper_w = center - (int(upper[-1]) if len(upper) else 0)
            lower_w = int(lower[0]) if len(lower) else len(profile) - center - 1
            return upper_w, lower_w

        pos_bias = side_widths(85)
        neg_bias = side_widths(-85)
        neutral = side_widths(0)
        self.assertLess(pos_bias[1], neutral[1])
        self.assertLess(neg_bias[0], neutral[0])

    def test_point_directional_width_visibly_broadens_near_source(self):
        img = _base(h=220, w=300)
        guide = [{"type": "point", "mode": "directional", "p": (70, 110), "p2": (250, 110)}]

        def visible_root_width(width):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                length=100,
                decay=100,
                width=width,
                softness=45,
                spread=12,
                count=1,
                density=0,
                variation=0,
                occlusion=0,
                seed=5,
            )
            profile = (out - img).mean(axis=2)[55:165, 105]
            return np.count_nonzero(profile > 0.08)

        self.assertGreater(visible_root_width(180), visible_root_width(18) * 2.0)

    def test_point_radial_variation_changes_individual_shaft_widths(self):
        img = _base(h=260, w=260)
        guide = [{"type": "point", "mode": "radial", "p": (130, 130)}]

        def angular_width_std(variation):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=90,
                length=100,
                decay=100,
                width=44,
                softness=50,
                count=10,
                density=0,
                variation=variation,
                occlusion=0,
                seed=41,
            )
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            theta = np.arctan2(yy - 130, xx - 130)
            r = np.sqrt((xx - 130) ** 2 + (yy - 130) ** 2)
            ring = (r > 90) & (r < 110)
            values = (out - img).mean(axis=2)[ring]
            angles = theta[ring]
            bins = np.linspace(-np.pi, np.pi, 361)
            profile = np.array([
                values[(angles >= bins[i]) & (angles < bins[i + 1])].mean()
                if np.any((angles >= bins[i]) & (angles < bins[i + 1])) else 0.0
                for i in range(len(bins) - 1)
            ])
            widths = []
            threshold = float(profile.max()) * 0.36
            active = profile > threshold
            doubled = np.concatenate([active, active])
            start = None
            for i, value in enumerate(doubled):
                if value and start is None:
                    start = i
                elif not value and start is not None:
                    if start < len(active):
                        widths.append(i - start)
                    start = None
            return float(np.std(widths[:10]))

        self.assertGreater(angular_width_std(100), angular_width_std(0) * 1.35)

    def test_edge_bias_makes_one_side_of_a_shaft_harder(self):
        img = _base(h=240, w=360)
        guide = [{"type": "line", "mode": "parallel", "p1": (-20, 120), "p2": (380, 120)}]

        def half_widths(edge_bias):
            out = light_rays.apply_light_rays(
                img,
                guide,
                intensity=70,
                width=35,
                softness=45,
                edge_bias=edge_bias,
                count=1,
                density=0,
                variation=0,
                occlusion=0,
                seed=5,
            )
            profile = (out - img).mean(axis=(1, 2))
            center = int(profile.argmax())
            half = float(profile.max()) * 0.5
            neg = np.where(profile[:center] < half)[0]
            pos = np.where(profile[center:] < half)[0]
            left = center - (int(neg[-1]) if len(neg) else 0)
            right = int(pos[0]) if len(pos) else len(profile) - center - 1
            return left, right

        left_hard = half_widths(-80)
        right_hard = half_widths(80)
        self.assertLess(left_hard[0], left_hard[1] * 0.45)
        self.assertLess(right_hard[1], right_hard[0] * 0.45)

    def test_dark_pixels_cast_soft_blocking_along_light_path(self):
        img = np.full((160, 260, 3), 0.72, dtype=np.float32)
        img[:, 100:118, :] = 0.02
        guide = [{"type": "line", "mode": "parallel", "p1": (-40, 80), "p2": (250, 80)}]
        clear = light_rays.apply_light_rays(
            img, guide, intensity=70, width=35, count=3, occlusion=0, seed=7)
        blocked = light_rays.apply_light_rays(
            img, guide, intensity=70, width=35, count=3, occlusion=60, seed=7)
        clear_diff = (clear - img).mean(axis=2)
        blocked_diff = (blocked - img).mean(axis=2)
        downstream_clear = clear_diff[70:90, 150:200].mean()
        downstream_blocked = blocked_diff[70:90, 150:200].mean()
        upstream_blocked = blocked_diff[70:90, 40:80].mean()
        self.assertLess(downstream_blocked, downstream_clear * 0.75)
        self.assertLess(downstream_blocked, upstream_blocked * 0.55)

    def test_reach_above_100_extends_radial_rays_toward_frame_edge(self):
        img = _base(h=200, w=320)
        guide = [{"type": "point", "mode": "radial", "p": (30, 100)}]
        short = light_rays.apply_light_rays(
            img, guide, intensity=80, length=60, decay=55, count=12, occlusion=0, seed=23)
        long = light_rays.apply_light_rays(
            img, guide, intensity=80, length=200, decay=80, count=12, occlusion=0, seed=23)
        edge_short = (short - img)[:, -32:, :].mean()
        edge_long = (long - img)[:, -32:, :].mean()
        self.assertGreater(edge_long, edge_short * 2.0)
        self.assertGreater(edge_long, 0.01)

    def test_hdr_values_are_not_clipped(self):
        img = np.full((130, 180, 3), 1.2, dtype=np.float32)
        out = light_rays.apply_light_rays(
            img,
            [{"type": "line", "mode": "parallel", "p1": (0, 65), "p2": (180, 65)}],
            intensity=80,
            occlusion=0,
        )
        self.assertGreater(float(out.max()), 1.2)
        self.assertEqual(out.dtype, np.float32)


class LightRaysEffectIntegrationTest(unittest.TestCase):
    def test_make_diff_converts_tcg_guides_and_renders(self):
        img = _base(h=128, w=128)
        eff = effects.LightRaysEffect()
        param = {
            "original_img_size": (128, 128),
            "disp_info": (0, 0, 1.0, 1.0, 1.0),
            "switch_light_rays": True,
            "light_ray_guides": [
                {"type": "line", "mode": "parallel", "p1": (-0.5, 0.0), "p2": (0.5, 0.0)}
            ],
            "light_ray_intensity": 80,
            "light_ray_occlusion": 0,
            "light_ray_count": 6,
        }
        efconfig = SimpleNamespace(resolution_scale=1.0, upstream_hash=0, disp_info=None)
        out = eff.make_diff(img, param, efconfig)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(float((out - img).max()), 0.0)

    def test_make_diff_uses_per_guide_intensity_when_global_is_zero(self):
        img = _base(h=128, w=128)
        eff = effects.LightRaysEffect()
        param = {
            "original_img_size": (128, 128),
            "disp_info": (0, 0, 1.0, 1.0, 1.0),
            "switch_light_rays": True,
            "light_ray_intensity": 0,
            "light_ray_guides": [
                {
                    "type": "line",
                    "mode": "parallel",
                    "p1": (-0.5, 0.0),
                    "p2": (0.5, 0.0),
                    "params": {"light_ray_intensity": 80},
                }
            ],
            "light_ray_occlusion": 0,
        }
        efconfig = SimpleNamespace(resolution_scale=1.0, upstream_hash=0, disp_info=None)
        out = eff.make_diff(img, param, efconfig)
        self.assertIsNotNone(out)
        self.assertGreater(float((out - img).max()), 0.0)

    def test_make_diff_keeps_width_consistent_when_preview_has_letterbox_padding(self):
        preview = _base(h=200, w=200)
        full = _base(h=300, w=400)
        param = {
            "original_img_size": (400, 300),
            "disp_info": (0, 0, 1.0, 0.75, 0.5),
            "switch_light_rays": True,
            "light_ray_guides": [
                {"type": "line", "mode": "parallel", "p1": (-0.6, 0.0), "p2": (0.6, 0.0)}
            ],
            "light_ray_intensity": 80,
            "light_ray_width": 42,
            "light_ray_softness": 50,
            "light_ray_count": 1,
            "light_ray_density": 0,
            "light_ray_variation": 0,
            "light_ray_occlusion": 0,
            "light_ray_seed": 21,
        }
        preview_out = effects.LightRaysEffect().make_diff(
            preview,
            param.copy(),
            SimpleNamespace(resolution_scale=0.5, upstream_hash=1, disp_info=(0, 0, 400, 300, 0.5)),
        )
        full_out = effects.LightRaysEffect().make_diff(
            full,
            param.copy(),
            SimpleNamespace(resolution_scale=1.0, upstream_hash=2, disp_info=(0, 0, 400, 300, 1.0)),
        )
        preview_profile = (preview_out - preview)[25:175, :, :].mean(axis=(1, 2))
        full_profile = (full_out - full).reshape(150, 2, 200, 2, 3).mean(axis=(1, 2, 3, 4))

        def half_width(profile):
            active = np.where(profile > float(profile.max()) * 0.5)[0]
            return int(active[-1] - active[0] + 1)

        self.assertLessEqual(abs(half_width(preview_profile) - half_width(full_profile)), 3)

    def test_make_diff_scales_light_rays_with_zoomed_view(self):
        img = _base(h=200, w=200)
        param = {
            "original_img_size": (400, 300),
            "disp_info": (0, 0, 400, 300, 0.5),
            "switch_light_rays": True,
            "light_ray_guides": [
                {"type": "line", "mode": "parallel", "p1": (-0.6, 0.0), "p2": (0.6, 0.0)}
            ],
            "light_ray_intensity": 80,
            "light_ray_width": 42,
            "light_ray_softness": 50,
            "light_ray_count": 1,
            "light_ray_density": 0,
            "light_ray_variation": 0,
            "light_ray_occlusion": 0,
            "light_ray_seed": 21,
        }

        unzoomed = effects.LightRaysEffect().make_diff(
            img,
            param.copy(),
            SimpleNamespace(resolution_scale=1.0, upstream_hash=1, disp_info=(0, 0, 400, 300, 0.5)),
        )
        zoomed = effects.LightRaysEffect().make_diff(
            img,
            param.copy(),
            SimpleNamespace(resolution_scale=1.0, upstream_hash=2, disp_info=(100, 50, 200, 150, 1.0)),
        )

        def half_width(out):
            profile = (out - img).mean(axis=2).mean(axis=1)
            active = np.where(profile > float(profile.max()) * 0.5)[0]
            return int(active[-1] - active[0] + 1)

        self.assertGreater(half_width(zoomed), half_width(unzoomed) * 1.35)

    def test_create_effects_registers_light_rays_in_lv2(self):
        effect_sets = effects.create_effects()
        self.assertIn("light_rays", effect_sets[2])

    def test_editor_sync_uses_toggle_state_not_identity_comparison(self):
        source = (PROJECT_ROOT / "main.py").read_text()
        start = source.index("        def _sync_effect_editors_impl(self):")
        end = source.index("        def is_liquify_editor_active(self):")
        body = source[start:end]
        self.assertIn("getattr(light_rays_btn, 'state', 'normal') == 'down'", body)
        self.assertNotIn("active is light_rays_btn", body)

    def test_light_rays_canvas_grabs_touch_for_drag_creation(self):
        source = (PROJECT_ROOT / "widgets" / "light_rays_canvas.py").read_text()
        self.assertIn("touch.grab(self)", source)
        self.assertIn("if touch.grab_current is not self:", source)
        self.assertIn("touch.ungrab(self)", source)

    def test_light_rays_canvas_copies_creation_params_to_new_guides(self):
        with (
            mock.patch("kivy.base.EventLoop.ensure_window"),
            mock.patch.object(LightRaysCanvas, "_refresh_markers"),
        ):
            canvas = LightRaysCanvas()
            canvas.set_creation_params({"light_ray_intensity": 77, "light_ray_width": 123})
            canvas.guides.append({
                "type": "point",
                "mode": "radial",
                "p": (0.0, 0.0),
                "params": canvas.creation_params.copy(),
            })
            params = canvas.get_guides()[0]["params"]
            self.assertEqual(params["light_ray_intensity"], 77.0)
            self.assertEqual(params["light_ray_width"], 123.0)

    def test_light_rays_canvas_keeps_projected_radial_p2_handle(self):
        with (
            mock.patch("kivy.base.EventLoop.ensure_window"),
            mock.patch.object(LightRaysCanvas, "_refresh_markers"),
        ):
            canvas = LightRaysCanvas(guides=[
                {"type": "point", "mode": "radial", "p": (0.0, 0.0), "p2": (0.2, 0.1)}
            ])
            parts = [part for part, _point in canvas._guide_points(canvas.guides[0])]
            self.assertEqual(parts, ["p", "p2"])
            self.assertEqual(canvas.get_guides()[0]["p2"], (0.2, 0.1))

    def test_line_editor_keeps_start_control_point_under_user_control(self):
        with (
            mock.patch("kivy.base.EventLoop.ensure_window"),
            mock.patch.object(LightRaysCanvas, "_refresh_markers"),
        ):
            canvas = LightRaysCanvas(guides=[
                {"type": "line", "mode": "parallel", "p1": (0.0, 0.0), "p2": (0.35, 0.0)}
            ])
            canvas.selected = 0
            canvas.selected_part = "p1"
            canvas._move_selected((0.12, 0.08))
            self.assertEqual(canvas.get_guides()[0]["p1"], (0.12, 0.08))

    def test_set2param_updates_only_selected_light_ray_guide_params(self):
        with (
            mock.patch("kivy.base.EventLoop.ensure_window"),
            mock.patch.object(LightRaysCanvas, "_refresh_markers"),
        ):
            eff = effects.LightRaysEffect()
            canvas = LightRaysCanvas(guides=[
                {"type": "point", "mode": "radial", "p": (0.0, 0.0), "params": {"light_ray_intensity": 10}},
                {"type": "point", "mode": "radial", "p": (0.2, 0.0), "params": {"light_ray_intensity": 20}},
            ])
            canvas.selected = 1
            eff.light_rays_canvas = canvas
            widget = _DummyWidget(editor_type="Point", editor_mode="Radial")
            widget.ids["slider_light_ray_intensity"].value = 88
            param = {
                "switch_light_rays": True,
                "light_ray_guides": canvas.get_guides(),
                "light_ray_editor_type": "Point",
                "light_ray_editor_mode": "Radial",
            }
            eff.set2param(param, widget)
            guides = param["light_ray_guides"]
            self.assertEqual(guides[0]["params"]["light_ray_intensity"], 10.0)
            self.assertEqual(guides[1]["params"]["light_ray_intensity"], 88.0)
            self.assertEqual(param["light_ray_selected"], 1)

    def test_set2param_preserves_projected_radial_projection_length(self):
        eff = effects.LightRaysEffect()
        widget = _DummyWidget(editor_type="Point", editor_mode="Radial")
        widget.ids["slider_light_ray_length"].value = 100
        param = {
            "switch_light_rays": True,
            "light_ray_guides": [{
                "type": "point",
                "mode": "radial",
                "p": (0.0, 0.0),
                "p2": (0.2, 0.0),
                "params": {
                    "light_ray_length": 30,
                    "light_ray_projection_length": 30,
                },
            }],
            "light_ray_selected": 0,
            "light_ray_editor_type": "Point",
            "light_ray_editor_mode": "Radial",
        }

        eff.set2param(param, widget)

        guide_params = param["light_ray_guides"][0]["params"]
        self.assertEqual(guide_params["light_ray_length"], 100.0)
        self.assertEqual(guide_params["light_ray_projection_length"], 30)

    def test_set2param_updates_last_selected_guide_when_editor_is_closed(self):
        eff = effects.LightRaysEffect()
        widget = _DummyWidget(editor_type="Point", editor_mode="Radial")
        widget.ids["slider_light_ray_intensity"].value = 88
        widget.ids["slider_light_ray_width"].value = 111
        param = {
            "switch_light_rays": True,
            "light_ray_guides": [
                {"type": "point", "mode": "radial", "p": (0.0, 0.0), "params": {"light_ray_intensity": 10}},
                {"type": "point", "mode": "radial", "p": (0.2, 0.0), "params": {"light_ray_intensity": 20}},
            ],
            "light_ray_selected": 1,
            "light_ray_editor_type": "Point",
            "light_ray_editor_mode": "Radial",
        }

        eff.set2param(param, widget)

        guides = param["light_ray_guides"]
        self.assertEqual(guides[0]["params"]["light_ray_intensity"], 10)
        self.assertEqual(guides[1]["params"]["light_ray_intensity"], 88.0)
        self.assertEqual(guides[1]["params"]["light_ray_width"], 111.0)
        self.assertEqual(param["light_ray_selected"], 1)

    def test_set2param_updates_first_light_ray_guide_without_selection(self):
        eff = effects.LightRaysEffect()
        widget = _DummyWidget(editor_type="Point", editor_mode="Radial")
        widget.ids["slider_light_ray_intensity"].value = 77
        param = {
            "switch_light_rays": True,
            "light_ray_guides": [
                {"type": "point", "mode": "radial", "p": (0.0, 0.0), "params": {"light_ray_intensity": 10}},
                {"type": "point", "mode": "radial", "p": (0.2, 0.0), "params": {"light_ray_intensity": 20}},
            ],
            "light_ray_selected": -1,
            "light_ray_editor_type": "Point",
            "light_ray_editor_mode": "Radial",
        }

        eff.set2param(param, widget)

        guides = param["light_ray_guides"]
        self.assertEqual(guides[0]["params"]["light_ray_intensity"], 77.0)
        self.assertEqual(guides[1]["params"]["light_ray_intensity"], 20)
        self.assertEqual(param["light_ray_selected"], 0)

    def test_canvas_active_params_use_first_light_ray_guide_without_selection(self):
        with (
            mock.patch("kivy.base.EventLoop.ensure_window"),
            mock.patch.object(LightRaysCanvas, "_refresh_markers"),
        ):
            canvas = LightRaysCanvas(guides=[
                {"type": "point", "mode": "radial", "p": (0.0, 0.0), "params": {"light_ray_intensity": 10}},
                {"type": "point", "mode": "radial", "p": (0.2, 0.0), "params": {"light_ray_intensity": 20}},
            ])
            changed = canvas.set_active_params({"light_ray_intensity": 66})

            guides = canvas.get_guides()
            self.assertTrue(changed)
            self.assertEqual(canvas.selected_index(), 0)
            self.assertEqual(guides[0]["params"]["light_ray_intensity"], 66.0)
            self.assertEqual(guides[1]["params"]["light_ray_intensity"], 20)

    def test_selecting_light_ray_guide_loads_its_params_into_widget(self):
        with (
            mock.patch("kivy.base.EventLoop.ensure_window"),
            mock.patch.object(LightRaysCanvas, "_refresh_markers"),
        ):
            eff = effects.LightRaysEffect()
            canvas = LightRaysCanvas(guides=[
                {
                    "type": "point",
                    "mode": "directional",
                    "p": (0.0, 0.0),
                    "p2": (0.2, 0.0),
                    "params": {"light_ray_intensity": 44, "light_ray_width": 99},
                }
            ])
            canvas.selected = 0
            eff.light_rays_canvas = canvas
            widget = _DummyWidget()
            param = {"light_ray_guides": canvas.get_guides()}
            self.assertTrue(eff.sync_selected_guide_to_widget(widget, param))
            self.assertEqual(widget.ids["spinner_light_ray_editor_type"].text, "Point")
            self.assertEqual(widget.ids["spinner_light_ray_editor_mode"].text, "Directional")
            self.assertEqual(widget.ids["slider_light_ray_intensity"].value, 44)
            self.assertEqual(widget.ids["slider_light_ray_width"].value, 99)

    def test_emission_change_updates_selected_or_single_existing_guide(self):
        with (
            mock.patch("kivy.base.EventLoop.ensure_window"),
            mock.patch.object(LightRaysCanvas, "_refresh_markers"),
        ):
            canvas = LightRaysCanvas(guides=[
                {"type": "line", "mode": "parallel", "p1": (0.0, 0.0), "p2": (0.3, 0.0)}
            ])
            changed = canvas.set_creation("Line", "Directional", apply_to_selected=True)
            self.assertTrue(changed)
            self.assertEqual(canvas.get_guides()[0]["mode"], "directional")

            canvas = LightRaysCanvas(guides=[
                {"type": "point", "mode": "radial", "p": (0.0, 0.0)}
            ])
            changed = canvas.set_creation("Point", "Directional", apply_to_selected=True)
            self.assertTrue(changed)
            guide = canvas.get_guides()[0]
            self.assertEqual(guide["mode"], "directional")
            self.assertIn("p2", guide)

    def test_light_ray_slider_labels_describe_effect(self):
        kv = (PROJECT_ROOT / "main.kv").read_text()
        for label in (
            'text: "Point Reach"',
            'text: "Shaft Width"',
            'text: "Edge Bias"',
            'text: "Shaft Count"',
            'text: "Texture Density"',
            'text: "Air Glow"',
            'text: "Tree Shadow"',
        ):
            self.assertIn(label, kv)
        self.assertIn("id: slider_light_ray_length", kv)
        self.assertIn("max: 200", kv)
        self.assertIn("id: slider_light_ray_width", kv)
        self.assertIn("id: slider_light_ray_edge_bias", kv)
        self.assertIn("id: slider_light_ray_count", kv)
        self.assertIn("max: 32", kv)
        self.assertIn("values: ['Parallel', 'Directional']", kv)

    def test_light_ray_defaults_are_clear_eight_beam_controls(self):
        defaults = effects.LightRaysEffect().get_param_dict({})
        self.assertEqual(defaults["light_ray_count"], 8)
        self.assertEqual(defaults["light_ray_density"], 10)
        self.assertEqual(defaults["light_ray_length"], 30)
        self.assertEqual(defaults["light_ray_decay"], 50)
        self.assertEqual(defaults["light_ray_width"], 65)
        self.assertEqual(defaults["light_ray_variation"], 45)
        self.assertEqual(defaults["light_ray_intensity"], 60)
        self.assertIn("light_ray_edge_bias", defaults)

    def test_light_ray_mode_spinner_has_two_point_modes(self):
        eff = effects.LightRaysEffect()
        widget = _DummyWidget()
        eff.set2widget(widget, {
            "switch_light_rays": True,
            "light_ray_editor_type": "Point",
            "light_ray_editor_mode": "Directional",
        })
        mode = widget.ids["spinner_light_ray_editor_mode"]
        self.assertEqual(mode.values, ["Radial", "Directional"])
        self.assertEqual(mode.text, "Directional")

    def test_light_ray_mode_spinner_has_two_line_modes(self):
        eff = effects.LightRaysEffect()
        widget = _DummyWidget(editor_type="Line", editor_mode="Radial")
        param = {}
        eff.set2param(param, widget)
        mode = widget.ids["spinner_light_ray_editor_mode"]
        self.assertEqual(mode.values, ["Parallel", "Directional"])
        self.assertEqual(mode.text, "Parallel")
        self.assertEqual(param["light_ray_editor_type"], "Line")
        self.assertEqual(param["light_ray_editor_mode"], "Parallel")


if __name__ == "__main__":
    unittest.main()
