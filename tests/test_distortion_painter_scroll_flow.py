import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DISTORTION_PAINTER_PATH = PROJECT_ROOT / "widgets" / "distortion_painter.py"
MAIN_PATH = PROJECT_ROOT / "main.py"
MAIN_KV_PATH = PROJECT_ROOT / "main.kv"
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
MASK_EDITOR2_PATH = PROJECT_ROOT / "widgets" / "mask_editor2.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


class DistortionPainterScrollFlowTest(unittest.TestCase):
    def test_scroll_changes_brush_size_before_recording_stroke(self):
        source_text = DISTORTION_PAINTER_PATH.read_text()
        node = _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_down")
        source = ast.get_source_segment(source_text, node)

        self.assertIn("touch.is_mouse_scrolling", source)
        self.assertIn("self._adjust_brush_size_from_scroll(touch)", source)
        self.assertLess(source.index("touch.is_mouse_scrolling"), source.index("self._start_stroke(touch)"))
        self.assertNotIn("self.current_image is None", source)

    def test_liquify_overlay_tracks_preview_widget_bounds(self):
        source_text = DISTORTION_PAINTER_PATH.read_text()
        parent_source = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_parent_changed"),
        )
        sync_source = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "_sync_to_image_widget_bounds"),
        )

        self.assertIn("self._sync_to_image_widget_bounds()", parent_source)
        self.assertIn("parent.bind(pos=self._sync_to_image_widget_bounds, size=self._sync_to_image_widget_bounds)", parent_source)
        self.assertIn("self.pos = self.image_widget.pos", sync_source)
        self.assertIn("self.size = self.image_widget.size", sync_source)

    def test_scroll_does_not_close_distortion_history(self):
        source_text = DISTORTION_PAINTER_PATH.read_text()
        move = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_move"),
        )
        up = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_up"),
        )

        self.assertIn("self._paint_touch_uid != touch.uid", move)
        self.assertIn("self._paint_touch_uid != touch.uid", up)
        self.assertIn("self.callback('end', self)", up)

    def test_liquify_move_flushes_buffer_before_pipeline_callback(self):
        source_text = DISTORTION_PAINTER_PATH.read_text()
        move = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_move"),
        )
        delayed_source = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "delayed_texture_update"),
        )

        self.assertLess(move.index("self.points_buffer.append(record)"), move.index("self.process_buffer()"))
        self.assertLess(move.index("self.process_buffer()"), move.index("self.callback('apply', self)"))
        self.assertNotIn("self.callback('apply', self)", delayed_source)

    def test_liquify_start_point_is_persisted_before_initial_resync_can_wipe_it(self):
        source_text = DISTORTION_PAINTER_PATH.read_text()
        start_source = ast.get_source_segment(
            source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "_start_stroke"),
        )

        self.assertLess(start_source.index("self.callback('start', self)"), start_source.index("self.recorded.append(record)"))
        self.assertLess(start_source.index("self.recorded.append(record)"), start_source.rindex("self.callback('apply', self)"))
        self.assertLess(start_source.index("self.points_buffer.append(record)"), start_source.rindex("self.callback('apply', self)"))

    def test_liquify_revision_drives_make_diff_hash(self):
        painter_source = DISTORTION_PAINTER_PATH.read_text()
        init_source = ast.get_source_segment(
            painter_source,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "__init__"),
        )
        process_source = ast.get_source_segment(
            painter_source,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "process_buffer"),
        )
        revision_source = ast.get_source_segment(
            painter_source,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "get_live_revision"),
        )
        effects_source = EFFECTS_PATH.read_text()
        make_diff_source = ast.get_source_segment(
            effects_source,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "make_diff"),
        )

        self.assertIn("self.live_revision = 0", init_source)
        self.assertIn("self.live_revision += 1", process_source)
        self.assertIn("return self.live_revision", revision_source)
        self.assertIn("get_live_revision", make_diff_source)
        self.assertIn("id(self.diff)", make_diff_source)

    def test_lv1_hash_change_forces_downstream_reset(self):
        pipeline_source = (PROJECT_ROOT / "pipeline.py").read_text()
        start = pipeline_source.index("def pipeline_lv1(")
        end = pipeline_source.index("def pipeline_lv2(", start)
        source = pipeline_source[start:end]

        self.assertIn('pre_hash = getattr(lv1[n], "hash", None)', source)
        self.assertIn('pre_hash != getattr(lv1[n], "hash", None)', source)
        self.assertIn("lv2reset = True", source)

    def test_brush_size_callback_updates_ui_without_history(self):
        source_text = MAIN_PATH.read_text()
        callback_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "distortion_callback"),
        )
        target_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_distortion_callback_target"),
        )
        brush_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_sync_distortion_brush_from_painter"),
        )

        self.assertIn("case 'start'", callback_source)
        self.assertIn("current_param['switch_distortion'] = True", callback_source)
        self.assertIn('switch = self.ids.get("switch_distortion")', callback_source)
        self.assertIn("switch.enabled = True", callback_source)
        self.assertIn("case 'brush_size'", callback_source)
        self.assertIn("self._sync_distortion_brush_from_painter(widget, current_param)", callback_source)
        self.assertIn("effect, current_param, owner_is_active = self._get_distortion_effect_and_param_for_painter(widget)", target_source)
        self.assertIn("if not owner_is_active:", target_source)
        self.assertIn("current_param['distortion_brush_size'] = widget.brush_size", brush_source)
        self.assertIn('slider = self.ids.get("slider_distortion_brush_size")', brush_source)
        self.assertIn("slider.set_slider_value(widget.brush_size)", brush_source)
        self.assertIn("self._remember_distortion_tool_values_from_widgets(current_param)", brush_source)
        self.assertNotIn('getattr(slider, "on_slider_value", None)', brush_source)
        self.assertNotIn("sync_value()", brush_source)

    def test_liquify_invalidates_active_composit_render_cache(self):
        source_text = MAIN_PATH.read_text()
        callback_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "distortion_callback"),
        )
        helper_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_request_active_liquify_mask_render_update"),
        )
        apply_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_apply_distortion_painter_params"),
        )

        self.assertIn("if self._is_mask2_enabled():", callback_source)
        self.assertIn("self._request_active_liquify_mask_render_update(redraw_pipeline=True)", callback_source)
        self.assertIn("else:", callback_source)
        self.assertIn("self.apply_effects_lv(1, 'distortion')", callback_source)
        self.assertIn("self._apply_distortion_painter_params(widget, effect, current_param)", callback_source)
        self.assertIn("current_param.update(widget.get_distortion_params())", apply_source)
        self.assertNotIn("effect.reeffect()", apply_source)
        self.assertLess(
            callback_source.rindex("self._apply_distortion_painter_params(widget, effect, current_param)"),
            callback_source.rindex("self._request_active_liquify_mask_render_update(redraw_pipeline=True)"),
        )
        self.assertLess(
            callback_source.index("self._apply_distortion_painter_params(widget, effect, current_param)"),
            callback_source.index("self.apply_effects_lv(1, 'distortion')"),
        )
        self.assertIn("editor.find_composit_mask(mask)", helper_source)
        self.assertIn("request_mask_render_update", helper_source)
        self.assertIn('reason="liquify"', helper_source)
        self.assertIn("refresh_visibility=False", helper_source)
        self.assertIn("redraw_overlay=False", helper_source)
        self.assertIn("redraw_pipeline=False", helper_source)
        self.assertIn("if redraw_pipeline:", helper_source)
        self.assertIn("self.start_draw_image(fast_display=False)", helper_source)

    def test_distortion_preview_touch_clears_text_focus(self):
        painter_source_text = DISTORTION_PAINTER_PATH.read_text()
        painter_source = ast.get_source_segment(
            painter_source_text,
            _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "on_touch_down"),
        )
        main_source_text = MAIN_PATH.read_text()
        callback_source = ast.get_source_segment(
            main_source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "distortion_callback"),
        )
        preview_touch_source = ast.get_source_segment(
            main_source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "on_image_touch_down"),
        )

        self.assertIn("self.callback('focus', self)", painter_source)
        self.assertLess(painter_source.index("self.callback('focus', self)"), painter_source.index("self._adjust_brush_size_from_scroll(touch)"))
        self.assertIn("case 'focus'", callback_source)
        self.assertIn("self._clear_text_input_focus()", callback_source)
        self.assertIn("self._clear_text_input_focus()", preview_touch_source)

    def test_liquify_painter_resyncs_on_preview_geometry_change(self):
        source_text = EFFECTS_PATH.read_text()
        make_key_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "_make_painter_ref_key"),
        )
        sync_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "_sync_distortion_painter_ref"),
        )
        view_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "_view_param"),
        )
        close_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "_close_distortion_painter"),
        )
        make_diff_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "make_diff"),
        )

        self.assertIn("view_param = self._view_param(param, efconfig=efconfig)", make_key_source)
        self.assertIn("params.get_disp_info(view_param)", make_key_source)
        self.assertIn("getattr(efconfig, 'upstream_hash', None)", make_key_source)
        self.assertIn("param.get('disp_info') is not None", view_source)
        self.assertIn("getattr(widget, 'primary_param', None)", view_source)
        self.assertIn("base = self._provider_view_param()", view_source)
        self.assertIn("getattr(efconfig, 'disp_info', None)", view_source)
        self.assertIn("params.set_disp_info(view_param, disp_info)", view_source)
        self.assertIn("distortion_painter = self.distortion_painter", sync_source)
        self.assertIn("if distortion_painter is None:", sync_source)
        self.assertIn("ref_key == self._painter_ref_key", sync_source)
        self.assertIn("self._view_param(param, efconfig=efconfig)", sync_source)
        self.assertIn("distortion_painter.set_ref_image(img, True)", sync_source)
        self.assertIn("if self.distortion_painter is distortion_painter:", sync_source)
        self.assertIn("self.diff = None", sync_source)
        self.assertIn("self.hash = None", sync_source)
        self.assertIn("self.diff = None", close_source)
        self.assertIn("self.hash = None", close_source)
        self.assertIn("self._sync_distortion_painter_ref(img, param, efconfig)", make_diff_source)
        self.assertIn("self._painter_ref_key", make_diff_source)
        open_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "_open_distortion_painter"),
        )
        self.assertIn("self.distortion_painter.is_recording = True", open_source)
        self.assertIn('add_widget(self.distortion_painter, index=0)', open_source)
        self.assertIn("self._bring_distortion_painter_to_front(widget)", open_source)

    def test_liquify_painter_is_forced_to_front_when_reopened(self):
        source_text = EFFECTS_PATH.read_text()
        front_source = ast.get_source_segment(
            source_text,
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "_bring_distortion_painter_to_front"),
        )

        self.assertIn("preview_widget.children[0] is painter", front_source)
        self.assertIn("preview_widget.remove_widget(painter)", front_source)
        self.assertIn("preview_widget.add_widget(painter, index=0)", front_source)

    def test_liquify_reset_button_calls_painter_directly(self):
        kv_source = MAIN_KV_PATH.read_text()
        main_source_text = MAIN_PATH.read_text()
        reset_source = ast.get_source_segment(
            main_source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "reset_distortion_painter_action"),
        )
        set2param_source = ast.get_source_segment(
            EFFECTS_PATH.read_text(),
            _load_class_function(EFFECTS_PATH, "DistortionEffect", "after_set2param"),
        )

        self.assertIn("id: button_distortion_reset", kv_source)
        self.assertIn("on_press: root.reset_distortion_painter_action()", kv_source)
        self.assertIn("painter.reset_image(notify=False)", reset_source)
        self.assertNotIn('button_distortion_reset"].state == "down"', set2param_source)

    def test_liquify_mask2_uses_active_distortion_target_not_primary_only(self):
        source_text = MAIN_PATH.read_text()
        callback_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "distortion_callback"),
        )
        reset_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "reset_distortion_painter_action"),
        )
        helper_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_get_active_distortion_effect_and_param"),
        )
        can_open_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "can_open_liquify_editor"),
        )
        close_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_close_inactive_distortion_painters"),
        )
        apply_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "apply_effects_lv"),
        )
        set_effect_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "set_effect_param"),
        )
        policy_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_mask_overlay_policy"),
        )
        active_effects_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_get_active_effects"),
        )
        disable_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_disable_mask2"),
        )
        target_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_distortion_callback_target"),
        )
        apply_params_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_apply_distortion_painter_params"),
        )
        brush_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_sync_distortion_brush_from_painter"),
        )

        self.assertIn("if mask_id is None and not self._is_mask2_enabled():", active_effects_source)
        self.assertIn("return (self.primary_effects, self.primary_param, None)", active_effects_source)
        self.assertIn("if not self._is_mask2_enabled():", helper_source)
        self.assertIn("return effect, self.primary_param", helper_source)
        self.assertIn("self._get_active_effects(", helper_source)
        self.assertIn("lv=1", helper_source)
        self.assertIn("subname='distortion'", helper_source)
        self.assertIn("active = editor.get_active_mask()", can_open_source)
        self.assertIn("active is None or not active.is_composit()", can_open_source)
        self.assertIn("self._distortion_callback_target(proc, widget)", callback_source)
        self.assertIn("self._get_distortion_effect_and_param_for_painter(widget)", target_source)
        self.assertIn("if not owner_is_active:", target_source)
        self.assertIn("self._close_inactive_distortion_painters(active_effect)", target_source)
        self.assertIn("current_param.update(widget.get_distortion_params())", apply_params_source)
        self.assertIn("current_param['distortion_brush_size'] = widget.brush_size", brush_source)
        self.assertNotIn("self.primary_param.update(widget.get_distortion_params())", callback_source)
        self.assertIn("if not self.can_open_liquify_editor():", reset_source)
        self.assertIn("effect, current_param = self._get_active_distortion_effect_and_param()", reset_source)
        self.assertIn("painter.reset_image(notify=False)", reset_source)
        self.assertIn("current_param['distortion_recorded'] = []", reset_source)
        self.assertIn("effect.reeffect()", reset_source)
        self.assertIn("self._request_active_liquify_mask_render_update(redraw_pipeline=True)", reset_source)
        self.assertNotIn("self.primary_effects[1].get('distortion')", reset_source)
        self.assertIn("effect._close_distortion_painter(param, self)", close_source)
        self.assertIn("self._close_inactive_distortion_painters(primary_distortion)", disable_source)
        self.assertIn("self._close_inactive_distortion_painters(None)", apply_source)
        self.assertIn("self._close_inactive_distortion_painters(current_effects[lv][effect])", set_effect_source)
        self.assertIn('lv == 1 and mask2_group == "distortion"', policy_source)
        self.assertIn('return "preserve"', policy_source)

    def test_liquify_callback_uses_painter_owner_and_ignores_inactive_painters(self):
        source_text = MAIN_PATH.read_text()
        callback_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "distortion_callback"),
        )
        target_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_distortion_callback_target"),
        )
        helper_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_get_distortion_effect_and_param_for_painter"),
        )

        self.assertIn("self._distortion_callback_target(proc, widget)", callback_source)
        self.assertIn("proc != 'focus'", target_source)
        self.assertIn("self._get_distortion_effect_and_param_for_painter(widget)", target_source)
        self.assertIn("if not owner_is_active:", target_source)
        self.assertIn("return", callback_source)
        self.assertIn("active_effect, active_param = self._get_active_distortion_effect_and_param()", helper_source)
        self.assertIn("for effect, param in self._iter_distortion_effect_targets():", helper_source)
        self.assertIn("getattr(effect, 'distortion_painter', None) is painter", helper_source)
        self.assertIn("effect is active_effect and param is active_param", helper_source)

    def test_mask2_effects_receive_liquify_callback(self):
        mask_source_text = MASK_EDITOR2_PATH.read_text()
        init_source = ast.get_source_segment(
            mask_source_text,
            _load_class_function(MASK_EDITOR2_PATH, "BaseMask", "__init__"),
        )
        factory_source = ast.get_source_segment(
            mask_source_text,
            _load_class_function(MASK_EDITOR2_PATH, "BaseMask", "_create_effects"),
        )

        self.assertIn("self.effects = self._create_effects()", init_source)
        self.assertIn("root = getattr(self.editor, 'root', None)", factory_source)
        self.assertIn("distortion_callback = getattr(root, 'distortion_callback', None)", factory_source)
        self.assertIn("view_param_provider = getattr(self.editor, 'get_effect_view_param', None)", factory_source)
        self.assertIn("effects.create_effects(", factory_source)
        self.assertIn("distortion_callback=distortion_callback if callable(distortion_callback) else None", factory_source)
        self.assertIn("view_param_provider=view_param_provider if callable(view_param_provider) else None", factory_source)

    def test_mask2_effect_view_param_is_reconstructed_from_editor_context(self):
        mask_source_text = MASK_EDITOR2_PATH.read_text()
        provider_source = ast.get_source_segment(
            mask_source_text,
            _load_class_function(MASK_EDITOR2_PATH, "MaskEditor2", "get_effect_view_param"),
        )
        effects_source_text = EFFECTS_PATH.read_text()
        create_source = ast.get_source_segment(
            effects_source_text,
            next(
                node for node in ast.parse(effects_source_text).body
                if isinstance(node, ast.FunctionDef) and node.name == "create_effects"
            ),
        )

        self.assertIn("with self._matrix_lock:", provider_source)
        self.assertIn("self._image_only_matrix if self._image_only_matrix is not None", provider_source)
        self.assertIn("'rotation': math.degrees(float(tcg_info.get('rotation', 0.0)))", provider_source)
        self.assertIn("'rotation2': math.degrees(float(tcg_info.get('rotation2', 0.0)))", provider_source)
        self.assertIn("'matrix': np.array(matrix, dtype=np.float64, copy=True)", provider_source)
        self.assertIn("view_param['img_size'] = view_param['original_img_size']", provider_source)
        self.assertIn("view_param_provider=None", create_source)
        self.assertIn("view_param_provider=view_param_provider", create_source)

    def test_liquify_mask2_locks_mask_editor_input_and_switches_target_on_selection(self):
        mask_source_text = MASK_EDITOR2_PATH.read_text()
        lock_source = ast.get_source_segment(
            mask_source_text,
            _load_class_function(MASK_EDITOR2_PATH, "MaskEditor2", "_liquify_editor_locks_input"),
        )
        down_source = ast.get_source_segment(
            mask_source_text,
            _load_class_function(MASK_EDITOR2_PATH, "MaskEditor2", "on_touch_down"),
        )
        active_source = ast.get_source_segment(
            mask_source_text,
            _load_class_function(MASK_EDITOR2_PATH, "MaskEditor2", "set_active_mask"),
        )

        self.assertIn("is_liquify_editor_active", lock_source)
        self.assertNotIn("_liquify_input_disabled", lock_source)
        self.assertIn("self._liquify_editor_locks_input()", down_source)
        self.assertIn('getattr(current_tab, "text", None) == "Li"', active_source)
        self.assertIn("self.root.apply_effects_lv(", active_source)
        self.assertIn('"distortion"', active_source)
        self.assertIn("defer_draw=True", active_source)

    def test_liquify_input_lock_does_not_use_mask_editor_disabled_property(self):
        main_source_text = MAIN_PATH.read_text()
        sync_source = ast.get_source_segment(
            main_source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_sync_liquify_mask_editor_input_lock"),
        )
        apply_source = ast.get_source_segment(
            main_source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "apply_effects_lv"),
        )

        self.assertIn("self._get_active_distortion_painter() is not None", sync_source)
        self.assertIn("self._unmount_mask_editor2_from_preview()", sync_source)
        self.assertIn("self._mount_mask_editor2_to_preview()", sync_source)
        self.assertNotIn(".disabled =", sync_source)
        self.assertNotIn("set_liquify_input_disabled", sync_source)
        self.assertIn("self._sync_liquify_mask_editor_input_lock()", apply_source)

    def test_mask_editor2_is_unmounted_from_preview_when_inactive_or_liquify_locked(self):
        source_text = MAIN_PATH.read_text()
        post_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "on_kv_post"),
        )
        enable_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_enable_mask2"),
        )
        disable_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_disable_mask2"),
        )
        mount_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_mount_mask_editor2_to_preview"),
        )
        unmount_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_unmount_mask_editor2_from_preview"),
        )

        self.assertIn("self._unmount_mask_editor2_from_preview()", post_source)
        self.assertIn("self._mount_mask_editor2_to_preview()", enable_source)
        self.assertIn("self._unmount_mask_editor2_from_preview()", disable_source)
        self.assertIn('getattr(current_tab, "text", None) == "Li"', disable_source)
        self.assertIn("self.apply_effects_lv(", disable_source)
        self.assertIn('"distortion"', disable_source)
        self.assertIn("defer_draw=True", disable_source)
        self.assertIn("preview.add_widget(editor, index=0)", mount_source)
        self.assertIn("preview.remove_widget(editor)", unmount_source)

    def test_liquify_brush_tool_values_survive_mask_target_widget_sync(self):
        source_text = MAIN_PATH.read_text()
        init_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "__init__"),
        )
        set2widget_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "set2widget_all"),
        )
        apply_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "apply_effects_lv"),
        )
        restore_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_restore_distortion_tool_values_to_widgets"),
        )
        remember_source = ast.get_source_segment(
            source_text,
            _load_class_function(MAIN_PATH, "MainWidget", "_remember_distortion_tool_values_from_widgets"),
        )

        self.assertIn("self._sticky_distortion_tool_values", init_source)
        self.assertIn("'distortion_brush_size': None", init_source)
        self.assertIn("'distortion_strength': None", init_source)
        self.assertIn("self._has_distortion_effect(_effects)", set2widget_source)
        self.assertIn("self._restore_distortion_tool_values_to_widgets(param)", set2widget_source)
        self.assertIn("self._remember_distortion_tool_values_from_widgets(current_param)", apply_source)
        self.assertIn("sticky[key] = value", remember_source)
        self.assertIn("param[key] = value", restore_source)
        self.assertIn("setter(value)", restore_source)


if __name__ == "__main__":
    unittest.main()
