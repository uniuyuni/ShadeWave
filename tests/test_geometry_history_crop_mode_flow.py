import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
PIPELINE_PATH = PROJECT_ROOT / "pipeline.py"
MAIN_PATH_TEXT = MAIN_PATH.read_text()
CORE_PATH = PROJECT_ROOT / "cores" / "core.py"
PARAMS_PATH = PROJECT_ROOT / "params.py"
MASK_EDITOR2_PATH = PROJECT_ROOT / "widgets" / "mask_editor2.py"


def _load_function(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} was not found")


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


def _load_class(path, class_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} was not found")


def _node_source(path, node):
    return ast.get_source_segment(path.read_text(), node)


def _attribute_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _assigned_sources(path, function_node, target_name):
    sources = []
    for node in ast.walk(function_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(_attribute_name(target) == target_name for target in targets):
            sources.append(_node_source(path, node.value))
    return sources


class GeometryHistoryCropModeFlowTest(unittest.TestCase):
    def test_history_redraw_syncs_crop_mode_before_rebuilding_crop_image(self):
        for function_name in ("_undo", "_redo", "_on_history_selected"):
            node = _load_function(MAIN_PATH, function_name)
            call_names = [
                call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            ]

            self.assertIn("_sync_editor_modes_after_history", call_names)
            self.assertIn("start_draw_image_and_crop", call_names)

        source = MAIN_PATH.read_text()
        self.assertIn(
            "self.primary_effects[0]['crop'].sync_crop_editor_mode_from_widget(self, self.primary_param)",
            source,
        )

    def test_crop_editor_mode_sync_does_not_write_crop_enable(self):
        sync = _load_class_function(EFFECTS_PATH, "CropEffect", "sync_crop_editor_mode_from_widget")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), sync)

        self.assertIn('widget.ids["effects"].current_tab.text == "Ge"', source)
        self.assertIn("self._open_crop_editor(param, widget)", source)
        self.assertIn("self._close_crop_editor(param, widget)", source)
        self.assertIn("self.sync_crop_editor_from_param(param)", source)
        self.assertNotIn("crop_enable", source)

    def test_crop_enable_is_not_saved_by_geometry_crop_or_vignette_effects(self):
        for class_name in ("GeometryEffect", "CropEffect", "VignetteEffect"):
            class_node = _load_class(EFFECTS_PATH, class_name)
            source = ast.get_source_segment(EFFECTS_PATH.read_text(), class_node)
            self.assertNotIn("crop_enable", source)

    def test_preview_crop_editing_is_runtime_pipeline_state(self):
        config_state = _load_function(PIPELINE_PATH, "_configure_preview_effect_config")
        source = ast.get_source_segment(PIPELINE_PATH.read_text(), config_state)
        crop_editing_sources = _assigned_sources(PIPELINE_PATH, config_state, "efconfig.crop_editing")

        self.assertIn('efconfig.current_tab = current_tab', source)
        self.assertIn('is_geometry_tab = current_tab == "Ge"', source)
        self.assertTrue(any("is_geometry_tab" in value for value in crop_editing_sources))
        self.assertNotIn("crop_enable", "".join(crop_editing_sources))

    def test_export_never_uses_geometry_editing_mode(self):
        export_pipeline = _load_function(PIPELINE_PATH, "export_pipeline")
        source = ast.get_source_segment(PIPELINE_PATH.read_text(), export_pipeline)

        self.assertIn("efconfig.crop_editing = False", source)

    def test_crop_dependent_effects_read_runtime_crop_editing(self):
        for class_name in ("GeometryEffect", "CropEffect", "VignetteEffect"):
            make_diff = _load_class_function(EFFECTS_PATH, class_name, "make_diff")
            source = ast.get_source_segment(EFFECTS_PATH.read_text(), make_diff)
            self.assertIn("getattr(efconfig, 'crop_editing', False)", source)
            self.assertNotIn("'crop_enable'", source)

    def test_zero_wrap_uses_runtime_crop_editing_not_param_crop_enable(self):
        apply_zero_wrap = _load_function(CORE_PATH, "apply_zero_wrap")
        source = ast.get_source_segment(CORE_PATH.read_text(), apply_zero_wrap)

        self.assertIn("crop_editing=False", source)
        self.assertIn("if not crop_editing:", source)
        self.assertNotIn("crop_enable", source)
        # クロップ編集中は param に格納されたコンテンツ四辺形からマスクを作る。
        self.assertIn("_zero_wrap_content_quad", source)
        self.assertIn("content_quad_mask", source)
        draw_image_core = _load_function(MAIN_PATH, "draw_image_core")
        draw_source = ast.get_source_segment(MAIN_PATH.read_text(), draw_image_core)
        self.assertIn('crop_editing = current_tab == "Ge"', draw_source)
        self.assertIn("crop_editing=crop_editing", draw_source)

    def test_mask_overlay_is_clipped_to_zero_wrap_image_area(self):
        draw_mask_image = _load_class_function(MASK_EDITOR2_PATH, "MaskEditor2", "draw_mask_image")
        draw_source = ast.get_source_segment(MASK_EDITOR2_PATH.read_text(), draw_mask_image)
        clip_overlay = _load_class_function(MASK_EDITOR2_PATH, "MaskEditor2", "_clip_mask_overlay_to_image_area")
        clip_source = ast.get_source_segment(MASK_EDITOR2_PATH.read_text(), clip_overlay)

        self.assertIn("_clip_mask_overlay_to_image_area(glayimg, disp_info)", draw_source)
        self.assertIn("core.crop_size_and_offset_from_texture", clip_source)
        self.assertIn("np.zeros_like(glayimg)", clip_source)
        self.assertNotIn("control_points", clip_source)

    def test_crop_enable_is_not_copied_into_history_runtime_special(self):
        source = PARAMS_PATH.read_text()

        self.assertIn("'crop_enable'", source)
        self.assertIn("DO_NOT_COPY_SPECIAL_PARAM", source)
        self.assertIn("if key in DO_NOT_COPY_SPECIAL_PARAM:", source)

    def test_geometry_history_captures_crop_state_changed_by_rotation_redraw(self):
        get_param_dict = _load_class_function(EFFECTS_PATH, "GeometryEffect", "get_param_dict")
        source = ast.get_source_segment(EFFECTS_PATH.read_text(), get_param_dict)

        self.assertIn("default_param['crop_rect']", source)
        self.assertIn("default_param['disp_info']", source)

    def test_zero_wrap_content_quad_runtime_special_and_clear(self):
        # ランタイム専用キーとして保存対象外であること（四辺形 + 変換キャンバス一辺）。
        params_text = PARAMS_PATH.read_text()
        self.assertIn("'_zero_wrap_content_quad'", params_text)
        self.assertIn("'_zero_wrap_canvas_size'", params_text)
        # GeometryEffect は full-preview 時に四辺形と canvas_size を格納する（通常表示はクリア）。
        store = _load_class_function(EFFECTS_PATH, "GeometryEffect", "_store_zero_wrap_quad")
        store_source = ast.get_source_segment(EFFECTS_PATH.read_text(), store)
        self.assertIn("_zero_wrap_content_quad", store_source)
        self.assertIn("_zero_wrap_canvas_size", store_source)
        self.assertIn("content_quad_norm", store_source)


class ZeroWrapQuadMathTest(unittest.TestCase):
    """フル依存環境でのみ実行する apply_zero_wrap / クォッド算出の動作テスト。"""

    def setUp(self):
        try:
            import numpy as np  # noqa: F401
            import cores.core as core  # noqa: F401
        except Exception as exc:  # numba 等が無い最小環境ではスキップ
            self.skipTest(f"cores.core unavailable: {exc}")

    def test_content_quad_and_zero_wrap_padding(self):
        import numpy as np
        import cores.core as core

        # 横長 300x150、回転 0 度。canvas=300、有効コンテンツ=300*150。
        h, w = 150, 300
        matrix, size = core.rotation_canvas_matrix((h, w), 0)
        quad = core.content_quad_norm((h, w), matrix, size, "affine")
        param = {"_zero_wrap_content_quad": quad.tolist()}
        img = np.ones((size, size, 3), dtype=np.float32)

        out, zero_count = core.apply_zero_wrap(img, param, crop_editing=True)
        expected_zero = size * size - w * h
        # fillConvexPoly の境界包含で数画素ずれるため許容誤差付き。
        self.assertLess(abs(zero_count - expected_zero), size * 3)
        # パディング画素は強制 0 化される。
        self.assertTrue(np.all(out[0, 0] == 0.0))

    def test_zero_rotation_square_has_no_padding(self):
        import numpy as np
        import cores.core as core

        matrix, size = core.rotation_canvas_matrix((200, 200), 0)
        quad = core.content_quad_norm((200, 200), matrix, size, "affine")
        param = {"_zero_wrap_content_quad": quad.tolist()}
        img = np.ones((size, size, 3), dtype=np.float32)
        out, zero_count = core.apply_zero_wrap(img, param, crop_editing=True)
        self.assertLess(zero_count, size * 3)

    def test_missing_quad_falls_back_without_masking(self):
        import numpy as np
        import cores.core as core

        # quad 未設定 + crop_editing=True → 旧挙動（乗算なし）。
        param = {"_zero_wrap_content_quad": None,
                 "original_img_size": (100, 100),
                 "disp_info": (0, 0, 100, 100, 1.0)}
        img = np.ones((100, 100, 3), dtype=np.float32)
        out, _ = core.apply_zero_wrap(img, param, crop_editing=True)
        self.assertTrue(np.all(out == 1.0))

    def test_non_crop_rotation_keeps_rectangular_mask_only(self):
        # 通常表示（crop_editing=False）はクロップ枠外の矩形黒塗りのみ。回転が効いていて
        # quad が（stale 等で）param に残っていても、枠内の reflect ミラー領域を斜めに
        # 削ってはならない（エクスポートはミラーのまま残るため、削るとプレビューと
        # エクスポートが不一致になり、後がけの Light Rays 等も斜めに切られる）。
        import numpy as np
        import cores.core as core
        import params

        h, w = 150, 300
        matrix, size = core.rotation_canvas_matrix((h, w), 30)
        quad = core.content_quad_norm((h, w), matrix, size, "affine")

        dx, dy, dw, dh = size * 0.15, size * 0.15, size * 0.7, size * 0.35
        tex_w, tex_h = 400, 200
        scale = tex_w / dw
        base = {"original_img_size": (size, size)}
        params.set_disp_info(base, (dx, dy, dw, dh, scale))

        img = np.ones((tex_h, tex_w, 3), dtype=np.float32)

        p_quad = dict(base)
        p_quad["_zero_wrap_content_quad"] = quad.tolist()
        p_quad["_zero_wrap_canvas_size"] = float(size)
        out_quad, zc_quad = core.apply_zero_wrap(img.copy(), p_quad, crop_editing=False)

        out_rect, zc_rect = core.apply_zero_wrap(img.copy(), dict(base), crop_editing=False)

        # quad の有無で通常表示の結果が変わらない（矩形マスクのみ）。
        self.assertTrue(np.array_equal(out_quad, out_rect))
        self.assertEqual(zc_quad, zc_rect)

    def test_store_zero_wrap_quad_cleared_when_not_full_preview(self):
        # 通常表示では quad をクリアする（quad は Ge タブ full-preview 専用。残すと
        # mask2 オーバーレイクリップが通常タブで菱形に誤クリップされる）。
        store = _load_class_function(EFFECTS_PATH, "GeometryEffect", "_store_zero_wrap_quad")
        store_source = ast.get_source_segment(EFFECTS_PATH.read_text(), store)
        self.assertIn("if not full_preview:", store_source)

    def test_non_crop_zero_rotation_matches_rectangular(self):
        # 回転なしのときは quad マスクが矩形フォールバックと一致（過剰マスクしない）。
        import numpy as np
        import cores.core as core
        import params

        h, w = 150, 300
        matrix, size = core.rotation_canvas_matrix((h, w), 0)
        quad = core.content_quad_norm((h, w), matrix, size, "affine")

        dx, dy, dw, dh = 0, (size - h) / 2, w, h  # コンテンツ矩形そのものを crop 窓に
        tex_w, tex_h = 400, 200
        scale = tex_w / dw
        base = {"original_img_size": (size, size)}
        params.set_disp_info(base, (dx, dy, dw, dh, scale))

        img = np.ones((tex_h, tex_w, 3), dtype=np.float32)

        p_quad = dict(base)
        p_quad["_zero_wrap_content_quad"] = quad.tolist()
        p_quad["_zero_wrap_canvas_size"] = float(size)
        out_quad, _ = core.apply_zero_wrap(img.copy(), p_quad, crop_editing=False)

        out_rect, _ = core.apply_zero_wrap(img.copy(), dict(base), crop_editing=False)

        self.assertLess(float(np.mean(np.abs(out_quad - out_rect))), 0.02)

    def test_non_crop_rotation_never_cuts_inside_crop_window(self):
        # 回転あり・crop 窓が回転コンテンツの内側に収まる通常表示では、
        # 画像領域（レターボックス以外）を 1px も削ってはならない。
        import numpy as np
        import cores.core as core
        import params

        W, H = 4000, 3000
        matrix, size, ttype = core.combined_rotation_canvas_matrix((H, W, 3), 10.0, 0, None)
        quad = core.content_quad_norm((H, W, 3), matrix, size, ttype)

        dw, dh = 2400, 1800
        dx, dy = (size - dw) / 2, (size - dh) / 2
        tex_w, tex_h = 1200, 900
        param = {"original_img_size": (W, H)}
        params.set_disp_info(param, (dx, dy, dw, dh, tex_w / dw))
        param["_zero_wrap_content_quad"] = quad.tolist()
        param["_zero_wrap_canvas_size"] = float(size)

        img = np.ones((tex_h, tex_w, 3), dtype=np.float32)
        out, _ = core.apply_zero_wrap(img, param, crop_editing=False)
        self.assertEqual(int(np.count_nonzero(out[..., 0] < 0.5)), 0)


if __name__ == "__main__":
    unittest.main()
