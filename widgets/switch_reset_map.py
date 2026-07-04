"""スイッチID → reset 対象効果の対応（MainWidget._switch_reset_targets と共有）。"""

# effects.HLSEffect.HLS_COLORS と同一順序を維持すること
HLS_COLORS = (
    "red",
    "skin",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "magenta",
)

BASE_SWITCH_TARGETS = {
    "switch_white_balance": (2, "color_temperature", None),
    "switch_exposure_contrast": (2, ["exposure", "contrast"], None),
    "switch_tone": (2, "tone", None),
    "switch_level": (2, "level", None),
    "switch_precence": (
        2,
        ["clarity", "texture", "microcontrast", "dehaze", "clahe"],
        None,
    ),
    "switch_saturation": (2, "vs_and_saturation", "saturation"),
    "switch_color_mixer": (2, "hls", None),
    "switch_unsharp_mask": (2, "unsharp_mask", None),
    "switch_vignette": (4, "vignette", None),
    "switch_lens_modifier": (0, "lens_modifier", None),
    "switch_tone_curves": (2, "curves", "tone_curves"),
    "switch_color_gradings": (2, "curves", "color_gradings"),
    "switch_color_curves": (2, "vs_and_saturation", "color_curves"),
    "switch_hue_vs_hue": (2, "vs_and_saturation", "HuevsHue"),
    "switch_hue_vs_lum": (2, "vs_and_saturation", "HuevsLum"),
    "switch_hue_vs_sat": (2, "vs_and_saturation", "HuevsSat"),
    "switch_lum_vs_lum": (2, "vs_and_saturation", "LumvsLum"),
    "switch_lum_vs_sat": (2, "vs_and_saturation", "LumvsSat"),
    "switch_sat_vs_lum": (2, "vs_and_saturation", "SatvsLum"),
    "switch_sat_vs_sat": (2, "vs_and_saturation", "SatvsSat"),
    "switch_ai_noise_reduction": (0, "ai_noise_reduction", None),
    "switch_light_noise_reduction": (2, "light_noise_reduction", None),
    "switch_details": (0, ["inpaint", "patchmatch_inpaint", "subpixel_shift", "exposure_fusion_debevec"], None),
    "switch_lut": (2, ["input_lut", "look_lut"], None),
    "switch_color_match": (0, "color_match", None),
    "switch_solid_color": (2, "solid_color", None),
    "switch_global": (2, "color_separation", None),
    "switch_fringe_removal": (0, "remove_chromatic_aberration", None),
    "switch_film_simulation": (2, "film_emulation", None),
    "switch_lens_simulator": (2, "lens_simulator", None),
    "switch_light_rays": (2, "light_rays", None),
    "switch_filters": (1, ["lensblur_filter", "scratch", "frosted_glass", "mosaic"], None),
    "switch_orton_effect": (2, "orton", None),
    "switch_glow_effect": (2, "glow", None),
    "switch_grain": (4, "grain", None),
    "switch_cross_filter": (0, "cross_filter", None),
    "switch_rotation": (0, "geometry", "rotation"),
    "switch_distortion_correction": (0, "geometry", "distortion_correction"),
    "switch_mask2_draw_effects": (3, "mask2", "mask2_draw_effects"),
    "switch_face": (1, "face", None),
    "switch_mask2_settings": (3, "mask2", "mask2_settings"),
    "switch_mask2_depth": (3, "mask2", "mask2_depth"),
    "switch_mask2_hue": (3, "mask2", "mask2_hue"),
    "switch_mask2_lum": (3, "mask2", "mask2_lum"),
    "switch_mask2_sat": (3, "mask2", "mask2_sat"),
    "switch_mask2_options": (3, "mask2", "mask2_options"),
    "switch_mask2_quick_select": (3, "mask2", "mask2_quick_select"),
    "switch_mask2_face": (3, "mask2", "mask2_face"),
    "switch_distortion": (1, "distortion", None),
    # lens_ghost は lv1/lv2 両方に同一インスタンスを登録。lv は reset_switch_defaults_for_label 側で
    # lens_ghost_level() に差し替える(ここの 2 はフォールバック)。
    "switch_lens_ghost": (2, "lens_ghost", None),
}


def build_switch_reset_targets():
    targets = dict(BASE_SWITCH_TARGETS)
    for color in HLS_COLORS:
        targets[f"switch_hls_{color}"] = (2, "hls", color)
    return targets


def flatten_targets_to_pipeline_ids(switch_ids):
    """パイプライン上の効果IDを列挙（重複は除去、順序維持）。"""
    targets = build_switch_reset_targets()
    ordered = []
    for sid in switch_ids:
        t = targets.get(sid)
        if t is None:
            continue
        _lv, eff, _sub = t
        chunk = eff if isinstance(eff, list) else [eff]
        for name in chunk:
            if name not in ordered:
                ordered.append(name)
    return ordered
