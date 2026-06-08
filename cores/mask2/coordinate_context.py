"""
MaskEditor2 の座標・参照画像まわりのみを保持（ウィンドウ Widget なし）。
params / cores と連携するメソッドは MaskEditor2 と同じシグネチャ。
"""
from __future__ import annotations

import numpy as np

import cores.core as core
import params
import cores.hlsrgb as hlsrgb
import macos as device


class Mask2CoordinateContext:
    """export / ヘッドレスパイプライン用。Kivy Widget を要しない。"""

    def __init__(self):
        self.pos = (0, 0)
        self.texture_size = (0, 0)
        self.tcg_info = None
        self.primary_param = None
        self.crop_image_rgb = None
        self.crop_image_hls = None
        self.original_image_rgb = None
        self.original_image_hls = None
        # mask Geometry: 画像 Geom のみの matrix を退避しておく。
        # HeadlessCompositMask が tcg_info['matrix'] を一時的に M_mask @ base
        # に置き換えて children を描画するときの base になる。
        # widgets/mask_editor2.py MaskEditor2._image_only_matrix と同じ役割。
        self._image_only_matrix = None

    def set_ref_image(self, crop_image, original_image=None):
        if self.crop_image_rgb is not crop_image:
            self.crop_image_rgb = crop_image
            self.crop_image_hls = None

        if self.original_image_rgb is not original_image:
            self.original_image_rgb = original_image
            self.original_image_hls = None

    def get_crop_image_hls(self):
        if self.crop_image_hls is None and self.crop_image_rgb is not None:
            self.crop_image_hls = hlsrgb.rgb_to_hlc_gain(self.crop_image_rgb)
            # Keep the RGB crop alive. Edge-refine and its debug views must use
            # the current zoom crop, even after HLS-based masks have run.
        return self.crop_image_hls

    def get_original_image_rgb(self):
        return self.original_image_rgb

    def get_original_image_hls(self):
        if self.original_image_hls is None and self.original_image_rgb is not None:
            self.original_image_hls = hlsrgb.rgb_to_hlc_gain(self.original_image_rgb)
        return self.original_image_hls

    @property
    def size(self):
        s = device.dpi_scale()
        return (self.texture_size[0] * s, self.texture_size[1] * s)

    def to_window(self, x, y):
        return (x, y)

    def set_texture_size(self, tx, ty):
        self.texture_size = (tx, ty)

    def set_primary_param(self, primary_param, disp_info):
        self.tcg_info = params.param_to_tcg_info(primary_param)
        params.set_disp_info(self.tcg_info, disp_info)
        # 画像 Geom のみの matrix を退避 (HeadlessCompositMask の get_mask_image が
        # 各 composit ごとに M_mask @ base で書き換える起点として使う)。
        self._image_only_matrix = np.array(self.tcg_info["matrix"], dtype=np.float64).copy()
        # mask_mesh_link_to_image=True の Composit が「画像 mesh の CP」を参照するため、
        # primary_param 自体への参照を保持しておく (control_points / mesh_size を読むため)。
        self.primary_param = primary_param

    def _call_with_image_only_matrix(self, func, *args, **kwargs):
        """func を「mask geom 抜きの image-only matrix」で一時実行する。
        SegmentMask / DepthMapMask / FaceMask / TargetTextMask など、
        follows_mask_geometry()==False なマスクの推論経路で使う。
        widgets/mask_editor2.py MaskEditor2._call_with_image_only_matrix と同じ流儀
        (headless では _matrix_lock 不要)。"""
        if self._image_only_matrix is None or self.tcg_info is None or "matrix" not in self.tcg_info:
            return func(*args, **kwargs)
        saved_matrix = self.tcg_info["matrix"]
        self.tcg_info["matrix"] = self._image_only_matrix
        try:
            return func(*args, **kwargs)
        finally:
            self.tcg_info["matrix"] = saved_matrix

    def get_hash_items(self):
        return (
            params.get_disp_info(self.tcg_info),
            self.tcg_info["rotation"] + self.tcg_info["rotation2"],
            self.tcg_info["flip_mode"],
            tuple(self.tcg_info["matrix"].flatten()),
        )

    def get_rotate_rad(self, rotate_rad):
        rad, flip = self.tcg_info["rotation2"], self.tcg_info["flip_mode"]
        angle_rad = rotate_rad + rad
        match flip:
            case 0:
                pass
            case 1:
                angle_rad = -angle_rad
            case 2:
                angle_rad = angle_rad + np.radians(90)
            case 3:
                angle_rad = angle_rad - np.radians(180)
        return self.tcg_info["rotation"] + angle_rad

    def get_image_size(self):
        return self.tcg_info["original_img_size"]

    def window_to_tcg_scale(self, x, y):
        return params.window_to_tcg_scale((x, y), self.tcg_info)

    def tcg_to_window_scale(self, x, y):
        return params.tcg_to_window_scale((x, y), self.tcg_info)

    def tcg_to_image_scale(self, x, y):
        return params.tcg_to_image_scale((x, y), self.tcg_info)

    def window_to_tcg(self, cx, cy):
        return params.window_to_tcg(cx, cy, self, self.texture_size, self.tcg_info, normalize=False)

    def tcg_to_window(self, cx, cy):
        return params.tcg_to_window(cx, cy, self, self.texture_size, self.tcg_info, normalize=False)

    def tcg_to_texture(self, cx, cy):
        disp_info = params.get_disp_info(self.tcg_info)
        imax = max(
            self.tcg_info["original_img_size"][0] / 2,
            self.tcg_info["original_img_size"][1] / 2,
        )
        cx, cy = params.center_rotate(cx, cy, self.tcg_info)
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - disp_info[0], cy - disp_info[1]
        cx, cy = cx * disp_info[4], cy * disp_info[4]
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(
            *self.texture_size, disp_info
        )
        cx, cy = cx + offset_x, cy + offset_y
        return (cx, cy)

    def tcg_to_full_image(self, cx, cy):
        imax = max(
            self.tcg_info["original_img_size"][0] / 2,
            self.tcg_info["original_img_size"][1] / 2,
        )
        cx, cy = params.center_rotate(cx, cy, self.tcg_info)
        cx, cy = cx + imax, cy + imax
        return (cx, cy)

    def tcg_to_crop_image(self, cx, cy):
        cx, cy = self.tcg_to_full_image(cx, cy)
        hls = self.get_crop_image_hls()
        shape_max = max(self.original_image_rgb.shape[0], self.original_image_rgb.shape[1])
        cx = cx * (hls.shape[1] / shape_max)
        cy = cy * (hls.shape[0] / shape_max)
        return (cx, cy)

    def tcg_to_original_image(self, cx, cy):
        h, w = self.get_original_image_rgb().shape[:2]
        cx, cy = cx + w * 0.5, cy + h * 0.5
        cx, cy = min(max(cx, 0), w), min(max(cy, 0), h)
        return (cx, cy)
