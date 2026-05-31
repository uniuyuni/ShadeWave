"""
マスク Mesh の texture_size stale 問題を再現する diagnostic script。
docs/mask-mesh-coordinate-investigation.md 第 5 章のシミュレーションを実コードで実行する。

実行: python3 docs/mask-mesh-coordinate-sample.py
"""
from __future__ import annotations

import math

# device.dpi_scale() の代用 (1.0 固定)
DPI = 1.0


def denorm_param(orig_w: int, orig_h: int, val):
    """params.denorm_param 相当 (各 dim 独立 orig_size 倍)。"""
    return (val[0] * orig_w, val[1] * orig_h)


def crop_size_and_offset_from_texture(tw: int, th: int, disp_info):
    """cores/core.py crop_size_and_offset_from_texture と同一。"""
    crop_aspect = disp_info[2] / disp_info[3]
    texture_aspect = tw / th
    if crop_aspect > texture_aspect:
        new_width = tw
        new_height = int(tw / crop_aspect)
    else:
        new_width = int(th * crop_aspect)
        new_height = th
    offset_x = (tw - new_width) // 2
    offset_y = (th - new_height) // 2
    return (new_width, new_height, offset_x, offset_y)


def tcg_to_window(tcg_x: float, tcg_y: float,
                  widget_size, widget_pos,
                  texture_size, tcg_info,
                  rotation_rad: float = 0.0):
    """params.tcg_to_window のミニ再現。rotation は単純 2D rotation のみ。
    Mask Geom matrix 等は省略。"""
    orig_w, orig_h = tcg_info["original_img_size"]
    imax = max(orig_w, orig_h) / 2.0

    # 1) normalize -> TCG px
    cx, cy = denorm_param(orig_w, orig_h, (tcg_x, tcg_y))

    # 2) center_rotate (rotation のみ)
    if rotation_rad != 0:
        cs, sn = math.cos(-rotation_rad), math.sin(-rotation_rad)
        cx, cy = cx * cs - cy * sn, cx * sn + cy * cs

    # 3) +imax で原点を左上に
    cx, cy = cx + imax, cy + imax

    # 4) crop offset を引いて preview スケール
    disp_info = tcg_info["disp_info"]
    cx, cy = cx - disp_info[0], cy - disp_info[1]
    cx, cy = cx * disp_info[4], cy * disp_info[4]

    # 5) texture offset
    _, _, offset_x, offset_y = crop_size_and_offset_from_texture(*texture_size, disp_info)
    cx, cy = cx + offset_x, cy + offset_y

    # 6) Y flip
    cy = texture_size[1] - cy

    # 7) widget margin
    margin_x = (widget_size[0] / DPI - texture_size[0]) / 2
    margin_y = (widget_size[1] / DPI - texture_size[1]) / 2
    cx, cy = cx + margin_x, cy + margin_y

    # 8) dpi scale + widget.pos
    cx, cy = cx * DPI, cy * DPI
    cx, cy = cx + widget_pos[0], cy + widget_pos[1]

    return (cx, cy)


def main():
    orig = (4000, 3000)
    imax = max(orig) / 2

    print("== シナリオ: ウィンドウリサイズで preview_widget が 512x512 -> 768x768 ==")
    print()

    # 起動時 (state-A)
    state_a = {
        "tcg_info": {
            "original_img_size": orig,
            "disp_info": (0, 0, orig[0], orig[1], 512 / max(orig)),  # scale = 512/4000 = 0.128
        },
        "texture_size": (512, 512),
        "widget_size": (512, 512),
        "widget_pos": (100, 100),
    }
    # リサイズ後 (state-B): preview_widget が 768x768 に。
    # refresh_preview_overlays で:
    #   - mask_editor.set_texture_size((768, 768))  → MaskEditor2 は更新される
    #   - geometry_effect.update_geometry_editor_texture_size() → 画像 mesh widget 更新
    #   - mask_mesh_editor.set_texture_size(...)  → ★ 呼ばれない (BUG)
    # disp_info[4] も新しい preview スケール 768/4000 で更新される (tcg_info は共有 dict)。
    state_b_correct = {
        "tcg_info": {
            "original_img_size": orig,
            "disp_info": (0, 0, orig[0], orig[1], 768 / max(orig)),  # scale = 768/4000 = 0.192
        },
        "texture_size": (768, 768),  # 正しく同期された場合 (= 画像 mesh widget の挙動)
        "widget_size": (768, 768),
        "widget_pos": (100, 100),
    }
    state_b_buggy = {
        # tcg_info は共有 dict なので disp_info は新スケールに更新されているが、
        # widget.texture_size はコンストラクタ時点 (512, 512) のまま (= マスク mesh 現状)
        "tcg_info": state_b_correct["tcg_info"],
        "texture_size": (512, 512),  # ★ stale (set_texture_size が呼ばれていない)
        "widget_size": (768, 768),
        "widget_pos": (100, 100),
    }

    test_points = [
        ("center  (0.0, 0.0)", (0.0, 0.0)),
        ("UL      (-0.5,-0.5)", (-0.5, -0.5)),
        ("UR      (+0.5,-0.5)", (+0.5, -0.5)),
        ("BL      (-0.5,+0.5)", (-0.5, +0.5)),
        ("BR      (+0.5,+0.5)", (+0.5, +0.5)),
    ]

    for name, (tx, ty) in test_points:
        a = tcg_to_window(tx, ty,
                          state_a["widget_size"], state_a["widget_pos"],
                          state_a["texture_size"], state_a["tcg_info"])
        b_ok = tcg_to_window(tx, ty,
                             state_b_correct["widget_size"], state_b_correct["widget_pos"],
                             state_b_correct["texture_size"], state_b_correct["tcg_info"])
        b_bug = tcg_to_window(tx, ty,
                              state_b_buggy["widget_size"], state_b_buggy["widget_pos"],
                              state_b_buggy["texture_size"], state_b_buggy["tcg_info"])

        print(f"  CP {name}")
        print(f"    state-A (起動時):                 window = ({a[0]:7.1f}, {a[1]:7.1f})")
        print(f"    state-B  正常 (画像 mesh):       window = ({b_ok[0]:7.1f}, {b_ok[1]:7.1f})")
        print(f"    state-B  stale (マスク mesh):    window = ({b_bug[0]:7.1f}, {b_bug[1]:7.1f})")
        diff = (b_bug[0] - b_ok[0], b_bug[1] - b_ok[1])
        print(f"    バグでのズレ                       = ({diff[0]:+.1f}, {diff[1]:+.1f}) px")
        print()

    print("== 解釈 ==")
    print("CP がそれぞれ数百 px ずつズレる。preview_widget (768x768) の表示領域の")
    print("外に出てしまう CP もある (画面外)。これがユーザー観察「ガイドが画面外」の正体。")
    print()
    print("修正: refresh_preview_overlays で self.mask_mesh_editor.set_texture_size(...) を呼ぶ。")


if __name__ == "__main__":
    main()
