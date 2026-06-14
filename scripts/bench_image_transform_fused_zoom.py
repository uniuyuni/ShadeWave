#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores import core


def timed_call(label, func, warmup, repeat):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times.append((time.perf_counter() - t0) * 1000.0)
    img, disp = result
    print(
        f"{label}: avg={np.mean(times):.3f}ms min={np.min(times):.3f}ms "
        f"max={np.max(times):.3f}ms shape={img.shape} disp={disp}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fused rotation+zoom preview backend.")
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--height", type=int, default=5120)
    parser.add_argument("--canvas-width", type=int, default=1600)
    parser.add_argument("--canvas-height", type=int, default=1000)
    parser.add_argument("--angle", type=float, default=17.0)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=890)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.random((args.height, args.width, 3), dtype=np.float32)
    matrix, size = core.rotation_canvas_matrix(image.shape, args.angle, 0)
    crop_rect = (0, 0, size, size)
    disp_info = (0, 0, size, size, args.canvas_width / size)
    center_pos = (size / 2, size / 2)

    print(f"input={image.shape} transform_canvas={size} canvas={args.canvas_width}x{args.canvas_height} zoom={args.zoom}")

    previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
    try:
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        fused = timed_call(
            "fused_zoom",
            lambda: core.transform_zoom_crop_image(
                image,
                matrix,
                size,
                size,
                disp_info,
                crop_rect,
                args.canvas_width,
                args.canvas_height,
                0,
                0,
                center_pos=center_pos,
                zoom_ratio=args.zoom,
            ),
            args.warmup,
            args.repeat,
        )

        def two_pass():
            rotated = core.rotation(image, args.angle, inter_mode="bilinear", border_mode="reflect")
            return core.crop_image(
                rotated,
                disp_info,
                crop_rect,
                args.canvas_width,
                args.canvas_height,
                0,
                0,
                True,
                center_pos=center_pos,
                zoom_ratio=args.zoom,
            )

        two = timed_call("two_pass_zoom", two_pass, 1, max(3, args.repeat // 4))
        diff = np.abs(two[0] - fused[0])
        print(f"diff_vs_two_pass: mean={float(np.mean(diff)):.6f} max={float(np.max(diff)):.6f}")
    finally:
        if previous is None:
            os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
        else:
            os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
