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

import params
from cores import core
from cores.distortion_correction.trapezoid_correction_3d import calculate_trapezoid_homography


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
    parser = argparse.ArgumentParser(description="Benchmark fused rotation+perspective+crop preview backend.")
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--height", type=int, default=5120)
    parser.add_argument("--canvas-width", type=int, default=1600)
    parser.add_argument("--canvas-height", type=int, default=1000)
    parser.add_argument("--angle", type=float, default=8.0)
    parser.add_argument("--horizontal", type=float, default=18.0)
    parser.add_argument("--vertical", type=float, default=8.0)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=901)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.random((args.height, args.width, 3), dtype=np.float32)
    size = max(args.width, args.height)
    half_size = size / 2
    focal_length = size * (0.5 + 20 * 0.025)
    matrix_param = {}
    params.set_matrix(matrix_param, None)
    H = calculate_trapezoid_homography(
        size,
        size,
        horizontal=args.horizontal * 0.5,
        vertical=args.vertical * 0.5,
        focal_length=focal_length,
    )
    params.add_matrix(matrix_param, H, offset=(half_size, half_size))
    matrix, transform_size, transform_type = core.combined_rotation_canvas_matrix(
        image.shape,
        args.angle,
        0,
        matrix_param["matrix"],
    )
    disp_info = (0, 0, transform_size, transform_size, args.canvas_width / transform_size)
    crop_rect = (0, 0, transform_size, transform_size)

    print(
        f"input={image.shape} transform_canvas={transform_size} canvas={args.canvas_width}x{args.canvas_height} "
        f"angle={args.angle} h={args.horizontal} v={args.vertical}"
    )

    previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
    try:
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        fused = timed_call(
            "fused_perspective",
            lambda: core.transform_crop_image(
                image,
                matrix,
                transform_size,
                transform_size,
                disp_info,
                args.canvas_width,
                args.canvas_height,
                border_mode="reflect",
                transform_type=transform_type,
            ),
            args.warmup,
            args.repeat,
        )

        def two_pass():
            transformed = core.rotation(
                image,
                args.angle,
                matrix=matrix_param["matrix"],
                inter_mode="bilinear",
                border_mode="reflect",
            )
            return core.crop_image(
                transformed,
                disp_info,
                crop_rect,
                args.canvas_width,
                args.canvas_height,
                0,
                0,
                False,
            )

        two = timed_call("two_pass_perspective", two_pass, 1, max(3, args.repeat // 3))
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
