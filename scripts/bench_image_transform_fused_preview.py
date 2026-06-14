#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import image_transform_adapter


def timed_call(label, func, warmup, repeat):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"{label}: avg={np.mean(times):.3f}ms min={np.min(times):.3f}ms "
        f"max={np.max(times):.3f}ms shape={result.shape}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fused rotation+crop preview backend.")
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--height", type=int, default=5120)
    parser.add_argument("--canvas-width", type=int, default=1600)
    parser.add_argument("--canvas-height", type=int, default=1000)
    parser.add_argument("--angle", type=float, default=17.0)
    parser.add_argument("--crop-scale", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--seed", type=int, default=789)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.random((args.height, args.width, 3), dtype=np.float32)
    size = max(args.width, args.height)
    center = (int(args.width / 2), int(args.height / 2))
    matrix = cv2.getRotationMatrix2D(center, args.angle, 1.0)
    matrix[0, 2] += (size / 2) - center[0]
    matrix[1, 2] += (size / 2) - center[1]

    crop_w = max(1, int(size * args.crop_scale))
    crop_h = max(1, int(size * args.crop_scale * args.canvas_height / args.canvas_width))
    crop_w = min(crop_w, size)
    crop_h = min(crop_h, size)
    crop_x = max(0, (size - crop_w) // 2)
    crop_y = max(0, (size - crop_h) // 2)
    draw_w = args.canvas_width
    draw_h = max(1, int(round(draw_w * crop_h / crop_w)))
    if draw_h > args.canvas_height:
        draw_h = args.canvas_height
        draw_w = max(1, int(round(draw_h * crop_w / crop_h)))
    offset_x = (args.canvas_width - draw_w) // 2
    offset_y = (args.canvas_height - draw_h) // 2

    fused_kwargs = dict(
        matrix=matrix,
        source_rect=(crop_x, crop_y, crop_w, crop_h),
        transform_width=size,
        transform_height=size,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        draw_width=draw_w,
        draw_height=draw_h,
        offset_x=offset_x,
        offset_y=offset_y,
        transform_type="affine",
        interpolation="area",
        border_mode="reflect",
    )
    transform_kwargs = dict(
        matrix=matrix,
        canvas_width=size,
        canvas_height=size,
        transform_type="affine",
        interpolation="linear",
        border_mode="reflect",
    )
    crop_kwargs = dict(
        source_rect=(crop_x, crop_y, crop_w, crop_h),
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        draw_width=draw_w,
        draw_height=draw_h,
        offset_x=offset_x,
        offset_y=offset_y,
        interpolation="area",
    )

    print(f"input={image.shape} transform_canvas={size} crop={fused_kwargs['source_rect']} canvas={args.canvas_width}x{args.canvas_height}")

    previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
    try:
        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        print(f"metal_status={image_transform_adapter.backend_status()}")
        if image_transform_adapter.backend_status().backend != "effect_backends._image_transform_metal":
            print("metal_unavailable")
            return 0

        fused = timed_call(
            "fused_metal",
            lambda: image_transform_adapter.transform_crop_to_canvas(image, **fused_kwargs),
            args.warmup,
            args.repeat,
        )

        def two_pass():
            transformed = image_transform_adapter.transform_to_canvas(image, **transform_kwargs)
            return image_transform_adapter.fit_crop_to_canvas(transformed, **crop_kwargs)

        two = timed_call("two_pass_metal", two_pass, 1, max(3, args.repeat // 3))
        diff = np.abs(two - fused)
        print(f"diff_vs_two_pass: mean={float(np.mean(diff)):.6f} max={float(np.max(diff)):.6f}")
    finally:
        if previous is None:
            os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
        else:
            os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
