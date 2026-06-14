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
    parser = argparse.ArgumentParser(description="Benchmark image transform backends.")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--canvas-width", type=int, default=1200)
    parser.add_argument("--canvas-height", type=int, default=800)
    parser.add_argument("--crop-scale", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.random((args.height, args.width, 3), dtype=np.float32)
    crop_w = max(1, int(args.width * args.crop_scale))
    crop_h = max(1, int(args.height * args.crop_scale))
    crop_x = max(0, (args.width - crop_w) // 2)
    crop_y = max(0, (args.height - crop_h) // 2)
    draw_w = args.canvas_width
    draw_h = max(1, int(round(draw_w * crop_h / crop_w)))
    if draw_h > args.canvas_height:
        draw_h = args.canvas_height
        draw_w = max(1, int(round(draw_h * crop_w / crop_h)))
    offset_x = (args.canvas_width - draw_w) // 2
    offset_y = (args.canvas_height - draw_h) // 2

    kwargs = dict(
        source_rect=(crop_x, crop_y, crop_w, crop_h),
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        draw_width=draw_w,
        draw_height=draw_h,
        offset_x=offset_x,
        offset_y=offset_y,
        interpolation="area",
    )
    print(f"input={image.shape} kwargs={kwargs}")

    previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
    previous_area_mode = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_AREA_MODE")
    try:
        os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
        os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_AREA_MODE", None)
        print("area_mode=exact")
        print(f"auto_status={image_transform_adapter.backend_status()}")
        auto = timed_call(
            "auto",
            lambda: image_transform_adapter.fit_crop_to_canvas(image, **kwargs),
            args.warmup,
            args.repeat,
        )

        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "reference"
        print(f"reference_status={image_transform_adapter.backend_status()}")
        reference = timed_call(
            "reference",
            lambda: image_transform_adapter.fit_crop_to_canvas(image, **kwargs),
            args.warmup,
            args.repeat,
        )
        auto_diff = np.abs(reference - auto)
        print(f"auto_diff: mean={float(np.mean(auto_diff)):.6f} max={float(np.max(auto_diff)):.6f}")

        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"
        print(f"metal_status={image_transform_adapter.backend_status()}")
        if image_transform_adapter.backend_status().backend != "effect_backends._image_transform_metal":
            print("metal_unavailable")
            return 0
        metal = timed_call(
            "metal",
            lambda: image_transform_adapter.fit_crop_to_canvas(image, **kwargs),
            args.warmup,
            args.repeat,
        )
        diff = np.abs(reference - metal)
        print(f"diff: mean={float(np.mean(diff)):.6f} max={float(np.max(diff)):.6f}")
    finally:
        if previous is None:
            os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
        else:
            os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = previous
        if previous_area_mode is None:
            os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_AREA_MODE", None)
        else:
            os.environ["PLATYPUS_IMAGE_TRANSFORM_AREA_MODE"] = previous_area_mode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
