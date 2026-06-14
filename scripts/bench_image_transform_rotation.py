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
    parser = argparse.ArgumentParser(description="Benchmark transform_to_canvas rotation backends.")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--angle", type=float, default=17.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.random((args.height, args.width, 3), dtype=np.float32)
    size = max(args.width, args.height)
    center = (int(args.width / 2), int(args.height / 2))
    matrix = cv2.getRotationMatrix2D(center, args.angle, 1.0)
    matrix[0, 2] += (size / 2) - center[0]
    matrix[1, 2] += (size / 2) - center[1]
    kwargs = dict(
        matrix=matrix,
        canvas_width=size,
        canvas_height=size,
        transform_type="affine",
        interpolation="linear",
        border_mode="reflect",
    )
    print(f"input={image.shape} canvas={size} angle={args.angle}")

    previous = os.environ.get("PLATYPUS_IMAGE_TRANSFORM_BACKEND")
    try:
        os.environ.pop("PLATYPUS_IMAGE_TRANSFORM_BACKEND", None)
        print(f"auto_status={image_transform_adapter.backend_status()}")
        auto = timed_call(
            "auto",
            lambda: image_transform_adapter.transform_to_canvas(image, **kwargs),
            args.warmup,
            args.repeat,
        )

        os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "reference"
        print(f"reference_status={image_transform_adapter.backend_status()}")
        reference = timed_call(
            "reference",
            lambda: image_transform_adapter.transform_to_canvas(image, **kwargs),
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
            lambda: image_transform_adapter.transform_to_canvas(image, **kwargs),
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
