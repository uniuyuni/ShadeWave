#!/usr/bin/env python
"""Benchmark CrossFilter backends on a synthetic image."""

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

from effect_backends import cross_filter_adapter, cross_filter_reference


def make_peak_image(height: int, width: int, peaks: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.float32)
    rng = np.random.default_rng(123)
    margin_y = max(1, height // 10)
    margin_x = max(1, width // 10)
    for _ in range(peaks):
        y = int(rng.integers(margin_y, max(margin_y + 1, height - margin_y)))
        x = int(rng.integers(margin_x, max(margin_x + 1, width - margin_x)))
        image[y, x] = rng.uniform(2.5, 5.0, size=3).astype(np.float32)
    return image


def timed_call(label: str, func, image: np.ndarray, kwargs: dict, warmup: int, repeat: int) -> None:
    for _ in range(warmup):
        func(image, **kwargs)
    values = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(image, **kwargs)
        values.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"{label:10s} min={min(values):8.3f}ms "
        f"avg={sum(values) / len(values):8.3f}ms "
        f"max={max(values):8.3f}ms "
        f"out_max={float(np.max(result)) if result is not None else 0.0:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--peaks", type=int, default=3)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--points", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    kwargs = dict(
        num_points=args.points,
        length=args.length,
        angle_deg=20.0,
        threshold=1.0,
        intensity=0.25,
        spectral_strength=0.2,
        line_thickness=1.0,
        min_distance=max(8, min(args.width, args.height) // 30),
        randomness=0.0,
        speed_factor=4,
    )
    image = make_peak_image(args.height, args.width, args.peaks)

    print(f"image={args.width}x{args.height} peaks={args.peaks} length={args.length} points={args.points}")
    print(f"default_status={cross_filter_adapter.backend_status()}")
    timed_call("reference", cross_filter_reference.apply_cross_filter, image, kwargs, args.warmup, args.repeat)

    previous = os.environ.get("PLATYPUS_CROSS_FILTER_BACKEND")
    try:
        os.environ["PLATYPUS_CROSS_FILTER_BACKEND"] = "cpu"
        print(f"cpu_status={cross_filter_adapter.backend_status()}")
        timed_call("cpu", cross_filter_adapter.apply_cross_filter, image, kwargs, args.warmup, args.repeat)

        os.environ["PLATYPUS_CROSS_FILTER_BACKEND"] = "metal"
        status = cross_filter_adapter.backend_status()
        print(f"metal_status={status}")
        if status.backend == "effect_backends._cross_filter_metal":
            timed_call("metal", cross_filter_adapter.apply_cross_filter, image, kwargs, args.warmup, args.repeat)
    finally:
        if previous is None:
            os.environ.pop("PLATYPUS_CROSS_FILTER_BACKEND", None)
        else:
            os.environ["PLATYPUS_CROSS_FILTER_BACKEND"] = previous

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
