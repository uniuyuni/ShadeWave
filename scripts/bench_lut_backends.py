#!/usr/bin/env python
"""Benchmark 3D LUT backends (Metal vs NumPy reference) on a synthetic image."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import lut_adapter, lut_reference


def make_identity_table(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    table = np.empty((size, size, size, 3), dtype=np.float32)
    # BGR layout: table[a=B, b=G, c=R] = [R, G, B] = [axis[c], axis[b], axis[a]]
    for a in range(size):
        for b in range(size):
            for c in range(size):
                table[a, b, c] = (axis[c], axis[b], axis[a])
    return table


def timed(label: str, func, image, table, domain, size, warmup: int, repeat: int) -> np.ndarray:
    for _ in range(warmup):
        result = func(image, table, domain, size)
    values = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(image, table, domain, size)
        values.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"{label:10s} min={min(values):8.3f}ms "
        f"avg={sum(values) / len(values):8.3f}ms "
        f"max={max(values):8.3f}ms "
        f"out_max={float(np.max(result)):.4f}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=3672)
    parser.add_argument("--height", type=int, default=2748)
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    print("backend_status:", lut_adapter.backend_status())
    print(f"image={args.width}x{args.height} lut_size={args.size}")

    rng = np.random.default_rng(0)
    image = rng.random((args.height, args.width, 3)).astype(np.float32)
    table = make_identity_table(args.size)
    domain = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    ref = timed("reference", lut_reference.apply_lut3d, image, table, domain, args.size, args.warmup, args.repeat)
    if lut_adapter.native_available():
        out = timed("metal", lut_adapter.apply_lut3d, image, table, domain, args.size, args.warmup, args.repeat)
        print(f"max|metal-reference|={float(np.max(np.abs(out - ref))):.3e}")
    else:
        print("metal backend unavailable; skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
