#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from effect_backends import colour_functions_reference
from effect_backends import colour_functions_adapter


def timed_call(label, func, warmup, repeat):
    for _ in range(warmup):
        func()
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"{label}: avg={np.mean(times):.3f}ms min={np.min(times):.3f}ms "
        f"max={np.max(times):.3f}ms shape={getattr(result, 'shape', None)}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark colour display transform hot path.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--src", default="ProPhoto RGB")
    parser.add_argument("--dst", default="sRGB")
    parser.add_argument("--cat", default="CAT16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.normal(0.25, 0.55, (args.height, args.width, 3)).astype(np.float32)
    image[: max(1, args.height // 8)] *= 3.0
    image[max(1, args.height // 8): max(2, args.height // 4)] -= 0.35
    basis = colour_functions_adapter.display_color_transform_basis(args.src, args.dst, args.cat)
    print(f"backend_status: {colour_functions_adapter.backend_status()}")

    if args.native_only:
        timed_call(
            "canonical_display_color_transform",
            lambda: colour_functions_adapter.display_color_transform(image, args.src, args.dst, args.cat),
            args.warmup,
            args.repeat,
        )
        timed_call(
            "cached_basis_apply_display_color_transform",
            lambda: colour_functions_adapter.apply_display_color_transform(image, basis, args.dst),
            args.warmup,
            args.repeat,
        )
        return 0

    def matrix_only():
        return (image.reshape(-1, 3) @ basis).reshape(image.shape)

    linear = timed_call("matrix_only", matrix_only, args.warmup, args.repeat)
    compressed = timed_call(
        "compress_negative_display_gamut",
        lambda: colour_functions_reference.compress_negative_display_gamut(linear),
        args.warmup,
        args.repeat,
    )
    timed_call(
        "encode_display_output",
        lambda: colour_functions_reference.encode_display_output(compressed, args.dst),
        args.warmup,
        args.repeat,
    )
    timed_call(
        "reference_apply_display_color_transform",
        lambda: colour_functions_reference.apply_display_color_transform(image, basis, args.dst),
        args.warmup,
        args.repeat,
    )
    timed_call(
        "canonical_display_color_transform",
        lambda: colour_functions_adapter.display_color_transform(image, args.src, args.dst, args.cat),
        args.warmup,
        args.repeat,
    )
    timed_call(
        "cached_basis_apply_display_color_transform",
        lambda: colour_functions_adapter.apply_display_color_transform(image, basis, args.dst),
        args.warmup,
        args.repeat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
