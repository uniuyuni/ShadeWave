import argparse
import os
import pathlib
import sys
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--height", type=int, default=2560)
    parser.add_argument("--canvas-width", type=int, default=1600)
    parser.add_argument("--canvas-height", type=int, default=1000)
    parser.add_argument("--strength", type=float, default=30.0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"

    import effects  # noqa: F401
    from cores.distortion_correction.lens_distortion import correct_lens_distortion
    from effect_backends import image_transform_adapter

    rng = np.random.default_rng(123)
    image = rng.random((args.height, args.width, 3), dtype=np.float32)
    matrix = np.eye(3, dtype=np.float64)
    source_rect = (0, 0, args.width, args.height)
    kwargs = dict(
        matrix=matrix,
        source_rect=source_rect,
        transform_width=args.width,
        transform_height=args.height,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        draw_width=args.canvas_width,
        draw_height=args.canvas_height,
        offset_x=0,
        offset_y=0,
        transform_type="perspective",
        interpolation="linear",
        border_mode="constant",
    )

    status = image_transform_adapter.backend_status()
    if not status.native:
        raise SystemExit(f"Metal backend unavailable: {status.detail}")

    for _ in range(args.warmup):
        image_transform_adapter.transform_crop_to_canvas(
            image,
            **kwargs,
            lens_strength=args.strength,
            lens_scale=1.0,
        )

    fused_times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        image_transform_adapter.transform_crop_to_canvas(
            image,
            **kwargs,
            lens_strength=args.strength,
            lens_scale=1.0,
        )
        fused_times.append((time.perf_counter() - start) * 1000.0)

    two_pass_times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        lens = correct_lens_distortion(
            image,
            args.strength,
            interpolation="bilinear",
            grid_size=4,
            scale=1.0,
        )
        image_transform_adapter.fit_crop_to_canvas(
            lens,
            source_rect,
            args.canvas_width,
            args.canvas_height,
            args.canvas_width,
            args.canvas_height,
            0,
            0,
            "linear",
        )
        two_pass_times.append((time.perf_counter() - start) * 1000.0)

    fused_avg = float(np.mean(fused_times))
    two_pass_avg = float(np.mean(two_pass_times))
    print(f"fused_lens_ms={fused_avg:.3f}")
    print(f"two_pass_lens_ms={two_pass_avg:.3f}")
    print(f"speedup={two_pass_avg / fused_avg:.2f}x")


if __name__ == "__main__":
    main()
