#!/usr/bin/env python3
"""Replay two Draw Quick Select debug NPZs and measure their seam union gap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores.mask2 import draw_quick_select, edge_refine


def _load_and_solve(path: Path):
    data = np.load(path, allow_pickle=True)
    strokes = list(data["strokes"]) if "strokes" in data else None
    result = draw_quick_select.compute_draw_support(
        data["guide"],
        data["mask"],
        float(data["radius"]),
        float(data["strength"]),
        seed_mask=data["seed_mask"] if "seed_mask" in data else None,
        draw_strokes=strokes,
        pixel_scale=float(data["pixel_scale"]) if "pixel_scale" in data else 1.0,
    )
    refined = edge_refine._compose_refined_mask(
        data["mask"],
        result.support,
        True,
        guide=data["guide"],
        natural_edge=True,
        edge_lock=float(data["strength"]),
    )
    planes = {name: value for name, value in result.debug_planes}
    return data, result, refined, planes


def _component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(labels - 1)


def _write_overlay(path: Path, guide, support_a, support_b, shared, gap, candidate):
    image = np.clip(guide * 255.0, 0, 255).astype(np.uint8)
    vis = image.astype(np.float32, copy=True)

    def overlay(mask, color, alpha):
        nonlocal vis
        color_arr = np.asarray(color, dtype=np.float32)
        vis[mask] = vis[mask] * (1.0 - alpha) + color_arr * alpha

    overlay(support_a, (255, 40, 40), 0.35)
    overlay(support_b, (40, 120, 255), 0.35)
    overlay(shared, (255, 255, 255), 0.20)
    overlay(gap, (255, 235, 0), 0.95)

    ys, xs = np.where(candidate)
    if ys.size:
        pad = 25
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(vis.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(vis.shape[1], int(xs.max()) + pad + 1)
        vis = vis[y0:y1, x0:x1]

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.4)
    parser.add_argument("--edge-near", type=int, default=2)
    parser.add_argument("--seam-radius", type=float, default=4.0)
    parser.add_argument("--max-gap-ratio", type=float, default=None)
    parser.add_argument("--overlay", type=Path, default=None)
    args = parser.parse_args()

    data_a, result_a, refined_a, planes_a = _load_and_solve(args.first)
    data_b, result_b, refined_b, planes_b = _load_and_solve(args.second)

    alpha_sum = np.clip(refined_a + refined_b, 0.0, 1.0)
    candidate = result_a.candidate | result_b.candidate
    edge = edge_refine._draw_snap_edge_strength(data_a["guide"])
    near_edge = edge >= float(args.edge_threshold)
    if args.edge_near > 0 and np.any(near_edge):
        near_edge = cv2.dilate(
            near_edge.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=int(args.edge_near),
        ) > 0

    dist_a = distance_transform_edt(~result_a.support)
    dist_b = distance_transform_edt(~result_b.support)
    shared = (
        (dist_a <= float(args.seam_radius))
        & (dist_b <= float(args.seam_radius))
        & candidate
        & near_edge
    )
    gap = shared & (alpha_sum < float(args.alpha_threshold))

    alpha_mask = (alpha_sum >= float(args.alpha_threshold)) & candidate
    metrics = {
        "first": str(args.first),
        "second": str(args.second),
        "first_support_px": int(np.count_nonzero(result_a.support)),
        "second_support_px": int(np.count_nonzero(result_b.support)),
        "first_edge_restore_px": int(np.count_nonzero(planes_a.get("edge_restore", 0))),
        "second_edge_restore_px": int(np.count_nonzero(planes_b.get("edge_restore", 0))),
        "shared_seam_px": int(np.count_nonzero(shared)),
        "gap_px": int(np.count_nonzero(gap)),
        "gap_ratio": float(np.count_nonzero(gap) / max(1, np.count_nonzero(shared))),
        "alpha_components": _component_count(alpha_mask),
    }

    if args.overlay is not None:
        _write_overlay(
            args.overlay,
            data_a["guide"],
            result_a.support,
            result_b.support,
            shared,
            gap,
            candidate,
        )
        metrics["overlay"] = str(args.overlay)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.max_gap_ratio is not None and metrics["gap_ratio"] > args.max_gap_ratio:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
