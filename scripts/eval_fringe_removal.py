import argparse
import time
from pathlib import Path
from typing import Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from cores.fringe_removal import remove_chromatic_aberration


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def compute_edge_mask(rgb: np.ndarray) -> np.ndarray:
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)
    thresh = max(0.02, float(np.percentile(grad, 86)))
    edge = (grad >= thresh).astype(np.uint8)
    return edge


def compute_aberration_mask(rgb: np.ndarray, sat_thresh: float = 0.12) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1), cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0] * 2.0
    s = hsv[:, :, 1]
    purple = ((h >= 230) & (h <= 310))
    green = ((h >= 40) & (h <= 90))
    return ((purple | green) & (s > sat_thresh)).astype(np.uint8)


def compute_protected_purple_mask(rgb: np.ndarray, edge: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1), cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0] * 2.0
    s = hsv[:, :, 1]
    purple = ((h >= 240) & (h <= 305) & (s > 0.35)).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(purple, connectivity=8)
    protect = np.zeros_like(purple)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 120:
            continue
        comp = labels == idx
        if float(edge[comp].mean()) < 0.25:
            protect[comp] = 1
    return protect


def ecas_like(rgb: np.ndarray) -> float:
    edge = compute_edge_mask(rgb)
    aberr = compute_aberration_mask(rgb)
    edge_sum = float(edge.sum())
    if edge_sum < 1:
        return 0.0
    return float((edge * aberr).sum() / edge_sum)


def delta_e_like(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    idx = mask > 0
    if not np.any(idx):
        return 0.0
    diff = a[idx] - b[idx]
    return float(np.sqrt((diff * diff).sum(axis=1)).mean())


def collect_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])


def guess_category(path: Path) -> str:
    name = path.stem.lower()
    if "flower" in name or "purple" in name or "violet" in name:
        return "purple_object"
    if "lowlight" in name or "night" in name or "iso" in name:
        return "lowlight_noise"
    if "branch" in name or "sky" in name or "backlit" in name:
        return "backlit_edge"
    return "uncategorized"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fringe removal behavior.")
    parser.add_argument("input", type=Path, help="Image file or directory.")
    parser.add_argument("--purple-amount", type=float, default=1.8)
    parser.add_argument("--green-amount", type=float, default=1.5)
    parser.add_argument("--fringe-width", type=int, default=4)
    parser.add_argument("--edge-threshold", type=float, default=0.10)
    args = parser.parse_args()

    images = collect_images(args.input)
    if not images:
        raise ValueError(f"no images found under: {args.input}")

    rows: List[Dict[str, object]] = []
    for image_path in images:
        src = read_rgb(image_path)
        t0 = time.perf_counter()
        dst = remove_chromatic_aberration(
            src,
            purple_amount=args.purple_amount,
            green_amount=args.green_amount,
            fringe_width=args.fringe_width,
            edge_threshold=args.edge_threshold,
            lateral_correction=False,
            min_saturation=0.1,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        edge = compute_edge_mask(src)
        aberr_src = compute_aberration_mask(src)
        protected = compute_protected_purple_mask(src, edge)
        non_target = ((edge == 0) & (aberr_src == 0)).astype(np.uint8)

        row = {
            "path": str(image_path),
            "category": guess_category(image_path),
            "ecas_before": ecas_like(src),
            "ecas_after": ecas_like(dst),
            "protected_shift": delta_e_like(src, dst, protected),
            "non_target_shift": delta_e_like(src, dst, non_target),
            "runtime_ms": float(elapsed_ms),
        }
        rows.append(row)

    print("path,category,ecas_before,ecas_after,ecas_improve,protected_shift,non_target_shift,runtime_ms")
    for row in rows:
        improve = row["ecas_before"] - row["ecas_after"]
        print(
            f"{row['path']},{row['category']},"
            f"{row['ecas_before']:.5f},{row['ecas_after']:.5f},{improve:.5f},"
            f"{row['protected_shift']:.5f},{row['non_target_shift']:.5f},{row['runtime_ms']:.2f}"
        )

    categories = sorted(set(row["category"] for row in rows))
    print("\ncategory_summary")
    for category in categories:
        chunk = [r for r in rows if r["category"] == category]
        ecas_imp = np.mean([r["ecas_before"] - r["ecas_after"] for r in chunk])
        protected_shift = np.mean([r["protected_shift"] for r in chunk])
        non_target_shift = np.mean([r["non_target_shift"] for r in chunk])
        runtime = np.mean([r["runtime_ms"] for r in chunk])
        print(
            f"{category}: "
            f"ecas_improve={ecas_imp:.5f}, "
            f"protected_shift={protected_shift:.5f}, "
            f"non_target_shift={non_target_shift:.5f}, "
            f"runtime_ms={runtime:.2f}"
        )


if __name__ == "__main__":
    main()
