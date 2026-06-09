"""Label-free regression metrics for Draw Quick Select (min-cut).

The goal is to replace the eyeball loop ("does this one npz look right?") with an
objective, reproducible metric suite that runs ``compute_draw_support`` over the
whole ``edge_refine_debug/qs_input_*.npz`` corpus and flags regressions WITHOUT
needing per-image hand labels.

All metrics are computable from a dump alone:

* ``support_hint_ratio`` / areas    -- gross size sanity (no collapse / explosion).
* ``edge_boundary_frac``            -- does the support boundary sit on real image
                                       edges (i.e. did it actually snap)?
* ``outside_*`` / ``overgrowth``    -- growth past the drawn footprint, and growth
                                       in *featureless* regions (= inflation).
* ``far_blob_px``                   -- support disconnected from the seed core
                                       (a runaway grab; should be ~0).
* ``deterministic`` / ``idempotence_iou`` -- the solve is stable and a fixed point.

Used by ``tests/test_edge_refine.py`` (corpus baseline test) and the
``scripts/draw_qs_corpus.py`` CLI. Keep this module Kivy-free and side-effect free
so it imports cleanly in headless/test contexts.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover - scipy is a hard dependency in practice
    distance_transform_edt = None

from cores.mask2 import draw_quick_select, edge_refine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "edge_refine_debug"

# Edge-strength thresholds mirror scripts/check_draw_qs_pair_union.py so the
# harness and that tool speak the same language.
EDGE_T = 0.4
EDGE_NEAR = 2
# A support pixel further than this from the drawn hint counts as "outside growth".
HINT_THRESH = 0.02

# Metrics whose value is a stable function of (code, npz). These go into the
# golden baseline and are compared with tolerance. runtime_ms is intentionally
# excluded (machine dependent); it is asserted only against a ceiling.
STABLE_METRICS = (
    "support_px",
    "hint_px",
    "band_px",
    "comp_count",
    "support_hint_ratio",
    "edge_boundary_frac",
    "edge_boundary_median",
    "outside_px",
    "outside_no_edge_px",
    "outside_overgrowth_dist",
    "far_blob_px",
    "deterministic",
    "idempotence_iou",
)


# --- dump IO -----------------------------------------------------------------
def _normalize_strokes(raw_strokes) -> List[SimpleNamespace]:
    strokes: List[SimpleNamespace] = []
    for raw in raw_strokes:
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
        if isinstance(raw, dict):
            points = np.asarray(raw.get("points", []), dtype=np.float32)
            pts = (
                [(float(x), float(y)) for x, y in points[:, :2]]
                if points.size
                else []
            )
            strokes.append(
                SimpleNamespace(
                    points=pts,
                    size=float(raw.get("size", 1.0)),
                    soft=float(raw.get("soft", 100.0)),
                    is_erasing=bool(raw.get("is_erasing", False)),
                )
            )
        else:
            strokes.append(raw)
    return strokes


def load_dump(path) -> dict:
    """Load a ``qs_input_*.npz`` into the kwargs ``compute_draw_support`` expects.

    Mirrors ``tests/test_edge_refine.py:_load_qs_input`` but stands alone so the
    metrics module has no test dependency.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    files = set(getattr(data, "files", []))
    strokes = _normalize_strokes(list(data["strokes"])) if "strokes" in files else []
    seed_mask = None
    if "seed_mask" in files:
        sm = data["seed_mask"]
        seed_mask = sm if getattr(sm, "size", 0) else None
    name = path.stem
    if name.startswith("qs_input_"):
        name = name[len("qs_input_"):]
    return {
        "name": name,
        "path": str(path),
        "guide": np.asarray(data["guide"], dtype=np.float32),
        "mask": np.asarray(data["mask"], dtype=np.float32),
        "seed_mask": seed_mask,
        "radius": float(data["radius"]),
        "strength": float(data["strength"]),
        "pixel_scale": float(data["pixel_scale"]) if "pixel_scale" in files else 1.0,
        "strokes": strokes,
    }


def corpus_paths(glob: str = "qs_input_*.npz") -> List[Path]:
    return sorted(CORPUS_DIR.glob(glob))


def _solver_module(solver: Optional[str] = None):
    value = (solver or os.environ.get("QS_DRAW_SOLVER") or "").strip().lower()
    if not value and os.environ.get("QS_DRAW_V2", "").strip().lower() in {"1", "true", "yes", "on"}:
        value = "v2"
    if value in {"v2", "2"}:
        from cores.mask2 import draw_quick_select_v2

        return draw_quick_select_v2
    return draw_quick_select


# --- solve -------------------------------------------------------------------
def _solve_support(dump, *, solver: Optional[str] = None):
    module = _solver_module(solver)
    return module.compute_draw_support(
        dump["guide"],
        dump["mask"],
        dump["radius"],
        dump["strength"],
        seed_mask=dump.get("seed_mask"),
        draw_strokes=dump.get("strokes"),
        pixel_scale=dump.get("pixel_scale", 1.0),
    )


def solve(dump, *, solver: Optional[str] = None) -> dict:
    """Run the production solve + matte and return arrays the metrics consume."""
    res = _solve_support(dump, solver=solver)
    refined = edge_refine._compose_refined_mask(
        dump["mask"],
        res.support,
        True,
        guide=dump["guide"],
        natural_edge=True,
        edge_lock=float(dump["strength"]),
    )
    return {
        "support": np.asarray(res.support, dtype=bool),
        "candidate": np.asarray(res.candidate, dtype=bool),
        "seed": np.asarray(res.seed, dtype=bool),
        "refined": np.asarray(refined, dtype=np.float32),
        "planes": {name: value for name, value in res.debug_planes},
        "edge_lock": float(getattr(res, "edge_lock", dump.get("strength", 60.0))),
    }


# --- metric helpers ----------------------------------------------------------
def _iou(a, b) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def _edge_strength(guide, shape) -> np.ndarray:
    edge = edge_refine._draw_snap_edge_strength(guide)
    if edge is None:
        return np.zeros(shape, dtype=np.float32)
    return np.clip(np.asarray(edge, dtype=np.float32), 0.0, 1.0)


def _near_edge(edge: np.ndarray) -> np.ndarray:
    near = edge >= EDGE_T
    if EDGE_NEAR > 0 and np.any(near):
        near = cv2.dilate(
            near.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=int(EDGE_NEAR),
        ) > 0
    return near


def _boundary(support: np.ndarray) -> np.ndarray:
    if not np.any(support):
        return np.zeros_like(support, dtype=bool)
    grad = cv2.morphologyEx(
        support.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    return grad > 0


def _far_blob_px(support: np.ndarray, seed: np.ndarray) -> int:
    if not np.any(support):
        return 0
    n_labels, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    if n_labels <= 2:
        return 0
    seed_on = seed & support
    keep_labels = set(int(v) for v in np.unique(labels[seed_on])) - {0}
    if not keep_labels:
        areas = np.bincount(labels.reshape(-1), minlength=n_labels)
        areas[0] = 0
        keep_labels = {int(areas.argmax())}
    keep = np.isin(labels, list(keep_labels))
    return int(np.count_nonzero(support & ~keep))


def _component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    n_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return int(n_labels - 1)


# --- zoom/scaling -------------------------------------------------------------
def _scale_strokes(strokes, scale: float) -> List[SimpleNamespace]:
    out: List[SimpleNamespace] = []
    for stroke in strokes or []:
        pts = np.asarray(getattr(stroke, "points", []), dtype=np.float32)
        out.append(SimpleNamespace(
            points=[(float(x) * scale, float(y) * scale) for x, y in pts[:, :2]],
            size=float(getattr(stroke, "size", 1.0)) * scale,
            soft=float(getattr(stroke, "soft", 100.0)),
            is_erasing=bool(getattr(stroke, "is_erasing", False)),
        ))
    return out


def scaled_dump(dump, scale: float) -> dict:
    """Return a logically equivalent dump scaled by ``scale``."""
    scale = float(scale)
    h, w = dump["mask"].shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    guide = cv2.resize(dump["guide"], (nw, nh), interpolation=interp)
    mask = cv2.resize(dump["mask"], (nw, nh), interpolation=interp)
    seed = dump.get("seed_mask")
    if seed is not None:
        seed = cv2.resize(seed.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    out = dict(dump)
    out.update({
        "guide": guide.astype(np.float32, copy=False),
        "mask": mask.astype(np.float32, copy=False),
        "seed_mask": seed,
        "radius": float(dump["radius"]) * scale,
        "pixel_scale": float(dump.get("pixel_scale", 1.0)) * scale,
        "strokes": _scale_strokes(dump.get("strokes"), scale),
    })
    return out


def zoom_metrics_for_dump(
        dump,
        *,
        solver: Optional[str] = None,
        scales=(2.0, 0.5)) -> dict:
    """Solve scaled versions and compare support after resampling to base size."""
    base = np.asarray(_solve_support(dump, solver=solver).support, dtype=bool)
    h, w = base.shape[:2]
    out = {}
    for scale in scales:
        sd = scaled_dump(dump, float(scale))
        ss = np.asarray(_solve_support(sd, solver=solver).support, dtype=bool)
        back = cv2.resize(ss.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        key = str(scale).replace(".", "_")
        out[f"zoom_iou_{key}x"] = round(_iou(base, back), 6)
        out[f"zoom_diff_px_{key}x"] = int(np.count_nonzero(base ^ back))
    return out


# --- per-dump metrics --------------------------------------------------------
def metrics_for_dump(
        dump,
        *,
        solver: Optional[str] = None,
        determinism: bool = True,
        idempotence: bool = True,
        zoom: bool = False) -> dict:
    """Compute the full label-free metric set for one dump."""
    t0 = time.perf_counter()
    res = _solve_support(dump, solver=solver)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    support = np.asarray(res.support, dtype=bool)
    candidate = np.asarray(res.candidate, dtype=bool)
    seed = np.asarray(res.seed, dtype=bool)
    hint = np.asarray(dump["mask"], dtype=np.float32) > HINT_THRESH
    shape = support.shape

    edge = _edge_strength(dump["guide"], shape)
    near_edge = _near_edge(edge)

    support_px = int(np.count_nonzero(support))
    hint_px = int(np.count_nonzero(hint))

    boundary = _boundary(support)
    boundary_px = int(np.count_nonzero(boundary))
    if boundary_px:
        edge_boundary_frac = float(np.count_nonzero(boundary & near_edge) / boundary_px)
        edge_boundary_median = float(np.median(edge[boundary]))
    else:
        edge_boundary_frac = 0.0
        edge_boundary_median = 0.0

    outside = support & ~hint
    outside_px = int(np.count_nonzero(outside))
    outside_no_edge_px = int(np.count_nonzero(outside & ~near_edge))
    if outside_px and distance_transform_edt is not None:
        dist = distance_transform_edt(~hint)
        outside_overgrowth_dist = float(dist[outside].max())
    else:
        outside_overgrowth_dist = 0.0

    far_blob_px = _far_blob_px(support, seed)

    deterministic = True
    if determinism:
        support2 = np.asarray(_solve_support(dump, solver=solver).support, dtype=bool)
        deterministic = bool(np.array_equal(support, support2))

    idempotence_iou = 1.0
    if idempotence:
        dump2 = dict(dump)
        dump2["mask"] = support.astype(np.float32)
        support3 = np.asarray(_solve_support(dump2, solver=solver).support, dtype=bool)
        idempotence_iou = _iou(support, support3)

    metrics = {
        "support_px": support_px,
        "hint_px": hint_px,
        "band_px": int(np.count_nonzero(candidate)),
        "comp_count": _component_count(support),
        "support_hint_ratio": float(support_px / hint_px) if hint_px else 0.0,
        "edge_boundary_frac": round(edge_boundary_frac, 6),
        "edge_boundary_median": round(edge_boundary_median, 6),
        "outside_px": outside_px,
        "outside_no_edge_px": outside_no_edge_px,
        "outside_overgrowth_dist": round(outside_overgrowth_dist, 4),
        "far_blob_px": far_blob_px,
        "deterministic": bool(deterministic),
        "idempotence_iou": round(idempotence_iou, 6),
        "runtime_ms": round(runtime_ms, 1),
        "solver": (solver or os.environ.get("QS_DRAW_SOLVER") or ("v2" if os.environ.get("QS_DRAW_V2") else "v1")),
    }
    if zoom:
        metrics.update(zoom_metrics_for_dump(dump, solver=solver))
    return metrics


# --- regression comparison (shared by the corpus test and the sweep CLI) -----
# Areas use a tolerance relative to the drawn footprint; ratios/fractions use an
# absolute tolerance; structural metrics (determinism / comp_count) are exact.
RATIO_ABS_TOL = 0.03
FRAC_ABS_TOL = 0.05
DIST_ABS_TOL = 3.0
AREA_REL_TOL = 0.01  # fraction of hint_px
AREA_MIN_TOL = 50
FAR_BLOB_ABS_TOL = 8
IDEMPOTENCE_DROP_TOL = 0.03

_AREA_METRICS = ("support_px", "hint_px", "band_px", "outside_px", "outside_no_edge_px")


def compare(baseline_metrics: dict, current: dict) -> List[dict]:
    """Return the list of regressions of ``current`` against ``baseline_metrics``.

    Empty list == no regression. Used by the corpus baseline test (fail on any)
    and by the sweep CLI (a knob is CORE if perturbing it produces regressions).
    """
    regressions: List[dict] = []
    hint_px = max(1, int(baseline_metrics.get("hint_px", current.get("hint_px", 1))))
    area_tol = max(AREA_MIN_TOL, AREA_REL_TOL * hint_px)

    def add(metric, b, c, why):
        regressions.append({"metric": metric, "baseline": b, "current": c, "reason": why})

    for metric in STABLE_METRICS:
        if metric not in baseline_metrics or metric not in current:
            continue
        b = baseline_metrics[metric]
        c = current[metric]
        if metric == "deterministic":
            if not c:
                add(metric, b, c, "solve became non-deterministic")
        elif metric == "idempotence_iou":
            if c < b - IDEMPOTENCE_DROP_TOL:
                add(metric, b, c, f"idempotence dropped >{IDEMPOTENCE_DROP_TOL}")
        elif metric == "comp_count":
            if c != b:
                add(metric, b, c, "component count changed")
        elif metric == "support_hint_ratio":
            if abs(c - b) > RATIO_ABS_TOL:
                add(metric, b, c, f"ratio moved >{RATIO_ABS_TOL}")
        elif metric in ("edge_boundary_frac", "edge_boundary_median"):
            if abs(c - b) > FRAC_ABS_TOL:
                add(metric, b, c, f"moved >{FRAC_ABS_TOL}")
        elif metric == "outside_overgrowth_dist":
            if abs(c - b) > DIST_ABS_TOL:
                add(metric, b, c, f"moved >{DIST_ABS_TOL}px")
        elif metric == "far_blob_px":
            if c > b + FAR_BLOB_ABS_TOL:
                add(metric, b, c, "far blob grew")
        elif metric in _AREA_METRICS:
            if abs(c - b) > area_tol:
                add(metric, b, c, f"area moved >{area_tol:.0f}px")
    return regressions


def stable_metrics(metrics: dict) -> dict:
    """Project a full metric dict down to the baseline-stored (stable) subset."""
    return {k: metrics[k] for k in STABLE_METRICS if k in metrics}


def report(paths=None, **kw) -> Dict[str, dict]:
    """Compute metrics for every dump path. Returns {name: metrics}."""
    if paths is None:
        paths = corpus_paths()
    out: Dict[str, dict] = {}
    for path in paths:
        dump = load_dump(path)
        out[dump["name"]] = metrics_for_dump(dump, **kw)
    return out


# --- seam-pair metric (generalized scripts/check_draw_qs_pair_union.py) -------
def pair_metrics(
    dump_a,
    dump_b,
    *,
    solver: Optional[str] = None,
    alpha_threshold: float = 0.5,
    edge_threshold: float = EDGE_T,
    edge_near: int = EDGE_NEAR,
    seam_radius: float = 4.0,
) -> dict:
    """Two strokes that share a seam should not leave an alpha gap on it.

    Generalizes ``check_draw_qs_pair_union.py`` so any declared pair can be a
    corpus check, not just the one hard-coded snow/sky pair.
    """
    sa = solve(dump_a, solver=solver)
    sb = solve(dump_b, solver=solver)
    support_a = sa["support"]
    support_b = sb["support"]
    candidate = sa["candidate"] | sb["candidate"]
    edge = _edge_strength(dump_a["guide"], support_a.shape)
    near_edge = edge >= float(edge_threshold)
    if edge_near > 0 and np.any(near_edge):
        near_edge = cv2.dilate(
            near_edge.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=int(edge_near),
        ) > 0

    if distance_transform_edt is None:
        raise RuntimeError("scipy is required for pair_metrics")
    dist_a = distance_transform_edt(~support_a)
    dist_b = distance_transform_edt(~support_b)
    shared = (
        (dist_a <= float(seam_radius))
        & (dist_b <= float(seam_radius))
        & candidate
        & near_edge
    )
    alpha_sum = np.clip(sa["refined"] + sb["refined"], 0.0, 1.0)
    gap = shared & (alpha_sum < float(alpha_threshold))
    alpha_mask = (alpha_sum >= float(alpha_threshold)) & candidate
    shared_px = int(np.count_nonzero(shared))
    return {
        "shared_seam_px": shared_px,
        "gap_px": int(np.count_nonzero(gap)),
        "gap_ratio": round(float(np.count_nonzero(gap) / max(1, shared_px)), 6),
        "alpha_components": _component_count(alpha_mask),
    }


# --- contact sheets ----------------------------------------------------------
def _normalize_guide_for_display(guide: np.ndarray) -> np.ndarray:
    arr = np.asarray(guide, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        p1, p99 = np.percentile(finite, [1.0, 99.0])
    else:
        p1, p99 = 0.0, 1.0
    out = np.clip((arr - p1) / max(float(p99 - p1), 1e-6), 0.0, 1.0)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    return (out[..., :3] * 255.0).astype(np.uint8)


def _label_panel(rgb: np.ndarray, text: str) -> np.ndarray:
    img = np.asarray(rgb, dtype=np.uint8).copy()
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 520), 26), (0, 0, 0), -1)
    cv2.putText(
        img,
        str(text)[:80],
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def contact_sheet_for_dump(
        dump,
        *,
        solvers=("v1", "v2"),
        thumb_size=320) -> np.ndarray:
    """Return a visual V1/V2 comparison sheet for one dump."""
    guide = _normalize_guide_for_display(dump["guide"])
    hint = np.asarray(dump["mask"], dtype=np.float32) > HINT_THRESH
    panels = [_label_panel(guide, f"{dump.get('name', 'dump')} guide")]
    edge = _edge_strength(dump["guide"], hint.shape)
    edge_rgb = cv2.applyColorMap((np.clip(edge, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    panels.append(_label_panel(cv2.cvtColor(edge_rgb, cv2.COLOR_BGR2RGB), "raw edge"))
    for solver in solvers:
        solved = solve(dump, solver=solver)
        support = solved["support"]
        diff = guide.copy()
        tint = np.zeros_like(diff)
        tint[hint & support] = (0, 220, 80)
        tint[hint & ~support] = (255, 45, 45)
        tint[~hint & support] = (70, 130, 255)
        diff = (diff.astype(np.float32) * 0.45 + tint.astype(np.float32) * 0.55).astype(np.uint8)
        boundary = _boundary(support)
        diff[boundary] = (0, 255, 255)
        ratio = float(np.count_nonzero(support) / max(1, np.count_nonzero(hint)))
        panels.append(_label_panel(diff, f"{solver} kept/removed ratio={ratio:.3f}"))

        zmet = zoom_metrics_for_dump(dump, solver=solver, scales=(2.0,))
        sd = scaled_dump(dump, 2.0)
        s2 = np.asarray(_solve_support(sd, solver=solver).support, dtype=bool)
        back = cv2.resize(s2.astype(np.uint8), hint.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        zdiff = np.zeros_like(guide)
        zdiff[support & back] = (70, 220, 80)
        zdiff[support ^ back] = (255, 60, 60)
        panels.append(_label_panel(zdiff, f"{solver} zoom2 IoU={zmet['zoom_iou_2_0x']:.3f}"))

    thumbs = []
    for panel in panels:
        thumbs.append(cv2.resize(panel, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA))
    cols = min(4, len(thumbs))
    rows = []
    for idx in range(0, len(thumbs), cols):
        row = thumbs[idx:idx + cols]
        while len(row) < cols:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)
