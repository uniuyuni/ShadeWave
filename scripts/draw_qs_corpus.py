#!/usr/bin/env python3
"""Corpus regression tool for Draw Quick Select (min-cut).

Turns the eyeball loop into objective, reproducible metrics over the whole
``edge_refine_debug/qs_input_*.npz`` corpus.

Subcommands::

    report  [--glob G] [--names ...] [--solver v1|v2|v3] [--zoom]
            [--json] [--no-determinism] [--no-idempotence]
        Print the metric table (or JSON) for the selected dumps.

    baseline [--glob G] [--names ...] [--solver v1|v2|v3]
        Write/update tests/fixtures/draw_qs_baseline.json. With --names it merges
        only those entries; otherwise it rewrites the whole baseline.

    add <name> [--from PATH]
        Enroll a freshly captured npz as qs_input_<name>.npz and baseline it.
        Without --from it picks the newest qs_input_*.npz in edge_refine_debug/.
        This is the 1-command "capture failing case -> baseline" step.

    sweep <KNOB> <values...> [--glob G]
        Perturb a QS_* env knob and report which dumps/metrics move relative to
        the default. A knob whose perturbation moves no metric is a removal
        CANDIDATE; one that moves >=1 metric on >=2 dumps is CORE.

    sweep-continuity [--glob G] [--names ...] [--solver v1|v2|v3] [--json]
        Diagnose control cliffs: sweep EdgeLock/Radius/Edge Bias over a fine grid
        and report max_step_frac (cliff indicator) / range_frac per dump.

    contact-v2 [--glob G] [--names ...] [--out DIR]
        Write V1/V2/zoom contact sheets for visual QA.

    pair <A> <B> [--solver v1|v2|v3]
        Report seam/alpha gap metrics for two opposite-side dumps.

    export-labels [--names ...] [--solver v1|v2|v3] [--out DIR]
        Export npz dumps to PNGs that can be edited into golden masks.

    label-report [--names ...] [--solver v1|v2|v3] [--label-dir DIR]
        Compare solver output to edited ``*_expected.png`` masks.

    label-diff [--names ...] [--solver v1|v2|v3] [--label-dir DIR] [--out DIR]
        Write visual label diffs for edited ``*_expected.png`` masks.

Capture loop (to grow the corpus):
    1. Run the app with QS_DUMP_INPUT=edge_refine_debug (or PLATYPUS_DEBUG_EDGE_REFINE=1).
    2. Reproduce the bad stroke; a new qs_input_NNN.npz appears.
    3. python scripts/draw_qs_corpus.py add <descriptive_name>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cores.mask2 import draw_qs_metrics as M

BASELINE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "draw_qs_baseline.json"
DEFAULT_GLOB = "qs_input_*.npz"


# --- path selection ----------------------------------------------------------
def _select_paths(glob: str, names) -> list[Path]:
    if names:
        paths = []
        for name in names:
            p = M.CORPUS_DIR / f"qs_input_{name}.npz"
            if not p.exists():
                raise SystemExit(f"missing dump: {p}")
            paths.append(p)
        return paths
    return M.corpus_paths(glob)


def _newest_dump() -> Path | None:
    candidates = sorted(
        M.CORPUS_DIR.glob(DEFAULT_GLOB), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


# --- baseline IO -------------------------------------------------------------
def _load_baseline() -> dict:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return {}


def _write_baseline(data: dict) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


# --- pretty print ------------------------------------------------------------
_TABLE_COLS = (
    ("ratio", "support_hint_ratio", "{:.3f}"),
    ("edge_b", "edge_boundary_frac", "{:.3f}"),
    ("z2", "zoom_iou_2_0x", "{:.3f}"),
    ("out_px", "outside_px", "{:d}"),
    ("out_noedge", "outside_no_edge_px", "{:d}"),
    ("overgrow", "outside_overgrowth_dist", "{:.1f}"),
    ("far", "far_blob_px", "{:d}"),
    ("comp", "comp_count", "{:d}"),
    ("nodes", "graph_nodes", "{:d}"),
    ("lock", "edge_lock_effective", "{:.1f}"),
    ("det", "deterministic", "{}"),
    ("idem", "idempotence_iou", "{:.3f}"),
    ("ms", "runtime_ms", "{:.0f}"),
)

_LABEL_COLS = (
    ("iou", "label_iou", "{:.3f}"),
    ("b_f1", "label_boundary_f1", "{:.3f}"),
    ("prec", "label_precision", "{:.3f}"),
    ("recall", "label_recall", "{:.3f}"),
    ("fp", "label_fp_px", "{:d}"),
    ("fn", "label_fn_px", "{:d}"),
    ("pred", "label_pred_px", "{:d}"),
    ("truth", "label_truth_px", "{:d}"),
)


def _print_table(rep: dict) -> None:
    header = f"{'dump':14s} " + " ".join(f"{label:>10s}" for label, _, _ in _TABLE_COLS)
    print(header)
    for name in sorted(rep):
        met = rep[name]
        cells = []
        for _, key, fmt in _TABLE_COLS:
            val = met.get(key)
            cells.append(fmt.format(val) if val is not None else "-")
        print(f"{name:14s} " + " ".join(f"{c:>10s}" for c in cells))


def _print_label_table(rep: dict) -> None:
    header = f"{'dump':14s} " + " ".join(f"{label:>10s}" for label, _, _ in _LABEL_COLS)
    print(header)
    for name in sorted(rep):
        met = rep[name]
        cells = []
        for _, key, fmt in _LABEL_COLS:
            val = met.get(key)
            cells.append(fmt.format(val) if val is not None else "-")
        print(f"{name:14s} " + " ".join(f"{c:>10s}" for c in cells))


# --- subcommands -------------------------------------------------------------
def cmd_report(args) -> int:
    paths = _select_paths(args.glob, args.names)
    rep = M.report(
        paths,
        solver=args.solver,
        zoom=args.zoom,
        determinism=not args.no_determinism,
        idempotence=not args.no_idempotence,
    )
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_table(rep)
    return 0


def cmd_baseline(args) -> int:
    paths = _select_paths(args.glob, args.names)
    rep = M.report(paths, solver=args.solver)
    stable = {name: M.stable_metrics(met) for name, met in rep.items()}
    if args.names:
        out = _load_baseline()
        out.update(stable)
    else:
        out = stable
    _write_baseline(out)
    print(f"wrote {len(stable)} entr{'y' if len(stable) == 1 else 'ies'} "
          f"({len(out)} total) -> {BASELINE_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def cmd_add(args) -> int:
    src = Path(args.from_) if args.from_ else _newest_dump()
    if src is None or not src.exists():
        raise SystemExit("no source npz found (pass --from PATH)")
    dst = M.CORPUS_DIR / f"qs_input_{args.name}.npz"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
        print(f"copied {src.name} -> {dst.name}")
    rep = M.report([dst])
    base = _load_baseline()
    base.update({name: M.stable_metrics(met) for name, met in rep.items()})
    _write_baseline(base)
    print(f"baselined '{args.name}' ({len(base)} total). Metrics:")
    _print_table(rep)
    return 0


def _report_via_subprocess(env_overrides: dict, glob: str, solver: str | None = None) -> dict:
    env = dict(os.environ)
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    cmd = [
        sys.executable, str(Path(__file__)), "report",
        "--json", "--glob", glob, "--no-determinism", "--no-idempotence",
    ]
    if solver:
        cmd.extend(["--solver", solver])
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    if proc.returncode != 0:
        raise RuntimeError(f"report subprocess failed:\n{proc.stderr}")
    out = proc.stdout
    start = out.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON in report output:\n{out}")
    return json.loads(out[start:])


def cmd_sweep(args) -> int:
    knob = args.knob
    print(f"# sweep {knob}: default vs {args.values}")
    default = _report_via_subprocess({knob: None}, args.glob, args.solver)
    any_core = False
    for value in args.values:
        perturbed = _report_via_subprocess({knob: value}, args.glob, args.solver)
        moved = []
        for name in sorted(default):
            if name not in perturbed:
                continue
            regs = M.compare(default[name], perturbed[name])
            for r in regs:
                moved.append(f"{name}:{r['metric']}({r['baseline']}->{r['current']})")
        dumps_moved = len({m.split(':')[0] for m in moved})
        tag = "CORE" if dumps_moved >= 2 else ("weak" if dumps_moved == 1 else "no-op")
        if dumps_moved >= 2:
            any_core = True
        print(f"\n{knob}={value}  [{tag}] {dumps_moved} dump(s) moved")
        for m in moved:
            print(f"    {m}")
    print(f"\n=> {knob} is {'CORE (keep)' if any_core else 'a removal CANDIDATE'}")
    return 0


def cmd_sweep_continuity(args) -> int:
    """Diagnose control cliffs: how jumpy is the output as each knob is swept.

    For every selected dump it sweeps EdgeLock offset / Radius / Edge Bias over a
    fine grid and prints, per control, ``max_step_frac`` (the largest support jump
    between adjacent grid points, normalized by the drawn hint -- a cliff) and
    ``range_frac`` (total travel). A predictable control has a small step even
    when the range is large. This is the measurement Phase 1 de-cliffs against.
    """
    paths = _select_paths(args.glob, args.names)
    rows = {}
    for path in paths:
        dump = M.load_dump(path)
        rows[dump["name"]] = M.continuity_metrics_for_dump(dump, solver=args.solver)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    controls = M.CONTINUITY_CONTROLS
    header = f"{'dump':14s} " + " ".join(
        f"{c[:6] + '_step':>12s} {c[:6] + '_rng':>10s}" for c in controls)
    print(header)
    worst = 0.0
    for name in sorted(rows):
        met = rows[name]
        cells = []
        for c in controls:
            step = met.get(f"{c}_max_step_frac")
            rng = met.get(f"{c}_range_frac")
            if step is not None:
                worst = max(worst, step)
            cells.append(f"{step:>12.4f}" if step is not None else f"{'-':>12s}")
            cells.append(f"{rng:>10.4f}" if rng is not None else f"{'-':>10s}")
        print(f"{name:14s} " + " ".join(cells))
    print("\n# *_step = max support jump between adjacent grid points / hint_px (cliff indicator)")
    print(f"# worst max_step_frac across selected dumps: {worst:.4f}")
    return 0


def cmd_contact_v2(args) -> int:
    paths = _select_paths(args.glob, args.names)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        dump = M.load_dump(path)
        sheet = M.contact_sheet_for_dump(dump, solvers=("v1", "v2"))
        out_path = out_dir / f"{dump['name']}_v1_v2.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(out_path.relative_to(PROJECT_ROOT))
    return 0


def cmd_pair(args) -> int:
    a = M.load_dump(M.CORPUS_DIR / f"qs_input_{args.a}.npz")
    b = M.load_dump(M.CORPUS_DIR / f"qs_input_{args.b}.npz")
    met = M.pair_metrics(
        a,
        b,
        solver=args.solver,
        alpha_threshold=args.alpha_threshold,
        seam_radius=args.seam_radius,
    )
    if args.json:
        print(json.dumps(met, indent=2, sort_keys=True))
    else:
        for key, value in met.items():
            print(f"{key}: {value}")
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet = M.pair_contact_sheet(
            a,
            b,
            solver=args.solver,
            alpha_threshold=args.alpha_threshold,
            seam_radius=args.seam_radius,
        )
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"wrote: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        arr = np.asarray(arr, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def _support_overlay(dump, solver: str) -> tuple[np.ndarray, np.ndarray]:
    guide = M._normalize_guide_for_display(dump["guide"])
    hint = np.asarray(dump["mask"], dtype=np.float32) > M.HINT_THRESH
    solved = M.solve(dump, solver=solver)
    support = np.asarray(solved["support"], dtype=bool)

    tint = np.zeros_like(guide)
    tint[hint & support] = (0, 220, 80)
    tint[hint & ~support] = (255, 45, 45)
    tint[~hint & support] = (70, 130, 255)
    overlay = (guide.astype(np.float32) * 0.55 + tint.astype(np.float32) * 0.45).astype(np.uint8)
    boundary = M._boundary(support)
    overlay[boundary] = (255, 255, 0)
    return support, overlay


def _label_diff_overlay(dump, expected_path: Path, roi_path: Path | None, solver: str) -> np.ndarray:
    guide = M._normalize_guide_for_display(dump["guide"])
    solved = M.solve(dump, solver=solver)
    support = np.asarray(solved["support"], dtype=bool)
    expected = M.load_label_mask(expected_path, support.shape)
    if roi_path is not None and roi_path.exists():
        roi = M.load_label_mask(roi_path, support.shape)
    else:
        roi = np.ones(support.shape, dtype=bool)

    pred = support & roi
    truth = expected & roi
    tint = np.zeros_like(guide)
    tint[pred & truth] = (0, 220, 80)
    tint[pred & ~truth] = (255, 45, 45)
    tint[~pred & truth] = (45, 110, 255)
    overlay = (guide.astype(np.float32) * 0.45 + tint.astype(np.float32) * 0.55).astype(np.uint8)
    pred_boundary = M._boundary(pred)
    truth_boundary = M._boundary(truth)
    overlay[truth_boundary] = (255, 255, 255)
    overlay[pred_boundary] = (255, 255, 0)
    return overlay


def cmd_export_labels(args) -> int:
    paths = _select_paths(args.glob, args.names)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        dump = M.load_dump(path)
        name = dump["name"]
        guide = M._normalize_guide_for_display(dump["guide"])
        hint = np.clip(np.asarray(dump["mask"], dtype=np.float32), 0.0, 1.0)
        support, overlay = _support_overlay(dump, args.solver)

        _write_png(out_dir / f"{name}_guide.png", guide)
        _write_png(out_dir / f"{name}_hint.png", (hint * 255.0).astype(np.uint8))
        _write_png(out_dir / f"{name}_{args.solver}_support.png", support.astype(np.uint8) * 255)
        _write_png(out_dir / f"{name}_overlay.png", overlay)

        expected_path = out_dir / f"{name}_expected.png"
        if args.overwrite_expected or not expected_path.exists():
            _write_png(expected_path, support.astype(np.uint8) * 255)
        if args.roi:
            roi_path = out_dir / f"{name}_eval_roi.png"
            if args.overwrite_expected or not roi_path.exists():
                _write_png(roi_path, np.full(support.shape, 255, dtype=np.uint8))
        print(out_dir.relative_to(PROJECT_ROOT) / f"{name}_expected.png")
    return 0


def cmd_label_report(args) -> int:
    paths = _select_paths(args.glob, args.names)
    label_dir = Path(args.label_dir)
    if not label_dir.is_absolute():
        label_dir = PROJECT_ROOT / label_dir
    rep = {}
    for path in paths:
        dump = M.load_dump(path)
        expected = label_dir / f"{dump['name']}_expected.png"
        if not expected.exists():
            if args.require_all:
                raise SystemExit(f"missing expected label: {expected}")
            continue
        roi = label_dir / f"{dump['name']}_eval_roi.png"
        rep[dump["name"]] = M.label_metrics_for_dump(
            dump,
            expected,
            roi_path=roi if roi.exists() else None,
            solver=args.solver,
            boundary_tolerance=args.boundary_tolerance,
        )
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        _print_label_table(rep)
    return 0


def cmd_label_diff(args) -> int:
    paths = _select_paths(args.glob, args.names)
    label_dir = Path(args.label_dir)
    if not label_dir.is_absolute():
        label_dir = PROJECT_ROOT / label_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        dump = M.load_dump(path)
        expected = label_dir / f"{dump['name']}_expected.png"
        if not expected.exists():
            if args.require_all:
                raise SystemExit(f"missing expected label: {expected}")
            continue
        roi = label_dir / f"{dump['name']}_eval_roi.png"
        overlay = _label_diff_overlay(
            dump,
            expected,
            roi if roi.exists() else None,
            args.solver,
        )
        out_path = out_dir / f"{dump['name']}_label_diff.png"
        _write_png(out_path, overlay)
        print(out_path.relative_to(PROJECT_ROOT))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="print metric table / JSON")
    p_report.add_argument("--glob", default=DEFAULT_GLOB)
    p_report.add_argument("--names", nargs="*")
    p_report.add_argument("--solver", choices=("v1", "v2", "v3", "v4"), default=None)
    p_report.add_argument("--zoom", action="store_true")
    p_report.add_argument("--json", action="store_true")
    p_report.add_argument("--no-determinism", action="store_true")
    p_report.add_argument("--no-idempotence", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_base = sub.add_parser("baseline", help="write/update the golden baseline JSON")
    p_base.add_argument("--glob", default=DEFAULT_GLOB)
    p_base.add_argument("--names", nargs="*")
    p_base.add_argument("--solver", choices=("v1", "v2", "v3", "v4"), default=None)
    p_base.set_defaults(func=cmd_baseline)

    p_add = sub.add_parser("add", help="enroll a captured npz and baseline it")
    p_add.add_argument("name")
    p_add.add_argument("--from", dest="from_", default=None)
    p_add.set_defaults(func=cmd_add)

    p_sweep = sub.add_parser("sweep", help="perturb a QS_* knob, show metric movement")
    p_sweep.add_argument("knob")
    p_sweep.add_argument("values", nargs="+")
    p_sweep.add_argument("--glob", default=DEFAULT_GLOB)
    p_sweep.add_argument("--solver", choices=("v1", "v2", "v3", "v4"), default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    p_cont = sub.add_parser("sweep-continuity",
                            help="diagnose control cliffs (EdgeLock/Radius/Edge Bias)")
    p_cont.add_argument("--glob", default=DEFAULT_GLOB)
    p_cont.add_argument("--names", nargs="*")
    p_cont.add_argument("--solver", choices=("v1", "v2", "v3", "v4"), default=None)
    p_cont.add_argument("--json", action="store_true")
    p_cont.set_defaults(func=cmd_sweep_continuity)

    p_contact = sub.add_parser("contact-v2", help="write V1/V2 contact sheets")
    p_contact.add_argument("--glob", default=DEFAULT_GLOB)
    p_contact.add_argument("--names", nargs="*")
    p_contact.add_argument("--out", default="edge_refine_debug/v2_contact")
    p_contact.set_defaults(func=cmd_contact_v2)

    p_pair = sub.add_parser("pair", help="measure opposite-side seam gap")
    p_pair.add_argument("a")
    p_pair.add_argument("b")
    p_pair.add_argument("--solver", choices=("v1", "v2", "v3", "v4"), default=None)
    p_pair.add_argument("--alpha-threshold", type=float, default=0.5)
    p_pair.add_argument("--seam-radius", type=float, default=4.0)
    p_pair.add_argument("--out", default=None)
    p_pair.add_argument("--json", action="store_true")
    p_pair.set_defaults(func=cmd_pair)

    p_export = sub.add_parser("export-labels", help="write editable golden-mask PNGs")
    p_export.add_argument("--glob", default=DEFAULT_GLOB)
    p_export.add_argument("--names", nargs="*")
    p_export.add_argument("--solver", choices=("v1", "v2", "v3", "v4"), default="v2")
    p_export.add_argument("--out", default="edge_refine_debug/label_exports")
    p_export.add_argument("--overwrite-expected", action="store_true")
    p_export.add_argument("--roi", action="store_true",
                          help="also create an all-white <name>_eval_roi.png")
    p_export.set_defaults(func=cmd_export_labels)

    p_label = sub.add_parser("label-report", help="compare against edited expected PNGs")
    p_label.add_argument("--glob", default=DEFAULT_GLOB)
    p_label.add_argument("--names", nargs="*")
    p_label.add_argument("--solver", choices=("v1", "v2", "v3", "v4"), default="v2")
    p_label.add_argument("--label-dir", default="edge_refine_debug/label_exports")
    p_label.add_argument("--boundary-tolerance", type=float, default=2.0)
    p_label.add_argument("--require-all", action="store_true")
    p_label.add_argument("--json", action="store_true")
    p_label.set_defaults(func=cmd_label_report)

    p_diff = sub.add_parser("label-diff", help="write visual diffs against edited expected PNGs")
    p_diff.add_argument("--glob", default=DEFAULT_GLOB)
    p_diff.add_argument("--names", nargs="*")
    p_diff.add_argument("--solver", choices=("v1", "v2", "v3", "v4"), default="v2")
    p_diff.add_argument("--label-dir", default="edge_refine_debug/label_exports")
    p_diff.add_argument("--out", default="edge_refine_debug/label_eval")
    p_diff.add_argument("--require-all", action="store_true")
    p_diff.set_defaults(func=cmd_label_diff)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
