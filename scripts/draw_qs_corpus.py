#!/usr/bin/env python3
"""Corpus regression tool for Draw Quick Select (min-cut).

Turns the eyeball loop into objective, reproducible metrics over the whole
``edge_refine_debug/qs_input_*.npz`` corpus.

Subcommands::

    report  [--glob G] [--names ...] [--solver v1|v2] [--zoom]
            [--json] [--no-determinism] [--no-idempotence]
        Print the metric table (or JSON) for the selected dumps.

    baseline [--glob G] [--names ...] [--solver v1|v2]
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

    contact-v2 [--glob G] [--names ...] [--out DIR]
        Write V1/V2/zoom contact sheets for visual QA.

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
    ("det", "deterministic", "{}"),
    ("idem", "idempotence_iou", "{:.3f}"),
    ("ms", "runtime_ms", "{:.0f}"),
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="print metric table / JSON")
    p_report.add_argument("--glob", default=DEFAULT_GLOB)
    p_report.add_argument("--names", nargs="*")
    p_report.add_argument("--solver", choices=("v1", "v2"), default=None)
    p_report.add_argument("--zoom", action="store_true")
    p_report.add_argument("--json", action="store_true")
    p_report.add_argument("--no-determinism", action="store_true")
    p_report.add_argument("--no-idempotence", action="store_true")
    p_report.set_defaults(func=cmd_report)

    p_base = sub.add_parser("baseline", help="write/update the golden baseline JSON")
    p_base.add_argument("--glob", default=DEFAULT_GLOB)
    p_base.add_argument("--names", nargs="*")
    p_base.add_argument("--solver", choices=("v1", "v2"), default=None)
    p_base.set_defaults(func=cmd_baseline)

    p_add = sub.add_parser("add", help="enroll a captured npz and baseline it")
    p_add.add_argument("name")
    p_add.add_argument("--from", dest="from_", default=None)
    p_add.set_defaults(func=cmd_add)

    p_sweep = sub.add_parser("sweep", help="perturb a QS_* knob, show metric movement")
    p_sweep.add_argument("knob")
    p_sweep.add_argument("values", nargs="+")
    p_sweep.add_argument("--glob", default=DEFAULT_GLOB)
    p_sweep.add_argument("--solver", choices=("v1", "v2"), default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    p_contact = sub.add_parser("contact-v2", help="write V1/V2 contact sheets")
    p_contact.add_argument("--glob", default=DEFAULT_GLOB)
    p_contact.add_argument("--names", nargs="*")
    p_contact.add_argument("--out", default="edge_refine_debug/v2_contact")
    p_contact.set_defaults(func=cmd_contact_v2)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
