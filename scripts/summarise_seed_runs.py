#!/usr/bin/env python3
"""Summarise seed-sweep log directories: mean and sample std over seeded runs.

Given one or more generic run directories that each contain one sub-directory
per seed (e.g. ``logs/lamaml/post-fix_me_til-2026-07-07_23-03-15-6051`` with
``0/``, ``39/``, ``55/`` inside), this reads each seed's ``terminal.log`` and
aggregates the final metrics printed on the ``SUMMARY_TE`` / ``SUMMARY_TR``
lines. When multiple run directories are given (e.g. one per algorithm),
results are printed side by side as rows of a single table.

The headline number is **signal-class F1** (``F1_cls``), computed as the
harmonic mean of ``cls_rec`` and ``cls_prec`` (the signal-only macro
recall/precision). This is deliberately *not* the ``cls_f1`` field
(``F1_CL`` below), which is the mean per-class F1 over all classes including
noise (i.e. f1_total) and does not equal the harmonic mean of the reported
recall/precision.

Table columns:
    F1_cls  -- signal-class F1 (harmonic mean of cls_rec, cls_prec)
    pd      -- probability of detection (detection recall)
    pfa     -- probability of false alarm
    F1_CL   -- continual-learning headline F1 (f1_total, all classes)
    BWT     -- backward transfer (validation split only)

Usage:
    python scripts/summarise_seed_runs.py logs/lamaml/<run-dir>
    python scripts/summarise_seed_runs.py logs/lamaml/<run-dir> logs/agem/<run-dir>
    python scripts/summarise_seed_runs.py logs/lamaml/<run-dir> logs/agem/<run-dir> \\
        --labels lamaml,agem --train --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

# Matches e.g. "cls_rec=0.5867" -> ("cls_rec", "0.5867")
_FIELD_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan)")

# Matches the "Backward: -0.0684" line written to each seed's results.txt.
_BWT_RE = re.compile(r"^Backward:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan)")


def _parse_summary_line(line: str) -> dict[str, float]:
    """Parse a "SUMMARY_TE cls_rec=.. cls_prec=.. ..." line into a dict."""
    out: dict[str, float] = {}
    for key, val in _FIELD_RE.findall(line):
        try:
            out[key] = float(val)
        except ValueError:
            out[key] = float("nan")
    return out


def _find_last_summary(log_path: str, tag: str) -> dict[str, float] | None:
    """Return the fields of the last ``SUMMARY_<tag>`` line in a log, if any."""
    prefix = "SUMMARY_" + tag
    found: dict[str, float] | None = None
    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith(prefix):
                found = _parse_summary_line(line)
    return found


def _read_bwt(seed_dir: str) -> float:
    """Parse the 'Backward:' (BWT) line from a seed's results.txt, else nan."""
    path = os.path.join(seed_dir, "results.txt")
    if not os.path.isfile(path):
        return float("nan")
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _BWT_RE.match(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return float("nan")
    return float("nan")


def _signal_f1(rec: float | None, prec: float | None) -> float:
    """Harmonic mean of signal recall and precision (0 if undefined)."""
    if rec is None or prec is None:
        return float("nan")
    if math.isnan(rec) or math.isnan(prec) or (rec + prec) == 0:
        return 0.0 if (rec == 0 and prec == 0) else float("nan")
    return 2.0 * rec * prec / (rec + prec)


def _infer_algo_label(run_dir: str) -> str:
    """Infer an algorithm label from a run dir's parent (e.g. logs/lamaml/<run> -> "lamaml")."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(run_dir.rstrip(os.sep))))
    return parent or run_dir


def _seed_sort_key(name: str):
    """Sort seed dir names numerically when possible, else lexically."""
    return (0, int(name)) if name.isdigit() else (1, name)


def discover_seed_runs(run_dir: str, tag: str) -> list[tuple[str, dict[str, float]]]:
    """Find (seed_name, summary_fields) for every seed sub-dir with a summary."""
    runs: list[tuple[str, dict[str, float]]] = []
    for name in sorted(os.listdir(run_dir), key=_seed_sort_key):
        sub = os.path.join(run_dir, name)
        if not os.path.isdir(sub):
            continue
        log_path = os.path.join(sub, "terminal.log")
        if not os.path.isfile(log_path):
            continue
        fields = _find_last_summary(log_path, tag)
        if fields is not None:
            runs.append((name, fields))
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and *sample* std (ddof=1). Std is nan for n<2."""
    clean = [v for v in values if not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(clean) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return mean, math.sqrt(var)


def summarise(run_dir: str, tag: str) -> dict:
    """Aggregate one split (TE or TR) across seeds into a structured result."""
    runs = discover_seed_runs(run_dir, tag)
    if not runs:
        raise SystemExit(
            f"No 'SUMMARY_{tag}' lines found under {run_dir!r} "
            "(expected one seed sub-dir per run, each with terminal.log)."
        )

    seeds = [name for name, _ in runs]
    # Per-seed derived signal F1 plus the raw fields we care about.
    per_seed_signal_f1 = [
        _signal_f1(f.get("cls_rec"), f.get("cls_prec")) for _, f in runs
    ]

    # Columns to report straight from the summary line (order preserved).
    raw_keys = ["cls_rec", "cls_prec", "det", "fa", "cls_f1"]
    columns: dict[str, list[float]] = {"signal_f1": per_seed_signal_f1}
    for key in raw_keys:
        columns[key] = [f.get(key, float("nan")) for _, f in runs]

    # BWT lives in each seed's results.txt (validation recall matrix), so it is
    # only meaningful for the validation split.
    if tag == "TE":
        columns["bwt"] = [_read_bwt(os.path.join(run_dir, name)) for name, _ in runs]

    stats = {name: mean_std(vals) for name, vals in columns.items()}
    return {
        "run_dir": run_dir,
        "tag": tag,
        "seeds": seeds,
        "n": len(seeds),
        "columns": columns,
        "stats": stats,
    }


# Table columns for the multi-algorithm summary: (header, underlying column key).
_TABLE_COLUMNS = [
    ("F1_cls", "signal_f1"),
    ("pd", "det"),
    ("pfa", "fa"),
    ("F1_CL", "cls_f1"),
    ("BWT", "bwt"),
]


def _fmt_cell(mean: float, std: float) -> str:
    """Format a "mean±std" cell, falling back to "nan" when undefined."""
    if math.isnan(mean):
        return "nan"
    std_s = "nan" if math.isnan(std) else f"{std:.4f}"
    return f"{mean:.4f}±{std_s}"


def print_algo_table(entries: list[tuple[str, dict]], tag: str) -> None:
    """Print one row per (label, summarise() result) with the shared metric columns."""
    split = "Validation" if tag == "TE" else "Training"
    # BWT is only computed for the validation split (see summarise()).
    columns = [
        (header, key) for header, key in _TABLE_COLUMNS if tag == "TE" or key != "bwt"
    ]

    label_width = max([len("algo")] + [len(label) for label, _ in entries]) + 2
    col_width = 16
    header_row = (
        f"{'algo':<{label_width}}"
        + "".join(f"{header:>{col_width}}" for header, _ in columns)
        + f"{'n':>5}"
    )

    print(f"Seed-sweep summary ({split}, SUMMARY_{tag})")
    print(header_row)
    print("-" * len(header_row))
    for label, result in entries:
        row = f"{label:<{label_width}}"
        for _, key in columns:
            mean, std = result["stats"].get(key, (float("nan"), float("nan")))
            row += f"{_fmt_cell(mean, std):>{col_width}}"
        row += f"{result['n']:>5}"
        print(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more log dirs, each containing one sub-dir per seed (e.g. one dir per algorithm).",
    )
    ap.add_argument(
        "--labels",
        help="Comma-separated row labels, one per run_dir (default: inferred from each run_dir's parent directory name).",
    )
    ap.add_argument(
        "--train",
        action="store_true",
        help="Also report the training split (SUMMARY_TR).",
    )
    ap.add_argument(
        "--only-train",
        action="store_true",
        help="Report only the training split, not validation.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table.",
    )
    args = ap.parse_args(argv)

    for run_dir in args.run_dirs:
        if not os.path.isdir(run_dir):
            ap.error(f"not a directory: {run_dir}")

    if args.labels:
        labels = [label.strip() for label in args.labels.split(",")]
        if len(labels) != len(args.run_dirs):
            ap.error(
                f"--labels has {len(labels)} entries but {len(args.run_dirs)} run_dirs were given"
            )
    else:
        labels = [_infer_algo_label(run_dir) for run_dir in args.run_dirs]

    tags = ["TR"] if args.only_train else ["TE"]
    if args.train and not args.only_train:
        tags.append("TR")

    results_by_tag = {
        tag: [
            (label, summarise(run_dir, tag))
            for label, run_dir in zip(labels, args.run_dirs)
        ]
        for tag in tags
    }

    if args.json:
        json.dump(
            {
                tag: [{"label": label, **result} for label, result in entries]
                for tag, entries in results_by_tag.items()
            },
            sys.stdout,
            indent=2,
        )
        print()
    else:
        for i, tag in enumerate(tags):
            if i:
                print()
            print_algo_table(results_by_tag[tag], tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
