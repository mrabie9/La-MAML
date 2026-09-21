#!/usr/bin/env python3
"""Summarise every model's seed-sweep run under a ``logs/00_sync`` group directory.

Given a group's ``saved_models`` directory (e.g.
``logs/00_sync/5e_CIL/saved_models``), this reads each model's seed subdirs
(``<model>/<run>/<seed>/seed_metrics.json``) and reports validation macro
recall/precision/F1 and backward transfer, meaned over seeds.

Rows are ordered by macro F1, lowest to highest, with two baselines pinned to
the ends regardless of their score: ``ft`` (lower-bound, no continual learning
strategy) always leads the table and ``iid2`` (upper-bound, joint/IID
training) always closes it.

When a model directory holds more than one run (e.g. mid seed-sweep
migration), the lexicographically last run directory name is used, which is
also the most recent since run directories are timestamp-prefixed.

Usage:
    python scripts/summarise_group_runs.py logs/00_sync/5e_CIL/saved_models
    python scripts/summarise_group_runs.py logs/00_sync/5e_CIL/saved_models --format latex
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

PINNED_TOP = "ft"
PINNED_BOTTOM = "iid2"

METRIC_KEYS = {
    "rec": "val_macro_rec",
    "prec": "val_macro_prec",
    "f1": "val_macro_f1",
    "bwt": "val_bwt_f1",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and *sample* std (ddof=1) of a list of floats; std is nan for n<2."""
    clean = [v for v in values if not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(clean) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    return mean, math.sqrt(var)


def find_latest_run_dir(model_dir: str) -> str:
    """Return the lexicographically last (most recent) completed run subdir.

    A run subdir only counts if it has a top-level ``results.txt``; smoke
    tests and abandoned/in-progress runs are left without one and are
    skipped, per the ``logs/00_sync`` promotion convention.
    """
    candidates = sorted(
        name
        for name in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, name))
        and os.path.isfile(os.path.join(model_dir, name, "results.txt"))
    )
    if not candidates:
        raise SystemExit(
            f"No completed run directories (with results.txt) found under {model_dir!r}"
        )
    return os.path.join(model_dir, candidates[-1])


def find_seed_dirs(run_dir: str) -> list[str]:
    """Return every seed subdir of a run dir that has a ``seed_metrics.json``."""
    seed_dirs = []
    for name in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, name)
        if os.path.isdir(sub) and os.path.isfile(
            os.path.join(sub, "seed_metrics.json")
        ):
            seed_dirs.append(sub)
    return seed_dirs


def summarise_model(model_dir: str) -> dict:
    """Aggregate one model's seeded runs into mean/std stats per metric."""
    run_dir = find_latest_run_dir(model_dir)
    seed_dirs = find_seed_dirs(run_dir)
    if not seed_dirs:
        raise SystemExit(f"No seed_metrics.json found under {run_dir!r}")

    per_seed_metrics = []
    for seed_dir in seed_dirs:
        with open(os.path.join(seed_dir, "seed_metrics.json")) as fh:
            per_seed_metrics.append(json.load(fh))

    stats = {}
    for column_name, json_key in METRIC_KEYS.items():
        values = [
            float(metrics.get(json_key, float("nan"))) for metrics in per_seed_metrics
        ]
        stats[column_name] = mean_std(values)

    return {
        "model": os.path.basename(model_dir),
        "run_dir": run_dir,
        "n": len(seed_dirs),
        "stats": stats,
    }


def order_rows(rows: list[dict]) -> list[dict]:
    """Sort by mean F1 ascending, pinning ft first and iid2 last regardless of score."""
    pinned_top = [row for row in rows if row["model"] == PINNED_TOP]
    pinned_bottom = [row for row in rows if row["model"] == PINNED_BOTTOM]
    middle = [row for row in rows if row["model"] not in (PINNED_TOP, PINNED_BOTTOM)]

    def f1_key(row: dict) -> float:
        mean, _ = row["stats"]["f1"]
        return math.inf if math.isnan(mean) else mean

    middle.sort(key=f1_key)
    return pinned_top + middle + pinned_bottom


def discover_model_rows(group_dir: str) -> list[dict]:
    """Summarise every model subdir found directly under a saved_models group dir."""
    model_names = sorted(
        name
        for name in os.listdir(group_dir)
        if os.path.isdir(os.path.join(group_dir, name))
    )
    if not model_names:
        raise SystemExit(f"No model directories found under {group_dir!r}")
    return [summarise_model(os.path.join(group_dir, name)) for name in model_names]


def _fmt_pct(mean: float, std: float) -> str:
    """Format a "mean ± std" percentage cell, falling back to "--" when undefined."""
    if math.isnan(mean):
        return "--"
    std_s = "--" if math.isnan(std) else f"{std * 100:.2f}"
    return f"{mean * 100:.2f} +/- {std_s}"


def _fmt_pct_latex(mean: float, std: float) -> str:
    """Format a "mean $\\pm$ std" percentage cell for LaTeX, falling back to "--"."""
    if math.isnan(mean):
        return "--"
    std_s = "--" if math.isnan(std) else f"{std * 100:.2f}"
    return f"{mean * 100:.2f} $\\pm$ {std_s}"


COLUMNS = [("Rec", "rec"), ("Prec", "prec"), ("F1", "f1"), ("BWT", "bwt")]


def render_terminal(rows: list[dict], group_label: str) -> str:
    """Render rows as a column-aligned Markdown table, readable straight in a terminal."""
    headers = ["Model"] + [header for header, _ in COLUMNS] + ["n"]
    table_rows = []
    for row in rows:
        cells = [row["model"]]
        for _, key in COLUMNS:
            mean, std = row["stats"][key]
            cells.append(_fmt_pct(mean, std))
        cells.append(str(row["n"]))
        table_rows.append(cells)

    widths = [
        max(len(header), *(len(row[i]) for row in table_rows))
        for i, header in enumerate(headers)
    ]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(w) for cell, w in zip(cells, widths)) + " |"

    lines = [
        f"## Seed-sweep summary: {group_label}",
        "",
        render_row(headers),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines.extend(render_row(cells) for cells in table_rows)
    return "\n".join(lines)


def render_latex(rows: list[dict], group_label: str) -> str:
    """Render rows as a LaTeX tabular (booktabs style)."""
    caption = f"Seed-sweep summary: {group_label}"
    label = "tab:" + group_label.lower().replace(" ", "-").replace("/", "-")
    col_spec = "l" + "c" * len(COLUMNS) + "c"
    headers = ["Model"] + [header for header, _ in COLUMNS] + ["$n$"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [row["model"].replace("_", r"\_")]
        for _, key in COLUMNS:
            mean, std = row["stats"][key]
            cells.append(_fmt_pct_latex(mean, std))
        cells.append(str(row["n"]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "group_dir",
        help="A saved_models directory under logs/00_sync, e.g. "
        "logs/00_sync/5e_CIL/saved_models",
    )
    ap.add_argument(
        "--format",
        choices=["terminal", "latex"],
        default="terminal",
        help="Output format (default: terminal).",
    )
    args = ap.parse_args(argv)

    if not os.path.isdir(args.group_dir):
        ap.error(f"not a directory: {args.group_dir}")

    rows = order_rows(discover_model_rows(args.group_dir))
    group_label = os.path.basename(os.path.dirname(os.path.abspath(args.group_dir)))

    if args.format == "latex":
        print(render_latex(rows, group_label))
    else:
        print(render_terminal(rows, group_label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
