#!/usr/bin/env python
"""Collect backward transfer (BWT) from continual-learning metrics on disk.

``scripts/plot_multi_algorithms.py`` defines **average forgetting** at each
checkpoint (after training task *k*) using the same peak-vs-current rule. The
**final** average forgetting is the last element of that curve (after all tasks).

This script reports that final value and **BWT** as its negative, matching the
convention described alongside ``plot_multi_algorithms`` average-forgetting plots:

- ``avg_forgetting_final``: last checkpoint average forgetting
- ``bwt``: ``-avg_forgetting_final``

Metrics are read from ``task*.npz`` under a ``metrics`` directory, identical to
the plotting workflow.

Usage:
    python scripts/collect_bwt_from_metrics.py --algo cmaml,hat

    python scripts/collect_bwt_from_metrics.py --algo cmaml

    python scripts/collect_bwt_from_metrics.py \\
        --algo cmaml,hat \\
        --metrics-dir logs/cmaml/run-a/0/metrics,logs/hat/run-b/0/metrics

    python scripts/collect_bwt_from_metrics.py \\
        --logs-root logs --run-index 0 --val-metric total_f1 \\
        --algo ewc,lamaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Sibling import: ``python scripts/collect_bwt_from_metrics.py`` puts this
# directory on ``sys.path`` first.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_multi_algorithms as plot_multi  # noqa: E402

TaskMetrics = Dict[str, Any]


def _comma_separated_list(argument_value: str) -> List[str]:
    """Split a single CLI token into stripped non-empty parts (comma-separated).

    Args:
        argument_value: Raw string from ``--algo`` or ``--metrics-dir``.

    Returns:
        List of trimmed substrings; empty entries are dropped.

    Usage:
        >>> _comma_separated_list("cmaml, hat , ewc")
        ['cmaml', 'hat', 'ewc']
        >>> _comma_separated_list("cmaml")
        ['cmaml']
    """
    return [part.strip() for part in argument_value.split(",") if part.strip()]


def resolve_val_metric_key(
    tasks: Sequence[TaskMetrics], val_metric_choice: str
) -> Tuple[str, str]:
    """Map CLI validation choice to ``task*.npz`` keys (same as plotter).

    Args:
        tasks: Loaded per-task metric dicts.
        val_metric_choice: ``total_f1`` or ``cls_recall``.

    Returns:
        Tuple of (npz key, human label).

    Usage:
        >>> resolve_val_metric_key([{"val_f1": np.array([1.0])}], "total_f1")
        ('val_f1', 'Total F1')
    """
    if val_metric_choice == "cls_recall":
        return "val_acc", "Cls recall"
    has_f1 = any("val_f1" in task for task in tasks)
    if has_f1:
        return "val_f1", "Total F1"
    print(
        "[WARN] val-metric=total_f1 but no 'val_f1'; using 'val_acc' (cls recall).",
        file=sys.stderr,
    )
    return "val_acc", "Cls recall"


def final_average_forgetting_and_bwt(
    tasks: Sequence[TaskMetrics], val_metric_key: str
) -> Tuple[float, float]:
    """Return final average forgetting and BWT (-forgetting).

    Args:
        tasks: Per-task metrics from ``load_metrics``.
        val_metric_key: e.g. ``val_f1`` or ``val_acc``.

    Returns:
        ``(avg_forgetting_final, bwt)`` where ``bwt == -avg_forgetting_final``.

    Usage:
        >>> import numpy as np
        >>> t0 = {"val_f1": np.array([0.9, 0.5])}
        >>> t1 = {"val_f1": np.array([0.9, 0.85])}
        >>> af, bwt = final_average_forgetting_and_bwt([t0, t1], "val_f1")
        >>> af >= 0
        True
    """
    _checkpoint_indices, avg_forgetting = plot_multi.compute_average_forgetting(
        tasks, val_metric_key
    )
    if avg_forgetting.size == 0:
        raise ValueError("No tasks: cannot compute forgetting.")
    final_forgetting = float(avg_forgetting[-1])
    return final_forgetting, -final_forgetting


def _prepare_runs(
    algos: Sequence[str],
    metrics_dirs: Sequence[Path] | None,
    logs_root: Path,
    run_index: int,
) -> List[plot_multi.AlgoRun]:
    """Resolve algorithms to ``AlgoRun`` (same rules as plotting script)."""
    if metrics_dirs and len(metrics_dirs) not in (0, len(algos)):
        raise SystemExit(
            "Number of --metrics-dir entries must be zero or match --algo count."
        )
    runs: List[plot_multi.AlgoRun] = []
    for idx, algo in enumerate(algos):
        if metrics_dirs and len(metrics_dirs) == len(algos):
            metrics_dir = metrics_dirs[idx]
        else:
            metrics_dir = plot_multi.find_metrics_dir_for_algo(
                algo, logs_root, run_index=run_index
            )
        metrics_dir = metrics_dir.resolve()
        if not metrics_dir.is_dir():
            raise SystemExit(f"Not a directory: {metrics_dir}")
        tasks = plot_multi.load_metrics(metrics_dir)
        name = plot_multi._resolve_algo_name(metrics_dir, explicit_name=algo)
        runs.append(
            plot_multi.AlgoRun(
                name=name,
                metrics_dir=metrics_dir,
                tasks=tasks,
                task_names=plot_multi.load_task_names(metrics_dir),
            )
        )
    return runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print final average forgetting and BWT (-forgetting) from metrics "
            "directories (same sources as plot_multi_algorithms)."
        )
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        help=(
            "Comma-separated algorithm names (e.g. 'cmaml,hat') or a single "
            "name. Whitespace around commas is ignored."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help=(
            "Optional comma-separated metrics directories (same order as --algo). "
            "If omitted, latest run under logs/<algo>/ is used per algorithm."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root for auto-discovering metrics (default: logs).",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Which run by recency when auto-discovering (0 = latest).",
    )
    parser.add_argument(
        "--val-metric",
        type=str,
        choices=("total_f1", "cls_recall"),
        default="total_f1",
        help="Validation signal for forgetting (default: total_f1).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write results as CSV.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=("algo", "bwt", "forgetting"),
        default="algo",
        help="Sort rows by algorithm name, BWT (desc), or forgetting (asc).",
    )
    return parser.parse_args()


def main() -> None:
    """Load metrics per algorithm and print final forgetting and BWT."""
    args = _parse_args()
    algorithm_names = _comma_separated_list(args.algo)
    if not algorithm_names:
        raise SystemExit("--algo must contain at least one algorithm name.")

    metrics_dirs_list: List[Path] | None
    if args.metrics_dir is None:
        metrics_dirs_list = None
    else:
        metrics_dirs_list = [
            Path(part) for part in _comma_separated_list(args.metrics_dir)
        ]

    runs = _prepare_runs(
        algos=algorithm_names,
        metrics_dirs=metrics_dirs_list,
        logs_root=args.logs_root,
        run_index=args.run_index,
    )

    rows: List[dict[str, str | float]] = []
    for run in runs:
        val_key, val_label = resolve_val_metric_key(run.tasks, args.val_metric)
        avg_f_final, bwt = final_average_forgetting_and_bwt(run.tasks, val_key)
        rows.append(
            {
                "algo": run.name,
                "val_metric": val_label,
                "val_key": val_key,
                "avg_forgetting_final": avg_f_final,
                "bwt": bwt,
                "metrics_dir": str(run.metrics_dir),
            }
        )

    if args.sort == "bwt":
        rows.sort(key=lambda row: float(row["bwt"]), reverse=True)
    elif args.sort == "forgetting":
        rows.sort(key=lambda row: float(row["avg_forgetting_final"]))
    else:
        rows.sort(key=lambda row: str(row["algo"]))

    header = f"{'algo':<12} {'val':<12} {'forget_final':>12} {'bwt':>12}  metrics_dir"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['algo']:<12} {row['val_metric']:<12} "
            f"{row['avg_forgetting_final']:12.6f} {row['bwt']:12.6f}  "
            f"{row['metrics_dir']}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "algo",
                    "val_metric",
                    "val_key",
                    "avg_forgetting_final",
                    "bwt",
                    "metrics_dir",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
