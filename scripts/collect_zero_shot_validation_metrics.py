#!/usr/bin/env python
"""Collect zero-shot validation metrics per task from continual-learning logs.

This script reads ``task*.npz`` files from one or more ``metrics`` directories and
extracts the zero-shot (pre-train) validation metrics written by ``main.py``:

- ``zero_shot_rec_cls``
- ``zero_shot_prec_cls``
- ``zero_shot_det``
- ``zero_shot_pfa``
- ``zero_shot_f1_cls`` (renamed in output to ``total_f1_zs``)

By default, one row is emitted per task checkpoint (task 0, task 1, ...). This is
the task-level series typically used to compute forward transfer (FWT).

Optionally, ``--include-per-task`` also exports the full zero-shot metric matrix
for all seen tasks at each checkpoint using the ``zero_shot_per_task_*`` arrays.

Usage:
    python scripts/collect_zero_shot_validation_metrics.py --algo cmaml

    python scripts/collect_zero_shot_validation_metrics.py \
        --algo cmaml,hat --logs-root logs --run-index 0 --csv outputs/zs.csv

    python scripts/collect_zero_shot_validation_metrics.py \
        --algo cmaml --metrics-dir logs/cmaml/my-run/0/metrics --include-per-task
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import plot_multi_algorithms as plot_multi  # noqa: E402

TaskMetrics = Dict[str, Any]


def _comma_separated_list(argument_value: str) -> List[str]:
    """Split a single CLI token into stripped non-empty values.

    Args:
        argument_value: Raw argument string (for example ``--algo cmaml,hat``).

    Returns:
        List of trimmed, non-empty values.

    Usage:
        >>> _comma_separated_list("cmaml, hat , ewc")
        ['cmaml', 'hat', 'ewc']
    """
    return [part.strip() for part in argument_value.split(",") if part.strip()]


def _safe_float(value: Any) -> float:
    """Convert metric-like values to ``float`` while preserving NaN.

    Args:
        value: Input scalar or array-like value.

    Returns:
        Float representation; ``nan`` on unsupported/missing values.

    Usage:
        >>> math.isnan(_safe_float(None))
        True
        >>> _safe_float(np.float64(0.25))
        0.25
    """
    if value is None:
        return float("nan")
    array_value = np.asarray(value)
    if array_value.size == 0:
        return float("nan")
    try:
        return float(array_value.reshape(-1)[0])
    except (TypeError, ValueError):
        return float("nan")


def _harmonic_mean_f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall using harmonic mean.

    Args:
        precision: Precision metric in [0, 1] or NaN.
        recall: Recall metric in [0, 1] or NaN.

    Returns:
        ``2 * P * R / (P + R)`` when defined, else NaN.

    Usage:
        >>> _harmonic_mean_f1(0.5, 0.5)
        0.5
    """
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    denominator = precision + recall
    if denominator <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / denominator)


def _extract_task_scalar_rows(run: plot_multi.AlgoRun) -> List[Dict[str, Any]]:
    """Build one row per task checkpoint with scalar zero-shot metrics.

    Args:
        run: Loaded algorithm run with per-task metric dictionaries.

    Returns:
        List of row dictionaries for each task in run order.

    Usage:
        >>> isinstance(_extract_task_scalar_rows, object)
        True
    """
    task_names = run.task_names or []
    rows: List[Dict[str, Any]] = []
    for task_index, task_metrics in enumerate(run.tasks):
        task_name = task_names[task_index] if task_index < len(task_names) else ""
        rec_cls = _safe_float(task_metrics.get("zero_shot_rec_cls"))
        prec_cls = _safe_float(task_metrics.get("zero_shot_prec_cls"))
        total_f1_zs = _safe_float(task_metrics.get("zero_shot_f1_cls"))
        f1_cls_from_rec_prec = _harmonic_mean_f1(prec_cls, rec_cls)
        rows.append(
            {
                "algo": run.name,
                "metrics_dir": str(run.metrics_dir),
                "task_index": task_index,
                "task_name": task_name,
                "zero_shot_rec_cls": rec_cls,
                "zero_shot_prec_cls": prec_cls,
                "zero_shot_f1_cls": f1_cls_from_rec_prec,
                "zero_shot_det": _safe_float(task_metrics.get("zero_shot_det")),
                "zero_shot_pfa": _safe_float(task_metrics.get("zero_shot_pfa")),
                "zero_shot_total_f1_zs": total_f1_zs,
            }
        )
    return rows


def _extract_per_task_matrix_rows(
    run: plot_multi.AlgoRun,
) -> List[Dict[str, Any]]:
    """Build full pre-train metric matrix rows for each checkpoint/task pair.

    Args:
        run: Loaded algorithm run.

    Returns:
        Row dictionaries with both ``checkpoint_task_index`` and
        ``evaluated_task_index``.

    Usage:
        >>> isinstance(_extract_per_task_matrix_rows, object)
        True
    """
    task_names = run.task_names or []
    rows: List[Dict[str, Any]] = []
    for checkpoint_task_index, task_metrics in enumerate(run.tasks):
        per_task_f1 = np.asarray(
            task_metrics.get("zero_shot_per_task_f1_cls", []), dtype=float
        ).reshape(-1)
        per_task_rec = np.asarray(
            task_metrics.get("zero_shot_per_task_rec_cls", []), dtype=float
        ).reshape(-1)
        per_task_prec = np.asarray(
            task_metrics.get("zero_shot_per_task_prec_cls", []), dtype=float
        ).reshape(-1)
        per_task_det = np.asarray(
            task_metrics.get("zero_shot_per_task_det", []), dtype=float
        ).reshape(-1)
        per_task_pfa = np.asarray(
            task_metrics.get("zero_shot_per_task_pfa", []), dtype=float
        ).reshape(-1)
        num_tasks_now = max(
            len(per_task_f1),
            len(per_task_rec),
            len(per_task_prec),
            len(per_task_det),
            len(per_task_pfa),
        )
        for evaluated_task_index in range(num_tasks_now):
            evaluated_task_name = (
                task_names[evaluated_task_index]
                if evaluated_task_index < len(task_names)
                else ""
            )
            rows.append(
                {
                    "algo": run.name,
                    "metrics_dir": str(run.metrics_dir),
                    "checkpoint_task_index": checkpoint_task_index,
                    "evaluated_task_index": evaluated_task_index,
                    "evaluated_task_name": evaluated_task_name,
                    "zero_shot_per_task_rec_cls": _safe_float(
                        per_task_rec[evaluated_task_index]
                        if evaluated_task_index < len(per_task_rec)
                        else float("nan")
                    ),
                    "zero_shot_per_task_prec_cls": _safe_float(
                        per_task_prec[evaluated_task_index]
                        if evaluated_task_index < len(per_task_prec)
                        else float("nan")
                    ),
                    "zero_shot_per_task_f1_cls": _safe_float(
                        per_task_f1[evaluated_task_index]
                        if evaluated_task_index < len(per_task_f1)
                        else float("nan")
                    ),
                    "zero_shot_per_task_det": _safe_float(
                        per_task_det[evaluated_task_index]
                        if evaluated_task_index < len(per_task_det)
                        else float("nan")
                    ),
                    "zero_shot_per_task_pfa": _safe_float(
                        per_task_pfa[evaluated_task_index]
                        if evaluated_task_index < len(per_task_pfa)
                        else float("nan")
                    ),
                }
            )
    return rows


def _prepare_runs(
    algorithm_names: Sequence[str],
    metrics_dirs: Sequence[Path] | None,
    logs_root: Path,
    run_index: int,
) -> List[plot_multi.AlgoRun]:
    """Resolve algorithms to loaded runs.

    Args:
        algorithm_names: Algorithm labels from CLI.
        metrics_dirs: Optional explicit metrics directories.
        logs_root: Root directory for auto-discovery.
        run_index: Recency index for auto-discovery (0 = latest).

    Returns:
        List of resolved runs with loaded task metrics.

    Usage:
        >>> isinstance(_prepare_runs, object)
        True
    """
    if metrics_dirs and len(metrics_dirs) not in (0, len(algorithm_names)):
        raise SystemExit(
            "Number of --metrics-dir entries must be zero or match --algo count."
        )
    runs: List[plot_multi.AlgoRun] = []
    for index, algorithm_name in enumerate(algorithm_names):
        if metrics_dirs and len(metrics_dirs) == len(algorithm_names):
            metrics_dir = metrics_dirs[index]
        else:
            metrics_dir = plot_multi.find_metrics_dir_for_algo(
                algorithm_name, logs_root, run_index=run_index
            )
        metrics_dir = metrics_dir.resolve()
        if not metrics_dir.is_dir():
            raise SystemExit(f"Not a directory: {metrics_dir}")
        tasks = plot_multi.load_metrics(metrics_dir)
        name = plot_multi._resolve_algo_name(metrics_dir, explicit_name=algorithm_name)
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
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.

    Usage:
        >>> isinstance(_parse_args, object)
        True
    """
    parser = argparse.ArgumentParser(
        description=(
            "Collect zero-shot validation metrics per task from task*.npz files."
        )
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        help=(
            "Comma-separated algorithm names (for example 'cmaml,hat') or a single "
            "name."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help=(
            "Optional comma-separated metrics directories in the same order as "
            "--algo. If omitted, latest run under logs/<algo>/ is used."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root path for auto-discovering metrics directories.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="Run recency index for auto-discovery (0 = latest).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-task scalar zero-shot rows.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path for per-task scalar zero-shot rows.",
    )
    parser.add_argument(
        "--include-per-task",
        action="store_true",
        help=(
            "Also export full zero-shot matrix rows (all seen tasks at each "
            "checkpoint)."
        ),
    )
    parser.add_argument(
        "--per-task-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV output path for --include-per-task rows. Ignored unless "
            "--include-per-task is set."
        ),
    )
    return parser.parse_args()


def _write_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    """Write a list of dictionaries to CSV.

    Args:
        rows: Table rows with homogeneous keys.
        output_path: Destination CSV path.

    Usage:
        >>> isinstance(_write_csv, object)
        True
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _nan_to_none(value: Any) -> Any:
    """Convert NaN floats recursively to ``None`` for JSON compatibility.

    Args:
        value: Potentially nested structure.

    Returns:
        JSON-safe structure with NaN replaced by ``None``.

    Usage:
        >>> _nan_to_none(float("nan")) is None
        True
    """
    if isinstance(value, dict):
        return {key: _nan_to_none(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    """Collect and print zero-shot validation metrics across task checkpoints."""
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
        algorithm_names=algorithm_names,
        metrics_dirs=metrics_dirs_list,
        logs_root=args.logs_root,
        run_index=args.run_index,
    )

    scalar_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    for run in runs:
        scalar_rows.extend(_extract_task_scalar_rows(run))
        if args.include_per_task:
            matrix_rows.extend(_extract_per_task_matrix_rows(run))

    scalar_rows.sort(key=lambda row: (str(row["algo"]), int(row["task_index"])))
    matrix_rows.sort(
        key=lambda row: (
            str(row["algo"]),
            int(row["checkpoint_task_index"]),
            int(row["evaluated_task_index"]),
        )
    )

    header = (
        f"{'algo':<12} {'task':>4} {'f1_cls':>10} {'rec_cls':>10} "
        f"{'prec_cls':>10} {'det':>10} {'pfa':>10} {'total_f1_zs':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in scalar_rows:
        print(
            f"{row['algo']:<12} {row['task_index']:4d} "
            f"{row['zero_shot_f1_cls']:10.6f} "
            f"{row['zero_shot_rec_cls']:10.6f} {row['zero_shot_prec_cls']:10.6f} "
            f"{row['zero_shot_det']:10.6f} {row['zero_shot_pfa']:10.6f} "
            f"{row['zero_shot_total_f1_zs']:12.6f}"
        )
    print(
        "\nNote: f1_cls is recomputed from rec_cls/prec_cls as 2PR/(P+R). "
        "total_f1_zs is the raw stored zero_shot_f1_cls from logs."
    )

    if args.csv is not None:
        _write_csv(scalar_rows, args.csv)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as json_file:
            json.dump(_nan_to_none(scalar_rows), json_file, indent=2)
            json_file.write("\n")
    if args.include_per_task and args.per_task_csv is not None:
        _write_csv(matrix_rows, args.per_task_csv)

    if args.include_per_task:
        print(f"\nCollected {len(matrix_rows)} full per-task zero-shot matrix rows.")
    if args.csv is not None:
        print(f"Saved scalar rows CSV to {args.csv}")
    if args.json is not None:
        print(f"Saved scalar rows JSON to {args.json}")
    if args.include_per_task and args.per_task_csv is not None:
        print(f"Saved per-task matrix CSV to {args.per_task_csv}")


if __name__ == "__main__":
    main()
