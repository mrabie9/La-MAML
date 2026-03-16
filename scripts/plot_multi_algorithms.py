"""Plot metrics from multiple algorithms in a single figure.

Each row in the figure corresponds to one algorithm, while each column is a
different type of plot. Metrics are read from the usual ``metrics`` directories
containing ``task*.npz`` files, as produced by ``main.py``.

The script can either:

- search under ``logs/{algo}/`` for the latest run's metrics directory, or
- use metrics directories explicitly provided by the user.

Usage:
    # Auto-discover latest runs under logs/{algo}/
    python scripts/plot_multi_algorithms.py \\
        --algo cmaml --algo hat --algo ewc

    # Use a specific run by index (0 = latest, 1 = second latest, ...)
    python scripts/plot_multi_algorithms.py \\
        --algo cmaml --algo hat --run-index 1

    # Explicit metrics directories for each algorithm
    python scripts/plot_multi_algorithms.py \\
        --algo cmaml --metrics-dir logs/cmaml/runA/0/metrics \\
        --algo hat   --metrics-dir logs/hat/runB/0/metrics \\
        -o plots/
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Use non-interactive backend only when saving to file (-o); otherwise plt.show()
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("-o", "--output-dir", type=Path, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.output_dir is not None:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


TaskMetrics = Dict[str, Any]


@dataclass
class AlgoRun:
    """Resolved information for a single algorithm run.

    Attributes:
        name: Short algorithm name to show in the plots.
        metrics_dir: Directory containing ``task*.npz`` metric files.
        tasks: Loaded per-task metrics dictionaries.
        task_names: Optional human-readable names for each task.
    """

    name: str
    metrics_dir: Path
    tasks: List[TaskMetrics]
    task_names: List[str] | None = None


def load_task_names(metrics_dir: Path) -> List[str] | None:
    """Load task names from a ``task_order.txt`` file if present.

    The file is expected to live inside ``metrics_dir`` and contain one task
    name per line in task index order.
    """
    order_file = metrics_dir / "task_order.txt"
    if not order_file.is_file():
        return None

    names: List[str] = []
    with order_file.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                names.append(stripped)
    return names or None


def _group_for_task_name(task_name: str) -> str:
    """Return a semantic group label for a task name."""
    name = task_name.lower()
    if "rcn" in name:
        return "rcn"
    if "deeprad" in name:
        return "deeprad"
    if "uclresm" in name:
        return "uclresm"
    return "other"


def get_task_color(task_idx: int, task_names: Sequence[str] | None = None):
    """Assign a colour for a task, grouping by task name when available."""
    if task_names is None or task_idx >= len(task_names):
        group1 = [3, 4, 6, 9]
        group2 = [0, 1, 8]
        group3 = [2, 5, 7]

        if task_idx in group1:
            cmap = plt.get_cmap("Blues")
            idx = group1.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group1)))
        if task_idx in group2:
            cmap = plt.get_cmap("Oranges")
            idx = group2.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group2)))
        if task_idx in group3:
            cmap = plt.get_cmap("Greens")
            idx = group3.index(task_idx)
            return cmap(0.4 + 0.6 * (idx / len(group3)))
        return f"C{task_idx % 10}"

    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(task_names):
        group = _group_for_task_name(name)
        groups.setdefault(group, []).append(idx)

    task_name = task_names[task_idx]
    group = _group_for_task_name(task_name)
    group_indices = groups.get(group, [task_idx])
    position_in_group = group_indices.index(task_idx)
    group_size = len(group_indices)

    def _shade(index: int, size: int) -> float:
        if size <= 1:
            return 0.7
        return 0.4 + 0.6 * (index / float(size))

    cmap_name = {
        "rcn": "Oranges",
        "deeprad": "Blues",
        "uclresm": "Greens",
        "other": "Greys",
    }[group]
    cmap = plt.get_cmap(cmap_name)
    return cmap(_shade(position_in_group, group_size))


def _task_index(filename: str) -> int:
    """Extract task number from filename like task0.npz or task12.npz.

    Args:
        filename: Name of the metrics file.

    Returns:
        The parsed task index or -1 if it cannot be parsed.

    Usage:
        >>> _task_index("task3.npz")
        3
    """
    match = re.match(r"task(\d+)\.npz", filename, re.IGNORECASE)
    if not match:
        return -1
    return int(match.group(1))


def load_metrics(metrics_dir: Path) -> List[TaskMetrics]:
    """Load all ``task*.npz`` files from a metrics directory.

    Args:
        metrics_dir: Path to the metrics directory.

    Returns:
        List of dictionaries, one per task, with at least the keys
        ``losses``, ``cls_tr_rec`` and ``val_acc`` when present.

    Raises:
        FileNotFoundError: If no ``task*.npz`` files are found.

    Usage:
        >>> tasks = load_metrics(Path("logs/cmaml/run/0/metrics"))
        >>> len(tasks) > 0
        True
    """
    task_files = sorted(
        metrics_dir.glob("task*.npz"),
        key=lambda p: _task_index(p.name),
    )
    if not task_files:
        raise FileNotFoundError(f"No task*.npz files in {metrics_dir}")

    tasks: List[TaskMetrics] = []
    for path in task_files:
        data = np.load(path, allow_pickle=True)
        task_data: TaskMetrics = {key: np.asarray(data[key]) for key in data.files}
        # Truncate validation metrics (e.g. val_acc, val_f1) in the same way as
        # scripts/plot_metrics.py so that only the final per-task values remain.
        task_idx = _task_index(path.name)
        num_tasks_seen = task_idx + 1
        for key in ("val_acc", "val_f1"):
            if key in task_data and len(task_data[key]) > num_tasks_seen:
                task_data[key] = task_data[key][-num_tasks_seen:]
        tasks.append(task_data)
    return tasks


def compute_average_forgetting(
    tasks: Sequence[TaskMetrics],
    val_metric_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute average forgetting over previous tasks at each checkpoint.

    This uses a single validation metric key (e.g. ``val_f1`` or ``val_acc``)
    consistently for peak and current values.

    For each task ``t``, peak metric is taken as ``tasks[t][val_metric_key][t]``
    after training task ``t``. After training task ``K > t``, the metric on
    task ``t`` is ``tasks[K][val_metric_key][t]``. Forgetting for task ``t`` at
    ``K`` is ``max(0, peak[t] - current_metric)`` and the average forgetting at
    ``K`` is the mean over all previous tasks.

    Args:
        tasks: Sequence of per-task metric dictionaries.
        val_metric_key: Key of the validation metric to use (e.g. ``val_f1`` or ``val_acc``).

    Returns:
        A tuple ``(x, avg_forgetting)`` where ``x`` is an array of checkpoint
        indices and ``avg_forgetting`` is the corresponding forgetting values.

    Usage:
        >>> x, f = compute_average_forgetting(tasks, \"val_acc\")
        >>> x.shape == f.shape
        True
    """
    n_tasks = len(tasks)
    if n_tasks == 0:
        return np.array([]), np.array([])

    # Peak metric after learning each task t.
    peak_vals: List[float] = []
    for t in range(n_tasks):
        metrics_t = tasks[t]
        metric_vals = metrics_t.get(val_metric_key)
        if metric_vals is None or len(metric_vals) <= t:
            continue
        peak_vals.append(float(metric_vals[t]))
    peak_vals_arr = np.asarray(peak_vals, dtype=float)
    avg_forgetting = np.zeros(n_tasks, dtype=float)

    for k in range(n_tasks):
        if k == 0:
            avg_forgetting[k] = 0.0
            continue
        forgets: List[float] = []
        for t in range(k):
            metrics_k = tasks[k]
            metric_vals_k = metrics_k.get(val_metric_key)
            if metric_vals_k is None or len(metric_vals_k) <= t:
                continue
            current_metric = float(metric_vals_k[t])
            forgets.append(max(0.0, float(peak_vals_arr[t] - current_metric)))
        avg_forgetting[k] = float(np.mean(forgets)) if forgets else 0.0

    return np.arange(n_tasks), avg_forgetting


def find_metrics_dir_for_algo(
    algo: str,
    logs_root: Path,
    run_index: int = 0,
) -> Path:
    """Find a metrics directory for an algorithm by run index (0 = latest).

    The function looks for directories named ``metrics`` under
    ``logs_root / algo`` that contain at least one ``task*.npz`` file, sorts
    them by parent run directory modification time (newest first), then
    returns the one at ``run_index`` (0 = latest, 1 = second latest, etc.).

    Args:
        algo: Algorithm name (subdirectory under ``logs_root``).
        logs_root: Root directory containing algorithm subdirectories.
        run_index: Which run to use: 0 for latest, 1 for second latest, etc.
            Default is 0 (latest run).

    Returns:
        Path to the metrics directory for the given algorithm and run index.

    Raises:
        FileNotFoundError: If no valid metrics directory is found, or
            ``run_index`` is out of range.

    Usage:
        >>> logs_root = Path("logs")
        >>> latest = find_metrics_dir_for_algo("cmaml", logs_root, run_index=0)
        >>> latest.name
        'metrics'
        >>> second = find_metrics_dir_for_algo("cmaml", logs_root, run_index=1)
    """
    algo_root = logs_root / algo
    if not algo_root.exists():
        raise FileNotFoundError(f"No logs found for algorithm '{algo}' under {logs_root}")

    candidate_dirs: List[Tuple[float, Path]] = []
    for metrics_dir in algo_root.rglob("metrics"):
        if not metrics_dir.is_dir():
            continue
        if not any(metrics_dir.glob("task*.npz")):
            continue
        # Use the parent run directory modification time as a proxy for recency.
        run_dir = metrics_dir.parent
        mtime = run_dir.stat().st_mtime
        candidate_dirs.append((mtime, metrics_dir))

    if not candidate_dirs:
        raise FileNotFoundError(f"No metrics directories with task*.npz for '{algo}' under {algo_root}")

    candidate_dirs.sort(key=lambda x: x[0], reverse=True)
    if run_index < 0 or run_index >= len(candidate_dirs):
        raise FileNotFoundError(
            f"Run index {run_index} out of range for '{algo}' "
            f"(found {len(candidate_dirs)} run(s); use 0 to {len(candidate_dirs) - 1})."
        )
    return candidate_dirs[run_index][1]


def _resolve_algo_name(metrics_dir: Path, explicit_name: str | None) -> str:
    """Resolve the algorithm name for a metrics directory.

    Args:
        metrics_dir: Directory containing metrics files.
        explicit_name: Optional name passed on the command line.

    Returns:
        A short name for labelling plots.
    """
    if explicit_name:
        return explicit_name
    # Fallback: use the algorithm directory name under logs/
    parts = metrics_dir.parts
    try:
        logs_index = parts.index("logs")
    except ValueError:
        return metrics_dir.parent.name
    if logs_index + 1 < len(parts):
        return parts[logs_index + 1]
    return metrics_dir.parent.name


def _prepare_algo_runs(
    algos: Sequence[str],
    metrics_dirs: Sequence[Path] | None,
    logs_root: Path,
    run_index: int = 0,
) -> List[AlgoRun]:
    """Resolve algorithms and metrics directories into a list of runs.

    Args:
        algos: Algorithm names from CLI.
        metrics_dirs: Optional list of metrics directories, same length as
            ``algos``. If empty or ``None``, metrics directories are discovered
            under ``logs_root/{algo}/`` using ``run_index``.
        logs_root: Root logs directory.
        run_index: When auto-discovering, which run to use: 0 = latest,
            1 = second latest, etc. Ignored when ``metrics_dirs`` is provided.

    Returns:
        List of ``AlgoRun`` instances ready for plotting.
    """
    runs: List[AlgoRun] = []

    if metrics_dirs and len(metrics_dirs) not in (0, len(algos)):
        raise SystemExit(
            "Number of --metrics-dir entries must be zero or match the number of --algo entries."
        )

    for idx, algo in enumerate(algos):
        if metrics_dirs and len(metrics_dirs) == len(algos):
            metrics_dir = metrics_dirs[idx]
        else:
            metrics_dir = find_metrics_dir_for_algo(algo, logs_root, run_index=run_index)

        metrics_dir = metrics_dir.resolve()
        if not metrics_dir.is_dir():
            raise SystemExit(f"Not a directory: {metrics_dir}")

        tasks = load_metrics(metrics_dir)
        task_names = load_task_names(metrics_dir)
        name = _resolve_algo_name(metrics_dir, explicit_name=algo)
        runs.append(
            AlgoRun(
                name=name,
                metrics_dir=metrics_dir,
                tasks=tasks,
                task_names=task_names,
            )
        )

    return runs


def _plot_loss_vs_steps(ax: plt.Axes, tasks: Sequence[TaskMetrics]) -> None:
    """Deprecated: kept for backwards compatibility, not used."""
    if not tasks:
        return


def _plot_train_acc_vs_steps(
    ax: plt.Axes,
    tasks: Sequence[TaskMetrics],
    task_names: Sequence[str] | None,
) -> None:
    """Plot training recall vs step for each task separately for an algorithm."""
    if not tasks:
        return
    for task_idx, task in enumerate(tasks):
        # Try both possible keys: prefer "cls_tr_rec" (new name), else fall back
        # to "tr_acc" for backwards compatibility with older runs.
        acc_values = task.get("cls_tr_rec")
        if acc_values is None:
            acc_values = task.get("tr_acc")
        if acc_values is None:
            raise KeyError("Neither 'cls_tr_rec' nor 'tr_acc' found in task metrics.")
        acc = np.asarray(acc_values, dtype=float)
        steps = np.arange(len(acc))
        ax.plot(
            steps,
            acc,
            color=get_task_color(task_idx, task_names),
            alpha=0.8,
            label=f"Task {task_idx}",
        )
    ax.set_ylabel("Train recall")
    ax.grid(True, alpha=0.3)


def _plot_val_acc_over_tasks(
    ax: plt.Axes,
    tasks: Sequence[TaskMetrics],
    task_names: Sequence[str] | None,
    val_metric_key: str,
    val_metric_label: str,
) -> None:
    """Plot a validation metric per task as more tasks are trained."""
    n_tasks = len(tasks)
    if n_tasks == 0:
        return

    x = np.arange(1, n_tasks + 1)
    for task_idx in range(n_tasks):
        ys = []
        for k in range(n_tasks):
            metric_vals = tasks[k].get(val_metric_key)
            if metric_vals is None or task_idx >= len(metric_vals):
                ys.append(np.nan)
            else:
                ys.append(float(metric_vals[task_idx]))
        ax.plot(
            x,
            ys,
            "o-",
            label=f"Task {task_idx}",
            color=get_task_color(task_idx, task_names),
            alpha=0.8,
        )

    ax.set_xlabel("After training up to task")
    ax.set_ylabel(f"Val {val_metric_label}")
    ax.grid(True, alpha=0.3)


def _plot_average_forgetting(
    ax: plt.Axes,
    tasks: Sequence[TaskMetrics],
    val_metric_key: str,
    val_metric_label: str,
) -> None:
    """Plot average forgetting vs checkpoint index for an algorithm."""
    x, avg_forgetting = compute_average_forgetting(tasks, val_metric_key)
    if x.size == 0:
        return
    ax.plot(x, avg_forgetting, "o-", color="C3")
    ax.set_xlabel("After training up to task")
    ax.set_ylabel(f"Avg forgetting ({val_metric_label})")
    ax.grid(True, alpha=0.3)


def _plot_final_validation_metrics(
    ax: plt.Axes,
    tasks: Sequence[TaskMetrics],
    task_names: Sequence[str] | None,
) -> None:
    """Plot final validation metrics (pfa, detection recall, cls rec) from last task.

    Bars are ordered with false alarm (pfa) first, followed by detection
    recall and classification recall.
    """
    if not tasks:
        return

    last = tasks[-1]
    val_acc = last.get("val_acc")
    val_det_acc = last.get("val_det_acc")
    val_det_fa = last.get("val_det_fa")

    if val_acc is None and val_det_acc is None and val_det_fa is None:
        return

    # Determine number of tasks from metrics that are present, using the
    # minimum length so that all plotted series align.
    lengths = [
        len(v)
        for v in (val_det_fa, val_det_acc, val_acc)
        if v is not None
    ]
    if not lengths:
        return
    n_tasks = min(lengths)
    task_indices = np.arange(n_tasks)
    colors = [get_task_color(i, task_names) for i in range(n_tasks)]
    width = 0.25

    # False alarm rate (pfa) first
    if val_det_fa is not None:
        fa_vals = np.asarray(val_det_fa, dtype=float)
        ax.bar(
            task_indices - width,
            fa_vals[:n_tasks],
            width=width,
            label="Pfa",
            color=colors,
            hatch="//",
        )

    # Detection recall
    if val_det_acc is not None:
        det_vals = np.asarray(val_det_acc, dtype=float)
        ax.bar(
            task_indices,
            det_vals[:n_tasks],
            width=width,
            label="Det recall",
            color=colors,
            hatch="..",
        )

    # Classification recall
    if val_acc is not None:
        cls_vals = np.asarray(val_acc, dtype=float)
        ax.bar(
            task_indices + width,
            cls_vals[:n_tasks],
            width=width,
            label="Cls recall",
            color=colors,
        )

    ax.set_xlabel("Task")
    ax.set_ylabel("Metric value")
    ax.set_xticks(task_indices)
    ax.grid(True, alpha=0.3, axis="y")


def plot_multi_algorithms(
    runs: Sequence[AlgoRun],
    output_dir: Path | None,
    same_y_limits: bool = False,
    val_metric_choice: str = "total_f1",
) -> None:
    """Create the multi-row, multi-column figure for all algorithms.

    Args:
        runs: Sequence of resolved algorithm runs.
        output_dir: Optional directory in which to save the combined figure.

    Usage:
        >>> plot_multi_algorithms(runs, Path("plots"))
    """
    if not runs:
        raise ValueError("No algorithms provided for plotting.")

    n_algos = len(runs)
    n_cols = 4

    # Use a larger figure when saving to file to allow detailed zooming.
    if output_dir is not None:
        col_width, row_height = 9, 4.5
        dpi = 600
    else:
        col_width, row_height = 4.0, 3.0
        dpi = 200

    fig, axes = plt.subplots(
        n_algos,
        n_cols,
        figsize=(col_width * n_cols, row_height * n_algos),
        squeeze=False,
        sharex="col",
        dpi=dpi,
    )

    def _resolve_val_metric_for_run(choice: str, run: AlgoRun) -> tuple[str, str]:
        """Resolve validation metric key and label for a single run."""
        if choice == "cls_recall":
            return "val_acc", "Cls recall"
        # choice == "total_f1": prefer F1 when available, fall back to recall.
        has_f1 = any("val_f1" in t for t in run.tasks)
        if has_f1:
            return "val_f1", "Total F1"
        print(
            f"[WARN] Requested val-metric=total_f1 but run '{run.name}' at "
            f"{run.metrics_dir} has no 'val_f1'; falling back to cls recall ('val_acc')."
        )
        return "val_acc", "Cls recall"

    # Resolve label for column titles from the first run.
    first_key, first_label = _resolve_val_metric_for_run(val_metric_choice, runs[0])

    for row_idx, run in enumerate(runs):
        val_metric_key, val_metric_label = _resolve_val_metric_for_run(val_metric_choice, run)
        row_axes = axes[row_idx]
        _plot_train_acc_vs_steps(row_axes[0], run.tasks, run.task_names)
        _plot_final_validation_metrics(row_axes[1], run.tasks, run.task_names)
        _plot_val_acc_over_tasks(
            row_axes[2],
            run.tasks,
            run.task_names,
            val_metric_key=val_metric_key,
            val_metric_label=val_metric_label,
        )
        _plot_average_forgetting(
            row_axes[3],
            run.tasks,
            val_metric_key=val_metric_key,
            val_metric_label=val_metric_label,
        )

        row_axes[0].set_ylabel(f"{run.name}\nTrain recall")

    # Column titles on the top row
    axes[0][0].set_title("Train recall vs step")
    axes[0][1].set_title("Final validation metrics")
    axes[0][2].set_title(f"Val {first_label} over tasks")
    axes[0][3].set_title(f"Average forgetting ({first_label})")

    # Legends on first row only: train recall vs step, final validation metrics.
    handles_train, _ = axes[0][0].get_legend_handles_labels()
    if handles_train:
        axes[0][0].legend(ncol=min(len(handles_train), 5), fontsize=8)

    handles_fv, _ = axes[0][1].get_legend_handles_labels()
    if handles_fv:
        axes[0][1].legend(ncol=len(handles_fv), fontsize=8)

    # Legend for val recall over tasks (first row only).
    handles_val, _ = axes[0][2].get_legend_handles_labels()
    if handles_val:
        axes[0][2].legend(ncol=min(len(handles_val), 4), fontsize=8)

    # Optionally enforce the same y-axis limits across rows for each column so
    # that comparisons between algorithms are visually consistent.
    if same_y_limits:
        for col_idx in range(n_cols):
            y_mins: List[float] = []
            y_maxs: List[float] = []
            for row_idx in range(n_algos):
                y_min, y_max = axes[row_idx][col_idx].get_ylim()
                y_mins.append(y_min)
                y_maxs.append(y_max)
            if not y_mins:
                continue
            col_min = min(y_mins)
            col_max = max(y_maxs)
            for row_idx in range(n_algos):
                axes[row_idx][col_idx].set_ylim(col_min, col_max)

    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "multi_algorithms_metrics.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
    else:
        plt.show()

    plt.close(fig)


def _load_n_epochs(metrics_dir: Path) -> int | None:
    """Optionally load ``n_epochs`` from training_parameters.json if needed.

    This is provided for potential extensions that require per-epoch
    aggregation. It is currently unused in the default plots.

    Args:
        metrics_dir: Metrics directory for a single run.

    Returns:
        ``n_epochs`` if present, otherwise ``None``.
    """
    params_file = metrics_dir.parent / "training_parameters.json"
    if not params_file.is_file():
        return None
    try:
        with open(params_file) as f:
            params = json.load(f)
        value = params.get("n_epochs")
        return int(value) if value is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        parents=[_pre_parser],
        description=(
            "Plot metrics from multiple algorithms in a single figure. "
            "Each row is an algorithm; each column is a plot type."
        ),
    )
    parser.add_argument(
        "--algo",
        type=str,
        action="append",
        dest="algos",
        required=True,
        help="Algorithm name to include (can be specified multiple times).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        action="append",
        dest="metrics_dirs",
        default=None,
        help=(
            "Metrics directory for an algorithm (can be specified multiple times). "
            "If omitted, the latest metrics dir under logs/{algo}/ is used."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root directory containing logs/{algo}/ subdirectories (default: logs).",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        metavar="N",
        help=(
            "When auto-discovering runs (no --metrics-dir), use the Nth run by recency: "
            "0 = latest (default), 1 = second latest, etc."
        ),
    )
    parser.add_argument(
        "--same-y-limits",
        action="store_true",
        help=(
            "Use the same y-axis limits across all rows for each column to "
            "make cross-algorithm comparisons easier."
        ),
    )
    parser.add_argument(
        "--val-metric",
        type=str,
        choices=("total_f1", "cls_recall"),
        default="total_f1",
        help=(
            "Validation metric to use for 'val over tasks' and average forgetting "
            "plots. Defaults to total_f1; falls back to cls_recall when F1 is "
            "not available for a run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for multi-algorithm metrics plotting."""
    args = _parse_args()

    metrics_dirs: List[Path] | None
    if args.metrics_dirs is None:
        metrics_dirs = None
    else:
        metrics_dirs = list(args.metrics_dirs)

    runs = _prepare_algo_runs(
        algos=list(args.algos),
        metrics_dirs=metrics_dirs,
        logs_root=args.logs_root,
        run_index=args.run_index,
    )
    print("Algorithms and metrics directories:")
    for run in runs:
        print(f"  {run.name}: {run.metrics_dir}")

    plot_multi_algorithms(
        runs=runs,
        output_dir=args.output_dir,
        same_y_limits=args.same_y_limits,
        val_metric_choice=args.val_metric,
    )


if __name__ == "__main__":
    main()

