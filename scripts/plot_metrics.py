"""Read and plot metrics from a run's metrics directory.

Metrics are saved by main.py under args.log_dir/metrics/ as task0.npz, task1.npz, ...
Each task*.npz contains:
  - losses: per-step/epoch loss for that task
  - tr_acc: per-step/epoch training recall
  - val_acc: validation recall per task (length = task_index + 1)
  - val_det_acc, val_det_fa: optional detection metrics (same shape as val_acc)

Usage:
    python scripts/plot_metrics.py logs/ctn/eclresm_test-2026-03-05_14-58-57-4091/0/metrics
    python scripts/plot_metrics.py logs/ctn/eclresm_test-2026-03-05_14-58-57-4091/0/metrics -o plots/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

# Use non-interactive backend only when saving to file (-o); otherwise plt.show() can open windows
_pre_parser = argparse.ArgumentParser()
_pre_parser.add_argument("-o", "--output-dir", type=Path, default=None)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.output_dir is not None:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_dir: Path) -> list[dict[str, Any]]:
    """Load all task*.npz files from metrics_dir into a list of dicts.

    Args:
        metrics_dir: Path to the metrics directory (contains task0.npz, task1.npz, ...).

    Returns:
        List of dicts, one per task, each with keys losses, tr_acc, val_acc,
        and optionally val_det_acc, val_det_fa.
    """
    task_files = sorted(
        metrics_dir.glob("task*.npz"),
        key=lambda p: _task_index(p.name),
    )
    if not task_files:
        raise FileNotFoundError(f"No task*.npz files in {metrics_dir}")

    tasks = []
    for path in task_files:
        data = np.load(path, allow_pickle=True)
        task_data = {key: np.asarray(data[key]) for key in data.files}
        task_idx = _task_index(path.name)
        # Metrics like val_acc / val_f1 might contain intermediate epoch
        # validations flattened. The correct length for the final validation
        # after this task is task_idx + 1.
        num_tasks_seen = task_idx + 1
        for key in ("val_acc", "val_f1"):
            if key in task_data and len(task_data[key]) > num_tasks_seen:
                task_data[key] = task_data[key][-num_tasks_seen:]
        tasks.append(task_data)
    return tasks


def _task_index(filename: str) -> int:
    """Extract task number from filename like task0.npz or task12.npz."""
    match = re.match(r"task(\d+)\.npz", filename, re.IGNORECASE)
    if not match:
        return -1
    return int(match.group(1))


def load_n_epochs(metrics_dir: Path) -> int | None:
    """Read n_epochs from training_parameters.json in the run directory (parent of metrics_dir)."""
    params_file = metrics_dir.parent / "training_parameters.json"
    if not params_file.is_file():
        return None
    try:
        with open(params_file) as f:
            params = json.load(f)
        return params.get("n_epochs")
    except (json.JSONDecodeError, TypeError):
        return None


def _aggregate_per_epoch(values: np.ndarray, n_epochs: int) -> np.ndarray:
    """Average per-step values into per-epoch (one value per epoch)."""
    n = len(values)
    steps_per_epoch = n // n_epochs
    if steps_per_epoch == 0:
        return np.array([np.mean(values)]) if n > 0 else np.array([])
    n_use = steps_per_epoch * n_epochs
    return np.mean(values[:n_use].reshape(n_epochs, steps_per_epoch), axis=1)


def load_task_names(metrics_dir: Path) -> list[str] | None:
    """Load task names from a ``task_order.txt`` file if present.

    The file is expected to live inside ``metrics_dir`` and contain one task
    name per line in task index order.

    Args:
        metrics_dir: Path to the metrics directory.

    Returns:
        A list of task names, or ``None`` if the file does not exist or is
        empty.
    """
    order_file = metrics_dir / "task_order.txt"
    if not order_file.is_file():
        return None

    names: list[str] = []
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
    """Assign a color for a task, grouping by task name when available.

    Tasks whose names contain ``rcn`` / ``deeprad`` / ``uclresm`` are coloured
    consistently within their group across plots. When no task names are
    provided, a static index-based grouping is used as a fallback.
    """
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

    # Name-based grouping.
    groups: dict[str, list[int]] = {}
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
        return 0.4 + 0.6 * (index / float(size - 1))

    cmap_name = {
        "rcn": "Blues",
        "deeprad": "Oranges",
        "uclresm": "Greens",
        "other": "Greys",
    }[group]
    cmap = plt.get_cmap(cmap_name)
    return cmap(_shade(position_in_group, group_size))


def plot_per_task_curves(
    tasks: list[dict[str, Any]],
    output_dir: Path | None,
    task_names: Sequence[str] | None = None,
) -> None:
    """Plot loss and training recall per task (steps within task)."""
    n_tasks = len(tasks)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for task_idx, task in enumerate(tasks):
        steps = np.arange(len(task["losses"]))
        c = get_task_color(task_idx, task_names)
        axes[0].plot(steps, task["losses"], label=f"Task {task_idx}", color=c, alpha=0.8)
        axes[1].plot(steps, task["tr_acc"], label=f"Task {task_idx}", color=c, alpha=0.8)

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per task (per step)")
    axes[0].legend(ncol=min(n_tasks, 5), fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Train recall")
    axes[1].set_xlabel("Step / epoch index")
    axes[1].set_title("Training cls recall per task")
    axes[1].legend(ncol=min(n_tasks, 5), fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "per_task_loss_and_acc.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_per_epoch_curves(
    tasks: list[dict[str, Any]],
    n_epochs: int,
    output_dir: Path | None,
    task_names: Sequence[str] | None = None,
) -> None:
    """Plot mean loss and mean training recall per epoch (one point per epoch per task)."""
    n_tasks = len(tasks)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for task_idx, task in enumerate(tasks):
        loss_ep = _aggregate_per_epoch(task["losses"], n_epochs)
        acc_ep = _aggregate_per_epoch(task["tr_acc"], n_epochs)
        epochs = np.arange(len(loss_ep))
        c = get_task_color(task_idx, task_names)
        axes[0].plot(epochs, loss_ep, "o-", label=f"Task {task_idx}", color=c, alpha=0.8)
        axes[1].plot(epochs, acc_ep, "o-", label=f"Task {task_idx}", color=c, alpha=0.8)

    axes[0].set_ylabel("Loss (mean)")
    axes[0].set_title("Loss per epoch (mean over steps)")
    axes[0].legend(ncol=min(n_tasks, 5), fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Train recall (mean)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Training cls recall per epoch (mean over steps)")
    axes[1].legend(ncol=min(n_tasks, 5), fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "per_epoch_loss_and_acc.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_final_validation(
    tasks: list[dict[str, Any]],
    output_dir: Path | None,
    task_names: Sequence[str] | None = None,
) -> None:
    """Plot final validation metrics from the last task, including Pfa.

    Bars per task are ordered as:
    - detection false alarm rate (Pfa)
    - detection recall
    - classification recall
    """
    if not tasks:
        return

    last = tasks[-1]
    val_acc = last.get("val_acc")
    val_det_acc = last.get("val_det_acc")
    val_det_fa = last.get("val_det_fa")

    if val_acc is None and val_det_acc is None and val_det_fa is None:
        return

    lengths = [
        len(v)
        for v in (val_det_fa, val_det_acc, val_acc)
        if v is not None
    ]
    if not lengths:
        return
    n_tasks = min(lengths)
    task_indices = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    colors = [get_task_color(i, task_names) for i in range(n_tasks)]

    # False alarm rate (Pfa) first.
    if val_det_fa is not None:
        fa_vals = np.asarray(val_det_fa, dtype=float)
        ax.bar(
            task_indices - width,
            fa_vals[:n_tasks],
            width=width,
            label="Det Pfa",
            color=colors,
            hatch="//",
        )

    # Detection recall.
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

    # Classification recall.
    if val_acc is not None:
        cls_vals = np.asarray(val_acc, dtype=float)
        ax.bar(
            task_indices + width,
            cls_vals[:n_tasks],
            width=width,
            label="Cls Recall",
            color=colors,
        )

    ax.set_xlabel("Task")
    ax.set_ylabel("Metric value")
    ax.set_title("Final validation metrics (after all tasks)")
    ax.set_xticks(task_indices)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "final_validation.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_validation_over_time(
    tasks: list[dict[str, Any]],
    output_dir: Path | None,
    task_names: Sequence[str] | None = None,
) -> None:
    """Plot validation recall per task as more tasks are trained (recall matrix)."""
    n_tasks = len(tasks)
    # After training task k, we have val_acc of length k+1 (tasks 0..k)
    x = np.arange(1, n_tasks + 1)  # "after task 0", "after task 1", ...

    fig, ax = plt.subplots(figsize=(9, 5))
    for task_idx in range(n_tasks):
        # For each task t, get its val acc from task files task_idx, task_idx+1, ..., n_tasks-1
        ys = []
        for k in range(n_tasks):
            val_acc = tasks[k]["val_acc"]
            if task_idx < len(val_acc):
                ys.append(val_acc[task_idx])
            else:
                ys.append(np.nan)
        ax.plot(
            x,
            ys,
            "o-",
            label=f"Task {task_idx}",
            color=get_task_color(task_idx, task_names),
            alpha=0.8,
        )

    ax.set_xlabel("After training up to task")
    ax.set_ylabel("Validation recall (classification)")
    ax.set_title("Validation recall per task over continual learning")
    ax.legend(ncol=min(n_tasks, 5), fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "val_acc_over_tasks.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def compute_average_forgetting(tasks: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute average forgetting over all previous tasks at each checkpoint.

    This uses classification F1 when available (``val_f1`` saved in metrics);
    otherwise it falls back to classification recall from ``val_acc``.

    For each task t, peak metric = metric[t] after training task t
    (i.e. tasks[t]['val_f1'][t] or tasks[t]['val_acc'][t]). After training
    task K > t, the metric on task t is tasks[K]['val_f1'][t] (or
    tasks[K]['val_acc'][t]). Forgetting for task t at K is
    max(0, peak[t] - current_metric).
    Average forgetting at K = mean over t in 0..K-1 (no previous tasks when K=0).

    Returns:
        x: checkpoint indices (0, 1, ..., n_tasks-1), "after training task K"
        avg_forgetting: average forgetting at each checkpoint (length n_tasks)
    """
    n_tasks = len(tasks)
    # Peak metric (F1 if present, else recall) after learning each task t.
    peak_vals = []
    for t in range(n_tasks):
        metrics_t = tasks[t]
        if "val_f1" in metrics_t and len(metrics_t["val_f1"]) > t:
            peak_vals.append(float(metrics_t["val_f1"][t]))
        else:
            peak_vals.append(float(metrics_t["val_acc"][t]))
    peak_vals_arr = np.asarray(peak_vals, dtype=float)
    avg_forgetting = np.zeros(n_tasks)
    for k in range(n_tasks):
        if k == 0:
            avg_forgetting[k] = 0.0
            continue
        forgets: list[float] = []
        for t in range(k):
            metrics_k = tasks[k]
            if "val_f1" in metrics_k and len(metrics_k["val_f1"]) > t:
                current_metric = float(metrics_k["val_f1"][t])
            else:
                current_metric = float(metrics_k["val_acc"][t])
            forgets.append(max(0.0, float(peak_vals_arr[t] - current_metric)))
        avg_forgetting[k] = float(np.mean(forgets)) if forgets else 0.0
    return np.arange(n_tasks), avg_forgetting


def plot_average_forgetting(tasks: list[dict[str, Any]], output_dir: Path | None) -> None:
    """Plot average forgetting over previous tasks vs checkpoint (after training task K)."""
    x, avg_forgetting = compute_average_forgetting(tasks)
    n_tasks = len(tasks)

    # Decide label based on whether F1 is available in metrics.
    has_f1 = any("val_f1" in t for t in tasks)
    y_label = "Average forgetting (Cls F1)" if has_f1 else "Average forgetting (Cls acc)"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, avg_forgetting, "o-", color="C0", linewidth=2, markersize=6)
    ax.set_xlabel("After training up to task")
    ax.set_ylabel(y_label)
    ax.set_title(f"Average forgetting on all previous tasks ({'Cls F1' if has_f1 else 'Cls acc'})")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "average_forgetting.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read and plot metrics from a run's metrics directory.",
    )
    parser.add_argument(
        "metrics_dir",
        type=Path,
        default=Path("logs/ctn/eclresm_test-2026-03-05_14-58-57-4091/0/metrics"),
        nargs="?",
        help="Path to the metrics directory (contains task0.npz, task1.npz, ...).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plot images. If not set, only display.",
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir.resolve()
    if not metrics_dir.is_dir():
        raise SystemExit(f"Not a directory: {metrics_dir}")

    import matplotlib
    if args.output_dir is None and matplotlib.get_backend().lower() == "agg":
        print(f"Non-interactive backend detected. Defaulting to saving plots in {metrics_dir}")
        args.output_dir = metrics_dir

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_metrics(metrics_dir)
    print(f"Loaded {len(tasks)} task(s) from {metrics_dir}")

    task_names = load_task_names(metrics_dir)

    plot_per_task_curves(tasks, args.output_dir, task_names=task_names)

    n_epochs = load_n_epochs(metrics_dir)
    if n_epochs is not None:
        plot_per_epoch_curves(tasks, n_epochs, args.output_dir, task_names=task_names)
    else:
        print("Skipping per-epoch plot (no n_epochs in training_parameters.json)")

    plot_final_validation(tasks, args.output_dir, task_names=task_names)
    plot_validation_over_time(tasks, args.output_dir, task_names=task_names)
    plot_average_forgetting(tasks, args.output_dir)

    if args.output_dir:
        print(f"Plots saved under {args.output_dir}")


if __name__ == "__main__":
    main()
