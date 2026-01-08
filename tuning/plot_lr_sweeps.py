"""Visualize learning-rate tuning results across models."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")  # Allows running in headless environments.
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - matplotlib not always installed.
    plt = None
    _HAS_MPL = False


MetricRow = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning-rate tuning sweeps across all models (filters to tuning/* datasets).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs/tuning"),
        help="Root directory that contains model tuning sessions.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_mean",
        help="Metric key to plot (e.g. val_mean, test_mean).",
    )
    parser.add_argument(
        "--lr-key",
        type=str,
        default="lr",
        help="Parameter name to treat as learning rate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/tuning/summary/lr_sweeps.png"),
        help="Output path for the faceted LR sweep plot.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=None,
        help="Output path for the model-vs-LR heatmap. Defaults next to --output.",
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=1,
        help="Minimum number of successful trials required to include a model.",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip heatmap generation.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots after saving (requires matplotlib).",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            return float(value)
    return None


def _iter_summary_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return root.rglob("summary.json")


def _resolve_model_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parents[1].name
    parts = rel.parts

    if len(parts) >= 2:
        return parts[0]
    return path.parents[1].name


def load_lr_metrics(root: Path, metric: str, lr_key: str) -> Dict[str, List[MetricRow]]:
    data: Dict[str, List[MetricRow]] = defaultdict(list)
    for summary_path in _iter_summary_files(root):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = _resolve_model_name(summary_path, root)
        for result in summary.get("results", []):
            if result.get("status") != "ok":
                continue
            params = result.get("params") or {}
            trial_params = result.get("trial_params") or {}
            lr_value = params.get(lr_key, trial_params.get(lr_key))
            lr = _safe_float(lr_value)
            score = _safe_float(result.get(metric))
            if lr is None or score is None:
                continue
            data[model].append((lr, score))
    return data


def aggregate_by_lr(rows: Iterable[MetricRow]) -> Dict[float, float]:
    grouped: Dict[float, List[float]] = defaultdict(list)
    for lr, score in rows:
        grouped[lr].append(score)
    return {lr: statistics.mean(scores) for lr, scores in grouped.items()}


def plot_lr_sweeps(
    model_rows: Dict[str, List[MetricRow]],
    metric: str,
    output: Path,
    show: bool,
) -> None:
    if not _HAS_MPL:
        print("Matplotlib is not available; skipping plot generation.")
        return

    models = sorted(model_rows)
    if not models:
        print("No tuning summaries found.")
        return

    cols = min(3, len(models))
    n_rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(5.0 * cols, 4.0 * n_rows), squeeze=False)
    fig.suptitle(f"Learning-rate sweeps ({metric})")

    for idx, model in enumerate(models):
        ax = axes[idx // cols][idx % cols]
        points = model_rows[model]
        lr_means = aggregate_by_lr(points)
        lrs_sorted = sorted(lr_means)
        means_sorted = [lr_means[lr] for lr in lrs_sorted]
        ax.scatter(
            [lr for lr, _ in points],
            [score for _, score in points],
            alpha=0.35,
            s=18,
        )
        if lrs_sorted:
            ax.plot(lrs_sorted, means_sorted, marker="o", linewidth=1.4)
        ax.set_xscale("log")
        ax.set_title(f"{model} (n={len(points)})")
        ax.set_xlabel(lr_label(model))
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.2)

    for idx in range(len(models), n_rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output, dpi=200)
    print(f"Saved LR sweep plot to {output}")

    if show:
        plt.show()
    plt.close(fig)


def lr_label(model: str) -> str:
    _ = model  # placeholder for future per-model labels
    return "lr"


def plot_lr_heatmap(
    model_rows: Dict[str, List[MetricRow]],
    metric: str,
    output: Path,
    show: bool,
) -> None:
    if not _HAS_MPL:
        print("Matplotlib is not available; skipping plot generation.")
        return

    models = sorted(model_rows)
    if not models:
        return

    lr_values: List[float] = sorted({lr for rows in model_rows.values() for lr, _ in rows})
    if not lr_values:
        return

    matrix: List[List[float]] = []
    for model in models:
        means = aggregate_by_lr(model_rows[model])
        row = [means.get(lr, float("nan")) for lr in lr_values]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(1.2 * len(lr_values) + 3, 0.6 * len(models) + 3))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(lr_values)))
    ax.set_xticklabels([f"{lr:g}" for lr in lr_values], rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("lr")
    ax.set_ylabel("model")
    ax.set_title(f"Mean {metric} by model and lr")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved LR heatmap to {output}")

    if show:
        plt.show()
    plt.close(fig)


def filter_models(model_rows: Dict[str, List[MetricRow]], min_trials: int) -> Dict[str, List[MetricRow]]:
    return {model: rows for model, rows in model_rows.items() if len(rows) >= min_trials}


def main() -> None:
    args = parse_args()
    model_rows = load_lr_metrics(args.root, args.metric, args.lr_key)
    model_rows = filter_models(model_rows, args.min_trials)

    if not model_rows:
        print("No successful tuning trials found/*.")
        return

    heatmap_output = args.heatmap_output
    if heatmap_output is None:
        heatmap_output = args.output.with_name(args.output.stem + "_heatmap.png")

    plot_lr_sweeps(model_rows, args.metric, args.output, args.show)

    if not args.no_heatmap:
        plot_lr_heatmap(model_rows, args.metric, heatmap_output, args.show)


if __name__ == "__main__":
    main()
