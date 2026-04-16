"""Plot forward-transfer metrics from a metrics JSON file.

This script reads a JSON file containing per-task metrics for multiple
algorithms and generates a line plot for one selected metric.

Usage:
    python scripts/plot_fwt_metrics.py \
        --json-path logs/full_experiments/one-shot_cil/fwt_metrics.json

    python scripts/plot_fwt_metrics.py \
        --json-path logs/full_experiments/one-shot_cil/fwt_metrics.json \
        --metric forward_transfer_total_f1_zs \
        --output-path logs/full_experiments/one-shot_cil/fwt_metrics_plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

MetricRecord = Dict[str, Any]
SeriesPoint = Tuple[int, Optional[float], str]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-path",
        type=Path,
        required=True,
        help="Path to the metrics JSON file.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="forward_transfer_total_f1_zs",
        help="Metric key to plot from each record.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to save the figure. "
            "Defaults to '<json_stem>_<metric>.png' next to the JSON file."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    return parser.parse_args()


def load_metrics(json_path: Path) -> List[MetricRecord]:
    """Load metrics records from JSON.

    Args:
        json_path: Path to JSON containing a list of metric records.

    Returns:
        List of metric dictionaries.

    Raises:
        FileNotFoundError: If the JSON path does not exist.
        ValueError: If JSON is not a list of records.
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, list):
        raise ValueError("Expected JSON root to be a list of metric records.")
    return loaded


def build_series_by_algo(
    records: List[MetricRecord], metric_name: str
) -> Dict[str, List[SeriesPoint]]:
    """Group metric points by algorithm and sort by task index.

    Args:
        records: Flat list of metric records.
        metric_name: Metric key to extract from each record.

    Returns:
        Mapping from algorithm name to ordered points:
        ``(task_index, metric_value, task_name)``.
    """
    series_by_algorithm: Dict[str, List[SeriesPoint]] = {}
    for record in records:
        algorithm_name = str(record.get("algo", "unknown"))
        task_index = int(record.get("task_index", -1))
        task_name = str(record.get("task_name", f"task_{task_index}"))
        raw_metric_value = record.get(metric_name, None)
        metric_value = float(raw_metric_value) if raw_metric_value is not None else None

        series_by_algorithm.setdefault(algorithm_name, []).append(
            (task_index, metric_value, task_name)
        )

    for algorithm_name, series in series_by_algorithm.items():
        series_by_algorithm[algorithm_name] = sorted(series, key=lambda point: point[0])
    return series_by_algorithm


def plot_series(
    series_by_algorithm: Dict[str, List[SeriesPoint]],
    metric_name: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Create and save the metric line plot.

    Args:
        series_by_algorithm: Algorithm-to-series mapping.
        metric_name: Metric key being plotted.
        output_path: Where to save the PNG.
        title: Optional custom chart title.

    Usage:
        >>> plot_series({"algo": [(0, 0.1, "t0")]}, "metric", Path("out.png"))
    """
    figure, axis = plt.subplots(figsize=(8, 4))

    all_task_indices = set()
    for algorithm_name in sorted(series_by_algorithm.keys()):
        series = series_by_algorithm[algorithm_name]
        x_positions = [point[0] for point in series]
        y_values = [point[1] for point in series]
        for task_index, _, _ in series:
            all_task_indices.add(task_index)

        axis.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=2.0,
            alpha=0.9,
            label=algorithm_name,
        )

    sorted_task_indices = sorted(all_task_indices)
    axis.set_xticks(sorted_task_indices)
    axis.set_xticklabels([str(task_index) for task_index in sorted_task_indices])
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    axis.set_xlabel("Task", fontsize=14)
    axis.set_ylabel("Forward Transfer", fontsize=14)
    # axis.set_title(title or f"{metric_name} by task and algorithm")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend(loc="best", ncol=2, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def main() -> None:
    """Run the plotting pipeline."""
    arguments = parse_args()
    default_output_path = (
        arguments.json_path.parent
        / f"{arguments.json_path.stem}_{arguments.metric}_plot.png"
    )
    output_path = arguments.output_path or default_output_path

    records = load_metrics(arguments.json_path)
    series_by_algorithm = build_series_by_algo(records, arguments.metric)
    plot_series(
        series_by_algorithm=series_by_algorithm,
        metric_name=arguments.metric,
        output_path=output_path,
        title=arguments.title,
    )
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
