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
PlotStyle = Dict[str, Any]
DEFAULT_ALGORITHM_COLOR_ORDER = [
    "agem",
    "bcl_dual",
    "eralg4",
    "ewc",
    "iid2",
    "la-er",
    "si",
    "ucl",
    "cmaml",
    "er_ring",
    "gem",
    "icarl",
    "lamaml",
    "lwf",
    "rwalk",
    "smaml",
]
PLOT_STYLES_BY_EXPERIMENT: Dict[str, PlotStyle] = {
    "default": {
        "figsize": (6, 3),
        "ylim": None,
        "legend_kwargs": {
            "loc": "best",
            "ncol": 3,
            "fontsize": 10,
            "columnspacing": 0.9,
            "labelspacing": 0.3,
            "framealpha": 0.5,
            "borderaxespad": 0.1,
            "borderpad": 0.2,
        },
    },
    "til": {
        "figsize": (7, 3.5),
        "ylim": (-0.08, 0.2),
        "legend_kwargs": {
            "loc": "best",
            "ncol": 6,
            "fontsize": 10,
            "columnspacing": 0.7,
            "labelspacing": 0.3,
            "framealpha": 0.5,
            "borderaxespad": 0.1,
            "borderpad": 0.2,
        },
    },
}
ALGORITHM_DISPLAY_NAMES: Dict[str, str] = {
    "agem": "A-GEM",
    "bcl_dual": "BCL-Dual",
    "cmaml": "C-MAML",
    "ctn": "CTN",
    "eralg4": "ER-Res",
    "er_ring": "ER-Ring",
    "ewc": "EWC",
    "gem": "GEM",
    "hat": "HAT",
    "icarl": "iCaRL",
    "iid2": "IID2",
    "la-er": "La-ER",
    "lamaml": "La-MAML",
    "lwf": "LwF",
    "packnet": "PackNet",
    "rwalk": "RWalk",
    "si": "SI",
    "smaml": "S-MAML",
    "ucl": "UCL",
}


def build_label_colors(labels: List[str]) -> Dict[str, Any]:
    """Build distinct colors for labels using the combined-layout palette.

    This mirrors the color-selection approach from
    ``scripts/plot_multi_algorithms.py`` used by ``--plot-layout together``.
    """
    if not labels:
        return {}

    qualitative_palette: List[Any] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        cmap_colors = getattr(cmap, "colors", None)
        if cmap_colors is None:
            continue
        qualitative_palette.extend(list(cmap_colors))

    def color_for_position(label_position: int, total_labels: int) -> Any:
        if label_position < len(qualitative_palette):
            return qualitative_palette[label_position]
        if total_labels <= 1:
            return plt.get_cmap("hsv")(0.0)
        return plt.get_cmap("hsv")(label_position / float(total_labels))

    colors_by_label: Dict[str, Any] = {}
    known_algorithm_to_palette_index = {
        algorithm_name: index
        for index, algorithm_name in enumerate(DEFAULT_ALGORITHM_COLOR_ORDER)
    }

    # Use fixed palette positions per known algorithm so color remains stable
    # across plots even when some algorithms are absent.
    for label in labels:
        if label in known_algorithm_to_palette_index:
            colors_by_label[label] = color_for_position(
                known_algorithm_to_palette_index[label],
                len(DEFAULT_ALGORITHM_COLOR_ORDER),
            )

    # Assign deterministic colors to unknown labels after the known block.
    next_palette_index = len(DEFAULT_ALGORITHM_COLOR_ORDER)
    for label in sorted(labels):
        if label in colors_by_label:
            continue
        colors_by_label[label] = color_for_position(
            next_palette_index, next_palette_index + 1
        )
        next_palette_index += 1

    return colors_by_label


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
            "Defaults to '<repo>/logs/00_sync/<experiment>/plots/<json_stem>_<metric>_plot.png'."
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


def resolve_default_output_path(json_path: Path, metric_name: str) -> Path:
    """Resolve the default PNG output path under logs/00_sync.

    Args:
        json_path: Input metrics JSON path.
        metric_name: Metric key used in the generated filename.

    Returns:
        Default destination path for the output plot.
    """
    json_parts = json_path.parts
    if "logs" in json_parts:
        logs_index = json_parts.index("logs")
        suffix_parts = list(json_parts[logs_index + 1 : -1])
        if suffix_parts:
            suffix_parts[0] = "00_sync"
            return (
                Path(*json_parts[:logs_index])
                / "logs"
                / Path(*suffix_parts)
                / "plots"
                / f"{json_path.stem}_{metric_name}_plot.png"
            )

    return json_path.parent / "plots" / f"{json_path.stem}_{metric_name}_plot.png"


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
        if algorithm_name.strip().lower() == "iid2":
            continue
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
    plot_style: PlotStyle,
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
    figure, axis = plt.subplots(figsize=plot_style["figsize"])

    sorted_algorithm_names = sorted(series_by_algorithm.keys())
    label_colors = build_label_colors(sorted_algorithm_names)
    all_task_indices = set()
    task_index_to_dataset_name: Dict[int, str] = {}
    for algorithm_name in sorted_algorithm_names:
        series = series_by_algorithm[algorithm_name]
        x_positions = [point[0] for point in series]
        y_values = [point[1] for point in series]
        for task_index, _, task_name in series:
            all_task_indices.add(task_index)
            if task_index not in task_index_to_dataset_name:
                task_name_parts = task_name.split("-", 1)
                if len(task_name_parts) == 2 and task_name_parts[1]:
                    dataset_name = task_name_parts[1]
                else:
                    dataset_name = task_name
                dataset_name_lower = dataset_name.lower()
                if "uclresm" in dataset_name_lower:
                    dataset_name = "(RML)"
                elif "deeprad" in dataset_name_lower:
                    dataset_name = "(DR)"
                elif "rcn" in dataset_name_lower:
                    dataset_name = "(RCN)"
                task_index_to_dataset_name[task_index] = dataset_name

        axis.plot(
            x_positions,
            y_values,
            marker=".",
            linewidth=2.0,
            alpha=0.8,
            label=ALGORITHM_DISPLAY_NAMES.get(algorithm_name, algorithm_name),
            color=label_colors.get(algorithm_name),
        )

    sorted_task_indices = sorted(all_task_indices)
    axis.set_xticks(sorted_task_indices)
    axis.set_xticklabels(
        [
            f"{task_index}\n{task_index_to_dataset_name.get(task_index, 'unknown')}"
            for task_index in sorted_task_indices
        ]
    )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    axis.set_xlabel("Task", fontsize=16)
    axis.set_ylabel("Forward Transfer", fontsize=16)
    # axis.set_title(title or f"{metric_name} by task and algorithm")
    if plot_style["ylim"] is not None:
        axis.set_ylim(*plot_style["ylim"])
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend(**plot_style["legend_kwargs"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.1)
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def main() -> None:
    """Run the plotting pipeline."""
    arguments = parse_args()
    default_output_path = resolve_default_output_path(
        json_path=arguments.json_path,
        metric_name=arguments.metric,
    )
    output_path = arguments.output_path or default_output_path

    records = load_metrics(arguments.json_path)
    series_by_algorithm = build_series_by_algo(records, arguments.metric)
    experiment_path_text = str(arguments.json_path).lower()
    style_key = "til" if "til" in experiment_path_text else "default"
    selected_plot_style = PLOT_STYLES_BY_EXPERIMENT[style_key]
    plot_series(
        series_by_algorithm=series_by_algorithm,
        metric_name=arguments.metric,
        output_path=output_path,
        plot_style=selected_plot_style,
        title=arguments.title,
    )
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
