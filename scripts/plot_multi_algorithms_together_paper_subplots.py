"""Export paper-ready subplot PNGs for combined ("together") multi-algorithm plots.

This script is a purpose-built alternative to `scripts/plot_multi_algorithms.py
--save-subplots` for the `--plot-layout together` case. It applies the same
legend/layout configuration used in `scripts/plot_fwt_metrics.py` so exported
subplots are consistent with the paper-ready forward-transfer figures.

Typical usage:
    python scripts/plot_multi_algorithms_together_paper_subplots.py \
        --runs-dir logs/00_sync/one-shot_CIL \
        -o logs/00_sync/one-shot_CIL/plots
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

REPO_ROOT = Path(__file__).resolve().parents[1]

PlotStyle = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory containing job_logs/ (example: a full_experiments run folder)."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing algorithm subfolders with metrics/ (example: logs/00_sync/full-til_10epochs_w-zs)."
        ),
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root logs directory used by plot_multi_algorithms discovery helpers.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="When auto-discovering per-algorithm metrics, choose this run by recency.",
    )
    parser.add_argument(
        "--train-metric",
        type=str,
        choices=("total_f1", "cls_recall"),
        default="cls_recall",
        help="Training metric used in the first panel (train vs step).",
    )
    parser.add_argument(
        "--val-metric",
        type=str,
        choices=("total_f1", "cls_recall"),
        default="total_f1",
        help="Validation metric used in mean-val and average-forgetting panels.",
    )
    parser.add_argument(
        "--include-iid2",
        action="store_true",
        help="Include iid2 runs (by default they are excluded for clarity).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated labels for runs (legend entries).",
    )
    parser.add_argument(
        "--labels-grouping",
        type=str,
        default=None,
        help="Optional comma-separated keywords used to group labels by color family.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save the combined PNG and individual subplot PNGs.",
    )
    return parser.parse_args()


def _resolve_val_metric_for_run(choice: str, run: Any) -> tuple[str, str]:
    """Resolve validation metric key and label for a single run.

    Mirrors the behavior in `plot_multi_algorithms.py`.
    """
    if choice == "cls_recall":
        return "val_acc", "Cls recall"

    has_f1 = any("val_f1" in t for t in run.tasks)
    if has_f1:
        return "val_f1", "Total F1"

    print(
        f"[WARN] Requested val-metric=total_f1 but run '{run.name}' at {run.metrics_dir} "
        "has no 'val_f1'; falling back to cls recall ('val_acc')."
    )
    return "val_acc", "Cls recall"


def _case_insensitive_detect_style(runs: Sequence[Any]) -> PlotStyle:
    """Detect til vs default style based on path/name substrings (case-insensitive)."""
    from scripts.plot_fwt_metrics import PLOT_STYLES_BY_EXPERIMENT

    experiment_path_text = " ".join(
        [str(run.metrics_dir) for run in runs] + [run.name for run in runs]
    ).lower()
    style_key = "til" if "til" in experiment_path_text else "default"
    return PLOT_STYLES_BY_EXPERIMENT[style_key]


def _export_subplot(
    *,
    fig: plt.Figure,
    output_path: Path,
    subplot_dpi: float,
    pad_pixels: float,
    fixed_subplot_height_pixels: float,
    axis: plt.Axes,
    axis_bbox_title: str,
    renderer: Any,
    export_bbox: Bbox,
    bbox_height_inches: float,
    export_dpi_floor: float = 72.0,
) -> float:
    """Save a single subplot using the axis-only bbox cropping logic."""
    height_fixed_dpi = fixed_subplot_height_pixels / bbox_height_inches
    export_dpi_calc = min(subplot_dpi, height_fixed_dpi)
    export_dpi_calc = max(export_dpi_floor, export_dpi_calc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=export_dpi_calc, bbox_inches=export_bbox)
    print(f"Saved subplot to {output_path}")
    _ = axis_bbox_title  # keep for debugging hooks
    return export_dpi_calc


def main() -> None:
    """Run the plotting pipeline and export paper-ready subplot PNGs."""
    # Ensure `scripts/` can be imported as a namespace package when this file is
    # executed via `python scripts/<this_script>.py`.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from scripts.plot_multi_algorithms import (  # pylint: disable=import-error
        _build_axis_only_export_bbox,
        _build_label_colors,
        _compute_mean_val_metric_over_tasks,
        _concat_train_metric_for_run,
        _discover_runs_from_algorithm_root,
        _discover_runs_from_run_dir,
        _mean_final_metric_for_run,
        _prepare_algo_runs,
        _resolve_train_x_axis_label,
        compute_average_forgetting,
    )

    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir is not None and args.runs_dir is not None:
        raise SystemExit("Use either --run-dir or --runs-dir, not both.")

    run_source_dir: Path | None = (
        args.runs_dir if args.runs_dir is not None else args.run_dir
    )
    if run_source_dir is None:
        raise SystemExit("Provide --run-dir or --runs-dir.")

    if (run_source_dir / "job_logs").is_dir():
        algorithm_names, metrics_dirs = _discover_runs_from_run_dir(run_source_dir)
    else:
        algorithm_names, metrics_dirs = _discover_runs_from_algorithm_root(
            run_source_dir
        )

    runs = _prepare_algo_runs(
        algos=algorithm_names,
        metrics_dirs=metrics_dirs,
        logs_root=args.logs_root,
        run_index=args.run_index,
    )

    if not args.include_iid2:
        runs = [run for run in runs if run.name.strip().lower() != "iid2"]
        if not runs:
            raise SystemExit(
                "No runs left after filtering iid2. Pass --include-iid2 to include it."
            )

    print("Algorithms and metrics directories:")
    for run in runs:
        print(f"  {run.name}: {run.metrics_dir}")

    label_list: list[str] | None = None
    if args.labels is not None:
        label_list = [part.strip() for part in args.labels.split(",") if part.strip()]

    labels_grouping: list[str] | None = None
    if args.labels_grouping is not None:
        labels_grouping = [
            part.strip().lower()
            for part in args.labels_grouping.split(",")
            if part.strip()
        ]

    # Styling config derived from plot_fwt_metrics.py.
    plot_style = _case_insensitive_detect_style(runs)
    legend_kwargs: Dict[str, Any] = plot_style.get("legend_kwargs", {}) or {}

    # Create combined-row figure.
    n_rows = 1
    n_cols = 4
    col_width, row_height = 9, 4.5
    dpi = 550
    fig_width = col_width * n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, row_height * n_rows),
        squeeze=False,
        dpi=dpi,
    )

    train_metric_label = "Train recall"
    row_axes = axes[0]

    first_key, first_label = _resolve_val_metric_for_run(args.val_metric, runs[0])

    # Legend labels/colors for algorithm lines.
    if label_list is not None:
        if len(label_list) < len(runs):
            raise SystemExit(
                "Too few --labels entries for the number of runs discovered."
            )
        run_labels = label_list
    else:
        run_labels = [run.name for run in runs]

    label_colors = _build_label_colors(run_labels, labels_grouping)

    # Panel 1: train recall vs step.
    for run_idx, run in enumerate(runs):
        run_label = run_labels[run_idx]
        if not run.tasks:
            continue
        train_series, train_metric_label = _concat_train_metric_for_run(
            run.tasks, args.train_metric
        )
        if train_series is None:
            continue
        if args.train_metric == "total_f1":
            x_values = np.arange(1, len(train_series) + 1)
        else:
            x_values = np.arange(len(train_series))
        row_axes[0].plot(
            x_values,
            train_series,
            label=run_label,
            color=label_colors.get(run_label, f"C{run_idx % 10}"),
            alpha=0.9,
        )
    row_axes[0].set_ylabel(train_metric_label)
    row_axes[0].set_xlabel(_resolve_train_x_axis_label(args.train_metric))
    row_axes[0].grid(True, alpha=0.3)

    # Panel 2: final validation metrics (bars for Pfa/Det/Cls recall).
    for run_idx, run in enumerate(runs):
        if not run.tasks:
            continue
        last = run.tasks[-1]
        mean_pfa = _mean_final_metric_for_run(last, "val_det_fa", len(run.tasks))
        mean_det = _mean_final_metric_for_run(last, "val_det_acc", len(run.tasks))
        mean_cls = _mean_final_metric_for_run(last, "val_acc", len(run.tasks))

        x_center = run_idx
        width = 0.2
        if mean_pfa is not None:
            row_axes[1].bar(
                x_center - width,
                mean_pfa,
                width=width,
                label="Pfa" if run_idx == 0 else None,
                color=label_colors.get(run_labels[run_idx], f"C{run_idx % 10}"),
                hatch="//",
                alpha=0.8,
            )
        if mean_det is not None:
            row_axes[1].bar(
                x_center,
                mean_det,
                width=width,
                label="Det recall" if run_idx == 0 else None,
                color=label_colors.get(run_labels[run_idx], f"C{run_idx % 10}"),
                hatch="..",
                alpha=0.8,
            )
        if mean_cls is not None:
            row_axes[1].bar(
                x_center + width,
                mean_cls,
                width=width,
                label="Cls recall" if run_idx == 0 else None,
                color=label_colors.get(run_labels[run_idx], f"C{run_idx % 10}"),
                alpha=0.8,
            )
    row_axes[1].set_ylabel("Metric value")
    row_axes[1].set_xticks(np.arange(len(runs)))
    row_axes[1].set_xticklabels([])
    row_axes[1].set_xlabel("")
    row_axes[1].grid(True, alpha=0.3, axis="y")

    # Panel 3: mean validation metric over tasks (one line per run).
    for run_idx, run in enumerate(runs):
        x_vals, y_vals = _compute_mean_val_metric_over_tasks(run.tasks, first_key)
        if x_vals.size == 0:
            continue
        row_axes[2].plot(
            x_vals,
            y_vals,
            "o-",
            label=run_labels[run_idx],
            color=label_colors.get(run_labels[run_idx], f"C{run_idx % 10}"),
            alpha=0.9,
        )
    row_axes[2].set_xlabel("After training up to task")
    row_axes[2].set_ylabel(f"Val {first_label}")
    row_axes[2].grid(True, alpha=0.3)

    # Panel 4: average forgetting over tasks (one line per run).
    for run_idx, run in enumerate(runs):
        val_metric_key, _ = _resolve_val_metric_for_run(args.val_metric, run)
        x_vals, y_vals = compute_average_forgetting(run.tasks, val_metric_key)
        if x_vals.size == 0:
            continue
        row_axes[3].plot(
            x_vals + 1,
            y_vals,
            "o-",
            label=run_labels[run_idx],
            color=label_colors.get(run_labels[run_idx], f"C{run_idx % 10}"),
            alpha=0.9,
        )
    row_axes[3].set_xlabel("After training up to task")
    row_axes[3].set_ylabel(f"Avg forgetting ({first_label})")
    row_axes[3].grid(True, alpha=0.3)

    # Titles for the combined panels (these will also be used for subplot filenames).
    axes[0][0].set_title(
        f"{train_metric_label} vs {_resolve_train_x_axis_label(args.train_metric).lower()}"
    )
    axes[0][1].set_title("Final validation metrics")
    axes[0][2].set_title(f"Mean val {first_label} over tasks")
    axes[0][3].set_title(f"Average forgetting ({first_label})")

    # Rebuild algorithm legends explicitly so legend_kwargs (ncol/spacing/etc.)
    # are applied consistently before export.
    for panel_idx in (0, 2, 3):
        handles, labels = row_axes[panel_idx].get_legend_handles_labels()
        if handles and labels and legend_kwargs:
            row_axes[panel_idx].legend(handles, labels, **legend_kwargs)

    fig.tight_layout()
    combined_path = output_dir / "multi_algorithms_metrics.png"
    fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {combined_path}")

    # Apply per-axis paper-ready export settings.
    subplot_dpi = max(300, dpi)
    subplot_pad_pixels = 8.0
    fixed_subplot_height_pixels = 1000.0
    subplot_left = 0.1
    subplot_right = 0.9
    subplot_bottom = 0.2
    subplot_top = 0.99

    subplot_figure_width_inches = float(plot_style.get("figsize", (8.0, 6.0))[0])
    subplot_figure_height_inches = float(plot_style.get("figsize", (8.0, 6.0))[1])
    original_figure_size_inches = fig.get_size_inches().copy()

    all_axes: List[plt.Axes] = [ax for row in axes for ax in row]
    renderer: Any = None
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Match the font/layout tuning used by plot_multi_algorithms' exports.
    legend_fontsize = float(legend_kwargs.get("fontsize", 12.0))
    for axis in all_axes:
        axis.xaxis.label.set_size(16)
        axis.yaxis.label.set_size(16)
        axis.tick_params(axis="both", labelsize=13)

        legend = axis.get_legend()
        if legend is not None:
            for legend_text in legend.get_texts():
                legend_text.set_fontsize(legend_fontsize)

    # Track original positions/titles so we can restore after exports.
    original_positions: Dict[plt.Axes, Any] = {
        ax: ax.get_position().frozen() for ax in all_axes
    }
    original_titles: Dict[plt.Axes, str] = {ax: ax.get_title() for ax in all_axes}

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            axis = axes[row_idx][col_idx]
            fig.set_size_inches(
                subplot_figure_width_inches, subplot_figure_height_inches, forward=True
            )

            axis.set_position(
                [
                    subplot_left,
                    subplot_bottom,
                    max(subplot_right - subplot_left, 1e-3),
                    max(subplot_top - subplot_bottom, 1e-3),
                ]
            )
            axis.set_anchor("W")
            if axis.lines:
                axis.margins(x=0.0)

            # Apply the til-specific ylim for the "Avg forgetting" panel.
            if (
                plot_style.get("ylim") is not None
                and "Avg forgetting" in axis.get_ylabel()
            ):
                axis.set_ylim(*plot_style["ylim"])

            subplot_title = axis.get_title().strip()
            if subplot_title:
                title_stem = re.sub(r"[^A-Za-z0-9]+", "_", subplot_title).strip("_")
                if not title_stem:
                    title_stem = "subplot"
            else:
                title_stem = "subplot"

            axis.set_title("")

            # Hide non-target axes to avoid overlap in the export bbox.
            for other_axis in all_axes:
                if other_axis is not axis:
                    other_axis.set_visible(False)

            # Ensure legend kwargs are applied at export time.
            handles, labels = axis.get_legend_handles_labels()
            if handles and labels and legend_kwargs:
                axis.legend(handles, labels, **legend_kwargs)

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axis_bbox_pixels = _build_axis_only_export_bbox(
                axis=axis, renderer=renderer, pad_pixels=subplot_pad_pixels
            )
            bbox = axis_bbox_pixels.transformed(fig.dpi_scale_trans.inverted())
            bbox_height_inches = max(float(bbox.height), 1e-6)

            subplot_path = (
                output_dir / f"{title_stem.lower()}_r{row_idx + 1}_c{col_idx + 1}.png"
            )
            export_bbox = bbox
            _export_subplot(
                fig=fig,
                output_path=subplot_path,
                subplot_dpi=subplot_dpi,
                pad_pixels=subplot_pad_pixels,
                fixed_subplot_height_pixels=fixed_subplot_height_pixels,
                axis=axis,
                axis_bbox_title=title_stem,
                renderer=renderer,
                export_bbox=export_bbox,
                bbox_height_inches=bbox_height_inches,
            )

            # Restore axes state.
            axis.set_position(original_positions[axis])
            axis.set_anchor("C")
            for other_axis in all_axes:
                if other_axis is not axis:
                    other_axis.set_visible(True)

    for axis in all_axes:
        axis.set_title(original_titles[axis])

    fig.set_size_inches(
        float(original_figure_size_inches[0]),
        float(original_figure_size_inches[1]),
        forward=True,
    )
    fig.canvas.draw()
    plt.close(fig)


if __name__ == "__main__":
    main()
