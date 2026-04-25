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
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PlotStyle = Dict[str, Any]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1))]
LINEWIDTHS = [2.4, 1.8, 1.8, 1.5, 1.5]

# Manual plot controls (set values to ``None`` to keep auto behavior).
# Keys: train, final_validation, mean_val, average_forgetting
PANEL_YLIM_OVERRIDES: Dict[str, tuple[float, float] | None] = {
    "train": None,
    "final_validation": None,
    "mean_val": (0, 0.9),
    "average_forgetting": (-0.8, 0.2),
}

# Manual legend ncol controls (set values to ``None`` to keep auto behavior).
# Keys: train, final_validation, mean_val, average_forgetting
PANEL_LEGEND_NCOL_OVERRIDES: Dict[str, int | None] = {
    "train": None,
    "final_validation": None,
    "mean_val": 6,
    "average_forgetting": 6,
}

# Global x-axis label spacing (distance from axis to xlabel text).
X_LABEL_PAD: float = 4.0
# Global tick-label font size (applies to both x and y axes).
TICK_LABEL_FONT_SIZE: float = 12.0
# Optional legend placement: move legends above line plots for consistent axes.
LEGEND_ABOVE_PLOT: bool = True
# Vertical offset used when LEGEND_ABOVE_PLOT=True.
LEGEND_ABOVE_BBOX_Y: float = 1.02


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

    style_key = _case_insensitive_detect_style_key(runs)
    return PLOT_STYLES_BY_EXPERIMENT[style_key]


def _case_insensitive_detect_style_key(runs: Sequence[Any]) -> str:
    """Detect style key based on path/name substrings (case-insensitive)."""
    experiment_path_text = " ".join(
        [str(run.metrics_dir) for run in runs] + [run.name for run in runs]
    ).lower()
    return "til" if "til" in experiment_path_text else "default"


def _build_export_legend_kwargs(
    base_legend_kwargs: Dict[str, Any], panel_key: str
) -> Dict[str, Any]:
    """Build legend kwargs for subplot export."""
    legend_kwargs = dict(base_legend_kwargs)
    manual_ncol = PANEL_LEGEND_NCOL_OVERRIDES.get(panel_key)
    if manual_ncol is not None:
        legend_kwargs["ncol"] = int(manual_ncol)
    if LEGEND_ABOVE_PLOT and panel_key in {"train", "mean_val", "average_forgetting"}:
        legend_kwargs["loc"] = "lower center"
        legend_kwargs["bbox_to_anchor"] = (0.5, LEGEND_ABOVE_BBOX_Y)
        legend_kwargs.setdefault("borderaxespad", 0.0)
    return legend_kwargs


def _format_task_dataset_label(task_name: str, fallback_task_number: int) -> str:
    """Format task dataset labels to mirror plot_fwt_metrics.py style."""
    task_name_parts = task_name.split("-", 1)
    if len(task_name_parts) == 2 and task_name_parts[1]:
        original_task_number = task_name_parts[0].lstrip("t")
        dataset_name = task_name_parts[1]
    else:
        original_task_number = str(fallback_task_number)
        dataset_name = task_name
    dataset_name_lower = dataset_name.lower()
    if "uclresm" in dataset_name_lower:
        return f"(RML{original_task_number})"
    if "deeprad" in dataset_name_lower:
        return f"(DR{original_task_number})"
    if "rcn" in dataset_name_lower:
        return f"(RCN{original_task_number})"
    return dataset_name


def _build_task_index_to_dataset_name(runs: Sequence[Any]) -> Dict[int, str]:
    """Build task-index label map for x-axis display."""
    task_index_to_dataset_name: Dict[int, str] = {}
    for run in runs:
        run_task_names = getattr(run, "task_names", None)
        for task_index, task in enumerate(run.tasks):
            if task_index in task_index_to_dataset_name:
                continue
            task_name = (
                str(run_task_names[task_index])
                if run_task_names is not None and task_index < len(run_task_names)
                else str(task.get("task_name", f"t{task_index}"))
            )
            task_index_to_dataset_name[task_index] = _format_task_dataset_label(
                task_name, task_index
            )
    return task_index_to_dataset_name


def _set_task_axis_like_fwt(
    axis: plt.Axes,
    task_positions: Sequence[int],
    task_index_to_dataset_name: Dict[int, str],
) -> None:
    """Set x ticks every 1 and labels like plot_fwt_metrics.py."""
    if not task_positions:
        return
    min_position = int(min(task_positions))
    max_position = int(max(task_positions))
    tick_positions = list(range(min_position, max_position + 1))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [
            f"{task_position}\n"
            f"{task_index_to_dataset_name.get(task_position - 1, 'unknown')}"
            for task_position in tick_positions
        ]
    )


def _save_independent_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    """Save one independent panel figure as both PDF and PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    base_output_path = output_path.with_suffix("")
    pdf_output_path = base_output_path.with_suffix(".pdf")
    png_output_path = base_output_path.with_suffix(".png")
    fig.savefig(pdf_output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved subplot to {pdf_output_path} and {png_output_path}")
    plt.close(fig)


def _build_experiment_prefix(run_source_dir: Path, runs: Sequence[Any]) -> str:
    """Build '<Multi/Single>-Epoch_<TIL/CIL>' prefix from run context."""
    experiment_path_parts = [str(run_source_dir)] + [
        str(run.metrics_dir) for run in runs
    ]
    experiment_path_text = " ".join(experiment_path_parts).lower()
    epoch_mode = "Single" if "one-shot" in experiment_path_text else "Multi"

    def has_mode_token(text: str, mode: str) -> bool:
        return bool(re.search(rf"(^|[-_/]){mode}($|[-_/])", text.lower()))

    # Prefer explicit mode from the user-provided run root.
    run_root_text = str(run_source_dir).lower()
    if has_mode_token(run_root_text, "til"):
        learning_setup = "TIL"
    elif has_mode_token(run_root_text, "cil"):
        learning_setup = "CIL"
    else:
        # Fallback: infer from discovered metrics directories.
        metrics_paths_text = " ".join(str(run.metrics_dir).lower() for run in runs)
        if has_mode_token(metrics_paths_text, "til"):
            learning_setup = "TIL"
        elif has_mode_token(metrics_paths_text, "cil"):
            learning_setup = "CIL"
        else:
            learning_setup = "TIL"
    return f"{epoch_mode}-Epoch_{learning_setup}"


def _line_style_for_index(index: int) -> Dict[str, Any]:
    """Return line width/style settings aligned with plot_fwt_metrics.py."""
    return {
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        "linewidth": LINEWIDTHS[index % len(LINEWIDTHS)],
        "alpha": 0.9,
    }


def _build_group_color_lookup_for_runs(
    run_names: Sequence[str],
    group_colors: Sequence[str],
) -> Dict[str, str]:
    """Build run->color mapping using FWT group ordering/color rules."""
    from scripts.plot_fwt_metrics import _build_ordered_groups
    from scripts.plot_algorithm_group_styles import group_sort_key

    sorted_run_names = sorted(run_names, key=group_sort_key)
    ordered_groups = _build_ordered_groups(sorted_run_names)
    run_to_color: Dict[str, str] = {}
    for group_index, (_, member_names) in enumerate(ordered_groups):
        group_color = group_colors[group_index % len(group_colors)]
        for member_name in member_names:
            run_to_color[member_name] = group_color
    return run_to_color


def main() -> None:
    """Run the plotting pipeline and export paper-ready subplot PNGs."""
    # Ensure `scripts/` can be imported as a namespace package when this file is
    # executed via `python scripts/<this_script>.py`.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from scripts.plot_multi_algorithms import (  # pylint: disable=import-error
        _compute_mean_val_metric_over_tasks,
        _concat_train_metric_for_run,
        _discover_runs_from_algorithm_root,
        _discover_runs_from_run_dir,
        _mean_final_metric_for_run,
        _prepare_algo_runs,
        _resolve_train_x_axis_label,
        compute_average_forgetting,
    )
    from scripts.plot_fwt_metrics import (
        ALGORITHM_DISPLAY_NAMES,
        GROUP_COLORS as fwt_group_colors,
    )
    from scripts.plot_algorithm_group_styles import group_sort_key
    from scripts.plot_style_overrides import resolve_legend_kwargs

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
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

    excluded_run_name_set = {"saved_models", "models", "plots", "figures"}
    if not args.include_iid2:
        excluded_run_name_set.add("iid2")
    runs = [
        run for run in runs if run.name.strip().lower() not in excluded_run_name_set
    ]
    if not runs:
        raise SystemExit(
            "No runs left after filtering wrapper directories/iid2. "
            "Pass --include-iid2 to include iid2."
        )
    runs = sorted(runs, key=lambda run: group_sort_key(run.name))

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
    dpi = 550
    figure_size = tuple(map(float, plot_style.get("figsize", (7.0, 3.5))))
    first_key, first_label = _resolve_val_metric_for_run(args.val_metric, runs[0])

    # Legend labels/colors for algorithm lines.
    if label_list is not None:
        if len(label_list) < len(runs):
            raise SystemExit(
                "Too few --labels entries for the number of runs discovered."
            )
        run_labels = label_list
    else:
        run_labels = [ALGORITHM_DISPLAY_NAMES.get(run.name, run.name) for run in runs]

    if labels_grouping:
        print(
            "[WARN] --labels-grouping is ignored in this script to preserve "
            "per-algorithm colors from plot_fwt_metrics.py."
        )

    algorithm_colors = _build_group_color_lookup_for_runs(
        [run.name for run in runs],
        fwt_group_colors,
    )
    task_index_to_dataset_name = _build_task_index_to_dataset_name(runs)
    style_key = _case_insensitive_detect_style_key(runs)
    experiment_prefix = _build_experiment_prefix(run_source_dir, runs)

    # Figure 1: train recall vs step.
    fig_train, axis_train = plt.subplots(figsize=figure_size, dpi=dpi)
    train_metric_label = "Train recall"
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
        axis_train.plot(
            x_values,
            train_series,
            label=run_label,
            color=algorithm_colors.get(run.name, f"C{run_idx % 10}"),
            **_line_style_for_index(run_idx),
        )
    axis_train.set_ylabel(train_metric_label, fontsize=16)
    axis_train.set_xlabel(
        _resolve_train_x_axis_label(args.train_metric),
        fontsize=16,
        labelpad=X_LABEL_PAD,
    )
    axis_train.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_train.grid(True, alpha=0.3)
    handles_train, labels_train = axis_train.get_legend_handles_labels()
    if handles_train and labels_train:
        export_legend_kwargs = _build_export_legend_kwargs(
            resolve_legend_kwargs(
                style_key=style_key,
                panel_key="train",
                base_legend_kwargs=legend_kwargs,
                run_count=len(runs),
            ),
            "train",
        )
        axis_train.legend(handles_train, labels_train, **export_legend_kwargs)
    manual_ylim_train = PANEL_YLIM_OVERRIDES.get("train")
    if manual_ylim_train is not None:
        axis_train.set_ylim(*manual_ylim_train)
    _save_independent_figure(
        fig_train,
        output_dir / f"{experiment_prefix}_TR-F1",
        dpi,
    )

    # Figure 2: final validation metrics (bars for Pfa/Det/Cls recall).
    fig_final, axis_final = plt.subplots(figsize=figure_size, dpi=dpi)
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
            axis_final.bar(
                x_center - width,
                mean_pfa,
                width=width,
                label="Pfa" if run_idx == 0 else None,
                color=algorithm_colors.get(run.name, f"C{run_idx % 10}"),
                hatch="//",
                alpha=0.8,
            )
        if mean_det is not None:
            axis_final.bar(
                x_center,
                mean_det,
                width=width,
                label="Det recall" if run_idx == 0 else None,
                color=algorithm_colors.get(run.name, f"C{run_idx % 10}"),
                hatch="..",
                alpha=0.8,
            )
        if mean_cls is not None:
            axis_final.bar(
                x_center + width,
                mean_cls,
                width=width,
                label="Cls recall" if run_idx == 0 else None,
                color=algorithm_colors.get(run.name, f"C{run_idx % 10}"),
                alpha=0.8,
            )
    axis_final.set_ylabel("Metric value", fontsize=16)
    axis_final.set_xticks(np.arange(len(runs)))
    axis_final.set_xticklabels([])
    axis_final.set_xlabel("")
    axis_final.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_final.grid(True, alpha=0.3, axis="y")
    handles_final, labels_final = axis_final.get_legend_handles_labels()
    if handles_final and labels_final:
        export_legend_kwargs = resolve_legend_kwargs(
            style_key=style_key,
            panel_key="final_validation",
            base_legend_kwargs=legend_kwargs,
            run_count=len(runs),
        )
        manual_ncol_final = PANEL_LEGEND_NCOL_OVERRIDES.get("final_validation")
        if manual_ncol_final is not None:
            export_legend_kwargs["ncol"] = int(manual_ncol_final)
        axis_final.legend(handles_final, labels_final, **export_legend_kwargs)
    manual_ylim_final = PANEL_YLIM_OVERRIDES.get("final_validation")
    if manual_ylim_final is not None:
        axis_final.set_ylim(*manual_ylim_final)
    _save_independent_figure(
        fig_final, output_dir / f"{experiment_prefix}_Final-VAL", dpi
    )

    # Figure 3: mean validation metric over tasks (one line per run).
    fig_mean, axis_mean = plt.subplots(figsize=figure_size, dpi=dpi)
    mean_task_positions: set[int] = set()
    for run_idx, run in enumerate(runs):
        x_vals, y_vals = _compute_mean_val_metric_over_tasks(run.tasks, first_key)
        if x_vals.size == 0:
            continue
        mean_task_positions.update(int(value) for value in x_vals.tolist())
        axis_mean.plot(
            x_vals,
            y_vals,
            label=run_labels[run_idx],
            color=algorithm_colors.get(run.name, f"C{run_idx % 10}"),
            **_line_style_for_index(run_idx),
        )
    axis_mean.set_xlabel("Task", fontsize=16, labelpad=X_LABEL_PAD)
    axis_mean.set_ylabel("F1 Score", fontsize=16)
    _set_task_axis_like_fwt(
        axis_mean,
        sorted(mean_task_positions),
        task_index_to_dataset_name,
    )
    axis_mean.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_mean.grid(True, alpha=0.3)
    handles_mean, labels_mean = axis_mean.get_legend_handles_labels()
    if handles_mean and labels_mean:
        export_legend_kwargs = _build_export_legend_kwargs(
            resolve_legend_kwargs(
                style_key=style_key,
                panel_key="mean_val",
                base_legend_kwargs=legend_kwargs,
                run_count=len(runs),
            ),
            "mean_val",
        )
        axis_mean.legend(handles_mean, labels_mean, **export_legend_kwargs)
    manual_ylim_mean = PANEL_YLIM_OVERRIDES.get("mean_val")
    if manual_ylim_mean is not None:
        axis_mean.set_ylim(*manual_ylim_mean)
    _save_independent_figure(
        fig_mean,
        output_dir / f"{experiment_prefix}_CL-F1",
        dpi,
    )

    # Figure 4: backward transfer over tasks (one line per run).
    fig_forgetting, axis_forgetting = plt.subplots(figsize=figure_size, dpi=dpi)
    forgetting_task_positions: set[int] = set()
    for run_idx, run in enumerate(runs):
        val_metric_key, _ = _resolve_val_metric_for_run(args.val_metric, run)
        x_vals, y_vals = compute_average_forgetting(run.tasks, val_metric_key)
        if x_vals.size == 0:
            continue
        backward_transfer_values = -y_vals
        shifted_x_values = x_vals + 1
        forgetting_task_positions.update(
            int(value) for value in shifted_x_values.tolist()
        )
        axis_forgetting.plot(
            shifted_x_values,
            backward_transfer_values,
            label=run_labels[run_idx],
            color=algorithm_colors.get(run.name, f"C{run_idx % 10}"),
            **_line_style_for_index(run_idx),
        )
    axis_forgetting.set_xlabel("Task", fontsize=16, labelpad=X_LABEL_PAD)
    axis_forgetting.set_ylabel("Backward Transfer", fontsize=16)
    _set_task_axis_like_fwt(
        axis_forgetting,
        sorted(forgetting_task_positions),
        task_index_to_dataset_name,
    )
    axis_forgetting.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    axis_forgetting.grid(True, alpha=0.3)
    manual_ylim_forgetting = PANEL_YLIM_OVERRIDES.get("average_forgetting")
    if manual_ylim_forgetting is not None:
        axis_forgetting.set_ylim(*manual_ylim_forgetting)
    elif plot_style.get("ylim") is not None:
        axis_forgetting.set_ylim(*plot_style["ylim"])
    handles_forgetting, labels_forgetting = axis_forgetting.get_legend_handles_labels()
    if handles_forgetting and labels_forgetting:
        export_legend_kwargs = _build_export_legend_kwargs(
            resolve_legend_kwargs(
                style_key=style_key,
                panel_key="average_forgetting",
                base_legend_kwargs=legend_kwargs,
                run_count=len(runs),
            ),
            "average_forgetting",
        )
        axis_forgetting.legend(
            handles_forgetting, labels_forgetting, **export_legend_kwargs
        )
    _save_independent_figure(
        fig_forgetting,
        output_dir / f"{experiment_prefix}_BWT",
        dpi,
    )


if __name__ == "__main__":
    main()
