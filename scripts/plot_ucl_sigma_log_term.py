"""Visualize the UCL sigma regularizer log term.

This script plots the term used in `R_sigma` from `background_knowledge/ucl_bresnet.tex`:

    f(r) = r - log(r + epsilon),  where r = (sigma_new^2) / (sigma_old^2)

It also overlays the isolated log contribution `-log(r + epsilon)` so the effect
of the logarithmic correction is easy to inspect.

Usage:
    source la-maml_env/bin/activate
    python scripts/plot_ucl_sigma_log_term.py

    python scripts/plot_ucl_sigma_log_term.py \
        --epsilon 1e-6 \
        --ratio-min 0.01 \
        --ratio-max 4.0 \
        --num-points 800 \
        --output-path plots/ucl_sigma_log_term.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.

    Usage:
        args = parse_args()
    """
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Numerical epsilon added inside log(r + epsilon).",
    )
    argument_parser.add_argument(
        "--ratio-min",
        type=float,
        default=1e-3,
        help="Minimum ratio r = (sigma_new^2)/(sigma_old^2). Must be > 0.",
    )
    argument_parser.add_argument(
        "--ratio-max",
        type=float,
        default=5.0,
        help="Maximum ratio r = (sigma_new^2)/(sigma_old^2).",
    )
    argument_parser.add_argument(
        "--num-points",
        type=int,
        default=1200,
        help="Number of points sampled between ratio-min and ratio-max.",
    )
    argument_parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("plots/ucl_sigma_log_term.png"),
        help="Destination file for the figure.",
    )
    return argument_parser.parse_args()


def compute_ucl_sigma_terms(
    ratio_values: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the UCL sigma regularizer components.

    Args:
        ratio_values: Ratio values r = (sigma_new^2)/(sigma_old^2).
        epsilon: Numerical stabilizer added in the logarithm.

    Returns:
        A tuple containing:
            - full_regularizer_values: f(r) = r - log(r + epsilon)
            - log_component_values: g(r) = -log(r + epsilon)

    Usage:
        full_term, log_term = compute_ucl_sigma_terms(ratios, epsilon=1e-8)
    """
    log_component_values = -np.log(ratio_values + epsilon)
    full_regularizer_values = ratio_values + log_component_values
    return full_regularizer_values, log_component_values


def validate_args(parsed_args: argparse.Namespace) -> None:
    """Validate argument constraints before plotting.

    Args:
        parsed_args: Parsed CLI namespace from `parse_args`.

    Raises:
        ValueError: If any argument is outside expected bounds.

    Usage:
        validate_args(args)
    """
    if parsed_args.epsilon <= 0:
        raise ValueError("--epsilon must be > 0.")
    if parsed_args.ratio_min <= 0:
        raise ValueError("--ratio-min must be > 0.")
    if parsed_args.ratio_max <= parsed_args.ratio_min:
        raise ValueError("--ratio-max must be larger than --ratio-min.")
    if parsed_args.num_points < 10:
        raise ValueError("--num-points must be at least 10.")


def main() -> None:
    """Run the visualization script."""
    parsed_args = parse_args()
    validate_args(parsed_args)

    ratio_values = np.linspace(
        parsed_args.ratio_min, parsed_args.ratio_max, parsed_args.num_points
    )
    full_regularizer_values, log_component_values = compute_ucl_sigma_terms(
        ratio_values=ratio_values,
        epsilon=parsed_args.epsilon,
    )

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        ratio_values,
        full_regularizer_values,
        label=r"$r - \log(r+\epsilon)$ (full term)",
        linewidth=2.2,
    )
    axis.plot(
        ratio_values,
        log_component_values,
        label=r"$-\log(r+\epsilon)$ (log component)",
        linestyle="--",
        linewidth=2.0,
    )
    axis.axvline(1.0, color="black", linestyle=":", linewidth=1.2, label=r"$r=1$")
    axis.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    axis.set_xlabel(r"$r$")
    axis.set_ylabel("Term value")
    axis.set_title("UCL variance-drift penalty shape")
    axis.grid(True, alpha=0.25)

    y_minimum, y_maximum = axis.get_ylim()
    annotation_y = y_minimum + 0.11 * (y_maximum - y_minimum)
    label_y = y_minimum + 0.09 * (y_maximum - y_minimum)
    available_left_span = 1.0 - parsed_args.ratio_min
    available_right_span = parsed_args.ratio_max - 1.0
    symmetric_available_span = min(available_left_span, available_right_span)
    center_offset = 0.10 * symmetric_available_span
    arrow_length = 0.65 * symmetric_available_span

    left_arrow_start_ratio = 1.0 - center_offset
    left_arrow_end_ratio = left_arrow_start_ratio - arrow_length
    right_arrow_start_ratio = 1.0 + center_offset
    right_arrow_end_ratio = right_arrow_start_ratio + arrow_length

    left_label_ratio = 0.5 * (left_arrow_start_ratio + left_arrow_end_ratio)
    right_label_ratio = 0.5 * (right_arrow_start_ratio + right_arrow_end_ratio)

    axis.annotate(
        "",
        xy=(left_arrow_end_ratio, annotation_y),
        xytext=(left_arrow_start_ratio, annotation_y),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.2,
            "connectionstyle": "arc3,rad=0.0",
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    axis.annotate(
        "",
        xy=(right_arrow_end_ratio, annotation_y),
        xytext=(right_arrow_start_ratio, annotation_y),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.2,
            "connectionstyle": "arc3,rad=0.0",
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )
    axis.text(
        left_label_ratio,
        label_y,
        "decreasing\nvariance",
        ha="center",
        va="top",
        fontsize=9,
    )
    axis.text(
        right_label_ratio,
        label_y,
        "increasing\nvariance",
        ha="center",
        va="top",
        fontsize=9,
    )

    axis.legend(loc="best")

    parsed_args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(parsed_args.output_path, dpi=660)
    print(f"Saved plot to: {parsed_args.output_path}")


if __name__ == "__main__":
    main()
