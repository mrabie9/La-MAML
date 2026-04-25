"""Create a Euclidean vs Riemannian step-geometry illustration for RWalk.

Usage:
    source la-maml_env/bin/activate
    python scripts/plot_rwalk_metric_geometry.py
    python scripts/plot_rwalk_metric_geometry.py --output-path plots/rwalk_metric_geometry.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for figure generation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Usage:
        arguments = parse_arguments()
    """
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("plots/rwalk_euclidean_vs_riemannian_ieee.png"),
        help="Output image path for the generated figure.",
    )
    argument_parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output image resolution in dots per inch.",
    )
    return argument_parser.parse_args()


def loss_landscape(theta_1: np.ndarray, theta_2: np.ndarray) -> np.ndarray:
    """Compute a curved-valley surrogate loss landscape.

    Args:
        theta_1: Grid values for the first parameter axis.
        theta_2: Grid values for the second parameter axis.

    Returns:
        np.ndarray: Scalar loss values over the mesh.

    Usage:
        loss_values = loss_landscape(theta_1_grid, theta_2_grid)
    """
    valley_center = 0.35 * (theta_1**2)
    valley_penalty = 2.4 * ((theta_2 - valley_center) ** 2)
    longitudinal_penalty = 0.18 * (theta_1**2)
    return valley_penalty + longitudinal_penalty


def normalized(vector_values: np.ndarray) -> np.ndarray:
    """Return a normalized copy of a vector.

    Args:
        vector_values: Input vector of shape (2,).

    Returns:
        np.ndarray: Unit-norm direction vector.

    Usage:
        unit_vector = normalized(np.array([1.0, 2.0]))
    """
    return vector_values / np.linalg.norm(vector_values)


def valley_tangent_and_normal(point_theta_1: float) -> tuple[np.ndarray, np.ndarray]:
    """Construct tangent and normal directions at a valley point.

    Args:
        point_theta_1: Horizontal coordinate at which to evaluate the valley geometry.

    Returns:
        tuple[np.ndarray, np.ndarray]: Unit tangent and unit normal vectors.

    Usage:
        tangent_direction, normal_direction = valley_tangent_and_normal(-0.8)
    """
    slope_value = 0.7 * point_theta_1
    tangent_direction = normalized(np.array([1.0, slope_value], dtype=float))
    normal_direction = normalized(np.array([-slope_value, 1.0], dtype=float))
    return tangent_direction, normal_direction


def add_step_arrow(
    axis: plt.Axes,
    start_point: np.ndarray,
    direction_vector: np.ndarray,
    step_length: float,
    color_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a parameter-step arrow and return its start/end points.

    Args:
        axis: Matplotlib axis where the arrow will be rendered.
        start_point: Arrow tail coordinate in parameter space.
        direction_vector: Unit direction vector.
        step_length: Arrow length in displayed coordinates.
        color_name: Arrow color.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrow start and end coordinates.

    Usage:
        tail, tip = add_step_arrow(axis, start, direction, 0.4, "#d55e00")
    """
    end_point = start_point + step_length * direction_vector
    axis.annotate(
        "",
        xy=(end_point[0], end_point[1]),
        xytext=(start_point[0], start_point[1]),
        arrowprops=dict(arrowstyle="->", lw=2.0, color=color_name),
    )
    return start_point, end_point


def setup_panel(axis: plt.Axes, title_text: str) -> None:
    """Apply shared panel formatting.

    Args:
        axis: Target axis.
        title_text: Panel title.

    Usage:
        setup_panel(axis, "Euclidean Metric")
    """
    axis.set_title(title_text, fontsize=9.8, pad=4.0)
    axis.set_xlabel(r"$\theta_1$", fontsize=9)
    axis.set_ylabel(r"$\theta_2$", fontsize=9)
    axis.tick_params(labelsize=7.5)
    axis.set_xlim(-1.5, 1.5)
    axis.set_ylim(-0.4, 1.6)
    axis.set_facecolor("#f9f9f9")


def render_figure(output_path: Path, dpi_value: int) -> None:
    """Render and save the Euclidean vs Riemannian geometry figure.

    Args:
        output_path: Destination image path.
        dpi_value: Export resolution.

    Usage:
        render_figure(Path("plots/figure.png"), dpi_value=600)
    """
    theta_1_values = np.linspace(-1.5, 1.5, 500)
    theta_2_values = np.linspace(-0.4, 1.6, 500)
    theta_1_grid, theta_2_grid = np.meshgrid(theta_1_values, theta_2_values)
    loss_values = loss_landscape(theta_1_grid, theta_2_grid)
    contour_levels = [0.10, 0.30, 0.55]

    figure, axes = plt.subplots(1, 2, figsize=(3.5, 2.25), sharex=True, sharey=True)
    contour_color = "#6d8fb3"
    orange_color = "#e69f00"
    red_color = "#d55e00"

    for panel_axis, panel_title in zip(
        axes, ("Euclidean Metric", "Riemannian Metric (Fisher-weighted)")
    ):
        panel_axis.contour(
            theta_1_grid,
            theta_2_grid,
            loss_values,
            levels=contour_levels,
            colors=contour_color,
            linewidths=1.2,
        )
        setup_panel(panel_axis, panel_title)

    start_theta_1 = -0.85
    start_theta_2 = 0.35 * (start_theta_1**2)
    start_point = np.array([start_theta_1, start_theta_2], dtype=float)
    tangent_direction, normal_direction = valley_tangent_and_normal(start_theta_1)

    euclidean_length = 0.62
    _, tangent_end_euclidean = add_step_arrow(
        axes[0], start_point, tangent_direction, euclidean_length, orange_color
    )
    _, normal_end_euclidean = add_step_arrow(
        axes[0], start_point, normal_direction, euclidean_length, red_color
    )

    axes[0].text(
        tangent_end_euclidean[0] + 0.03,
        tangent_end_euclidean[1] - 0.02,
        r"$\|\Delta\theta\|_2 = c$",
        color=orange_color,
        fontsize=7.4,
    )
    axes[0].text(
        normal_end_euclidean[0] + 0.03,
        normal_end_euclidean[1] + 0.02,
        r"$\|\Delta\theta\|_2 = c$",
        color=red_color,
        fontsize=7.4,
    )
    axes[0].text(
        tangent_end_euclidean[0] - 0.17,
        tangent_end_euclidean[1] - 0.18,
        "Small loss change",
        color="#333333",
        fontsize=7.2,
    )
    axes[0].text(
        normal_end_euclidean[0] - 0.10,
        normal_end_euclidean[1] + 0.11,
        "Large loss change",
        color="#333333",
        fontsize=7.2,
    )

    tangent_riemannian_length = 0.35
    normal_riemannian_length = 1.00
    _, tangent_end_riemannian = add_step_arrow(
        axes[1], start_point, tangent_direction, tangent_riemannian_length, orange_color
    )
    _, normal_end_riemannian = add_step_arrow(
        axes[1], start_point, normal_direction, normal_riemannian_length, red_color
    )

    axes[1].text(
        normal_end_riemannian[0] - 0.35,
        normal_end_riemannian[1] + 0.08,
        "Fisher metric stretches\nhigh-curvature directions",
        color="#333333",
        fontsize=7.2,
    )
    axes[1].text(
        tangent_end_riemannian[0] - 0.35,
        tangent_end_riemannian[1] - 0.20,
        "Equal Riemannian steps\n$\\rightarrow$ comparable behavior change",
        color="#333333",
        fontsize=7.2,
    )

    caption_text = (
        "The Riemannian metric assigns greater magnitude to steps that substantially "
        "alter the predictive distribution, providing a principled measure of parameter "
        "importance for continual learning (RWalk)."
    )
    figure.text(0.5, 0.01, caption_text, ha="center", va="bottom", fontsize=6.8)
    figure.subplots_adjust(left=0.11, right=0.99, top=0.86, bottom=0.27, wspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi_value)
    plt.close(figure)


def main() -> None:
    """Entry point for command-line execution.

    Usage:
        main()
    """
    parsed_arguments = parse_arguments()
    render_figure(
        output_path=parsed_arguments.output_path, dpi_value=parsed_arguments.dpi
    )
    print(f"Saved figure to: {parsed_arguments.output_path}")


if __name__ == "__main__":
    main()
