"""Shared style override helpers for plotting scripts.

This module centralizes legend-kwargs overrides so multiple scripts can apply
the same per-panel, per-run-count behavior.
"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_STYLE_FULL_LEGEND_KWARGS: Dict[str, Any] = {
    "loc": "best",
    "ncol": 3,
    "fontsize": 10,
    "columnspacing": 0.9,
    "labelspacing": 0.3,
    "framealpha": 0.5,
    "borderaxespad": 0.1,
    "borderpad": 0.2,
}

TIL_STYLE_FULL_LEGEND_KWARGS: Dict[str, Any] = {
    "loc": "best",
    "ncol": 6,
    "fontsize": 10,
    "columnspacing": 0.5,
    "labelspacing": 0.2,
    "framealpha": 0.5,
    "borderaxespad": 0.1,
    "borderpad": 0.2,
}

# Explicit per-plot legend kwargs by style and run-count thresholds.
# Keys in ``run_count_overrides`` are interpreted as minimum run counts.
LEGEND_KWARGS_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "default": {
        "train": {
            "base": dict(DEFAULT_STYLE_FULL_LEGEND_KWARGS),
            "run_count_overrides": {
                1: {"ncol": 2},
                6: {"ncol": 3},
                10: {"ncol": 4},
                14: {"ncol": 5},
            },
        },
        "mean_val": {
            "base": dict(DEFAULT_STYLE_FULL_LEGEND_KWARGS),
            "run_count_overrides": {
                1: {"ncol": 2},
                6: {"ncol": 3},
                10: {"ncol": 4},
                14: {"ncol": 5},
            },
        },
        "average_forgetting": {
            "base": dict(DEFAULT_STYLE_FULL_LEGEND_KWARGS),
            "run_count_overrides": {
                1: {"ncol": 2},
                6: {"ncol": 3},
                10: {"ncol": 4},
                14: {"ncol": 5},
            },
        },
        "final_validation": {
            "base": dict(DEFAULT_STYLE_FULL_LEGEND_KWARGS),
            "run_count_overrides": {
                1: {"ncol": 3},
                10: {"ncol": 4},
            },
        },
    },
    "til": {
        "train": {
            "base": dict(TIL_STYLE_FULL_LEGEND_KWARGS),
            "run_count_overrides": {
                1: {"ncol": 3},
                6: {"ncol": 4},
                10: {"ncol": 5},
                14: {"ncol": 6},
            },
        },
        "mean_val": {
            "base": dict(TIL_STYLE_FULL_LEGEND_KWARGS),
            "run_count_overrides": {
                1: {"ncol": 3},
                6: {"ncol": 4},
                10: {"ncol": 5},
                14: {"ncol": 6},
            },
        },
        "average_forgetting": {
            "base": dict(TIL_STYLE_FULL_LEGEND_KWARGS),
            "run_count_overrides": {
                1: {"ncol": 3},
                6: {"ncol": 4},
                10: {"ncol": 5},
                14: {"ncol": 6},
            },
        },
        "final_validation": {
            "base": dict(TIL_STYLE_FULL_LEGEND_KWARGS),
            "run_count_overrides": {
                1: {"ncol": 3},
                10: {"ncol": 4},
            },
        },
    },
}


def resolve_legend_kwargs(
    *,
    style_key: str,
    panel_key: str,
    base_legend_kwargs: Dict[str, Any] | None,
    run_count: int,
) -> Dict[str, Any]:
    """Resolve legend kwargs using shared per-panel/per-run overrides.

    Args:
        style_key: Style family key, e.g. ``default`` or ``til``.
        panel_key: Plot key (``train``, ``mean_val``, ``average_forgetting``, ``final_validation``).
        base_legend_kwargs: Baseline kwargs from the style source.
        run_count: Number of plotted runs/algorithms.

    Returns:
        Final kwargs dictionary to pass to ``Axes.legend``.
    """
    resolved: Dict[str, Any] = dict(base_legend_kwargs or {})
    style_overrides = LEGEND_KWARGS_OVERRIDES.get(
        style_key, LEGEND_KWARGS_OVERRIDES["default"]
    )
    panel_overrides = style_overrides.get(panel_key, {})
    resolved.update(panel_overrides.get("base", {}))

    run_overrides = panel_overrides.get("run_count_overrides", {})
    if run_overrides:
        selected_threshold = max(
            (threshold for threshold in run_overrides if run_count >= threshold),
            default=None,
        )
        if selected_threshold is not None:
            resolved.update(run_overrides[selected_threshold])

    return resolved
