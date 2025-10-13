"""Utility for inspecting La-MAML HAT tuning summary files.

This script reads the JSON output produced by ``tune_hat.py`` and provides
basic analytics together with an optional heatmap visualisation of the
validation scores across the explored hyper-parameter grid.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import matplotlib
    matplotlib.use("Agg")  # Allows running in headless environments.
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - matplotlib not always installed.
    plt = None
    _HAS_MPL = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse and visualise HAT tuning summary files produced by tune_hat.py",
    )
    parser.add_argument(
        "summary",
        type=Path,
        help="Path to a tuning summary JSON file (e.g. logs/tuning/hat/.../summary.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top trials to display in the textual report (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the generated visualisation (PNG). Defaults to <summary_dir>/hat_tuning_heatmap.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots with matplotlib after saving (has no effect if matplotlib is unavailable)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation and only print the textual analytics",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        params = item.get("params", {})
        raw_scores = item.get("val_per_task") or []
        val_scores = [float(score) for score in raw_scores]
        val_min = min(val_scores) if val_scores else float("nan")
        val_max = max(val_scores) if val_scores else float("nan")
        if len(val_scores) > 1:
            val_std = statistics.pstdev(val_scores)
        elif val_scores:
            val_std = 0.0
        else:
            val_std = float("nan")
        rows.append(
            {
                "trial": item.get("trial"),
                "status": item.get("status"),
                "gamma": params.get("gamma"),
                "lr": params.get("lr"),
                "smax": params.get("smax"),
                "val_mean": item.get("val_mean", float("nan")),
                "val_min": val_min,
                "val_max": val_max,
                "val_std": val_std,
                "duration_sec": item.get("duration_sec"),
                "log_dir": item.get("log_dir"),
            }
        )
    return rows


def fmt_float(value: Any, precision: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if isinstance(value, (int, float)):
        magnitude = abs(value)
        if magnitude and (magnitude < 10 ** -precision or magnitude >= 10 ** (precision + 1)):
            return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def print_header(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    print("\n=== HAT tuning summary ===")
    print(f"Config file      : {summary.get('config')}")
    print(f"Experiment name  : {summary.get('base_expt_name')}")
    print(f"Session directory: {summary.get('session_dir')}")
    print(f"Trials completed : {len(rows)} / {summary.get('num_trials')}\n")

    best = summary.get("best") or {}
    best_params = best.get("params") or {}
    print("Best trial (from summary)")
    print(f"  trial #{best.get('trial')} | val_mean={fmt_float(best.get('val_mean'))}")
    print(
        "  params: gamma={gamma}, lr={lr}, smax={smax}".format(
            gamma=best_params.get("gamma"),
            lr=best_params.get("lr"),
            smax=best_params.get("smax"),
        )
    )
    print(f"  duration: {fmt_float(best.get('duration_sec'), 2)} sec")


def print_top_trials(rows: Sequence[Dict[str, Any]], k: int) -> None:
    def sort_key(item: Dict[str, Any]) -> float:
        value = item.get("val_mean")
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
        return float("-inf")

    valid_rows = sorted(rows, key=sort_key, reverse=True)
    print("\nTop trials by validation mean")
    header = f"{'rank':>4} {'trial':>5} {'val_mean':>10} {'gamma':>8} {'lr':>10} {'smax':>8} {'val_std':>10} {'duration_s':>11}"
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(valid_rows[:k], start=1):
        print(
            f"{idx:>4} {row.get('trial', ''):>5} {fmt_float(row.get('val_mean')):>10} "
            f"{fmt_float(row.get('gamma'), 3):>8} {fmt_float(row.get('lr'), 5):>10} "
            f"{fmt_float(row.get('smax'), 3):>8} {fmt_float(row.get('val_std')):>10} "
            f"{fmt_float(row.get('duration_sec'), 2):>11}"
        )


def summarise_by_param(rows: Iterable[Dict[str, Any]], param: str) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(param)
        val_mean = row.get("val_mean")
        if value is None or not isinstance(val_mean, (int, float)) or math.isnan(val_mean):
            continue
        grouped[value].append(float(val_mean))

    summary_rows: List[Dict[str, Any]] = []
    for value, scores in grouped.items():
        arr = [float(score) for score in scores]
        summary_rows.append(
            {
                param: value,
                "count": len(arr),
                "mean": statistics.mean(arr),
                "std": statistics.pstdev(arr) if len(arr) > 1 else float("nan"),
                "best": max(arr),
            }
        )
    summary_rows.sort(key=lambda item: item["mean"], reverse=True)
    return summary_rows


def print_param_summaries(rows: Sequence[Dict[str, Any]]) -> None:
    for param in ("lr", "gamma", "smax"):
        summary_rows = summarise_by_param(rows, param)
        if not summary_rows:
            continue
        print(f"\nAverages grouped by {param}")
        header = f"{param:>12} {'count':>7} {'mean':>10} {'std':>10} {'best':>10}"
        print(header)
        print("-" * len(header))
        for item in summary_rows:
            print(
                f"{fmt_float(item[param], 5):>12} {item['count']:>7} {fmt_float(item['mean']):>10} "
                f"{fmt_float(item['std']):>10} {fmt_float(item['best']):>10}"
            )


def build_score_tensor(rows: Sequence[Dict[str, Any]]):
    gammas = sorted({row.get("gamma") for row in rows if row.get("gamma") is not None})
    lrs = sorted({row.get("lr") for row in rows if row.get("lr") is not None})
    smax_values = sorted({row.get("smax") for row in rows if row.get("smax") is not None})
    if not gammas or not lrs or not smax_values:
        return None

    score_tensor: List[List[List[float]]] = []
    for _ in smax_values:
        score_tensor.append([[float("nan") for _ in lrs] for _ in gammas])

    gamma_index = {value: idx for idx, value in enumerate(gammas)}
    lr_index = {value: idx for idx, value in enumerate(lrs)}
    smax_index = {value: idx for idx, value in enumerate(smax_values)}

    for row in rows:
        gamma = row.get("gamma")
        lr = row.get("lr")
        smax = row.get("smax")
        val_mean = row.get("val_mean")
        if None in (gamma, lr, smax) or not isinstance(val_mean, (int, float)) or math.isnan(val_mean):
            continue
        score_tensor[smax_index[smax]][gamma_index[gamma]][lr_index[lr]] = float(val_mean)

    return score_tensor, gammas, lrs, smax_values


def plot_heatmaps(score_tensor, gammas: Sequence[float], lrs: Sequence[float], smax_values: Sequence[float], output_path: Path, show: bool) -> None:
    if not _HAS_MPL:
        print("Matplotlib is not available; skipping plot generation.")
        return

    finite_scores = [
        value
        for matrix in score_tensor
        for row in matrix
        for value in row
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not finite_scores:
        print("No finite validation scores available for plotting.")
        return

    vmin, vmax = min(finite_scores), max(finite_scores)

    cols = min(len(smax_values), 3)
    rows = int(math.ceil(len(smax_values) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), squeeze=False)
    fig.suptitle("Validation mean heatmaps by smax")

    for idx, smax in enumerate(smax_values):
        ax = axes[idx // cols][idx % cols]
        matrix = score_tensor[idx]
        im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{lr:g}" for lr in lrs], rotation=45, ha="right")
        ax.set_yticks(range(len(gammas)))
        ax.set_yticklabels([f"{gamma:g}" for gamma in gammas])
        ax.set_xlabel("lr")
        ax.set_ylabel("gamma")
        ax.set_title(f"smax = {smax:g}")

        for gamma_idx, _ in enumerate(gammas):
            for lr_idx, _ in enumerate(lrs):
                value = matrix[gamma_idx][lr_idx]
                if isinstance(value, float) and math.isfinite(value):
                    ax.text(lr_idx, gamma_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="val_mean")

    for idx in range(len(smax_values), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    print(f"Saved heatmap visualisation to {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    rows = flatten_results(summary.get("results", []))

    print_header(summary, rows)
    print_top_trials(rows, args.top_k)
    print_param_summaries(rows)

    if args.no_plot:
        return

    tensor_info = build_score_tensor(rows)
    if tensor_info is None:
        print("Insufficient hyper-parameter coverage to build a heatmap.")
        return

    score_tensor, gammas, lrs, smax_values = tensor_info
    output_path = args.output
    if output_path is None:
        output_path = args.summary.parent / "hat_tuning_heatmap.png"

    plot_heatmaps(score_tensor, gammas, lrs, smax_values, output_path, args.show)


if __name__ == "__main__":
    main()
