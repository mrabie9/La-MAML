"""Estimate eval-vs-train time impact from log files.

This script parses La-MAML experiment logs that contain per-epoch timing lines of
the form::

    Task 0 Epoch 1/5 | ... | Epoch Time 73.25s (Eval 4.77s, Train 68.48s)

It then:

  * Aggregates total train and eval time across all tasks/epochs.
  * Estimates how long training would take if evaluation were run after every
    epoch instead of only when triggered by ``val_rate``.

The estimation assumes that the cost of an evaluation pass is approximately
constant and equal to the mean of the observed per-epoch eval times.

Usage:
    python scripts/estimate_eval_impact.py \\
        --log-path logs/full_experiments/full_experiments_20260309_182510.log
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class EpochTiming:
    """Container for per-epoch timing information.

    Attributes:
        task_id: Integer task identifier.
        epoch_index: One-based epoch index within the task.
        num_epochs: Total number of epochs for the task.
        total_time_s: Total wall-clock time for the epoch in seconds.
        eval_time_s: Time spent in evaluation during the epoch in seconds.
        train_time_s: Time spent in training during the epoch in seconds.
    """

    task_id: int
    epoch_index: int
    num_epochs: int
    total_time_s: float
    eval_time_s: float
    train_time_s: float


_EPOCH_LINE_RE = re.compile(
    r"Task\s+(?P<task_id>\d+)\s+Epoch\s+"
    r"(?P<epoch_idx>\d+)/(?P<num_epochs>\d+)"
    r".*?Epoch Time\s+(?P<total>[0-9.]+)s\s*"
    r"\(Eval\s+(?P<eval>[0-9.]+)s,\s*Train\s+(?P<train>[0-9.]+)s\)"
)


def _parse_epoch_timings(lines: Iterable[str]) -> List[EpochTiming]:
    """Parse epoch timing lines from a log file.

    Args:
        lines: Iterable of log file lines.

    Returns:
        A list of :class:`EpochTiming` records, one per epoch line found.
    """
    timings: List[EpochTiming] = []
    for line in lines:
        match = _EPOCH_LINE_RE.search(line)
        if not match:
            continue
        task_id = int(match.group("task_id"))
        epoch_idx = int(match.group("epoch_idx"))
        num_epochs = int(match.group("num_epochs"))
        total_time = float(match.group("total"))
        eval_time = float(match.group("eval"))
        train_time = float(match.group("train"))
        timings.append(
            EpochTiming(
                task_id=task_id,
                epoch_index=epoch_idx,
                num_epochs=num_epochs,
                total_time_s=total_time,
                eval_time_s=eval_time,
                train_time_s=train_time,
            )
        )
    return timings


def _summarise_current_training(timings: List[EpochTiming]) -> Tuple[float, float, float]:
    """Summarise current total, eval, and train time.

    Args:
        timings: Parsed epoch timing records.

    Returns:
        Tuple of (total_time_s, total_eval_time_s, total_train_time_s).
    """
    total_time = sum(t.total_time_s for t in timings)
    total_eval = sum(t.eval_time_s for t in timings)
    total_train = sum(t.train_time_s for t in timings)
    return total_time, total_eval, total_train


def _estimate_eval_every_epoch(timings: List[EpochTiming]) -> float:
    """Estimate total time if eval ran every epoch.

    The estimate keeps the observed train time fixed and assumes that:

      * Each epoch would still take the same amount of train time.
      * The cost of an eval pass is approximately equal to the mean of the
        observed per-epoch eval times (where eval_time_s > 0).

    Args:
        timings: Parsed epoch timing records.

    Returns:
        Estimated total wall-clock time (seconds) under eval-per-epoch.
    """
    if not timings:
        return 0.0

    num_epochs = len(timings)
    total_train = sum(t.train_time_s for t in timings)
    eval_epochs = [t for t in timings if t.eval_time_s > 0.0]
    if not eval_epochs:
        # No evals recorded; assume current total time is already train-only.
        return total_train

    mean_eval_time = sum(t.eval_time_s for t in eval_epochs) / float(len(eval_epochs))
    return total_train + mean_eval_time * float(num_epochs)


def _format_seconds(seconds: float) -> str:
    """Format seconds as a human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60.0
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the impact on total training time if evaluation is run "
            "after every epoch, based on an existing La-MAML log file."
        )
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to a log file, e.g. logs/full_experiments/full_experiments_20260309_182510.log.",
    )
    return parser


def main() -> None:
    """Entry point for the eval impact estimator CLI."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {args.log_path}")

    with args.log_path.open("r", encoding="utf-8") as handle:
        lines = list(handle)

    timings = _parse_epoch_timings(lines)
    if not timings:
        print(f"No epoch timing lines found in {args.log_path}")
        return

    total_time, total_eval, total_train = _summarise_current_training(timings)
    est_total_eval_every_epoch = _estimate_eval_every_epoch(timings)

    extra_time = est_total_eval_every_epoch - total_time
    factor = est_total_eval_every_epoch / total_time if total_time > 0.0 else 1.0
    percent_increase = (factor - 1.0) * 100.0

    num_epochs = len(timings)
    num_eval_epochs = sum(1 for t in timings if t.eval_time_s > 0.0)

    print(f"Log file: {args.log_path}")
    print(f"Epochs with timing info: {num_epochs}")
    print(f"Epochs with eval:        {num_eval_epochs}")
    print()
    print("Current run:")
    print(f"  Total time: {total_time:.1f}s ({_format_seconds(total_time)})")
    print(f"    Train:    {total_train:.1f}s ({_format_seconds(total_train)})")
    print(f"    Eval:     {total_eval:.1f}s ({_format_seconds(total_eval)})")
    print()
    print("If eval ran every epoch:")
    print(
        f"  Estimated total: {est_total_eval_every_epoch:.1f}s "
        f"({_format_seconds(est_total_eval_every_epoch)})"
    )
    print(
        f"  Extra time vs current: {extra_time:.1f}s "
        f"({_format_seconds(extra_time)})  "
        f"(+{percent_increase:.1f}% , x{factor:.3f})"
    )


if __name__ == "__main__":
    main()

