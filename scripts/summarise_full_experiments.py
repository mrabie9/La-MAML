#!/usr/bin/env python
"""Summarise full_experiments log files by algorithm performance.

Parses a `full_experiments_*.log` file and prints, for each algorithm:
- status (completed or failed)
- total classification accuracy
- total detection recall
- total detection false-alarm rate

Usage:
    python scripts/summarise_full_experiments.py \
        --log logs/full_experiments/full_experiments_20260305_175812.log

If --log is omitted, the most recent matching log in
`logs/full_experiments/` is used.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AlgoSummary:
    name: str
    status: str = "unknown"  # "completed", "failed", "skipped", "unknown"
    exit_code: Optional[int] = None
    total_acc: Optional[float] = None
    total_det: Optional[float] = None
    total_fa: Optional[float] = None


RUN_RE = re.compile(r"--- Running: base \+ (?P<algo>[\w\-]+) ---")
COMPLETED_RE = re.compile(r"Completed:\s+(?P<algo>[\w\-]+)\s+\(exit\s+(?P<code>\d+)\)")
ERROR_RE = re.compile(r"ERROR:\s+(?P<algo>[\w\-]+)\s+failed with exit code\s+(?P<code>\d+)")
TOTAL_ACC_RE = re.compile(r"Total Accuracy:\s+(?P<val>[0-9.]+)")
TOTAL_DET_RE = re.compile(r"Total Detection:\s+(?P<val>[0-9.]+)")
TOTAL_FA_RE = re.compile(r"Total False Alarm:\s+(?P<val>[0-9.]+)")


def _find_default_log() -> str:
    candidates = sorted(
        glob.glob(os.path.join("logs", "full_experiments", "full_experiments_*.log"))
    )
    if not candidates:
        raise SystemExit("No full_experiments_*.log files found.")
    return candidates[-1]


def parse_log(path: str) -> Dict[str, AlgoSummary]:
    summaries: Dict[str, AlgoSummary] = {}
    current_algo: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Running block
            m = RUN_RE.search(line)
            if m:
                current_algo = m.group("algo")
                summaries.setdefault(current_algo, AlgoSummary(name=current_algo))
                continue

            # Completion / error lines
            m = COMPLETED_RE.search(line)
            if m:
                algo = m.group("algo")
                code = int(m.group("code"))
                s = summaries.setdefault(algo, AlgoSummary(name=algo))
                s.status = "completed" if code == 0 else "failed"
                s.exit_code = code
                continue

            m = ERROR_RE.search(line)
            if m:
                algo = m.group("algo")
                code = int(m.group("code"))
                s = summaries.setdefault(algo, AlgoSummary(name=algo))
                s.status = "failed"
                s.exit_code = code
                continue

            # Metrics: only make sense in context of current_algo
            if current_algo is None:
                continue

            s = summaries.setdefault(current_algo, AlgoSummary(name=current_algo))

            m = TOTAL_ACC_RE.search(line)
            if m:
                s.total_acc = float(m.group("val"))
                continue

            m = TOTAL_DET_RE.search(line)
            if m:
                s.total_det = float(m.group("val"))
                continue

            m = TOTAL_FA_RE.search(line)
            if m:
                s.total_fa = float(m.group("val"))
                continue

    return summaries


def print_summary(summaries: Dict[str, AlgoSummary]) -> None:
    if not summaries:
        print("No algorithm runs found in log.")
        return

    # Sort algorithms alphabetically for stable output
    header = f"{'Algo':<10} {'Status':<10} {'Exit':<5} {'Recall':>8} {'Det':>8} {'FA':>8}"
    print(header)
    print("-" * len(header))
    for name in sorted(summaries.keys()):
        s = summaries[name]
        acc = f"{s.total_acc:.3f}" if s.total_acc is not None else "  -"
        det = f"{s.total_det:.3f}" if s.total_det is not None else "  -"
        fa = f"{s.total_fa:.3f}" if s.total_fa is not None else "  -"
        exit_str = "" if s.exit_code is None else str(s.exit_code)
        print(
            f"{s.name:<10} {s.status:<10} {exit_str:<5} {acc:>8} {det:>8} {fa:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise full_experiments log file by algorithm performance."
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to full_experiments_*.log (default: latest in logs/full_experiments/).",
    )
    args = parser.parse_args()

    log_path = args.log or _find_default_log()
    summaries = parse_log(log_path)
    print(f"Log: {log_path}")
    print_summary(summaries)


if __name__ == "__main__":
    main()

