#!/usr/bin/env python
"""Summarise full_experiments log files by algorithm performance.

Parses a `full_experiments_*.log` file and prints, for each algorithm:
- status (completed or failed)
- total classification accuracy (validation)
- total detection recall and false-alarm rate
- training precision and F1 (from last epoch line when present; left blank if
  the log only has "Train Acc | Val Acc" to avoid misleading values)
- time taken (from results line when completed, else sum of Epoch Time)

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
    status: str = "unknown"
    exit_code: Optional[int] = None
    cls_rec_tr: Optional[float] = None
    cls_prec_tr: Optional[float] = None
    cls_f1_tr: Optional[float] = None
    det_tr: Optional[float] = None
    fa_tr: Optional[float] = None
    cls_rec_te: Optional[float] = None
    cls_prec_te: Optional[float] = None
    cls_f1_te: Optional[float] = None
    det_te: Optional[float] = None
    fa_te: Optional[float] = None
    size_gb: Optional[float] = None
    time_sec: Optional[float] = None


RUN_RE = re.compile(r"--- Running: base \+ (?P<algo>[\w\-]+) ---")
COMPLETED_RE = re.compile(r"Completed:\s+(?P<algo>[\w\-]+)\s+\(exit\s+(?P<code>\d+)\)")
ERROR_RE = re.compile(r"ERROR:\s+(?P<algo>[\w\-]+)\s+failed with exit code\s+(?P<code>\d+)")
TOTAL_ACC_RE = re.compile(r"Total Accuracy:\s+(?P<val>[0-9.]+)")
TOTAL_DET_RE = re.compile(r"Total Detection:\s+(?P<val>[0-9.]+)")
TOTAL_FA_RE = re.compile(r"Total False Alarm:\s+(?P<val>[0-9.]+)")
# Results line: ... # val: ... # 2276.83
RESULTS_TIME_RE = re.compile(r"#\s+val:\s+[^#]+#\s+(?P<sec>[0-9.]+)\s*$")
# Epoch line with Prec and F1: Task 0 Epoch 1/1 | Loss 1.38 | Train Acc 0.47 | Prec 0.50 | F1 0.45 | Epoch Time ...
EPOCH_PREC_F1_RE = re.compile(
    r"Task\s+\d+\s+Epoch\s+\d+/\d+\s+\|\s+Loss\s+[0-9.]+\s+\|\s+Train Acc\s+[0-9.]+\s+\|\s+Prec\s+(?P<prec>[0-9.]+)\s+\|\s+F1\s+(?P<f1>[0-9.]+)\s+\|"
)
# Epoch Time for fallback when results line has no time (e.g. incomplete run)
EPOCH_TIME_RE = re.compile(r"Epoch Time\s+(?P<sec>[0-9.]+)s")
SUMMARY_TR_RE = re.compile(
    r"SUMMARY_TR\s+"
    r"(?:cls_rec=(?P<cls_rec>[0-9.]+)\s+)?"
    r"(?:cls_prec=(?P<cls_prec>[0-9.]+)\s+)?"
    r"(?:cls_f1=(?P<cls_f1>[0-9.]+)\s+)?"
    r"(?:det=(?P<det>[0-9.]+)\s+)?"
    r"(?:fa=(?P<fa>[0-9.]+))?"
)
SUMMARY_TE_RE = re.compile(
    r"SUMMARY_TE\s+"
    r"(?:cls_rec=(?P<cls_rec>[0-9.]+)\s+)?"
    r"(?:cls_prec=(?P<cls_prec>[0-9.]+)\s+)?"
    r"(?:cls_f1=(?P<cls_f1>[0-9.]+)\s+)?"
    r"(?:det=(?P<det>[0-9.]+)\s+)?"
    r"(?:fa=(?P<fa>[0-9.]+))?"
)
MODEL_SIZE_RE = re.compile(r"Model size:\s+(?P<gb>[0-9.]+)\s+GB")


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

            m = SUMMARY_TR_RE.search(line)
            if m:
                if m.group("cls_rec"):
                    s.cls_rec_tr = float(m.group("cls_rec"))
                if m.group("cls_prec"):
                    s.cls_prec_tr = float(m.group("cls_prec"))
                if m.group("cls_f1"):
                    s.cls_f1_tr = float(m.group("cls_f1"))
                if m.group("det"):
                    s.det_tr = float(m.group("det"))
                if m.group("fa"):
                    s.fa_tr = float(m.group("fa"))
                continue

            m = SUMMARY_TE_RE.search(line)
            if m:
                if m.group("cls_rec"):
                    s.cls_rec_te = float(m.group("cls_rec"))
                if m.group("cls_prec"):
                    s.cls_prec_te = float(m.group("cls_prec"))
                if m.group("cls_f1"):
                    s.cls_f1_te = float(m.group("cls_f1"))
                if m.group("det"):
                    s.det_te = float(m.group("det"))
                if m.group("fa"):
                    s.fa_te = float(m.group("fa"))
                continue

            m = MODEL_SIZE_RE.search(line)
            if m:
                s.size_gb = float(m.group("gb"))
                continue

            m = TOTAL_ACC_RE.search(line)
            if m:
                if s.cls_rec_te is None:
                    s.cls_rec_te = float(m.group("val"))
                continue

            m = TOTAL_DET_RE.search(line)
            if m:
                if s.det_te is None:
                    s.det_te = float(m.group("val"))
                continue

            m = TOTAL_FA_RE.search(line)
            if m:
                if s.fa_te is None:
                    s.fa_te = float(m.group("val"))
                continue

            m = EPOCH_PREC_F1_RE.search(line)
            if m:
                if s.cls_prec_tr is None:
                    s.cls_prec_tr = float(m.group("prec"))
                if s.cls_f1_tr is None:
                    s.cls_f1_tr = float(m.group("f1"))
                continue

            m = EPOCH_TIME_RE.search(line)
            if m:
                epoch_sec = float(m.group("sec"))
                s.time_sec = (s.time_sec or 0.0) + epoch_sec
                continue

            m = RESULTS_TIME_RE.search(line)
            if m:
                s.time_sec = float(m.group("sec"))
                continue

    return summaries


def _format_time(sec: Optional[float]) -> str:
    if sec is None:
        return "    -"
    if sec >= 3600:
        return f"{sec / 3600:.1f}h"
    if sec >= 60:
        return f"{sec / 60:.1f}m"
    return f"{sec:.0f}s"


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "  -"


def print_summary(summaries: Dict[str, AlgoSummary]) -> None:
    if not summaries:
        print("No algorithm runs found in log.")
        return

    w_algo, w_status, w_exit = 8, 10, 5
    w_num = 7
    w_time = 8
    # tr: rec prec f1 det fa | te: rec prec f1 det fa | Size_GB Time
    header = (
        f"{'Algo':<{w_algo}} {'Status':<{w_status}} {'Exit':<{w_exit}} "
        f"{'rec':>{w_num}} {'prec':>{w_num}} {'f1':>{w_num}} {'det':>{w_num}} {'fa':>{w_num}} "
        f"| "
        f"{'rec':>{w_num}} {'prec':>{w_num}} {'f1':>{w_num}} {'det':>{w_num}} {'fa':>{w_num}} "
        f"{'Size_GB':>{w_num}} {'Time':>{w_time}}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(summaries.keys()):
        s = summaries[name]
        size_str = f"{s.size_gb:.3f}" if s.size_gb is not None else "  -"
        time_str = _format_time(s.time_sec)
        exit_str = str(s.exit_code) if s.exit_code is not None else ""
        print(
            f"{s.name:<{w_algo}} {s.status:<{w_status}} {exit_str:<{w_exit}} "
            f"{_fmt(s.cls_rec_tr):>{w_num}} {_fmt(s.cls_prec_tr):>{w_num}} {_fmt(s.cls_f1_tr):>{w_num}} "
            f"{_fmt(s.det_tr):>{w_num}} {_fmt(s.fa_tr):>{w_num}} "
            f"| "
            f"{_fmt(s.cls_rec_te):>{w_num}} {_fmt(s.cls_prec_te):>{w_num}} {_fmt(s.cls_f1_te):>{w_num}} "
            f"{_fmt(s.det_te):>{w_num}} {_fmt(s.fa_te):>{w_num}} "
            f"{size_str:>{w_num}} {time_str:>{w_time}}"
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

