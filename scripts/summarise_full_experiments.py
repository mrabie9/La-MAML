#!/usr/bin/env python
"""Summarise full_experiments logs by algorithm performance.

Parses a `full_experiments_*.log` file and prints, for each algorithm:
- status (completed or failed)
- total classification accuracy (validation)
- total detection recall and false-alarm rate
- training precision and F1 (from last epoch line when present; left blank if
  the log only has "Train Acc | Val Acc" to avoid misleading values)
- time taken (from results line when completed, else sum of Epoch Time)

Time is computed per algorithm as follows:
- If the run completed: the value is taken from the final results line
  (regex RESULTS_TIME_RE matches "# val: ... # <seconds>" at end of line).
- If the run did not finish or that line is missing: time is the sum of all
  "Epoch Time <seconds>s" lines (EPOCH_TIME_RE) seen for that algorithm.

Usage:
    python scripts/summarise_full_experiments.py --log \
        logs/full_experiments/full_experiments_20260305_175812.log
    python scripts/summarise_full_experiments.py --log \
        logs/full_experiments/run_20260325_154832_lnx-elkk-1

If --log is omitted, the newest `run_*` directory under
`logs/full_experiments/` is used when available; otherwise the newest
`full_experiments_*.log` file is used.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


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
DISPATCH_RE = re.compile(
    r"--- Dispatching:\s+base \+\s+(?P<algo>[\w\-]+)\s+\(job log:\s+(?P<path>[^)]+)\)\s+---"
)
COMPLETED_RE = re.compile(r"Completed:\s+(?P<algo>[\w\-]+)\s+\(exit\s+(?P<code>\d+)\)")
ERROR_RE = re.compile(
    r"ERROR:\s+(?P<algo>[\w\-]+)\s+failed with exit code\s+(?P<code>\d+)"
)
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
LOG_TIMESTAMP_RE = re.compile(r"^\[(?P<stamp>\d{4}-\d{2}-\d{2}T[^]]+)\]")
LOGGING_TO_RE = re.compile(r"Logging to\s+(?P<path>\S+)")
RESULTS_DICT_RE = re.compile(r"'log_dir':\s*'(?P<path>[^']+)'")


def _find_default_log() -> str:
    run_dirs = sorted(glob.glob(os.path.join("logs", "full_experiments", "run_*")))
    if run_dirs:
        return run_dirs[-1]

    candidates = sorted(
        glob.glob(os.path.join("logs", "full_experiments", "full_experiments_*.log"))
    )
    if candidates:
        return candidates[-1]
    raise SystemExit("No full_experiments logs found.")


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


def parse_job_log(path: str, algo_name: Optional[str] = None) -> Dict[str, AlgoSummary]:
    summaries: Dict[str, AlgoSummary] = {}
    current_algo = algo_name

    with open(path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if current_algo is None:
                match = re.search(r"Running model:\s+(?P<algo>[\w\-]+)", line)
                if match:
                    current_algo = match.group("algo")
                    summaries.setdefault(current_algo, AlgoSummary(name=current_algo))
                    continue
                continue

            summary = summaries.setdefault(current_algo, AlgoSummary(name=current_algo))

            match = SUMMARY_TR_RE.search(line)
            if match:
                if match.group("cls_rec"):
                    summary.cls_rec_tr = float(match.group("cls_rec"))
                if match.group("cls_prec"):
                    summary.cls_prec_tr = float(match.group("cls_prec"))
                if match.group("cls_f1"):
                    summary.cls_f1_tr = float(match.group("cls_f1"))
                if match.group("det"):
                    summary.det_tr = float(match.group("det"))
                if match.group("fa"):
                    summary.fa_tr = float(match.group("fa"))
                continue

            match = SUMMARY_TE_RE.search(line)
            if match:
                if match.group("cls_rec"):
                    summary.cls_rec_te = float(match.group("cls_rec"))
                if match.group("cls_prec"):
                    summary.cls_prec_te = float(match.group("cls_prec"))
                if match.group("cls_f1"):
                    summary.cls_f1_te = float(match.group("cls_f1"))
                if match.group("det"):
                    summary.det_te = float(match.group("det"))
                if match.group("fa"):
                    summary.fa_te = float(match.group("fa"))
                continue

            match = MODEL_SIZE_RE.search(line)
            if match:
                summary.size_gb = float(match.group("gb"))
                continue

            match = TOTAL_ACC_RE.search(line)
            if match and summary.cls_rec_te is None:
                summary.cls_rec_te = float(match.group("val"))
                continue

            match = TOTAL_DET_RE.search(line)
            if match and summary.det_te is None:
                summary.det_te = float(match.group("val"))
                continue

            match = TOTAL_FA_RE.search(line)
            if match and summary.fa_te is None:
                summary.fa_te = float(match.group("val"))
                continue

            match = EPOCH_PREC_F1_RE.search(line)
            if match:
                if summary.cls_prec_tr is None:
                    summary.cls_prec_tr = float(match.group("prec"))
                if summary.cls_f1_tr is None:
                    summary.cls_f1_tr = float(match.group("f1"))
                continue

            match = EPOCH_TIME_RE.search(line)
            if match:
                epoch_sec = float(match.group("sec"))
                summary.time_sec = (summary.time_sec or 0.0) + epoch_sec
                continue

            match = RESULTS_TIME_RE.search(line)
            if match:
                summary.time_sec = float(match.group("sec"))
                continue

            match = re.search(r"Total runtime:\s+(?P<hours>[0-9.]+)\s+hours", line)
            if match:
                summary.time_sec = float(match.group("hours")) * 3600

    return summaries


def _to_float_or_none(raw_value: object) -> Optional[float]:
    """Convert an NP value/array to the latest finite float.

    Args:
        raw_value: Raw object loaded from an NPZ entry.

    Returns:
        Last finite numeric value if available, otherwise None.
    """
    array = np.asarray(raw_value)
    if array.size == 0:
        return None
    flattened_values = array.astype(float).reshape(-1)
    finite_values = flattened_values[np.isfinite(flattened_values)]
    if finite_values.size == 0:
        return None
    return float(finite_values[-1])


def _extract_latest_metric(
    metrics_data: np.lib.npyio.NpzFile, candidate_keys: List[str]
) -> Optional[float]:
    """Return the latest finite value from the first present metric key.

    Args:
        metrics_data: Opened task NPZ payload.
        candidate_keys: Ordered keys to try.

    Returns:
        Latest finite float from the first matching key, otherwise None.
    """
    for metric_key in candidate_keys:
        if metric_key in metrics_data:
            metric_value = _to_float_or_none(metrics_data[metric_key])
            if metric_value is not None:
                return metric_value
    return None


def _extract_log_dir_from_job_log(job_log_path: str) -> Optional[str]:
    """Extract the per-run logging directory from a job log.

    Args:
        job_log_path: Path to a single algorithm job log.

    Returns:
        Reported run directory path if found, else None.
    """
    extracted_log_dir: Optional[str] = None
    with open(job_log_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            logging_match = LOGGING_TO_RE.search(line)
            if logging_match:
                extracted_log_dir = logging_match.group("path").strip()
            results_match = RESULTS_DICT_RE.search(line)
            if results_match:
                extracted_log_dir = results_match.group("path").strip()
    return extracted_log_dir


def _resolve_existing_log_dir(job_log_path: str, raw_log_dir: str) -> Optional[Path]:
    """Resolve a raw run directory string to an existing path.

    Args:
        job_log_path: Path to the job log containing the directory hint.
        raw_log_dir: Directory string parsed from that job log.

    Returns:
        Existing directory path if found, otherwise None.
    """
    cleaned_relative_path = raw_log_dir.strip().replace("//", "/").lstrip("./")
    candidate_paths: List[Path] = []
    parsed_path = Path(cleaned_relative_path)
    if parsed_path.is_absolute():
        candidate_paths.append(parsed_path)
    else:
        candidate_paths.append(Path.cwd() / parsed_path)
        candidate_paths.append(Path(job_log_path).resolve().parent / parsed_path)
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return None


def _fill_missing_metrics_from_npz(
    summary: AlgoSummary, metrics_file_path: Path
) -> None:
    """Populate missing summary metrics from a task metrics NPZ.

    Args:
        summary: Algorithm summary object to augment in-place.
        metrics_file_path: Path to the selected task metrics file.

    Returns:
        None.
    """
    with np.load(metrics_file_path, allow_pickle=False) as metrics_data:
        train_recall = _extract_latest_metric(
            metrics_data, ["train_cls_rec", "train_rec", "cls_tr_rec", "rec"]
        )
        train_precision = _extract_latest_metric(
            metrics_data, ["train_cls_prec", "train_prec", "prec"]
        )
        train_f1 = _extract_latest_metric(
            metrics_data, ["train_f1", "train_f1_c", "f1_c", "f1_cls"]
        )
        train_detection = _extract_latest_metric(
            metrics_data, ["train_det_rec", "train_det", "det"]
        )
        train_false_alarm = _extract_latest_metric(
            metrics_data, ["train_det_pfa", "train_det_fa", "train_fa", "fa"]
        )

        validation_recall = _extract_latest_metric(
            metrics_data, ["val_cls_rec", "val_rec", "val_acc", "rec"]
        )
        validation_precision = _extract_latest_metric(
            metrics_data, ["val_cls_prec", "val_prec", "prec"]
        )
        validation_f1 = _extract_latest_metric(
            metrics_data, ["val_f1", "val_f1_c", "f1_c", "f1_cls"]
        )
        validation_detection = _extract_latest_metric(
            metrics_data, ["val_det_rec", "val_det_acc", "val_det", "det"]
        )
        validation_false_alarm = _extract_latest_metric(
            metrics_data, ["val_det_pfa", "val_det_fa", "val_fa", "fa"]
        )

    if summary.cls_rec_tr is None:
        summary.cls_rec_tr = train_recall
    if summary.cls_prec_tr is None:
        summary.cls_prec_tr = train_precision
    if summary.cls_f1_tr is None:
        summary.cls_f1_tr = train_f1
    if summary.det_tr is None:
        summary.det_tr = train_detection
    if summary.fa_tr is None:
        summary.fa_tr = train_false_alarm

    if summary.cls_rec_te is None:
        summary.cls_rec_te = validation_recall
    if summary.cls_prec_te is None:
        summary.cls_prec_te = validation_precision
    if summary.cls_f1_te is None:
        summary.cls_f1_te = validation_f1
    if summary.det_te is None:
        summary.det_te = validation_detection
    if summary.fa_te is None:
        summary.fa_te = validation_false_alarm


def _apply_metrics_fallback_from_job_log(
    summary: AlgoSummary, job_log_path: str
) -> None:
    """Use task metrics NPZ files when terminal log metrics are missing.

    Args:
        summary: Algorithm summary object to augment in-place.
        job_log_path: Path to the algorithm job log.

    Returns:
        None.
    """
    raw_log_directory = _extract_log_dir_from_job_log(job_log_path)
    if not raw_log_directory:
        return
    resolved_log_directory = _resolve_existing_log_dir(job_log_path, raw_log_directory)
    if resolved_log_directory is None:
        return
    task_metric_paths = sorted(
        resolved_log_directory.glob("metrics/task*.npz"),
        key=lambda path: int(path.stem.replace("task", "")),
    )
    if not task_metric_paths:
        return
    _fill_missing_metrics_from_npz(summary, task_metric_paths[-1])


def _merge_summary(into: AlgoSummary, source: AlgoSummary) -> None:
    for field_name in (
        "cls_rec_tr",
        "cls_prec_tr",
        "cls_f1_tr",
        "det_tr",
        "fa_tr",
        "cls_rec_te",
        "cls_prec_te",
        "cls_f1_te",
        "det_te",
        "fa_te",
        "size_gb",
        "time_sec",
    ):
        value = getattr(source, field_name)
        if value is not None:
            setattr(into, field_name, value)
    if source.exit_code is not None:
        into.exit_code = source.exit_code
        into.status = source.status


def parse_run_directory(path: str) -> Dict[str, AlgoSummary]:
    main_log_candidates = sorted(
        glob.glob(os.path.join(path, "full_experiments_*.log"))
    )
    if not main_log_candidates:
        raise SystemExit(f"No full_experiments_*.log found under run directory: {path}")
    main_log_path = main_log_candidates[-1]

    summaries = parse_log(main_log_path)
    job_logs_by_algo: Dict[str, str] = {}

    with open(main_log_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            dispatch_match = DISPATCH_RE.search(line)
            if dispatch_match:
                algo_name = dispatch_match.group("algo")
                job_logs_by_algo[algo_name] = dispatch_match.group("path").strip()

    if not job_logs_by_algo:
        for job_log_path in sorted(
            glob.glob(os.path.join(path, "job_logs", "job_*.log"))
        ):
            file_name = os.path.basename(job_log_path)
            job_match = re.match(
                r"job_(?P<algo>[\w\-]+)_\d{8}_\d{6}_\d+\.log", file_name
            )
            if not job_match:
                continue
            job_logs_by_algo[job_match.group("algo")] = job_log_path

    for algo_name, job_log_path in job_logs_by_algo.items():
        job_summaries = parse_job_log(job_log_path, algo_name=algo_name)
        if algo_name in job_summaries:
            summary = summaries.setdefault(algo_name, AlgoSummary(name=algo_name))
            _merge_summary(summary, job_summaries[algo_name])
            _apply_metrics_fallback_from_job_log(summary, job_log_path)

    return summaries


def parse_path(path: str) -> Dict[str, AlgoSummary]:
    if os.path.isdir(path):
        return parse_run_directory(path)
    return parse_log(path)


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


def _classification_f1_from_recall_precision(
    recall_value: Optional[float], precision_value: Optional[float]
) -> Optional[float]:
    if recall_value is None or precision_value is None:
        return None
    denominator = recall_value + precision_value
    if denominator == 0:
        return None
    return 2 * (recall_value * precision_value) / denominator


def _compute_concurrent_runtime_seconds(log_path: str) -> Optional[float]:
    """Estimate wall-clock concurrent runtime from file timestamps."""
    if os.path.isdir(log_path):
        candidates = sorted(glob.glob(os.path.join(log_path, "full_experiments_*.log")))
        if not candidates:
            return None
        coordinator_log_path = Path(candidates[-1])
    else:
        coordinator_log_path = Path(log_path)

    file_stat = coordinator_log_path.stat()
    creation_timestamp = getattr(file_stat, "st_birthtime", None)
    if creation_timestamp is None:
        with open(coordinator_log_path, "r", encoding="utf-8") as file_handle:
            first_line = file_handle.readline()
        timestamp_match = LOG_TIMESTAMP_RE.search(first_line)
        if timestamp_match:
            creation_timestamp = datetime.fromisoformat(
                timestamp_match.group("stamp")
            ).timestamp()
        else:
            creation_timestamp = file_stat.st_ctime
    modification_timestamp = file_stat.st_mtime
    concurrent_runtime_seconds = modification_timestamp - creation_timestamp
    return max(0.0, concurrent_runtime_seconds)


def print_summary(
    summaries: Dict[str, AlgoSummary],
    concurrent_runtime_seconds: Optional[float] = None,
) -> None:
    if not summaries:
        print("No algorithm runs found in log.")
        return

    w_algo, w_exit = 8, 5
    w_num = 7
    w_time = 8
    # tr: rec prec f1_c det fa f1 | te: rec prec f1_c det fa f1 | Size_GB Time
    header = (
        f"{'Algo':<{w_algo}} {'Exit':<{w_exit}} "
        f"{'rec':>{w_num}} {'prec':>{w_num}} {'F1_c':>{w_num}} "
        f"{'det':>{w_num}} {'fa':>{w_num}} {'f1':>{w_num}} "
        f"| "
        f"{'rec':>{w_num}} {'prec':>{w_num}} {'F1_c':>{w_num}} "
        f"{'det':>{w_num}} {'fa':>{w_num}} {'f1':>{w_num}} "
        f"{'Size_GB':>{w_num}} {'Time':>{w_time}}"
    )
    print(header)
    print("-" * len(header))
    sorted_summaries = sorted(
        summaries.values(),
        key=lambda summary: (
            summary.cls_f1_te is None,
            summary.cls_f1_te if summary.cls_f1_te is not None else float("inf"),
            summary.name,
        ),
    )
    for s in sorted_summaries:
        train_classification_f1 = _classification_f1_from_recall_precision(
            s.cls_rec_tr, s.cls_prec_tr
        )
        test_classification_f1 = _classification_f1_from_recall_precision(
            s.cls_rec_te, s.cls_prec_te
        )
        size_str = f"{s.size_gb:.3f}" if s.size_gb is not None else "  -"
        time_str = _format_time(s.time_sec)
        exit_str = str(s.exit_code) if s.exit_code is not None else ""
        print(
            f"{s.name:<{w_algo}} {exit_str:<{w_exit}} "
            f"{_fmt(s.cls_rec_tr):>{w_num}} {_fmt(s.cls_prec_tr):>{w_num}} "
            f"{_fmt(train_classification_f1):>{w_num}} "
            f"{_fmt(s.det_tr):>{w_num}} {_fmt(s.fa_tr):>{w_num}} {_fmt(s.cls_f1_tr):>{w_num}} "
            f"| "
            f"{_fmt(s.cls_rec_te):>{w_num}} {_fmt(s.cls_prec_te):>{w_num}} "
            f"{_fmt(test_classification_f1):>{w_num}} "
            f"{_fmt(s.det_te):>{w_num}} {_fmt(s.fa_te):>{w_num}} {_fmt(s.cls_f1_te):>{w_num}} "
            f"{size_str:>{w_num}} {time_str:>{w_time}}"
        )

    total_serial_seconds = sum(
        summary.time_sec
        for summary in summaries.values()
        if summary.time_sec is not None
    )
    if total_serial_seconds > 0:
        print("-" * len(header))
        print(f"Total serial time (all models): {_format_time(total_serial_seconds)}")
        if concurrent_runtime_seconds is not None:
            print(
                "Total concurrent time (coordinator log wall-clock): "
                f"{_format_time(concurrent_runtime_seconds)}"
            )


def _merge_many_summaries(
    summary_collections: List[Dict[str, AlgoSummary]],
) -> Dict[str, AlgoSummary]:
    merged_summaries: Dict[str, AlgoSummary] = {}
    for summary_collection in summary_collections:
        for algorithm_name, summary in summary_collection.items():
            merged_summary = merged_summaries.setdefault(
                algorithm_name, AlgoSummary(name=algorithm_name)
            )
            _merge_summary(merged_summary, summary)
    return merged_summaries


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
    parser.add_argument(
        "--logs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more log paths (files or run directories) to aggregate into a single "
            "summary. Example: --logs run_elkk1 run_elkk2"
        ),
    )
    args = parser.parse_args()

    if args.logs:
        log_paths = args.logs
    else:
        log_paths = [args.log or _find_default_log()]

    summary_collections: List[Dict[str, AlgoSummary]] = []
    total_concurrent_runtime_seconds: Optional[float] = 0.0
    for log_path in log_paths:
        summary_collections.append(parse_path(log_path))
        concurrent_runtime_seconds = _compute_concurrent_runtime_seconds(log_path)
        if concurrent_runtime_seconds is None:
            total_concurrent_runtime_seconds = None
        elif total_concurrent_runtime_seconds is not None:
            total_concurrent_runtime_seconds += concurrent_runtime_seconds

    merged_summaries = _merge_many_summaries(summary_collections)
    if len(log_paths) == 1:
        print(f"Log: {log_paths[0]}")
    else:
        print(f"Logs ({len(log_paths)}): {', '.join(log_paths)}")
    print_summary(
        merged_summaries, concurrent_runtime_seconds=total_concurrent_runtime_seconds
    )


if __name__ == "__main__":
    main()
