#!/usr/bin/env python3
"""Compute train-endpoint forgetting and train-BWT proxy.

This script implements the metric discussed in chat:
1) Parse per-task "peak train F1" from a job log using the epoch-complete lines.
2) Locate the matching run directory (and `results.pt`) from log metadata.
3) Rebuild the model/loader from saved args and evaluate the final checkpoint on
   each task's training split.
4) Compute forgetting per task as:
       peak_train_f1(task) - final_train_f1(task)
   and aggregate over tasks 0..T-2:
       avg_forgetting = mean(forgetting[:-1])
       train_bwt_proxy = -avg_forgetting
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from main import _split_eval_output, eval_class_tasks, eval_tasks  # noqa: E402
from scripts.collect_bwt_from_metrics import (  # noqa: E402
    _candidate_metrics_dirs_for_logged_output,
    _extract_algo_name_from_job_filename,
    _extract_logged_output_dir_from_log,
    _extract_model_name_from_log_header,
)

try:
    from torch.serialization import add_safe_globals  # type: ignore
except ImportError:  # pragma: no cover
    add_safe_globals = None


_TASK_EPOCH_F1_RE = re.compile(
    r"T(?P<task>\d+)\s+Ep\s+\d+/\d+\s+complete:\s+Prec\s+[0-9.]+\s+F1\s+(?P<f1>[0-9.]+)"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute train-BWT proxy from logs.")
    parser.add_argument(
        "--job-log",
        type=Path,
        default=None,
        help="Single job_*.log file to process.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory that contains job_logs/ (processes all job_*.log files).",
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=None,
        help=(
            "Suite directory under logs/full_experiments/ (e.g. "
            "logs/full_experiments/full-til_10epochs_w-zs). Processes all runs."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for final-checkpoint evaluation.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-model aggregate results.",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-task rows across all processed models.",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default=None,
        help=(
            "Optional comma-separated algorithm subset to process (for example "
            "'rwalk,icarl'). Matches job log-derived algorithm names."
        ),
    )
    return parser.parse_args()


def _resolve_device(device_choice: str) -> torch.device:
    if device_choice == "cpu":
        return torch.device("cpu")
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_peak_train_f1_from_job_log(job_log_path: Path) -> Dict[int, float]:
    """Parse latest epoch-complete train F1 per task from a job log."""
    peak_by_task: Dict[int, float] = {}
    with job_log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_text in log_file:
            match = _TASK_EPOCH_F1_RE.search(line_text)
            if match is None:
                continue
            task_index = int(match.group("task"))
            f1_value = float(match.group("f1"))
            peak_by_task[task_index] = f1_value
    if not peak_by_task:
        raise SystemExit(
            f"No task epoch-complete F1 lines found in log: {job_log_path}"
        )
    return peak_by_task


def _resolve_run_dir_from_job_log(job_log_path: Path) -> Path:
    algorithm_name = _extract_algo_name_from_job_filename(job_log_path)
    if algorithm_name is None:
        algorithm_name = _extract_model_name_from_log_header(job_log_path)
    if not algorithm_name:
        raise SystemExit(f"Could not infer algorithm name from: {job_log_path}")

    logged_output_dir = _extract_logged_output_dir_from_log(job_log_path)
    if logged_output_dir is None:
        raise SystemExit(f"Could not infer logged output dir from: {job_log_path}")

    candidate_metrics_dirs = _candidate_metrics_dirs_for_logged_output(
        logged_output_dir=logged_output_dir,
        algorithm_name=algorithm_name,
    )
    for metrics_dir in candidate_metrics_dirs:
        run_dir = metrics_dir.parent
        if (run_dir / "results.pt").is_file() and metrics_dir.is_dir():
            return run_dir
    raise SystemExit(
        "Could not resolve a run directory containing results.pt from job log. "
        f"Tried {len(candidate_metrics_dirs)} candidate metrics directories."
    )


def _discover_job_logs(
    job_log: Path | None, run_dir: Path | None, suite_dir: Path | None
) -> List[Path]:
    provided_inputs = sum(value is not None for value in (job_log, run_dir, suite_dir))
    if provided_inputs > 1:
        raise SystemExit("Use only one of --job-log, --run-dir, or --suite-dir.")
    if job_log is not None:
        resolved = job_log.resolve()
        if not resolved.is_file():
            raise SystemExit(f"--job-log does not exist: {resolved}")
        return [resolved]
    if run_dir is not None:
        resolved = run_dir.resolve()
        job_logs_dir = resolved / "job_logs"
        if not job_logs_dir.is_dir():
            raise SystemExit(f"--run-dir missing job_logs/: {resolved}")
        job_logs = sorted(job_logs_dir.glob("job_*.log"))
        if not job_logs:
            raise SystemExit(f"No job_*.log found in: {job_logs_dir}")
        return job_logs
    if suite_dir is not None:
        resolved = suite_dir.resolve()
        if not resolved.is_dir():
            raise SystemExit(f"--suite-dir does not exist: {resolved}")
        job_logs = sorted(resolved.glob("*/job_logs/job_*.log"))
        if not job_logs:
            raise SystemExit(f"No job_*.log found under: {resolved}")
        return job_logs

    default_suite = (_REPO_ROOT / "logs" / "full_experiments").resolve()
    job_logs = sorted(default_suite.glob("run_*/job_logs/job_*.log"))
    if not job_logs:
        raise SystemExit(
            "No default run_*/job_logs/job_*.log found. Pass --job-log, --run-dir, "
            "or --suite-dir."
        )
    return job_logs


def _load_results_bundle(
    results_path: Path,
) -> tuple[dict[str, Any], argparse.Namespace]:
    if add_safe_globals is not None:
        add_safe_globals([argparse.Namespace])
    try:
        bundle = torch.load(results_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(results_path, map_location="cpu")
    if not (isinstance(bundle, (tuple, list)) and len(bundle) >= 6):
        raise SystemExit(f"Unexpected checkpoint format: {results_path}")
    state_dict = bundle[2]
    saved_args = bundle[5]
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Checkpoint state_dict is not a dict: {results_path}")
    return state_dict, copy.deepcopy(saved_args)


def _instantiate_model_and_loader(
    run_args: argparse.Namespace,
    state_dict: dict[str, Any],
    device: torch.device,
):
    loader_module = importlib.import_module(f"dataloaders.{run_args.loader}")
    incremental_loader = loader_module.IncrementalLoader(run_args, seed=run_args.seed)
    n_inputs, n_outputs, n_tasks = incremental_loader.get_dataset_info()

    model_module = importlib.import_module(f"model.{run_args.model}")
    model = model_module.Net(n_inputs, n_outputs, n_tasks, run_args)

    model_state_dict = model.state_dict()
    filtered_state = {
        key: value for key, value in state_dict.items() if key in model_state_dict
    }
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    return model, incremental_loader


def _build_train_task_loaders(incremental_loader: Any) -> List[Any]:
    train_task_loaders: List[Any] = []
    for _ in range(incremental_loader.n_tasks):
        _task_info, train_loader, _val_loader, _test_loader = (
            incremental_loader.new_task()
        )
        train_task_loaders.append(train_loader)
    return train_task_loaders


def _extract_metric_at_index(metric_values: object, index: int) -> float:
    if isinstance(metric_values, (list, tuple)):
        if 0 <= index < len(metric_values):
            return float(metric_values[index])
        return float("nan")
    if metric_values is None:
        return float("nan")
    return float(metric_values)


def _evaluate_final_model_train_f1(
    model: torch.nn.Module,
    train_task_loaders: Sequence[Any],
    run_args: argparse.Namespace,
) -> Dict[int, float]:
    evaluator = (
        eval_class_tasks
        if run_args.loader == "class_incremental_loader"
        else eval_tasks
    )
    with torch.no_grad():
        eval_output = evaluator(model, list(train_task_loaders), run_args)
    _cls_rec, _cls_prec, cls_f1, _det, _fa = _split_eval_output(eval_output)
    return {
        task_index: _extract_metric_at_index(cls_f1, task_index)
        for task_index in range(len(train_task_loaders))
    }


def _rows_and_aggregate(
    peak_train_f1_by_task: Dict[int, float],
    final_train_f1_by_task: Dict[int, float],
) -> tuple[List[Dict[str, float]], float, float]:
    task_ids = sorted(set(peak_train_f1_by_task) & set(final_train_f1_by_task))
    rows: List[Dict[str, float]] = []
    for task_index in task_ids:
        peak_value = float(peak_train_f1_by_task[task_index])
        final_value = float(final_train_f1_by_task[task_index])
        forgetting_value = peak_value - final_value
        rows.append(
            {
                "task": float(task_index),
                "peak_train_f1": peak_value,
                "final_train_f1": final_value,
                "forgetting": forgetting_value,
            }
        )
    if len(rows) < 2:
        raise SystemExit(
            "Need at least 2 tasks to compute average forgetting over tasks 0..T-2."
        )
    forget_values = [row["forgetting"] for row in rows[:-1]]
    avg_forgetting = float(np.mean(forget_values))
    train_bwt_proxy = -avg_forgetting
    return rows, avg_forgetting, train_bwt_proxy


def _infer_algo_name(job_log_path: Path) -> str:
    algo_name = _extract_algo_name_from_job_filename(job_log_path)
    if algo_name:
        return algo_name
    header_name = _extract_model_name_from_log_header(job_log_path)
    if header_name:
        return header_name
    return "unknown"


def _parse_algo_subset(algos_arg: str | None) -> set[str] | None:
    """Parse a comma-separated algorithm list into a normalized subset."""
    if algos_arg is None:
        return None
    normalized = {
        token.strip().lower() for token in algos_arg.split(",") if token.strip()
    }
    if not normalized:
        raise SystemExit("--algos was provided but no algorithm names were parsed.")
    return normalized


def _filter_job_logs_by_algorithms(
    job_logs: Sequence[Path], selected_algorithms: set[str] | None
) -> List[Path]:
    """Filter discovered job logs to selected algorithms."""
    if selected_algorithms is None:
        return list(job_logs)
    return [
        job_log_path
        for job_log_path in job_logs
        if _infer_algo_name(job_log_path).lower() in selected_algorithms
    ]


def _evaluate_one_job_log(
    job_log_path: Path,
    device: torch.device,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    peak_train_f1_by_task = _parse_peak_train_f1_from_job_log(job_log_path)
    run_dir = _resolve_run_dir_from_job_log(job_log_path)
    results_path = run_dir / "results.pt"
    state_dict, run_args = _load_results_bundle(results_path)

    run_args.cuda = device.type == "cuda"
    run_args.log_dir = str(run_dir)
    model, incremental_loader = _instantiate_model_and_loader(
        run_args, state_dict, device
    )
    train_task_loaders = _build_train_task_loaders(incremental_loader)
    final_train_f1_by_task = _evaluate_final_model_train_f1(
        model=model,
        train_task_loaders=train_task_loaders,
        run_args=run_args,
    )
    per_task_rows, avg_forgetting, train_bwt_proxy = _rows_and_aggregate(
        peak_train_f1_by_task=peak_train_f1_by_task,
        final_train_f1_by_task=final_train_f1_by_task,
    )
    aggregate = {
        "algo": _infer_algo_name(job_log_path),
        "job_log": str(job_log_path),
        "run_dir": str(run_dir),
        "results": str(results_path),
        "n_tasks": len(per_task_rows),
        "avg_forgetting_0_to_t_minus_2": avg_forgetting,
        "train_bwt_proxy": train_bwt_proxy,
    }
    detailed_rows: List[Dict[str, Any]] = []
    for row in per_task_rows:
        detailed_rows.append(
            {
                "algo": aggregate["algo"],
                "job_log": str(job_log_path),
                "run_dir": str(run_dir),
                "task": int(row["task"]),
                "peak_train_f1": row["peak_train_f1"],
                "final_train_f1": row["final_train_f1"],
                "forgetting": row["forgetting"],
            }
        )
    return aggregate, detailed_rows


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    discovered_job_logs = _discover_job_logs(
        job_log=args.job_log,
        run_dir=args.run_dir,
        suite_dir=args.suite_dir,
    )
    selected_algorithms = _parse_algo_subset(args.algos)
    job_logs = _filter_job_logs_by_algorithms(discovered_job_logs, selected_algorithms)
    if selected_algorithms is None:
        print(f"Discovered {len(job_logs)} job log(s). Evaluating on {device}.")
    else:
        selected_list = ", ".join(sorted(selected_algorithms))
        print(
            f"Discovered {len(discovered_job_logs)} job log(s), selected "
            f"{len(job_logs)} after --algos filter ({selected_list}). "
            f"Evaluating on {device}."
        )
    if not job_logs:
        print("No job logs matched the requested algorithm subset.")
        return

    aggregate_rows: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, str]] = []
    for log_index, job_log_path in enumerate(job_logs, start=1):
        try:
            aggregate, detailed = _evaluate_one_job_log(job_log_path, device)
            aggregate_rows.append(aggregate)
            detailed_rows.extend(detailed)
            print(
                f"[{log_index}/{len(job_logs)}] {aggregate['algo']}: "
                f"train_bwt_proxy={aggregate['train_bwt_proxy']:.6f}"
            )
        except (Exception, SystemExit) as exc:  # pragma: no cover
            failed_rows.append({"job_log": str(job_log_path), "error": str(exc)})
            print(f"[{log_index}/{len(job_logs)}] SKIPPED {job_log_path}: {exc}")

    if not aggregate_rows:
        print("No complete job logs were processed successfully.")
        if failed_rows:
            print(f"Skipped jobs: {len(failed_rows)}")
            for failed in failed_rows:
                print(f"- {failed['job_log']}: {failed['error']}")
        return

    aggregate_rows.sort(key=lambda row: str(row["algo"]))
    print("")
    print(f"{'algo':<14} {'tasks':>5} {'avg_forgetting':>14} {'train_bwt_proxy':>16}")
    print("-" * 55)
    for row in aggregate_rows:
        print(
            f"{row['algo']:<14} {int(row['n_tasks']):5d} "
            f"{float(row['avg_forgetting_0_to_t_minus_2']):14.6f} "
            f"{float(row['train_bwt_proxy']):16.6f}"
        )

    if failed_rows:
        print(f"\nFailed jobs: {len(failed_rows)}")
        for failed in failed_rows:
            print(f"- {failed['job_log']}: {failed['error']}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "algo",
                "job_log",
                "run_dir",
                "results",
                "n_tasks",
                "avg_forgetting_0_to_t_minus_2",
                "train_bwt_proxy",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate_rows)
        print(f"saved_csv: {args.csv}")
    if args.details_csv is not None:
        args.details_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.details_csv.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "algo",
                "job_log",
                "run_dir",
                "task",
                "peak_train_f1",
                "final_train_f1",
                "forgetting",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_rows)
        print(f"saved_details_csv: {args.details_csv}")


if __name__ == "__main__":
    main()
