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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
_EVAL_CLS_VECTOR_RE = re.compile(r"Eval at Epoch\s+\d+:\s+cls\s+\[(?P<cls>[^\]]*)\]")
_SHARED_EVAL_VALIDATION_SPLIT = 0.3


@dataclass
class SharedLoaderContext:
    """Reusable loader-derived context shared across compatible models."""

    n_inputs: int
    n_outputs: int
    n_tasks: int
    train_task_loaders: List[Any]
    test_task_loaders: List[Any]
    get_samples_per_task_resolver: Any


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


def _parse_task_end_test_f1_from_job_log(job_log_path: Path) -> Dict[int, float]:
    """Parse per-task test F1 measured right after each task training step."""
    task_end_test_f1_by_task: Dict[int, float] = {}
    with job_log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_text in log_file:
            match = _EVAL_CLS_VECTOR_RE.search(line_text)
            if match is None:
                continue
            cls_raw_values = match.group("cls").strip()
            if not cls_raw_values:
                continue
            cls_tokens = [token.strip() for token in cls_raw_values.split(",")]
            cls_values = [float(token) for token in cls_tokens if token]
            if not cls_values:
                continue
            task_index = len(cls_values) - 1
            task_end_test_f1_by_task[task_index] = cls_values[task_index]
    if not task_end_test_f1_by_task:
        raise SystemExit(f"No per-epoch eval cls vectors found in log: {job_log_path}")
    return task_end_test_f1_by_task


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


def _build_loader_cache_key(run_args: argparse.Namespace) -> Tuple[Any, ...]:
    """Build a cache key for dataloader contexts that can be shared."""
    key_fields = (
        "loader",
        "data_path",
        "dataset",
        "task_order_files",
        "shuffle_tasks",
        "class_order",
        "increment",
        "validation",
        "workers",
        "samples_per_task",
        "seed",
        "classes_per_it",
        "nc_per_task",
        "nc_per_task_list",
        "data_scaling",
        "use_iq_aug_features",
        "iq_aug_feature_type",
        "test_batch_size",
    )
    return tuple(getattr(run_args, field_name, None) for field_name in key_fields)


def _build_shared_loader_context(run_args: argparse.Namespace) -> SharedLoaderContext:
    """Build a single loader context that can be reused by many models."""
    # Train-BWT evaluation is data-loading heavy; use more workers to reduce
    # loader bottlenecks when rebuilding task loaders.
    run_args.workers = 12
    run_args.validation = _SHARED_EVAL_VALIDATION_SPLIT
    loader_module = importlib.import_module(f"dataloaders.{run_args.loader}")
    incremental_loader = loader_module.IncrementalLoader(run_args, seed=run_args.seed)
    n_inputs, n_outputs, n_tasks = incremental_loader.get_dataset_info()
    train_task_loaders, test_task_loaders = _build_task_loaders(incremental_loader)
    return SharedLoaderContext(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_tasks=n_tasks,
        train_task_loaders=train_task_loaders,
        test_task_loaders=test_task_loaders,
        get_samples_per_task_resolver=incremental_loader.get_samples_per_task,
    )


def _get_or_build_shared_loader_context(
    run_args: argparse.Namespace,
    loader_context_cache: Dict[Tuple[Any, ...], SharedLoaderContext],
) -> SharedLoaderContext:
    """Return a cached shared loader context, building it once per key."""
    run_args.validation = _SHARED_EVAL_VALIDATION_SPLIT
    cache_key = _build_loader_cache_key(run_args)
    shared_context = loader_context_cache.get(cache_key)
    if shared_context is not None:
        print(
            f"CACHE_HIT loader={run_args.loader} dataset={run_args.dataset} "
            f"tasks={shared_context.n_tasks} workers={run_args.workers}"
        )
        return shared_context
    print(
        f"CACHE_MISS loader={run_args.loader} dataset={run_args.dataset} "
        f"validation={run_args.validation} workers={run_args.workers}"
    )
    shared_context = _build_shared_loader_context(run_args)
    loader_context_cache[cache_key] = shared_context
    return shared_context


def _instantiate_model(
    run_args: argparse.Namespace,
    state_dict: dict[str, Any],
    device: torch.device,
    shared_loader_context: SharedLoaderContext,
):
    # Some learners (for example iCaRL) expect this resolver on args and
    # assert when samples_per_task <= 0 if it is missing.
    run_args.get_samples_per_task = shared_loader_context.get_samples_per_task_resolver

    model_module = importlib.import_module(f"model.{run_args.model}")
    model = model_module.Net(
        shared_loader_context.n_inputs,
        shared_loader_context.n_outputs,
        shared_loader_context.n_tasks,
        run_args,
    )

    model_state_dict = model.state_dict()
    filtered_state = {
        key: value for key, value in state_dict.items() if key in model_state_dict
    }
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    return model


def _build_task_loaders(incremental_loader: Any) -> tuple[List[Any], List[Any]]:
    train_task_loaders: List[Any] = []
    test_task_loaders: List[Any] = []
    for _ in range(incremental_loader.n_tasks):
        _task_info, train_loader, _val_loader, _test_loader = (
            incremental_loader.new_task()
        )
        train_task_loaders.append(train_loader)
        test_task_loaders.append(_test_loader)
    return train_task_loaders, test_task_loaders


def _extract_metric_at_index(metric_values: object, index: int) -> float:
    if isinstance(metric_values, (list, tuple)):
        if 0 <= index < len(metric_values):
            return float(metric_values[index])
        return float("nan")
    if metric_values is None:
        return float("nan")
    return float(metric_values)


def _evaluate_final_model_f1_by_task(
    model: torch.nn.Module,
    task_loaders: Sequence[Any],
    run_args: argparse.Namespace,
) -> Dict[int, float]:
    evaluator = (
        eval_class_tasks
        if run_args.loader == "class_incremental_loader"
        else eval_tasks
    )
    with torch.no_grad():
        eval_output = evaluator(model, list(task_loaders), run_args)
    _cls_rec, _cls_prec, cls_f1, _det, _fa = _split_eval_output(eval_output)
    return {
        task_index: _extract_metric_at_index(cls_f1, task_index)
        for task_index in range(len(task_loaders))
    }


def _rows_and_aggregate(
    train_f1_after_task_by_task: Dict[int, float],
    final_train_f1_by_task: Dict[int, float],
    test_f1_after_task_by_task: Dict[int, float],
    final_test_f1_by_task: Dict[int, float],
) -> tuple[List[Dict[str, float]], float, float, float]:
    task_ids = sorted(
        set(train_f1_after_task_by_task)
        & set(final_train_f1_by_task)
        & set(test_f1_after_task_by_task)
        & set(final_test_f1_by_task)
    )
    rows: List[Dict[str, float]] = []
    for task_index in task_ids:
        train_f1_after_task = float(train_f1_after_task_by_task[task_index])
        final_train_f1 = float(final_train_f1_by_task[task_index])
        test_f1_after_task = float(test_f1_after_task_by_task[task_index])
        final_test_f1 = float(final_test_f1_by_task[task_index])
        representational_forgetting = train_f1_after_task - final_train_f1
        test_forgetting = test_f1_after_task - final_test_f1
        generalisation_shift = test_forgetting - representational_forgetting
        rows.append(
            {
                "task": float(task_index),
                "train_f1_after_task": train_f1_after_task,
                "final_train_f1": final_train_f1,
                "test_f1_after_task": test_f1_after_task,
                "final_test_f1": final_test_f1,
                "representational_forgetting": representational_forgetting,
                "test_forgetting": test_forgetting,
                "generalisation_shift": generalisation_shift,
            }
        )
    if len(rows) < 2:
        raise SystemExit(
            "Need at least 2 tasks to compute average forgetting over tasks 0..T-2."
        )
    representational_forgetting_values = [
        row["representational_forgetting"] for row in rows[:-1]
    ]
    test_forgetting_values = [row["test_forgetting"] for row in rows[:-1]]
    generalisation_shift_values = [row["generalisation_shift"] for row in rows[:-1]]
    avg_representational_forgetting = float(np.mean(representational_forgetting_values))
    avg_test_forgetting = float(np.mean(test_forgetting_values))
    avg_generalisation_shift = float(np.mean(generalisation_shift_values))
    return (
        rows,
        avg_representational_forgetting,
        avg_test_forgetting,
        avg_generalisation_shift,
    )


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
    loader_context_cache: Dict[Tuple[Any, ...], SharedLoaderContext],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    train_f1_after_task_by_task = _parse_peak_train_f1_from_job_log(job_log_path)
    test_f1_after_task_by_task = _parse_task_end_test_f1_from_job_log(job_log_path)
    run_dir = _resolve_run_dir_from_job_log(job_log_path)
    results_path = run_dir / "results.pt"
    state_dict, run_args = _load_results_bundle(results_path)

    run_args.cuda = device.type == "cuda"
    run_args.log_dir = str(run_dir)
    shared_loader_context = _get_or_build_shared_loader_context(
        run_args, loader_context_cache
    )
    model = _instantiate_model(run_args, state_dict, device, shared_loader_context)
    final_train_f1_by_task = _evaluate_final_model_f1_by_task(
        model=model,
        task_loaders=shared_loader_context.train_task_loaders,
        run_args=run_args,
    )
    final_test_f1_by_task = _evaluate_final_model_f1_by_task(
        model=model,
        task_loaders=shared_loader_context.test_task_loaders,
        run_args=run_args,
    )
    (
        per_task_rows,
        avg_representational_forgetting,
        avg_test_forgetting,
        avg_generalisation_shift,
    ) = _rows_and_aggregate(
        train_f1_after_task_by_task=train_f1_after_task_by_task,
        final_train_f1_by_task=final_train_f1_by_task,
        test_f1_after_task_by_task=test_f1_after_task_by_task,
        final_test_f1_by_task=final_test_f1_by_task,
    )
    train_bwt_proxy = -avg_representational_forgetting
    aggregate = {
        "algo": _infer_algo_name(job_log_path),
        "job_log": str(job_log_path),
        "run_dir": str(run_dir),
        "results": str(results_path),
        "n_tasks": len(per_task_rows),
        "avg_representational_forgetting_0_to_t_minus_2": avg_representational_forgetting,
        "avg_test_forgetting_0_to_t_minus_2": avg_test_forgetting,
        "avg_generalisation_shift_0_to_t_minus_2": avg_generalisation_shift,
        "train_bwt_proxy": train_bwt_proxy,
        "test_bwt": -avg_test_forgetting,
    }
    detailed_rows: List[Dict[str, Any]] = []
    for row in per_task_rows:
        detailed_rows.append(
            {
                "algo": aggregate["algo"],
                "job_log": str(job_log_path),
                "run_dir": str(run_dir),
                "task": int(row["task"]),
                "train_f1_after_task": row["train_f1_after_task"],
                "final_train_f1": row["final_train_f1"],
                "test_f1_after_task": row["test_f1_after_task"],
                "final_test_f1": row["final_test_f1"],
                "representational_forgetting": row["representational_forgetting"],
                "test_forgetting": row["test_forgetting"],
                "generalisation_shift": row["generalisation_shift"],
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
    loader_context_cache: Dict[Tuple[Any, ...], SharedLoaderContext] = {}
    for log_index, job_log_path in enumerate(job_logs, start=1):
        try:
            aggregate, detailed = _evaluate_one_job_log(
                job_log_path, device, loader_context_cache
            )
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
    print(
        f"{'algo':<14} {'tasks':>5} {'avg_rf':>12} {'avg_tf':>12} {'avg_gs':>12} "
        f"{'train_bwt':>12}"
    )
    print("-" * 74)
    for row in aggregate_rows:
        print(
            f"{row['algo']:<14} {int(row['n_tasks']):5d} "
            f"{float(row['avg_representational_forgetting_0_to_t_minus_2']):12.6f} "
            f"{float(row['avg_test_forgetting_0_to_t_minus_2']):12.6f} "
            f"{float(row['avg_generalisation_shift_0_to_t_minus_2']):12.6f} "
            f"{float(row['train_bwt_proxy']):12.6f}"
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
                "avg_representational_forgetting_0_to_t_minus_2",
                "avg_test_forgetting_0_to_t_minus_2",
                "avg_generalisation_shift_0_to_t_minus_2",
                "train_bwt_proxy",
                "test_bwt",
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
                "train_f1_after_task",
                "final_train_f1",
                "test_f1_after_task",
                "final_test_f1",
                "representational_forgetting",
                "test_forgetting",
                "generalisation_shift",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_rows)
        print(f"saved_details_csv: {args.details_csv}")


if __name__ == "__main__":
    main()
