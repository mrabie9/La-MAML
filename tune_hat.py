#!/usr/bin/env python3
"""Hyperparameter tuning harness for the HAT learner."""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import random
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

import parser as file_parser
from main import life_experience, life_experience_iid
from utils import misc_utils

DEFAULT_GRID: Dict[str, List[float]] = {
    "lr": [1e-3, 1e-2, 1e-1],
    "gamma": [1.0, 1.3, 1.8],
    "smax": [25, 50, 100],
    # "grad_clip_norm": [5.0, 10.0],
}

DEFAULT_TYPE_HINTS: Dict[str, type] = {
    "lr": float,
    "gamma": float,
    "smax": float,
    "grad_clip_norm": float,
}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run grid or random search over HAT hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_all.yaml",
        help="Base YAML config to start from.",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="PARAM=V1,V2,...",
        help="Grid specification. May be provided multiple times.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PARAM=VALUE",
        help="Override applied to all trials (e.g. --override n_epochs=10).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Randomly sample this many combinations from the grid.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=0,
        help="Seed for shuffling or sampling trial order.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Evaluate at most this many trials (after sampling/shuffling).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to the base seed for each trial index.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="logs/tuning/hat",
        help="Directory where aggregated tuning summaries are stored.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned trials without running them.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the full grid even when every combination is evaluated.",
    )
    parser.add_argument(
        "--keep-expt-name",
        action="store_true",
        help="Do not alter the experiment name from the base config.",
    )
    return parser


def get_reference(key: str, base_args: argparse.Namespace) -> Any:
    if hasattr(base_args, key):
        return getattr(base_args, key)
    return DEFAULT_TYPE_HINTS.get(key)


def coerce_value(raw: str, reference: Any) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None

    if isinstance(reference, bool):
        return lowered in {"1", "true", "yes", "on"}
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, str):
        return value
    if isinstance(reference, type):
        if reference is bool:
            return lowered in {"1", "true", "yes", "on"}
        if reference is int:
            return int(value)
        if reference is float:
            return float(value)
        return reference(value)

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_grid_specs(specs: Sequence[str], base_args: argparse.Namespace) -> Dict[str, List[Any]]:
    space: Dict[str, List[Any]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid grid spec '{spec}'. Use PARAM=v1,v2,... format.")
        key, raw_values = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid grid spec '{spec}'.")
        values: List[Any] = []
        for item in raw_values.split(","):
            item = item.strip()
            if not item:
                continue
            reference = get_reference(key, base_args)
            values.append(coerce_value(item, reference))
        if not values:
            raise ValueError(f"No values provided for grid parameter '{key}'.")
        space[key] = values
    return space


def parse_override_specs(specs: Sequence[str], base_args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid override '{spec}'. Use PARAM=value format.")
        key, raw_value = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{spec}'.")
        reference = get_reference(key, base_args)
        overrides[key] = coerce_value(raw_value, reference)
    return overrides


def expand_trials(
    space: Dict[str, List[Any]],
    num_samples: int | None,
    search_seed: int,
    max_trials: int | None,
    shuffle: bool,
) -> List[Dict[str, Any]]:
    if not space:
        return [{}]

    keys = sorted(space)
    grid_iter = itertools.product(*(space[key] for key in keys))
    all_combos = [dict(zip(keys, values)) for values in grid_iter]

    rng = random.Random(search_seed)

    if num_samples is not None and num_samples < len(all_combos):
        indices = list(range(len(all_combos)))
        rng.shuffle(indices)
        selected = indices[:num_samples]
        combos = [all_combos[i] for i in selected]
    else:
        combos = all_combos
        if shuffle:
            rng.shuffle(combos)

    if max_trials is not None:
        combos = combos[:max_trials]
    return combos or [{}]


def format_value_for_slug(value: Any) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        abs_val = abs(value)
        if abs_val >= 1:
            txt = f"{value:.3f}".rstrip("0").rstrip(".")
        else:
            txt = f"{value:.0e}".replace("+", "")
        return txt.replace("-", "m").replace(".", "p")
    return str(value).replace(" ", "")


def slugify_params(params: Dict[str, Any], max_length: int = 80) -> str:
    if not params:
        return "base"
    parts = []
    for key in sorted(params):
        encoded = format_value_for_slug(params[key])
        encoded = encoded.replace("/", "_")
        if len(encoded) > 12:
            encoded = encoded[:12]
        parts.append(f"{key}-{encoded}")
    slug = "_".join(parts)
    if len(slug) > max_length:
        slug = slug[:max_length]
    return slug


def extract_final_scores(tensorlike: Any) -> List[float]:
    if isinstance(tensorlike, torch.Tensor):
        if tensorlike.numel() == 0:
            return []
        last = tensorlike[-1]
        if last.ndim == 0:
            return [float(last.item())]
        return [float(x) for x in last.tolist()]
    if isinstance(tensorlike, list) and tensorlike:
        last = tensorlike[-1]
        if isinstance(last, (list, tuple)):
            return [float(x) for x in last]
        return [float(last)]
    return []


def compute_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def run_single_trial(
    base_args: argparse.Namespace,
    constant_overrides: Dict[str, Any],
    trial_overrides: Dict[str, Any],
    trial_idx: int,
    session_timestamp: str,
    runs_root: Path,
    seed_offset: int,
    keep_expt_name: bool,
) -> Dict[str, Any]:
    args = deepcopy(base_args)
    merged = dict(constant_overrides)
    merged.update(trial_overrides)
    for key, value in merged.items():
        setattr(args, key, value)

    if getattr(args, "model", "hat") != "hat":
        args.model = "hat"

    args.log_dir = str(runs_root)
    args.seed = int(getattr(base_args, "seed", 0) + seed_offset + trial_idx)

    trial_slug = slugify_params(trial_overrides)
    if not keep_expt_name:
        base_name = getattr(base_args, "expt_name", "hat")
        args.expt_name = f"{base_name}_tune_{trial_idx:03d}_{trial_slug}"[:120]

    trial_timestamp = f"{session_timestamp}-trial{trial_idx:03d}"

    misc_utils.init_seed(args.seed)
    log_dir, tf_dir = misc_utils.log_dir(args, trial_timestamp)
    args.log_dir = log_dir
    args.tf_dir = tf_dir

    loader_mod = importlib.import_module(f"dataloaders.{args.loader}")
    loader = loader_mod.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()

    model_mod = importlib.import_module(f"model.{args.model}")
    model = model_mod.Net(n_inputs, n_outputs, n_tasks, args)

    if getattr(args, "cuda", False) and torch.cuda.is_available():
        model = model.cuda()

    try:
        if args.model == "iid2":
            (
                result_val_t,
                result_val_a,
                result_test_t,
                result_test_a,
                result_val_det_a,
                result_val_det_fa,
                result_test_det_a,
                result_test_det_fa,
                spent,
            ) = life_experience_iid(model, loader, args)
        else:
            (
                result_val_t,
                result_val_a,
                result_test_t,
                result_test_a,
                result_val_det_a,
                result_val_det_fa,
                result_test_det_a,
                result_test_det_fa,
                spent,
            ) = life_experience(model, loader, args)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_scores = extract_final_scores(result_val_a)
    test_scores = extract_final_scores(result_test_a)
    val_det_scores = extract_final_scores(result_val_det_a)
    val_pfa_scores = extract_final_scores(result_val_det_fa)
    test_det_scores = extract_final_scores(result_test_det_a)
    test_pfa_scores = extract_final_scores(result_test_det_fa)

    val_mean = compute_mean(val_scores)
    det_mean = compute_mean(val_det_scores)
    pfa_mean = compute_mean(val_pfa_scores)
    if val_det_scores and val_pfa_scores:
        score = val_mean * det_mean * (1.0 - pfa_mean)
    else:
        score = val_mean

    return {
        "status": "ok",
        "trial": trial_idx,
        "log_dir": log_dir,
        "tf_dir": tf_dir,
        "params": merged,
        "trial_params": dict(trial_overrides),
        "fixed_params": dict(constant_overrides),
        "val_per_task": val_scores,
        "val_mean": val_mean,
        "val_det_per_task": val_det_scores,
        "val_det_mean": det_mean,
        "val_pfa_per_task": val_pfa_scores,
        "val_pfa_mean": pfa_mean,
        "test_per_task": test_scores,
        "test_mean": compute_mean(test_scores),
        "test_det_per_task": test_det_scores,
        "test_det_mean": compute_mean(test_det_scores),
        "test_pfa_per_task": test_pfa_scores,
        "test_pfa_mean": compute_mean(test_pfa_scores),
        "score": score,
        "duration_sec": float(spent),
    }


def dump_summary(session_dir: Path, summary: Dict[str, Any], successes: List[Dict[str, Any]]) -> None:
    summary_path = session_dir / "summary.json"
    session_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if not successes:
        return

    field_names = [
        "trial",
        "score",
        "val_mean",
        "val_det_mean",
        "val_pfa_mean",
        "test_mean",
        "test_det_mean",
        "test_pfa_mean",
        "duration_sec",
        "log_dir",
    ]
    param_keys = sorted({key for trial in successes for key in trial["params"].keys()})
    field_names.extend(param_keys)
    csv_path = session_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for trial in successes:
            row = {
                "trial": trial["trial"],
                "score": trial.get("score"),
                "val_mean": trial.get("val_mean"),
                "val_det_mean": trial.get("val_det_mean"),
                "val_pfa_mean": trial.get("val_pfa_mean"),
                "test_mean": trial.get("test_mean"),
                "test_det_mean": trial.get("test_det_mean"),
                "test_pfa_mean": trial.get("test_pfa_mean"),
                "duration_sec": trial["duration_sec"],
                "log_dir": trial["log_dir"],
            }
            for key in param_keys:
                row[key] = trial["params"].get(key)
            writer.writerow(row)


def main() -> None:
    cli = build_cli().parse_args()

    base_args = file_parser.parse_args_from_yaml(cli.config)
    if getattr(base_args, "model", "hat") != "hat":
        base_args.model = "hat"

    constant_overrides = parse_override_specs(cli.override, base_args)

    if cli.grid:
        search_space = parse_grid_specs(cli.grid, base_args)
    else:
        search_space = {key: values[:] for key, values in DEFAULT_GRID.items()}

    for key in list(search_space.keys()):
        if key in constant_overrides:
            del search_space[key]

    trials = expand_trials(search_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle)

    if cli.dry_run:
        print("Planned trials (dry-run):")
        for idx, tr in enumerate(trials):
            merged = dict(constant_overrides)
            merged.update(tr)
            print(f"  #{idx:03d}: {merged}")
        print(f"Total: {len(trials)} trials")
        return

    session_timestamp = misc_utils.get_date_time()
    session_dir = Path(cli.output_root) / session_timestamp
    runs_root = session_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for idx, trial_params in enumerate(trials):
        try:
            outcome = run_single_trial(
                base_args,
                constant_overrides,
                trial_params,
                idx,
                session_timestamp,
                runs_root,
                cli.seed_offset,
                cli.keep_expt_name,
            )
        except Exception as exc:  # pylint: disable=broad-except
            trace = traceback.format_exc()
            outcome = {
                "status": "failed",
                "trial": idx,
                "params": dict(constant_overrides, **trial_params),
                "error": str(exc),
                "traceback": trace,
            }
            print(f"Trial {idx} failed: {exc}")
        else:
            print(
                f"Trial {idx} finished | score={outcome['score']:.4f} | params={trial_params}"
            )
        results.append(outcome)

    successes = [r for r in results if r.get("status") == "ok"]
    best = max(successes, key=lambda r: r["score"]) if successes else None

    summary = {
        "config": str(Path(cli.config).resolve()),
        "base_expt_name": getattr(base_args, "expt_name", "hat"),
        "session_dir": str(session_dir.resolve()),
        "timestamp": session_timestamp,
        "fixed_overrides": constant_overrides,
        "search_space": search_space,
        "num_trials": len(trials),
        "results": results,
        "best": best,
    }

    dump_summary(session_dir, summary, successes)

    if best:
        print(
            f"Best trial #{best['trial']} | score={best['score']:.4f} | params={best['trial_params']}"
        )
        print(f"Logs stored in: {best['log_dir']}")
    else:
        print("No successful trials were recorded.")


if __name__ == "__main__":
    main()
