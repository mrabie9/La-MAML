#!/usr/bin/env python3
"""Reusable hyperparameter tuning harness utilities."""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import random
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch
import sys
sys.path.append("/home/lunet/wsmr11/repos/La-MAML")  # to import from parent directory
import parser as file_parser
from main import life_experience, life_experience_iid
from utils import misc_utils

Grid = Dict[str, List[Any]]
TypeHints = Dict[str, type]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_config_chain(model_name: str, preset_default: str | None) -> List[str]:
    """Return the default stack of config fragments for a model."""

    chain: List[str] = []
    base_cfg = REPO_ROOT / "configs/base.yaml"
    if base_cfg.exists():
        chain.append(str(base_cfg))
    model_cfg = REPO_ROOT / "configs/models" / f"{model_name}.yaml"
    if model_cfg.exists():
        chain.append(str(model_cfg))
    if not chain and preset_default:
        preset_path = Path(preset_default)
        if not preset_path.is_absolute():
            preset_path = REPO_ROOT / preset_path
        chain.append(str(preset_path))
    return chain


@dataclass
class TuningPreset:
    """Configuration wrapper for building a tuning entrypoint."""

    model_name: str
    description: str | None = None
    default_config: str = "config_all.yaml"
    default_output_root: str | None = None
    default_grid: Grid | None = None
    type_hints: TypeHints = field(default_factory=dict)
    grid_factory: Callable[[argparse.Namespace], Grid] | None = None

    def resolve_description(self) -> str:
        if self.description:
            return self.description
        return f"Run grid or random search over {self.model_name.upper()} hyperparameters."

    def resolve_output_root(self) -> str:
        if self.default_output_root:
            return self.default_output_root
        return f"~/repos/La-MAML/logs/tuning/{self.model_name}"


def build_cli(preset: TuningPreset) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=preset.resolve_description(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="Base YAML config file(s) applied in order. Defaults to the shared base"
        " config plus the model-specific fragment when --config is omitted.",
    )
    parser.add_argument(
        "--config-dir",
        action="append",
        default=[str(REPO_ROOT / "configs/tuning_defaults.yaml")],
        metavar="DIR",
        help="Directory of YAML config fragments to apply (alphabetical order).",
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
        "--vary-seed",
        action="store_true",
        help="If set, add the trial index to the base seed.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=preset.resolve_output_root(),
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
    parser.add_argument(
        "--lr-first",
        action="store_true",
        help="Tune the learning rate first, then tune remaining parameters.",
    )
    parser.add_argument(
        "--lr-key",
        type=str,
        default="lr",
        help="Comma-separated parameter names treated as learning rates for --lr-first.",
    )
    parser.add_argument(
        "--tune-only",
        action="append",
        default=[],
        metavar="PARAM",
        help="Restrict the grid to specific hyperparameter(s). May be repeated or comma-separated.",
    )
    return parser


def get_reference(key: str, base_args: argparse.Namespace, type_hints: TypeHints) -> Any:
    if hasattr(base_args, key):
        return getattr(base_args, key)
    return type_hints.get(key)


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


def parse_grid_specs(
    specs: Sequence[str],
    base_args: argparse.Namespace,
    type_hints: TypeHints,
) -> Grid:
    space: Grid = {}
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
            reference = get_reference(key, base_args, type_hints)
            values.append(coerce_value(item, reference))
        if not values:
            raise ValueError(f"No values provided for grid parameter '{key}'.")
        space[key] = values
    return space


def parse_override_specs(
    specs: Sequence[str],
    base_args: argparse.Namespace,
    type_hints: TypeHints,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid override '{spec}'. Use PARAM=value format.")
        key, raw_value = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{spec}'.")
        reference = get_reference(key, base_args, type_hints)
        overrides[key] = coerce_value(raw_value, reference)
    return overrides


def expand_trials(
    space: Grid,
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


def parse_lr_keys(raw: str) -> List[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    return keys or ["lr"]


def parse_tune_only(specs: Sequence[str]) -> List[str]:
    keys: List[str] = []
    seen: set[str] = set()
    for spec in specs:
        for item in spec.split(","):
            key = item.strip()
            if not key or key in seen:
                continue
            keys.append(key)
            seen.add(key)
    return keys


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
    vary_seed: bool,
    keep_expt_name: bool,
    model_name: str,
) -> Dict[str, Any]:
    args = deepcopy(base_args)
    merged = dict(constant_overrides)
    merged.update(trial_overrides)
    for key, value in merged.items():
        setattr(args, key, value)

    args.model = model_name

    args.log_dir = str(runs_root)
    seed_base = int(getattr(base_args, "seed", 0) + seed_offset)
    args.seed = seed_base + (trial_idx if vary_seed else 0)

    trial_slug = slugify_params(trial_overrides)
    if not keep_expt_name:
        base_name = getattr(base_args, "expt_name", model_name)
        args.expt_name = f"{base_name}_tune_{trial_idx:03d}_{trial_slug}"[:120]

    trial_timestamp = f"{session_timestamp}-trial{trial_idx:03d}"

    misc_utils.init_seed(args.seed)
    log_dir, tf_dir = misc_utils.log_dir(args, trial_timestamp)
    args.log_dir = log_dir
    args.tf_dir = tf_dir
    if hasattr(args, "data_path"):
        data_path = Path(args.data_path).expanduser()
        if not data_path.is_absolute() and not data_path.exists():
            candidate = REPO_ROOT / data_path
            if candidate.exists():
                args.data_path = str(candidate)

    loader_mod = importlib.import_module(f"dataloaders.{args.loader}")
    loader = loader_mod.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)

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
    if any("stage" in trial for trial in successes):
        field_names.insert(1, "stage")
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
            if "stage" in field_names:
                row["stage"] = trial.get("stage")
            for key in param_keys:
                row[key] = trial["params"].get(key)
            writer.writerow(row)


def run_tuning(preset: TuningPreset) -> None:
    cli = build_cli(preset).parse_args()

    config_sources: List[str] = []
    # Apply defaults first so explicit model configs override them.
    config_sources.extend(cli.config_dir)
    config_sources.extend(cli.config)
    if not config_sources:
        config_sources = _default_config_chain(preset.model_name, preset.default_config)

    base_args = file_parser.parse_args_from_yaml(config_sources)
    if getattr(base_args, "model", preset.model_name) != preset.model_name:
        base_args.model = preset.model_name

    constant_overrides = parse_override_specs(cli.override, base_args, preset.type_hints)

    if cli.grid:
        search_space = parse_grid_specs(cli.grid, base_args, preset.type_hints)
    elif preset.grid_factory is not None:
        search_space = preset.grid_factory(base_args)
    elif preset.default_grid is not None:
        search_space = {key: values[:] for key, values in preset.default_grid.items()}
    else:
        search_space = {}

    for key in list(search_space.keys()):
        if key in constant_overrides:
            del search_space[key]

    tune_only = parse_tune_only(cli.tune_only)
    if tune_only:
        missing = [key for key in tune_only if key not in search_space]
        if missing:
            raise ValueError(
                "Tune-only parameter(s) not found in the search space: "
                + ", ".join(missing)
            )
        search_space = {key: search_space[key] for key in tune_only}

    lr_keys = parse_lr_keys(cli.lr_key)
    lr_first = bool(cli.lr_first)
    lr_space = {key: search_space[key] for key in lr_keys if key in search_space}

    if lr_first and not lr_space:
        print(
            "LR-first tuning requested but no learning-rate keys exist in the search space."
            " Proceeding with the full grid."
        )
        lr_first = False

    full_search_space = {key: values[:] for key, values in search_space.items()}
    trials = expand_trials(search_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle)

    if cli.dry_run:
        print("Planned trials (dry-run):")
        if lr_first:
            lr_trials = expand_trials(lr_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle)
            rest_space = {k: v for k, v in search_space.items() if k not in lr_space}
            rest_trials = expand_trials(rest_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle)
            total_trials = len(lr_trials) + len(rest_trials)
            for idx, tr in enumerate(lr_trials):
                merged = dict(constant_overrides)
                merged.update(tr)
                print(f"  [lr] #{idx:03d}: {merged}")
            offset = len(lr_trials)
            for idx, tr in enumerate(rest_trials):
                merged = dict(constant_overrides)
                merged.update(tr)
                print(f"  [rest] #{idx + offset:03d}: {merged}")
            print(f"Total: {total_trials} trials")
        else:
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

    def run_trials(
        trial_list: List[Dict[str, Any]],
        overrides: Dict[str, Any],
        stage: str | None,
        start_idx: int,
    ) -> List[Dict[str, Any]]:
        stage_results: List[Dict[str, Any]] = []
        for offset, trial_params in enumerate(trial_list):
            trial_idx = start_idx + offset
            try:
                outcome = run_single_trial(
                    base_args,
                    overrides,
                    trial_params,
                    trial_idx,
                    session_timestamp,
                    runs_root,
                    cli.seed_offset,
                    cli.vary_seed,
                    cli.keep_expt_name,
                    preset.model_name,
                )
            except Exception as exc:  # pylint: disable=broad-except
                trace = traceback.format_exc()
                outcome = {
                    "status": "failed",
                    "trial": trial_idx,
                    "params": dict(overrides, **trial_params),
                    "error": str(exc),
                    "traceback": trace,
                }
                if stage:
                    outcome["stage"] = stage
                print(f"Trial {trial_idx} failed: {exc}")
            else:
                if stage:
                    outcome["stage"] = stage
                    print(
                        f"[{stage}] Trial {trial_idx} finished | score={outcome['score']:.4f} |"
                        f" params={trial_params}"
                    )
                else:
                    print(
                        f"Trial {trial_idx} finished | score={outcome['score']:.4f} |"
                        f" params={trial_params}"
                    )
            stage_results.append(outcome)
        return stage_results

    lr_first_best: Dict[str, Any] | None = None
    if lr_first:
        lr_trials = expand_trials(lr_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle)
        results.extend(run_trials(lr_trials, constant_overrides, "lr", 0))
        lr_successes = [r for r in results if r.get("status") == "ok" and r.get("stage") == "lr"]
        lr_best = max(lr_successes, key=lambda r: r["score"]) if lr_successes else None
        if lr_best:
            lr_first_best = {key: lr_best["trial_params"].get(key) for key in lr_space}
            stage2_overrides = dict(constant_overrides, **lr_first_best)
            rest_space = {k: v for k, v in search_space.items() if k not in lr_space}
            rest_trials = expand_trials(rest_space, cli.num_samples, cli.search_seed, cli.max_trials, cli.shuffle)
            results.extend(run_trials(rest_trials, stage2_overrides, "rest", len(lr_trials)))
        else:
            print("LR-first stage recorded no successful trials; skipping remaining parameters.")
    else:
        results.extend(run_trials(trials, constant_overrides, None, 0))

    successes = [r for r in results if r.get("status") == "ok"]
    best = max(successes, key=lambda r: r["score"]) if successes else None

    resolved_chain = [str(Path(path).resolve()) for path in config_sources]
    summary = {
        "config": resolved_chain[-1] if resolved_chain else None,
        "config_chain": resolved_chain,
        "base_expt_name": getattr(base_args, "expt_name", preset.model_name),
        "session_dir": str(session_dir.resolve()),
        "timestamp": session_timestamp,
        "fixed_overrides": constant_overrides,
        "search_space": full_search_space,
        "lr_first": lr_first,
        "lr_first_keys": lr_keys if lr_first else None,
        "lr_first_best": lr_first_best,
        "num_trials": len(results),
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


def make_main(preset: TuningPreset) -> Callable[[], None]:
    def _main() -> None:
        run_tuning(preset)

    return _main


__all__ = [
    "Grid",
    "TypeHints",
    "TuningPreset",
    "build_cli",
    "run_tuning",
    "make_main",
]
