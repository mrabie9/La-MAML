#!/usr/bin/env python3
"""Evaluate saved La-MAML checkpoints on the validation split.

This script scans the `logs/` directory (or a user-specified location),
loads each `results.pt` bundle saved during training, rebuilds the corresponding
model and dataloader using the stored training arguments, and reports the
validation accuracies.

Examples
--------
Evaluate every run under the default `logs/` directory:

    python evaluate_logs.py

Evaluate a specific run (directory or direct path to `results.pt`):

    python evaluate_logs.py --run logs/ucl/ucl_loss_test-2025-10-16_16-37-57-7845/0

Force CPU inference and override the evaluation batch size:

    python evaluate_logs.py --device cpu --batch-size 256
"""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

# Newer PyTorch releases enable "weights only" deserialization by default,
# which blocks loading argparse.Namespace objects embedded in our checkpoints.
# Allowlist that class when the helper is available so we can keep using the
# stored Namespace safely for local experiments.
try:  # pragma: no cover - depends on torch runtime
    from torch.serialization import add_safe_globals  # type: ignore
except ImportError:  # pragma: no cover
    add_safe_globals = None

from main import eval_class_tasks, eval_tasks


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root directory containing experiment folders",
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Specific run directory or path to a results.pt file",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Substring filter; only paths containing this value are evaluated",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the validation batch size (per run)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Override the dataset root when rebuilding the dataloader",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort on the first run that fails to evaluate",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        return torch.device("cuda")

    return torch.device("cpu")


def iter_result_files(opts: argparse.Namespace) -> Iterable[Path]:
    if opts.run is not None:
        run_path = opts.run
        if run_path.is_dir():
            run_path = run_path / "results.pt"
        if run_path.suffix != ".pt":
            raise FileNotFoundError(f"Expected a .pt file, got: {run_path}")
        if not run_path.exists():
            raise FileNotFoundError(run_path)
        yield run_path.resolve()
        return

    root = opts.logs_root
    if not root.exists():
        raise FileNotFoundError(f"Logs root not found: {root}")

    for results_file in sorted(root.rglob("results.pt")):
        if opts.pattern and opts.pattern not in str(results_file):
            continue
        yield results_file.resolve()


def build_runtime_args(saved_args, run_dir: Path, device: torch.device, opts: argparse.Namespace):
    args = copy.deepcopy(saved_args)
    # Update paths/device flags for the current environment
    args.log_dir = str(run_dir)
    args.cuda = device.type == "cuda"
    if opts.data_path is not None:
        args.data_path = str(opts.data_path)
    if opts.batch_size is not None:
        args.eval_batch_size = opts.batch_size
    elif not hasattr(args, "eval_batch_size"):
        # Fall back to a reasonable default if it was never set during training
        args.eval_batch_size = getattr(args, "test_batch_size", getattr(args, "batch_size", 64))
    return args


def instantiate_model_and_loader(args, state_dict, device: torch.device):
    LoaderModule = importlib.import_module(f"dataloaders.{args.loader}")
    loader = LoaderModule.IncrementalLoader(args, seed=getattr(args, "seed", 0))
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()

    ModelModule = importlib.import_module(f"model.{args.model}")
    model = ModelModule.Net(n_inputs, n_outputs, n_tasks, args)
    load_state_into_model(model, state_dict)
    model.to(device)
    return model, loader


def evaluate_run(results_path: Path, device: torch.device, opts: argparse.Namespace):
    if add_safe_globals is not None:  # pragma: no branch - lightweight guard
        add_safe_globals([argparse.Namespace])

    try:
        bundle = torch.load(results_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only flag
        bundle = torch.load(results_path, map_location="cpu")
    try:
        (result_val_t, result_val_a, state_dict, _val_stats, _one_liner, saved_args) = bundle
    except ValueError as exc:  # pragma: no cover - defensive unpack
        raise RuntimeError(f"Unexpected checkpoint structure at {results_path}") from exc

    run_dir = results_path.parent
    args = build_runtime_args(saved_args, run_dir, device, opts)
    model, loader = instantiate_model_and_loader(args, state_dict, device)

    # print(model)

    evaluator = eval_class_tasks if args.loader == "class_incremental_loader" else eval_tasks

    with torch.no_grad():
        val_tasks = loader.get_tasks("val")
        task_accuracies: Sequence[float] = evaluator(model, val_tasks, args, eval_epistemic=True)

    overall = sum(task_accuracies) / len(task_accuracies) if task_accuracies else float("nan")
    metadata = {
        "run": str(run_dir),
        "tasks": list(task_accuracies),
        "overall": overall,
    }

    # Reduce the footprint of returned tensors, if any
    if isinstance(result_val_t, torch.Tensor):
        metadata["saved_val_tasks"] = result_val_t.cpu().numpy().tolist()
    if isinstance(result_val_a, torch.Tensor):
        metadata["saved_val_history"] = result_val_a.cpu().numpy().tolist()

    return metadata


def load_state_into_model(model: torch.nn.Module, raw_state_dict):
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in raw_state_dict.items() if k in model_state}
    skipped_keys = sorted(set(raw_state_dict) - set(filtered_state))

    if skipped_keys:
        sample = ", ".join(skipped_keys[:3])
        if len(skipped_keys) > 3:
            sample += ", ..."
        print(
            f"  Skipping {len(skipped_keys)} unexpected key(s) when loading checkpoint: {sample}")

    incompat = model.load_state_dict(filtered_state, strict=False)

    if getattr(incompat, "missing_keys", None):
        sample = ", ".join(incompat.missing_keys[:3])
        if len(incompat.missing_keys) > 3:
            sample += ", ..."
        print(f"  Missing {len(incompat.missing_keys)} key(s) in checkpoint: {sample}")

    if getattr(incompat, "unexpected_keys", None):
        sample = ", ".join(incompat.unexpected_keys[:3])
        if len(incompat.unexpected_keys) > 3:
            sample += ", ..."
        print(f"  Still found {len(incompat.unexpected_keys)} unexpected key(s): {sample}")


def main() -> int:
    opts = parse_cli()
    device = resolve_device(opts.device)

    results: List[Path] = list(iter_result_files(opts))
    if not results:
        print("No results.pt files found with the provided filters", file=sys.stderr)
        return 1

    print(f"Evaluating {len(results)} run(s) on {device}...\n")

    for results_path in results:
        try:
            metrics = evaluate_run(results_path, device, opts)
        except Exception as exc:  # pragma: no cover - robust evaluation loop
            print(f"[FAILED] {results_path}: {exc}")
            if opts.stop_on_error:
                raise
            continue

        run = metrics["run"]
        overall = metrics["overall"]
        per_task = ", ".join(f"{acc:.4f}" for acc in metrics["tasks"])
        print(f"Run: {run}")
        print(f"  Overall val accuracy: {overall:.4f}")
        print(f"  Task-wise accuracies: [{per_task}]\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
