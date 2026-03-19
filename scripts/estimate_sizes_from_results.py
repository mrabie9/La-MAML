#!/usr/bin/env python3
"""Compute model and memory-buffer sizes from saved results.pt (e.g. from runs before sizes were logged).

Loads each results.pt bundle, rebuilds the model using the stored args and loader
to get dataset dimensions, then reports model size and memory buffer size in the
same format as save_results in main.py. Use this for runs that were saved before
the change that added size logging to save_results.

Examples
--------
Report sizes for a single run:

    python scripts/estimate_sizes_from_results.py --run logs/my_experiment/0

Scan all runs under logs/:

    python scripts/estimate_sizes_from_results.py --logs-root logs

Filter by path substring:

    python scripts/estimate_sizes_from_results.py --logs-root logs --pattern gem
"""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
from pathlib import Path
from typing import Iterable
import pickle

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from main import estimate_memory_buffer_size_bytes

try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None


def parse_cli() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root directory to search for results.pt files",
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Single run: path to a results.pt file or to a directory containing results.pt",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Only process paths containing this substring",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Also print an approximate size breakdown of the results.pt contents.",
    )
    return parser.parse_args()


def iter_result_files(opts: argparse.Namespace) -> Iterable[Path]:
    """Yield paths to results.pt files to process."""
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


def build_args_for_sizes(saved_args: object, run_dir: Path) -> object:
    """Build an args object suitable for instantiating loader and model."""
    args = copy.deepcopy(saved_args)
    args.log_dir = str(run_dir)
    # Prefer CPU so we don't require GPU or move data to device
    args.cuda = False
    return args


def instantiate_model(args: object) -> object:
    """Build the model using saved args and loader to get dataset dimensions."""
    LoaderModule = importlib.import_module(f"dataloaders.{args.loader}")
    loader = LoaderModule.IncrementalLoader(args, seed=getattr(args, "seed", 0))
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()

    ModelModule = importlib.import_module(f"model.{args.model}")
    model = ModelModule.Net(n_inputs, n_outputs, n_tasks, args)
    return model


def _load_results_bundle(results_path: Path):
    """Load a results.pt bundle from disk."""
    if add_safe_globals is not None:
        add_safe_globals([argparse.Namespace])

    try:
        bundle = torch.load(results_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(results_path, map_location="cpu")
    return bundle


def _tensor_storage_bytes(obj) -> int:
    """Recursively sum tensor storage (in bytes) for a nested structure."""
    if torch.is_tensor(obj):
        return obj.element_size() * obj.numel()
    if isinstance(obj, dict):
        return sum(_tensor_storage_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_storage_bytes(v) for v in obj)
    return 0


def describe_checkpoint_sizes(results_path: Path, bundle) -> list[tuple[str, float]]:
    """Return a human-readable size breakdown for results.pt contents.

    Sizes are reported in megabytes and focus on tensor-heavy components.
    """
    try:
        result_val_t, result_val_a, state_dict, val_stats, one_liner, saved_args = (
            bundle
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Unexpected checkpoint structure at {results_path}"
        ) from exc

    entries: list[tuple[str, float]] = []
    entries.append(
        ("result_val_t (tensors)", _tensor_storage_bytes(result_val_t) / (1024**2))
    )
    entries.append(
        ("result_val_a (tensors)", _tensor_storage_bytes(result_val_a) / (1024**2))
    )
    entries.append(
        ("state_dict (tensors)", _tensor_storage_bytes(state_dict) / (1024**2))
    )
    entries.append(
        ("val_stats (tensors)", _tensor_storage_bytes(val_stats) / (1024**2))
    )

    # Rough pickle-based sizes for non-tensor payloads.
    try:
        one_liner_mb = len(pickle.dumps(one_liner)) / (1024**2)
    except Exception:
        one_liner_mb = 0.0
    try:
        args_mb = len(pickle.dumps(saved_args)) / (1024**2)
    except Exception:
        args_mb = 0.0
    entries.append(("one_liner (serialized)", one_liner_mb))
    entries.append(("args (serialized)", args_mb))
    return entries


def compute_sizes(results_path: Path, bundle=None) -> tuple[float, float]:
    """Rebuild model and return (model_size_gb, memory_buffer_size_gb)."""
    if bundle is None:
        bundle = _load_results_bundle(results_path)
    try:
        (
            _result_val_t,
            _result_val_a,
            _state_dict,
            _val_stats,
            _one_liner,
            saved_args,
        ) = bundle
    except ValueError as exc:
        raise RuntimeError(
            f"Unexpected checkpoint structure at {results_path}"
        ) from exc

    run_dir = results_path.parent
    args = build_args_for_sizes(saved_args, run_dir)
    model = instantiate_model(args)

    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_gb = size_bytes / (1024**3)
    buffer_bytes = estimate_memory_buffer_size_bytes(model)
    buffer_gb = buffer_bytes / (1024**3)
    return size_gb, buffer_gb


def main() -> int:
    opts = parse_cli()
    results_list = list(iter_result_files(opts))
    if not results_list:
        print("No results.pt files found.", file=sys.stderr)
        return 1

    for results_path in results_list:
        try:
            bundle = _load_results_bundle(results_path)
            model_gb, mem_gb = compute_sizes(results_path, bundle=bundle)
        except Exception as exc:
            print(f"[FAILED] {results_path}: {exc}", file=sys.stderr)
            continue

        print(f"Run: {results_path.parent}")
        print(f"Model size: {model_gb:.4f} GB")
        print(f"Memory buffer size: {mem_gb:.4f} GB")
        print(f"  # sizes: model_gb={model_gb:.4f} mem_gb={mem_gb:.4f}")

        if opts.breakdown:
            file_mb = results_path.stat().st_size / (1024**2)
            print(f"  results.pt on disk: {file_mb:.2f} MB")
            for label, size_mb in describe_checkpoint_sizes(results_path, bundle):
                print(f"    - {label}: {size_mb:.2f} MB")
            print()
        else:
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
