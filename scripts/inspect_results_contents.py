#!/usr/bin/env python3
"""Inspect what is stored inside a results.pt checkpoint.

This script focuses on the `args` object to determine whether a serialized
IncrementalLoader (dataloader) can be recovered, and to summarise what is
inside it.

Usage
-----
Inspect a single checkpoint:

    python scripts/inspect_results_contents.py --results-pt logs/ucl/exp/0/results.pt
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - depends on torch runtime
    from torch.serialization import add_safe_globals  # type: ignore
except ImportError:  # pragma: no cover
    add_safe_globals = None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-pt",
        type=Path,
        required=True,
        help="Path to a results.pt file to inspect.",
    )
    return parser.parse_args()


def _load_bundle(path: Path) -> Any:
    """Load the raw bundle from results.pt."""
    if add_safe_globals is not None:
        # Allow argparse.Namespace and similar objects inside the checkpoint.
        import argparse as _argparse

        add_safe_globals([_argparse.Namespace])

    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location="cpu")
    return bundle


def _unpack_bundle(bundle: Any):
    """Unpack the standard results.pt tuple."""
    try:
        result_val_t, result_val_a, state_dict, val_stats, one_liner, saved_args = (
            bundle
        )
    except ValueError as exc:
        raise RuntimeError(
            "Unexpected checkpoint structure; expected a 6-tuple."
        ) from exc
    return result_val_t, result_val_a, state_dict, val_stats, one_liner, saved_args


def _sizeof_mb(obj: Any) -> float:
    """Approximate serialized size of an object in megabytes."""
    try:
        return len(pickle.dumps(obj)) / (1024**2)
    except Exception:
        return 0.0


def inspect_results(results_path: Path) -> None:
    """Inspect a single results.pt file and print a summary."""
    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    print(f"Inspecting: {results_path}")
    print(f"On disk size: {results_path.stat().st_size / (1024**2):.2f} MB\n")

    bundle = _load_bundle(results_path)
    result_val_t, result_val_a, state_dict, val_stats, one_liner, saved_args = (
        _unpack_bundle(bundle)
    )

    print("Bundle components:")
    print(f"  result_val_t: type={type(result_val_t)}")
    print(f"  result_val_a: type={type(result_val_a)}")
    print(
        f"  state_dict:   type={type(state_dict)}, n_keys={len(state_dict) if isinstance(state_dict, dict) else 'N/A'}"
    )
    print(f"  val_stats:    type={type(val_stats)}")
    print(f"  one_liner:    type={type(one_liner)}")
    print(f"  args:         type={type(saved_args)}\n")

    print("Approximate serialized sizes (MB):")
    print(
        f"  state_dict (tensors only): {_tensor_storage_bytes(state_dict) / (1024**2):.2f}"
    )
    print(
        f"  val_stats (tensors only):  {_tensor_storage_bytes(val_stats) / (1024**2):.2f}"
    )
    print(f"  one_liner (pickled):       {_sizeof_mb(one_liner):.2f}")
    print(f"  args (pickled):            {_sizeof_mb(saved_args):.2f}\n")

    # Try to recover a serialized IncrementalLoader from args via the bound method.
    bound = getattr(saved_args, "get_samples_per_task", None)
    loader = getattr(bound, "__self__", None) if bound is not None else None

    print("Dataloader recovery:")
    if loader is None:
        print("  No loader instance found via args.get_samples_per_task.")
    else:
        print(f"  Recovered loader instance: type={type(loader)}")
        task_names = getattr(loader, "task_names", None)
        n_tasks = getattr(loader, "n_tasks", None)
        print(f"  loader.n_tasks:    {n_tasks}")
        if task_names is not None:
            print(f"  loader.task_names: {list(task_names)}")
        else:
            print("  loader.task_names: <missing>")

        # Show how much of args' pickle size is attributable to the loader.
        print("\n  Size breakdown within args (approximate, MB):")
        loader_mb = _sizeof_mb(loader)
        args_mb = _sizeof_mb(saved_args)
        print(f"    loader (pickled): {loader_mb:.2f}")
        print(f"    args total:        {args_mb:.2f}")
        print(f"    args minus loader: {max(args_mb - loader_mb, 0.0):.2f}")


def _tensor_storage_bytes(obj: Any) -> int:
    """Recursively sum tensor storage (in bytes) for a nested structure."""
    if torch.is_tensor(obj):
        return obj.element_size() * obj.numel()
    if isinstance(obj, dict):
        return sum(_tensor_storage_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_storage_bytes(v) for v in obj)
    return 0


def main() -> int:
    """Entry point."""
    opts = _parse_args()
    try:
        inspect_results(opts.results_pt)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
