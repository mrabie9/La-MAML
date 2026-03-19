#!/usr/bin/env python3
"""Backfill task_order.txt for existing runs using saved results.pt bundles.

This script is intended for runs that were created before ``task_order.txt``
was introduced.  For each requested algorithm, it:

* finds the latest run directory under ``logs/{algo}/`` containing
  ``results.pt``,
* loads the saved ``args`` from that bundle,
* reconstructs the corresponding ``IncrementalLoader``, and
* writes a ``metrics/task_order.txt`` file listing task names in order.

Task names are taken from ``loader.task_names`` when available; otherwise they
fall back to simple ``task{idx}`` labels.

Usage
-----
Process one or more algorithms, using the default ``logs`` root and the latest
run per algorithm:

    python scripts/backfill_task_order_from_results.py --algos rwalk ucl

Process the latest three runs per algorithm:

    python scripts/backfill_task_order_from_results.py --algos rwalk ucl --n-latest 3

Specify a different logs root:

    python scripts/backfill_task_order_from_results.py \\
        --logs-root logs/ctn \\
        --algos rwalk ucl

By default, existing ``task_order.txt`` files are not overwritten.  To force
re-generation:

    python scripts/backfill_task_order_from_results.py --algos rwalk --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import torch
import numpy as np

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - older torch
    add_safe_globals = None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line options namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root directory containing logs/{algo}/ subdirectories (default: logs).",
    )
    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        metavar="ALGO",
        required=True,
        help="Algorithms to process (subdirectories under logs_root).",
    )
    parser.add_argument(
        "--n-latest",
        type=int,
        default=1,
        help="Number of latest runs per algorithm to process (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing task_order.txt instead of skipping.",
    )
    return parser.parse_args()


def _iter_results_for_algo(algo: str, logs_root: Path) -> Iterable[Path]:
    """Yield all results.pt files for a given algorithm under logs_root.

    Args:
        algo: Algorithm name (subdirectory under logs_root).
        logs_root: Root logs directory.

    Yields:
        Paths to results.pt files.
    """
    algo_root = logs_root / algo
    if not algo_root.exists():
        return
    for results_file in algo_root.rglob("results.pt"):
        if results_file.is_file():
            yield results_file.resolve()


def _find_latest_results_for_algo(
    algo: str, logs_root: Path, n_latest: int
) -> List[Path]:
    """Return paths to the latest results.pt files for an algorithm.

    Determined by the modification time of the parent run directory.

    Args:
        algo: Algorithm name.
        logs_root: Root logs directory.

    Returns:
        List of up to ``n_latest`` paths to results.pt files, sorted newest first.
    """
    candidates: List[Tuple[float, Path]] = []
    for results_path in _iter_results_for_algo(algo, logs_root):
        run_dir = results_path.parent
        try:
            mtime = run_dir.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, results_path))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [path for (_mtime, path) in candidates[: max(1, n_latest)]]


def _load_results_bundle(results_path: Path):
    """Load a results.pt bundle from disk.

    Args:
        results_path: Path to the results.pt file.

    Returns:
        The deserialized bundle object.
    """
    if add_safe_globals is not None:
        # Allow argparse.Namespace objects in saved args.
        add_safe_globals([argparse.Namespace])

    try:
        bundle = torch.load(results_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(results_path, map_location="cpu")
    return bundle


def _unpack_results_bundle(bundle):
    """Unpack a results.pt bundle into its components.

    Expects a tuple ``(result_val_t, result_val_a, state_dict, val_stats,
    one_liner, saved_args)``.

    Args:
        bundle: Deserialized results.pt bundle.

    Returns:
        Tuple ``(result_val_t, result_val_a, state_dict, val_stats, one_liner, saved_args)``.

    Raises:
        RuntimeError: If the bundle has an unexpected structure.
    """
    try:
        result_val_t, result_val_a, state_dict, val_stats, one_liner, saved_args = (
            bundle
        )
    except ValueError as exc:
        raise RuntimeError(
            "Unexpected checkpoint structure; cannot unpack results.pt"
        ) from exc
    return result_val_t, result_val_a, state_dict, val_stats, one_liner, saved_args


def _extract_loader_from_args(saved_args: object):
    """Return the serialized IncrementalLoader instance from saved args, if any.

    Historical checkpoints attach the loader instance indirectly via a bound
    method on ``args`` (e.g. ``args.get_samples_per_task``).  We use that
    reference so we can recover the original per-task names based on filenames.
    """
    bound = getattr(saved_args, "get_samples_per_task", None)
    loader = getattr(bound, "__self__", None) if bound is not None else None
    return loader


def _derive_task_names_from_loader(loader) -> List[str]:
    """Derive human-readable task names from a loader instance if available.

    If the loader exposes ``task_names`` we use those directly. For historical
    runs where only raw datasets were stored, we infer a dataset label per task
    (e.g. ``rcn``, ``deeprad``, ``uclresm``) from the data shape and sample
    count, then build names such as ``rcn_task0``.
    """
    task_names: Sequence[str] | None = getattr(loader, "task_names", None)
    if task_names:
        return [str(name) for name in task_names]

    n_tasks = int(getattr(loader, "n_tasks", 0))

    def _infer_label_for_task(task_idx: int) -> str:
        try:
            x_train = loader.train_dataset[task_idx][1]
        except Exception:
            return "task"

        # Normalise to numpy for easier shape handling.
        if torch.is_tensor(x_train):
            arr = x_train.detach().cpu().numpy()
        else:
            arr = np.asarray(x_train)

        num_samples = int(arr.shape[0]) if arr.ndim >= 1 else int(arr.size)

        # Heuristic: uclresm has 3 channels, rcn has >=30k samples, deeprad less.
        num_channels = None
        if arr.ndim >= 2:
            num_channels = int(arr.shape[1])

        if num_channels == 3:
            return "uclresm"
        if num_samples >= 30000:
            return "rcn"
        return "deeprad"

    names: List[str] = []
    for idx in range(n_tasks):
        label = _infer_label_for_task(idx)
        names.append(f"{label}_task{idx}")
    return names


def _infer_num_tasks_from_results(result_val_t, result_val_a) -> int:
    """Infer the number of tasks from validation results stored in results.pt.

    This inspects the per-task validation history and returns the number of
    tasks present at the final checkpoint.
    """
    import torch as _torch  # local import to avoid polluting module namespace

    # Prefer result_val_a, which stores accuracy history.
    candidate = result_val_a if result_val_a is not None else result_val_t

    if candidate is None:
        return 0

    # Tensor case: assume last dimension indexes tasks.
    if _torch.is_tensor(candidate):
        if candidate.ndim >= 1:
            return int(candidate.shape[-1])
        return int(candidate.numel())

    # Sequence (e.g. list of per-checkpoint arrays/lists).
    if isinstance(candidate, (list, tuple)) and candidate:
        last = candidate[-1]
        if _torch.is_tensor(last):
            return int(last.numel())
        if isinstance(last, (list, tuple)):
            return len(last)

    return 0


def _derive_task_names_from_results(result_val_t, result_val_a) -> List[str]:
    """Derive generic task names from results.pt contents.

    The historical checkpoints do not store human-readable task labels, only
    per-task metrics.  For those runs we fall back to ``task0``, ``task1``,
    ... based on the inferred number of tasks.
    """
    n_tasks = _infer_num_tasks_from_results(result_val_t, result_val_a)
    return [f"task{idx}" for idx in range(n_tasks)]


def _write_task_order(
    metrics_dir: Path, task_names: Sequence[str], overwrite: bool
) -> None:
    """Write task_order.txt in a metrics directory.

    Args:
        metrics_dir: Directory containing task*.npz files.
        task_names: Ordered list of task names.
        overwrite: Whether to overwrite an existing file.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    order_file = metrics_dir / "task_order.txt"
    if order_file.exists() and not overwrite:
        print(f"  Skipping existing {order_file} (use --overwrite to replace).")
        return

    with order_file.open("w", encoding="utf-8") as f:
        for name in task_names:
            f.write(str(name) + "\n")
    print(f"  Wrote {order_file} with {len(task_names)} task(s).")


def process_algo(algo: str, logs_root: Path, overwrite: bool, n_latest: int) -> None:
    """Process a single algorithm: locate latest runs and backfill task_order.txt.

    Args:
        algo: Algorithm name.
        logs_root: Root logs directory.
        overwrite: Whether to overwrite existing task_order.txt.
        n_latest: Number of latest runs to process.
    """
    results_files = _find_latest_results_for_algo(algo, logs_root, n_latest=n_latest)
    if not results_files:
        print(f"[{algo}] No results.pt found under {logs_root / algo}")
        return

    for results_path in results_files:
        run_dir = results_path.parent
        metrics_dir = run_dir / "metrics"
        print(f"[{algo}] Run: {run_dir}")

        if not metrics_dir.is_dir():
            print(f"  No metrics directory at {metrics_dir}, skipping.")
            continue

        try:
            bundle = _load_results_bundle(results_path)
            (
                result_val_t,
                result_val_a,
                _state_dict,
                _val_stats,
                _one_liner,
                saved_args,
            ) = _unpack_results_bundle(bundle)

            # Prefer the serialized loader (via bound method on args) so that we
            # can recover original task/file names. Fall back to generic names
            # inferred from the results tensors if the loader is unavailable.
            loader = _extract_loader_from_args(saved_args)
            if loader is not None:
                task_names = _derive_task_names_from_loader(loader)
            else:
                task_names = _derive_task_names_from_results(result_val_t, result_val_a)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  Failed to derive task order from {results_path}: {exc}")
            continue

        _write_task_order(metrics_dir, task_names, overwrite=overwrite)


def main() -> int:
    """Entry point for the backfill script."""
    opts = _parse_args()
    logs_root: Path = opts.logs_root

    if not logs_root.exists():
        print(f"Logs root not found: {logs_root}", file=sys.stderr)
        return 1

    for algo in opts.algos:
        process_algo(
            algo,
            logs_root,
            overwrite=opts.overwrite,
            n_latest=max(1, int(opts.n_latest)),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
