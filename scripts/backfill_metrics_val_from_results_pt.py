#!/usr/bin/env python3
"""Backfill `metrics/*/task*.npz` `val_acc` from a `results.pt` checkpoint.

The training loop in `main.py` saves per-task metrics under:
`<run_dir>/metrics/task{i}.npz`.

Some runs may have missing or stale `val_acc` vectors in those `.npz`
files. This script reconstructs the validation history from the saved
`results.pt` bundle and then compares/overwrites `val_acc` in the metrics
directory.

Usage
-----
Compare only (no writes):
    python scripts/backfill_metrics_val_from_results_pt.py \
        --results-pt logs/rwalk/some_run/0/results.pt \
        --compare-only

Rebuild mismatched/missing `val_acc`:
    python scripts/backfill_metrics_val_from_results_pt.py \
        --results-pt logs/rwalk/some_run/0/results.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-pt",
        type=Path,
        required=True,
        help="Path to a single results.pt file.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Override metrics directory (default: sibling `metrics/`).",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=None,
        help="Only backfill a single task index (e.g. 0..n_tasks-1).",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Do not write files; only compare and report mismatches.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite val_acc even when arrays already match.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for comparing floats.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for comparing floats.",
    )
    return parser.parse_args()


def _load_results_pt(results_path: Path) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Load a `results.pt` checkpoint bundle.

    Args:
        results_path: Path to the `results.pt` file.

    Returns:
        The loaded checkpoint tuple:
        `(result_val_t, result_val_a, state_dict, val_stats, one_liner, args)`.

    Raises:
        FileNotFoundError: If `results_path` does not exist.
        ValueError: If the checkpoint does not match the expected structure.
    """
    if not results_path.is_file():
        raise FileNotFoundError(f"results.pt not found: {results_path}")
    try:
        bundle = torch.load(results_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(results_path, map_location="cpu")
    if not isinstance(bundle, (tuple, list)) or len(bundle) < 2:
        raise ValueError(
            f"Unexpected results.pt bundle type: {type(bundle)} (len={len(bundle) if hasattr(bundle, '__len__') else 'N/A'})"
        )
    # main.py saves: (result_val_t, result_val_a, state_dict, val_stats, one_liner, args)
    return tuple(bundle)  # type: ignore[return-value]


def _to_numpy_2d(arr: Any) -> np.ndarray:
    """Convert a tensor/list into a 2D numpy float array.

    Args:
        arr: Tensor-like input with shape `(N, M)`.

    Returns:
        A float64 numpy array with shape `(N, M)`.

    Raises:
        ValueError: If `arr` does not have 2 dimensions.
    """
    if torch.is_tensor(arr):
        np_arr = arr.detach().cpu().numpy()
    else:
        np_arr = np.asarray(arr)
    if np_arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {np_arr.shape}")
    return np_arr.astype(np.float64, copy=False)


def _to_numpy_1d_int(arr: Any) -> np.ndarray:
    """Convert a tensor/list into a 1D int numpy array.

    Args:
        arr: Tensor-like input.

    Returns:
        A 1D numpy int64 array.
    """
    if torch.is_tensor(arr):
        np_arr = arr.detach().cpu().numpy()
    else:
        np_arr = np.asarray(arr)
    np_arr = np_arr.reshape(-1)
    # main.py builds result_val_t from a python list of ints, then wraps with torch.Tensor.
    # That might come back as floats depending on torch serialization; round safely.
    return np.rint(np_arr).astype(np.int64, copy=False)


def _iter_task_indices(
    metrics_dir: Path, n_tasks: int, task_idx: int | None
) -> Iterable[int]:
    """Yield which task indices to process.

    Args:
        metrics_dir: Directory containing `task*.npz` files.
        n_tasks: Total number of tasks inferred from `results.pt`.
        task_idx: If provided, backfill only this task index.

    Returns:
        An iterable of task indices to process.
    """
    if task_idx is not None:
        if not (0 <= task_idx < n_tasks):
            raise ValueError(
                f"--task-idx={task_idx} out of range for n_tasks={n_tasks}"
            )
        yield task_idx
        return

    task_files = sorted(
        metrics_dir.glob("task*.npz"), key=lambda p: _task_index(p.name)
    )
    if task_files:
        for task_file in task_files:
            idx = _task_index(task_file.name)
            if idx >= 0:
                yield idx
        return

    for idx in range(n_tasks):
        yield idx


def _task_index(filename: str) -> int:
    """Extract task number from `task{idx}.npz` filename.

    Args:
        filename: Name like `task0.npz` or `task12.npz`.

    Returns:
        The parsed task index, or `-1` if it can't be parsed.
    """
    # Be liberal: accept task0.npz and task12.npz; scripts use case-insensitive.
    import re

    match = re.match(r"task(\d+)\.npz", filename, flags=re.IGNORECASE)
    if not match:
        return -1
    return int(match.group(1))


def _reconstruct_val_acc_for_task(
    result_val_t: np.ndarray,
    result_val_a: np.ndarray,
    task_i: int,
    *,
    exclude_last_final_validation: bool,
) -> np.ndarray:
    """Reconstruct flattened `val_acc` history for metrics/task{task_i}.npz.

    In `main.py`, the metrics file flattens across evaluation snapshots done
    while training task index `task_i`. Each evaluation snapshot returns a
    vector of length (`task_i + 1`) containing validation recall per task
    learned so far, in training-history order.

    Args:
        result_val_t: Per-evaluation task id vector from `results.pt`.
        result_val_a: Per-evaluation validation tensor from `results.pt`,
            padded to a dense 2D array.
        task_i: Task index corresponding to `metrics/task{task_i}.npz`.
        exclude_last_final_validation: If True, drop the last evaluation row
            for this task (continual-learning runs). If False, keep all rows
            (single-round `iid2` runs).

    Returns:
        Reconstructed flattened `val_acc` vector for the task metrics file.
    """
    if result_val_a.shape[1] <= task_i:
        raise ValueError(
            f"result_val_a has max_tasks={result_val_a.shape[1]}, cannot index task {task_i}"
        )

    # `result_val_t` is the current task id used in the training loop. In this
    # loader it matches the training step index, which is also used in the
    # metrics file numbering.
    selected_rows = np.nonzero(result_val_t == task_i)[0]
    if selected_rows.size == 0:
        return np.asarray([], dtype=np.float64)

    # For continual-learning runs in `main.py`, `metrics/task{task_i}.npz`
    # stores only the validation snapshots that happen *inside* the epoch
    # loop: `if (ep % args.val_rate) == 0: ...`.
    #
    # The "final validation" after the epoch loop is included in `results.pt`
    # but is not appended into the per-task `result_acc_val` list that gets
    # flattened into `metrics/task{task_i}.npz`. In chronological order stored
    # in `results.pt`, this corresponds to the last evaluation row for this
    # task.
    if exclude_last_final_validation:
        selected_rows = selected_rows[:-1]
        if selected_rows.size == 0:
            return np.asarray([], dtype=np.float64)

    # main.py flattens by concatenating each snapshot vector in the same
    # order snapshots were appended during training of task_i.
    vectors = [result_val_a[row_idx, : task_i + 1] for row_idx in selected_rows]
    return np.concatenate(vectors, axis=0).astype(np.float64, copy=False)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load an `.npz` file into a dictionary.

    Args:
        path: Path to the `.npz` file.

    Returns:
        Dictionary mapping keys in the archive to numpy arrays.
    """
    data = np.load(path, allow_pickle=True)
    payload: Dict[str, np.ndarray] = {key: np.asarray(data[key]) for key in data.files}
    return payload


def _save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    """Save a dictionary of arrays into an `.npz` file.

    Args:
        path: Output `.npz` file path.
        payload: Key->array dictionary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> bool:
    """Compare two arrays using `np.allclose`.

    Args:
        a: First array.
        b: Second array.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        True if arrays have identical shapes and match within tolerance.
    """
    if a.shape != b.shape:
        return False
    if a.size == 0 and b.size == 0:
        return True
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def _ensure_minimal_payload(
    existing: Dict[str, np.ndarray] | None, val_acc: np.ndarray
) -> Dict[str, np.ndarray]:
    """Create or update the `.npz` payload for a task.

    Args:
        existing: Existing loaded payload, or None if the file is missing.
        val_acc: Reconstructed flattened validation recall history.

    Returns:
        Payload dictionary to save to `task{i}.npz`.
    """
    if existing is None:
        # Minimal keys so plotting scripts can safely load the dict.
        return {
            "losses": np.asarray([], dtype=np.float64),
            "cls_tr_rec": np.asarray([], dtype=np.float64),
            "val_acc": val_acc,
        }

    updated = dict(existing)
    updated["val_acc"] = val_acc
    return updated


def main() -> int:
    """Run the backfill comparison/extraction.

    Returns:
        Process exit code. `0` means success, non-zero means mismatches in
        `--compare-only` mode.
    """
    args = _parse_args()

    results_path: Path = args.results_pt.resolve()
    if args.metrics_dir is not None:
        metrics_dir = args.metrics_dir.resolve()
    else:
        metrics_dir = results_path.parent / "metrics"

    bundle = _load_results_pt(results_path)
    result_val_t, result_val_a = bundle[0], bundle[1]
    # remaining fields are currently unused, but we keep the unpack for clarity:
    _state_dict, _val_stats, _one_liner, _saved_args = (
        bundle[2],
        bundle[3],
        bundle[4],
        bundle[5],
    )

    result_val_t_int = _to_numpy_1d_int(result_val_t)
    result_val_a_2d = _to_numpy_2d(result_val_a)
    n_tasks = int(result_val_a_2d.shape[1])
    exclude_last_final_validation = getattr(_saved_args, "model", None) != "iid2"

    task_indices = list(
        _iter_task_indices(metrics_dir, n_tasks=n_tasks, task_idx=args.task_idx)
    )
    if not task_indices:
        print("No task indices found to process.", file=sys.stderr)
        return 1

    print(f"Loaded results.pt: {results_path}")
    print(f"Reconstructing val_acc for n_tasks={n_tasks} into {metrics_dir}")

    mismatches = 0
    written = 0
    missing_files = 0

    for task_i in task_indices:
        reconstructed_val_acc = _reconstruct_val_acc_for_task(
            result_val_t=result_val_t_int,
            result_val_a=result_val_a_2d,
            task_i=task_i,
            exclude_last_final_validation=exclude_last_final_validation,
        )

        metrics_path = metrics_dir / f"task{task_i}.npz"
        existing_payload: Dict[str, np.ndarray] | None = None
        existing_val_acc: np.ndarray | None = None
        if metrics_path.exists():
            existing_payload = _load_npz(metrics_path)
            if "val_acc" in existing_payload:
                existing_val_acc = np.asarray(
                    existing_payload["val_acc"], dtype=np.float64
                )
        else:
            missing_files += 1

        needs_write = args.force
        if not args.force:
            if existing_val_acc is None:
                needs_write = True
            else:
                needs_write = not _compare_arrays(
                    existing_val_acc,
                    reconstructed_val_acc,
                    rtol=args.rtol,
                    atol=args.atol,
                )

        ok = existing_val_acc is None or _compare_arrays(
            existing_val_acc, reconstructed_val_acc, rtol=args.rtol, atol=args.atol
        )

        if not ok:
            mismatches += 1

        if needs_write:
            if args.compare_only:
                print(
                    f"[COMPARE-ONLY] task{task_i}: mismatch or missing val_acc "
                    f"(metrics_exists={metrics_path.exists()}, "
                    f"existing_len={None if existing_val_acc is None else existing_val_acc.size}, "
                    f"reconstructed_len={reconstructed_val_acc.size})"
                )
            else:
                payload_to_save = _ensure_minimal_payload(
                    existing_payload, reconstructed_val_acc
                )
                _save_npz(metrics_path, payload_to_save)
                written += 1
                print(
                    f"Wrote task{task_i}.npz val_acc "
                    f"(existing_len={None if existing_val_acc is None else existing_val_acc.size}, "
                    f"reconstructed_len={reconstructed_val_acc.size})"
                )
        else:
            print(f"task{task_i}: val_acc matches (len={reconstructed_val_acc.size})")

    if args.compare_only:
        if mismatches > 0 or missing_files > 0:
            print(
                f"COMPARE ONLY: found {mismatches} mismatch task(s) and {missing_files} missing metric file(s).",
                file=sys.stderr,
            )
            return 2
        print("COMPARE ONLY: all tasks match.")
        return 0

    print(
        f"Done. written={written}, mismatches={mismatches}, missing_files={missing_files}. "
        f"metrics_dir={metrics_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
