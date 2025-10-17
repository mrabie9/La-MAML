"""Utility to consolidate split task datasets into single ``.npz`` files.

The previous helper only handled two hard coded files.  This version walks a
set of data roots (``data/`` by default), discovers directories whose names
match ``task{number}``, and merges any ``.npz`` shards found inside each task
directory into one task level archive (``task{number}.npz`` in the parent
folder).

Each split file (for example ``train.npz`` or ``test.npz``) is loaded and the
arrays it contains are renamed using a consistent ``<name>_<split>`` pattern.
For instance ``train.npz`` with keys ``X`` and ``y`` becomes ``X_train`` and
``y_train`` in the consolidated file.  Extra metadata keys are namespaced in
the same way, so collisions between splits are avoided.

Usage examples
--------------

    # Combine every task under data/rff
    python utils/combine_npz.py data/rff

    # Combine multiple roots and overwrite any existing taskX.npz files
    python utils/combine_npz.py data/rff data/custom --overwrite
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping

import numpy as np


# Match directories such as ``task0`` or ``Task12``.
TASK_DIR_PATTERN = re.compile(r"task\d+", re.IGNORECASE)

# Normalise common split names to a consistent suffix used in the outputs.
SPLIT_ALIASES: Mapping[str, str] = {
    "train": "train",
    "training": "train",
    "test": "test",
    "testing": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "eval": "val",
}

# Canonical representations for frequently used array keys.
KEY_ALIASES: Mapping[str, str] = {
    "x": "X",
    "xtr": "X",
    "xte": "X",
    "x_train": "X_train",
    "x_test": "X_test",
    "xval": "X_val",
    "x_valid": "X_val",
    "x_validation": "X_val",
    "y": "y",
    "ytr": "y",
    "yte": "y",
    "y_train": "y_train",
    "y_test": "y_test",
    "yval": "y_val",
    "y_valid": "y_val",
    "y_validation": "y_val",
    "labels": "y",
    "label": "y",
}


def normalise_split_name(name: str) -> str:
    """Return a lower case, canonical split alias."""

    return SPLIT_ALIASES.get(name.lower(), name.lower())


def canonical_key(name: str) -> str:
    """Map a key to its canonical representation if we recognise it."""

    return KEY_ALIASES.get(name.lower(), name)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load an ``.npz`` archive into memory, returning a plain dictionary."""

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_output_key(key: str, split: str) -> str:
    """Derive the output key name for a given array and split."""

    canonical = canonical_key(key)
    split_alias = normalise_split_name(split) if split else ""

    if canonical.lower().startswith("x_") or canonical.lower().startswith("y_"):
        base = canonical
    elif canonical == "X":
        base = "X"
    elif canonical == "y":
        base = "y"
    else:
        base = canonical

    if not split_alias:
        return base

    # Avoid duplicating suffixes if the key already encodes the split.
    if base.lower().endswith(f"_{split_alias}"):
        return base

    return f"{base}_{split_alias}"


def combine_task_directory(task_dir: Path) -> Dict[str, np.ndarray]:
    """Merge all ``.npz`` files found inside ``task_dir``."""

    combined: Dict[str, np.ndarray] = {}

    for npz_file in sorted(task_dir.glob("*.npz")):
        split = npz_file.stem
        split_alias = normalise_split_name(split)
        arrays = load_npz(npz_file)

        for key, array in arrays.items():
            out_key = build_output_key(key, split_alias)

            if out_key in combined:
                raise ValueError(
                    f"Duplicate key '{out_key}' derived from {npz_file} inside {task_dir}"
                )

            combined[out_key] = array

    return combined


def discover_task_directories(roots: Iterable[Path]) -> Iterator[Path]:
    """Yield every directory matching the ``task{number}`` pattern."""

    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("task*"):
            if candidate.is_dir() and TASK_DIR_PATTERN.fullmatch(candidate.name):
                yield candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=["data"],
        help="One or more directories to scan for task folders (default: data)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate task archives even if the consolidated file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the work that would be performed without writing any files.",
    )
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        help="Store outputs without compression (defaults to compressed archives).",
    )
    parser.set_defaults(compress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    roots = [Path(r).resolve() for r in args.roots]
    task_dirs = list(discover_task_directories(roots))

    if not task_dirs:
        print("No task directories found.")
        return

    total_written = 0
    for task_dir in task_dirs:
        output_path = task_dir.parent / f"{task_dir.name}.npz"

        if output_path.exists() and not args.overwrite:
            print(f"Skipping {output_path} (already exists).")
            continue

        sources = sorted(task_dir.glob("*.npz"))

        if not sources:
            print(f"No split files found in {task_dir}; skipping.")
            continue

        if args.dry_run:
            print(f"Would write {output_path} from {[p.name for p in sources]}")
            continue

        combined = combine_task_directory(task_dir)

        if not combined:
            print(f"No arrays found in {task_dir}; skipping.")
            continue

        save_fn = np.savez_compressed if args.compress else np.savez
        save_fn(output_path, **combined)
        total_written += 1

        rel_sources = ", ".join(p.name for p in sources)
        print(f"Wrote {output_path} ({rel_sources}).")

    if not args.dry_run:
        print(f"Finished combining {total_written} task archives.")


if __name__ == "__main__":
    main()
