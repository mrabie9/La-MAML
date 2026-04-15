#!/usr/bin/env python
"""Move model-specific job output directories under a global run directory.

This utility inspects `job_logs/job_*.log` under a provided full-experiments run
directory, extracts each job's `Logging to ...` path, and moves those job output
directories into `<run_dir>/model-specific_logs/`.

Usage:
    # Preview planned moves only.
    python scripts/move_model_specific_logs.py \
        --run-dir logs/full_experiments/full-til_10epochs_w-zs/full-til_A_run_20260403_111257_lnx-elkk-2 \
        --dry-run

    # Perform moves.
    python scripts/move_model_specific_logs.py \
        --run-dir logs/full_experiments/full-til_10epochs_w-zs/full-til_A_run_20260403_111257_lnx-elkk-2

    # Overwrite existing destination directories if needed.
    python scripts/move_model_specific_logs.py \
        --run-dir logs/full_experiments/full-til_10epochs_w-zs/full-til_A_run_20260403_111257_lnx-elkk-2 \
        --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LOGGING_TO_RE = re.compile(r"Logging to\s+(?P<path>\S+)")


@dataclass(frozen=True)
class MoveCandidate:
    """Container for a planned source -> destination directory move.

    Attributes:
        source_directory: Existing model-specific directory to move.
        destination_directory: Target path under model-specific_logs.
    """

    source_directory: Path
    destination_directory: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Move model-specific job output directories into "
            "<run_dir>/model-specific_logs."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Full-experiments run directory containing job_logs/job_*.log.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned move operations without changing the filesystem.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination directory when it already exists.",
    )
    return parser.parse_args()


def extract_logged_directory(job_log_path: Path) -> Optional[str]:
    """Extract the last `Logging to ...` directory from a job log.

    Args:
        job_log_path: Path to a single `job_*.log` file.

    Returns:
        The extracted directory string or None when unavailable.
    """
    extracted_directory: Optional[str] = None
    with job_log_path.open("r", encoding="utf-8", errors="replace") as file_handle:
        for line_text in file_handle:
            logging_match = LOGGING_TO_RE.search(line_text)
            if logging_match:
                extracted_directory = logging_match.group("path").strip()
    return extracted_directory


def resolve_logged_directory(
    raw_logged_directory: str,
    repository_root: Path,
    job_log_path: Path,
) -> Optional[Path]:
    """Resolve a raw log directory string to an absolute path.

    Args:
        raw_logged_directory: Directory string parsed from a job log.
        repository_root: Repository root directory.
        job_log_path: Path to the job log that reported the directory.

    Returns:
        The normalized absolute directory path when resolvable, else None.
    """
    normalized_raw_path = raw_logged_directory.strip().replace("//", "/")
    parsed_path = Path(normalized_raw_path)
    candidate_paths: List[Path] = []

    if parsed_path.is_absolute():
        candidate_paths.append(parsed_path.resolve())
    else:
        candidate_paths.append((repository_root / parsed_path).resolve())
        candidate_paths.append((job_log_path.resolve().parent / parsed_path).resolve())

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    return None


def is_subpath(child_path: Path, parent_path: Path) -> bool:
    """Check whether `child_path` is equal to or within `parent_path`.

    Args:
        child_path: Candidate child path.
        parent_path: Candidate parent path.

    Returns:
        True when `child_path` is inside `parent_path`, otherwise False.
    """
    try:
        child_path.relative_to(parent_path)
        return True
    except ValueError:
        return False


def discover_move_candidates(
    run_directory: Path,
    destination_root: Path,
    repository_root: Path,
) -> Tuple[List[MoveCandidate], Dict[str, int]]:
    """Discover unique source directories to move for a run.

    Args:
        run_directory: Full global run directory.
        destination_root: `<run_dir>/model-specific_logs` target directory.
        repository_root: Repository root directory.

    Returns:
        Tuple of:
            - ordered list of unique move candidates
            - stats dictionary for discovery diagnostics
    """
    job_logs_directory = run_directory / "job_logs"
    if not job_logs_directory.is_dir():
        raise SystemExit(f"Missing job_logs directory: {job_logs_directory}")

    unique_candidates: Dict[Path, MoveCandidate] = {}
    discovery_stats = {
        "job_logs_total": 0,
        "missing_logging_to": 0,
        "missing_source": 0,
        "invalid_source": 0,
        "deduplicated_sources": 0,
    }
    repository_logs_root = (repository_root / "logs").resolve()

    for job_log_path in sorted(job_logs_directory.glob("job_*.log")):
        discovery_stats["job_logs_total"] += 1
        raw_logged_directory = extract_logged_directory(job_log_path)
        if raw_logged_directory is None:
            discovery_stats["missing_logging_to"] += 1
            continue

        source_directory = resolve_logged_directory(
            raw_logged_directory=raw_logged_directory,
            repository_root=repository_root,
            job_log_path=job_log_path,
        )
        if source_directory is None:
            discovery_stats["missing_source"] += 1
            continue
        if not source_directory.is_dir():
            discovery_stats["missing_source"] += 1
            continue

        resolved_run_directory = run_directory.resolve()
        resolved_destination_root = destination_root.resolve()
        if source_directory == resolved_run_directory or is_subpath(
            source_directory, resolved_destination_root
        ):
            discovery_stats["invalid_source"] += 1
            continue

        try:
            relative_source_path = source_directory.relative_to(repository_logs_root)
            destination_directory = resolved_destination_root / relative_source_path
        except ValueError:
            destination_directory = (
                resolved_destination_root
                / f"{source_directory.parent.name}_{source_directory.name}"
            )
        if source_directory in unique_candidates:
            discovery_stats["deduplicated_sources"] += 1
            continue
        unique_candidates[source_directory] = MoveCandidate(
            source_directory=source_directory,
            destination_directory=destination_directory,
        )

    return list(unique_candidates.values()), discovery_stats


def execute_move_plan(
    candidates: Sequence[MoveCandidate],
    destination_root: Path,
    dry_run: bool,
    overwrite: bool,
) -> Dict[str, int]:
    """Execute (or preview) move operations.

    Args:
        candidates: Planned unique move operations.
        destination_root: `<run_dir>/model-specific_logs` directory.
        dry_run: Whether to preview operations only.
        overwrite: Whether to replace existing destination directories.

    Returns:
        Dictionary with outcome counters.
    """
    move_stats = {
        "planned": len(candidates),
        "moved": 0,
        "skipped_exists": 0,
        "failed": 0,
    }

    if not dry_run:
        destination_root.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        source_directory = candidate.source_directory
        destination_directory = candidate.destination_directory

        if destination_directory.exists():
            if not overwrite:
                move_stats["skipped_exists"] += 1
                print(f"SKIP (destination exists): {destination_directory}")
                continue
            if dry_run:
                print(f"DRY-RUN REMOVE: {destination_directory}")
            else:
                shutil.rmtree(destination_directory)

        if dry_run:
            print(f"DRY-RUN MOVE: {source_directory} -> {destination_directory}")
            move_stats["moved"] += 1
            continue

        try:
            shutil.move(str(source_directory), str(destination_directory))
            move_stats["moved"] += 1
            print(f"MOVED: {source_directory} -> {destination_directory}")
        except Exception as error:  # noqa: BLE001
            move_stats["failed"] += 1
            print(
                "FAILED: "
                f"{source_directory} -> {destination_directory} "
                f"({error})"
            )

    return move_stats


def print_summary(
    run_directory: Path,
    destination_root: Path,
    discovery_stats: Dict[str, int],
    move_stats: Dict[str, int],
    dry_run: bool,
) -> None:
    """Print concise operation summary.

    Args:
        run_directory: Processed run directory.
        destination_root: Destination root directory.
        discovery_stats: Discovery phase counters.
        move_stats: Move phase counters.
        dry_run: Whether execution was dry-run.

    Returns:
        None.
    """
    print("")
    print("Summary")
    print(f"  run_dir: {run_directory.resolve()}")
    print(f"  destination_root: {destination_root.resolve()}")
    print(f"  mode: {'dry-run' if dry_run else 'apply'}")
    print(f"  job_logs_total: {discovery_stats['job_logs_total']}")
    print(f"  missing_logging_to: {discovery_stats['missing_logging_to']}")
    print(f"  missing_source: {discovery_stats['missing_source']}")
    print(f"  invalid_source: {discovery_stats['invalid_source']}")
    print(f"  deduplicated_sources: {discovery_stats['deduplicated_sources']}")
    print(f"  planned_moves: {move_stats['planned']}")
    print(f"  moved: {move_stats['moved']}")
    print(f"  skipped_exists: {move_stats['skipped_exists']}")
    print(f"  failed: {move_stats['failed']}")


def main() -> None:
    """Run the model-specific log relocation workflow.

    Usage:
        python scripts/move_model_specific_logs.py --run-dir <path> --dry-run
    """
    args = parse_args()
    run_directory = args.run_dir.resolve()
    if not run_directory.is_dir():
        raise SystemExit(f"--run-dir is not a directory: {run_directory}")

    destination_root = run_directory / "model-specific_logs"
    repository_root = Path(__file__).resolve().parent.parent
    candidates, discovery_stats = discover_move_candidates(
        run_directory=run_directory,
        destination_root=destination_root,
        repository_root=repository_root,
    )
    move_stats = execute_move_plan(
        candidates=candidates,
        destination_root=destination_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print_summary(
        run_directory=run_directory,
        destination_root=destination_root,
        discovery_stats=discovery_stats,
        move_stats=move_stats,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
