#!/usr/bin/env python3
"""Run every hyperparameter tuning entrypoint sequentially."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from tuning.presets import TUNING_PRESETS
except ImportError:  # pragma: no cover
    TUNING_PRESETS = {}


def discover_tuning_scripts(root: Path) -> "OrderedDict[str, Path]":
    """Return an ordered dict mapping model names to script paths."""
    scripts: "OrderedDict[str, Path]" = OrderedDict()
    for script_path in sorted(root.glob("tune_*.py")):
        if not script_path.is_file():
            continue
        name = script_path.stem[5:]
        if not name:
            continue
        scripts[name] = script_path
    return scripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all tuning scripts (tune_*.py inside the tuning directory) one after another."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke the tuning scripts.",
    )
    parser.add_argument(
        "--scripts-root",
        default="tuning",
        help="Path to the directory that contains tune_*.py files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names to run (matching tune_<name>.py).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Model names that should be skipped.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List detected tuning scripts and exit without running them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running other scripts after a failure.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to every tuning script (use -- to separate).",
    )
    return parser.parse_args()


def format_listing(models: Iterable[Tuple[str, Path]]) -> str:
    lines = []
    for name, path in models:
        description = ""
        preset = TUNING_PRESETS.get(name)
        if preset is not None:
            description = preset.resolve_description()
        detail = f"{name:<15} -> {path}"
        if description:
            detail = f"{detail} ({description})"
        lines.append(detail)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    scripts_root = (repo_root / args.scripts_root).resolve()

    scripts = discover_tuning_scripts(scripts_root)
    if not scripts:
        raise SystemExit(f"No tuning scripts found under {scripts_root}.")

    selected: "OrderedDict[str, Path]" = OrderedDict()
    if args.models:
        missing = sorted({name for name in args.models if name not in scripts})
        if missing:
            raise SystemExit(f"Unknown model(s): {', '.join(missing)}")
        for name in args.models:
            selected[name] = scripts[name]
    else:
        selected = OrderedDict(scripts)

    for name in args.exclude:
        selected.pop(name, None)

    if not selected:
        raise SystemExit("There are no tuning scripts to run after filtering.")

    if args.list:
        print(format_listing(selected.items()))
        return 0

    forwarded = list(args.script_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    successes: List[str] = []
    failures: List[str] = []
    total = len(selected)

    for idx, (name, script_path) in enumerate(selected.items(), start=1):
        cmd = [args.python, str(script_path)]
        cmd.extend(forwarded)
        printable = shlex.join(cmd)
        print(f"[{idx}/{total}] Running {name}: {printable}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(repo_root))
        if result.returncode == 0:
            successes.append(name)
            continue
        failures.append(name)
        print(f"-> {name} failed with exit code {result.returncode}.")
        if not args.keep_going:
            break

    if args.dry_run:
        return 0

    if failures:
        print(f"Completed with failures. Successful: {successes}. Failed: {failures}.")
        return 1

    print(f"All tuning scripts finished successfully: {successes}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
