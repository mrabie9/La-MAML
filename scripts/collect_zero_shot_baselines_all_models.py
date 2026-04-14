#!/usr/bin/env python3
"""Collect untrained zero-shot baselines for all model configs.

This script evaluates each model configuration under ``configs/models`` in an
untrained setting (zero-shot only) and writes a combined JSON baseline file.

Usage:
    python scripts/collect_zero_shot_baselines_all_models.py \
        --output logs/full_experiments/run_20260403_111437_lnx-elkk-1/zs_baseline.json

    python scripts/collect_zero_shot_baselines_all_models.py \
        --mode cil \
        --output logs/full_experiments/run_20260403_111437_lnx-elkk-1/zs_baseline_cil.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import parser as file_parser  # noqa: E402
from utils import misc_utils  # noqa: E402

from evaluate_zero_shot_untrained import (  # noqa: E402
    _apply_device_override,
    _build_model_and_loader,
    _collect_untrained_zero_shot,
    _nan_to_none,
)

Row = Dict[str, Any]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the baseline aggregation script.

    Returns:
        Namespace populated with base config path, models directory, optional
        model name filter, device choice, output path, and failure handling flag.

    """
    parser = argparse.ArgumentParser(
        description="Collect untrained zero-shot baselines across model configs."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Base YAML config applied to every model.",
    )
    parser.add_argument(
        "--mode",
        choices=("til", "cil"),
        default="til",
        help=(
            "Experiment mode used to select model configs when --models-dir is not "
            "provided explicitly."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-model YAML files. Defaults to "
            "configs/models/<mode>."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Optional comma-separated model names to include (without .yaml). "
            "If omitted, all YAML files in --models-dir are evaluated."
        ),
    )
    parser.add_argument(
        "--include-iid2",
        action="store_true",
        help=(
            "When auto-discovering models (--models omitted), include iid2.yaml "
            "(excluded by default). Explicit --models lists always honor every name."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device selection for evaluation runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for aggregated baseline rows.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one model evaluation fails.",
    )
    return parser.parse_args()


def _selected_model_config_paths(
    models_dir: Path,
    model_names_csv: str | None,
    include_iid2: bool,
) -> List[Path]:
    """Resolve YAML paths for models to evaluate.

    When ``model_names_csv`` is ``None``, every ``*.yaml`` under ``models_dir`` is
    used; ``iid2`` is omitted unless ``include_iid2`` is true. When the user lists
    models explicitly, every requested name is honored (including ``iid2``).

    Args:
        models_dir: Directory containing per-model YAML files.
        model_names_csv: Comma-separated stems (no ``.yaml``), or ``None`` for
            auto-discovery.
        include_iid2: If false and ``model_names_csv`` is ``None``, skip
            ``iid2.yaml``. Ignored when names are listed explicitly.

    Returns:
        Sorted list of resolved config paths.

    Raises:
        FileNotFoundError: If ``models_dir`` is missing or an explicit name has no
            matching file.

    """
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if model_names_csv is None:
        paths = sorted(path for path in models_dir.glob("*.yaml") if path.is_file())
        if not include_iid2:
            paths = [path for path in paths if path.stem != "iid2"]
        return paths

    selected_names = {
        item.strip() for item in model_names_csv.split(",") if item.strip()
    }
    paths: List[Path] = []
    for name in sorted(selected_names):
        model_path = models_dir / f"{name}.yaml"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model config: {model_path}")
        paths.append(model_path)
    return paths


def _resolve_models_dir(mode: str, models_dir_override: Path | None) -> Path:
    """Resolve model config directory from mode and optional explicit override.

    Args:
        mode: Experiment mode, either ``"til"`` or ``"cil"``.
        models_dir_override: Optional explicit model directory path from CLI.

    Returns:
        Directory path that should be scanned for ``*.yaml`` model configs.

    """
    if models_dir_override is not None:
        return models_dir_override
    return Path("configs/models") / mode


def _load_args_for_model(base_config: Path, model_config: Path) -> argparse.Namespace:
    """Load and merge YAML parse args for one model (base + model overlay).

    Applies batch-size-scaled learning rate adjustment to match training defaults.

    Args:
        base_config: Path to shared base YAML (e.g. ``configs/base.yaml``).
        model_config: Path to the model-specific YAML.

    Returns:
        Parsed namespace ready for model and dataloader construction.

    """
    config_chain = [str(base_config), str(model_config)]
    run_args = file_parser.parse_args_from_yaml(config_chain)
    run_args.lr = misc_utils.scale_learning_rate_for_batch_size(
        run_args.lr, run_args.batch_size
    )
    return run_args


def _collect_one_model(
    base_config: Path,
    model_config: Path,
    device_choice: str,
) -> List[Row]:
    """Run untrained zero-shot evaluation for a single model config file.

    Args:
        base_config: Shared base YAML path.
        model_config: Model YAML path to evaluate.
        device_choice: One of ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        List of per-task metric rows (same schema as
        ``evaluate_zero_shot_untrained``).

    """
    run_args = _load_args_for_model(base_config, model_config)
    _apply_device_override(run_args, device_choice)
    model, loader = _build_model_and_loader(run_args)
    rows, _matrix_rows = _collect_untrained_zero_shot(
        model=model,
        loader=loader,
        run_args=run_args,
        include_cumulative_matrix=False,
    )
    return rows


def main() -> None:
    """Evaluate all selected model YAMLs without training and write combined JSON.

    Auto-discovery skips ``iid2`` unless ``--include-iid2`` is set; an explicit
    ``--models iid2`` always includes that config.

    Usage:
        From the repo root, after activating ``la-maml_env``::

            python scripts/collect_zero_shot_baselines_all_models.py \\
                --output logs/my_run/zs_baseline.json

        Limit to named models (``iid2`` is included when listed)::

            python scripts/collect_zero_shot_baselines_all_models.py \\
                --models cmaml,iid2 \\
                --output logs/my_run/zs_baseline.json

        Run against CIL model configs::

            python scripts/collect_zero_shot_baselines_all_models.py \\
                --mode cil \\
                --output logs/my_run/zs_baseline_cil.json

    """
    args = _parse_args()
    if not args.base_config.is_file():
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    resolved_models_dir = _resolve_models_dir(
        mode=args.mode,
        models_dir_override=args.models_dir,
    )

    model_config_paths = _selected_model_config_paths(
        resolved_models_dir,
        args.models,
        include_iid2=args.include_iid2,
    )
    if not model_config_paths:
        raise SystemExit("No model configs found to evaluate.")

    all_rows: List[Row] = []
    failures: List[str] = []

    for model_config_path in model_config_paths:
        model_name = model_config_path.stem
        print(f"[baseline] evaluating {model_name} using {model_config_path}")
        try:
            rows = _collect_one_model(
                base_config=args.base_config,
                model_config=model_config_path,
                device_choice=args.device,
            )
            all_rows.extend(rows)
            print(f"[baseline] {model_name}: collected {len(rows)} task rows")
        except Exception as exc:  # pragma: no cover - robust batch run
            message = f"{model_name}: {exc}"
            failures.append(message)
            print(f"[baseline] FAILED {message}")
            if args.fail_fast:
                raise

    all_rows.sort(key=lambda row: (str(row.get("algo", "")), int(row.get("task", -1))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(_nan_to_none(all_rows), output_file, indent=2)
        output_file.write("\n")

    print(f"\nSaved {len(all_rows)} baseline rows to {args.output}")
    if failures:
        print("\nSome models failed:")
        for failure_message in failures:
            print(f"- {failure_message}")


if __name__ == "__main__":
    main()
