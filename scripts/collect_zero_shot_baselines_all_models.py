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
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import parser as file_parser  # noqa: E402
from utils import misc_utils  # noqa: E402

from evaluate_zero_shot_untrained import (  # noqa: E402
    _apply_device_override,
    _collect_untrained_zero_shot,
    _nan_to_none,
)

Row = Dict[str, Any]


class _CachedTaskLoader:
    """Replay precomputed test tasks via the IncrementalLoader interface.

    This adapter lets each model evaluation iterate over the exact same task
    sequence without rebuilding the underlying dataset loader.
    """

    def __init__(
        self, task_infos: Sequence[Dict[str, Any]], test_loaders: Sequence[Any]
    ):
        self._task_infos = [dict(task_info) for task_info in task_infos]
        self._test_loaders = list(test_loaders)
        self.n_tasks = len(self._test_loaders)
        self._current_task = 0

    def new_task(self) -> Tuple[Dict[str, Any], None, None, Any]:
        """Return the next cached task tuple expected by evaluators."""
        if self._current_task >= self.n_tasks:
            raise Exception("No more tasks available.")
        task_info = dict(self._task_infos[self._current_task])
        test_loader = self._test_loaders[self._current_task]
        self._current_task += 1
        return task_info, None, None, test_loader


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


def _load_shared_loader_args(
    base_config: Path, model_config_paths: Sequence[Path]
) -> argparse.Namespace:
    """Load parse args used to build the shared cached task loader.

    The shared loader must match the selected mode/model configs. We infer loader
    type from model configs and require that all selected models agree on it.

    Args:
        base_config: Path to shared base YAML.
        model_config_paths: Selected model YAML paths.

    Returns:
        Parsed namespace used to construct the shared dataloader.

    Raises:
        ValueError: If selected model configs resolve to mixed loader types.

    """
    if not model_config_paths:
        raise ValueError("At least one model config is required for loader setup.")

    loader_names: set[str] = set()
    for model_config_path in model_config_paths:
        model_args = file_parser.parse_args_from_yaml(
            [str(base_config), str(model_config_path)]
        )
        loader_names.add(str(model_args.loader))

    if len(loader_names) != 1:
        sorted_loader_names = ", ".join(sorted(loader_names))
        raise ValueError(
            "Selected models resolve to multiple loader types "
            f"({sorted_loader_names}). Evaluate homogeneous loaders together."
        )

    shared_loader_args = file_parser.parse_args_from_yaml(
        [str(base_config), str(model_config_paths[0])]
    )
    shared_loader_args.lr = misc_utils.scale_learning_rate_for_batch_size(
        shared_loader_args.lr, shared_loader_args.batch_size
    )
    return shared_loader_args


def _build_shared_loader(
    base_config: Path,
    model_config_paths: Sequence[Path],
) -> tuple[Any, tuple[int, int, int], str]:
    """Build one shared loader and return dataset shape metadata.

    Args:
        base_config: Shared base YAML path.

    Returns:
        Tuple of ``(shared_loader, (n_inputs, n_outputs, n_tasks), loader_name)``.

    """
    shared_loader_args = _load_shared_loader_args(base_config, model_config_paths)
    misc_utils.init_seed(shared_loader_args.seed)
    loader_module = importlib.import_module(f"dataloaders.{shared_loader_args.loader}")
    shared_loader = loader_module.IncrementalLoader(
        shared_loader_args, seed=shared_loader_args.seed
    )
    dataset_info = shared_loader.get_dataset_info()
    return shared_loader, dataset_info, str(shared_loader_args.loader)


def _cache_loader_tasks(shared_loader: Any) -> tuple[List[Dict[str, Any]], List[Any]]:
    """Extract and cache per-task metadata and test loaders once.

    Args:
        shared_loader: Loader built from ``configs/base.yaml``.

    Returns:
        Pair ``(task_infos, test_loaders)`` suitable for replay.

    """
    task_infos: List[Dict[str, Any]] = []
    test_loaders: List[Any] = []
    for _ in range(shared_loader.n_tasks):
        task_info, _train_loader, _val_loader, test_loader = shared_loader.new_task()
        task_infos.append(dict(task_info))
        test_loaders.append(test_loader)
    return task_infos, test_loaders


def _build_model_from_shared_dataset_info(
    run_args: argparse.Namespace,
    dataset_info: tuple[int, int, int],
) -> Any:
    """Instantiate an untrained model using shared loader dataset dimensions.

    Args:
        run_args: Model runtime args (base + model config).
        dataset_info: Tuple ``(n_inputs, n_outputs, n_tasks)``.

    Returns:
        Untrained model instance on the chosen device.

    """
    n_inputs, n_outputs, n_tasks = dataset_info
    model_module = importlib.import_module(f"model.{run_args.model}")
    model = model_module.Net(n_inputs, n_outputs, n_tasks, run_args)
    if run_args.cuda:
        try:
            model.cuda()
        except RuntimeError:
            run_args.cuda = False
            model.cpu()
    return model


def _collect_one_model(
    base_config: Path,
    model_config: Path,
    device_choice: str,
    cached_task_infos: Sequence[Dict[str, Any]],
    cached_test_loaders: Sequence[Any],
    shared_dataset_info: tuple[int, int, int],
    shared_loader_name: str,
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
    # Evaluator type must match how cached tasks were produced.
    run_args.loader = shared_loader_name
    model = _build_model_from_shared_dataset_info(run_args, shared_dataset_info)
    loader = _CachedTaskLoader(cached_task_infos, cached_test_loaders)
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
    shared_loader, shared_dataset_info, shared_loader_name = _build_shared_loader(
        args.base_config, model_config_paths
    )
    cached_task_infos, cached_test_loaders = _cache_loader_tasks(shared_loader)

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
                cached_task_infos=cached_task_infos,
                cached_test_loaders=cached_test_loaders,
                shared_dataset_info=shared_dataset_info,
                shared_loader_name=shared_loader_name,
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
