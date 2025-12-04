"""Preset definitions for hyperparameter tuning entrypoints."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Iterable, List

from tuning.hyperparam_tuner import Grid, TuningPreset

TypeSpec = Dict[str, Any]
GridSpec = Dict[str, TypeSpec]


COMMON_TYPE_HINTS: Dict[str, type] = {
    "lr": float,
    "memory_strength": float,
    "n_memories": int,
    "memories": int,
    "replay_batch_size": float,
    "beta": float,
    "gamma": float,
    "s": float,
    "batches_per_example": float,
    "meta_lr": float,
    "update_lr": float,
    "opt_lr": float,
    "opt_wt": float,
    "alpha_init": float,
    "mean_eta": float,
    "std_init": float,
    "train_mc_iters": int,
    "bgd_optimizer": str,
    "bcl_temperature": float,
    "bcl_memory_strength": float,
    "bcl_n_memories": int,
    "bcl_inner_steps": int,
    "ctn_lr": float,
    "ctn_beta": float,
    "ctn_n_meta": int,
    "ctn_inner_steps": int,
    "ctn_temperature": float,
    "ctn_n_memories": int,
    "ctn_alpha_init": float,
    "ctn_memory_strength": float,
    "ctn_task_emb": int,
    "lr_rho": float,
    "alpha": float,
    "ratio": float,
    "clipgrad": float,
    "momentum": float,
    "weight_decay": float,
    "lamb": float,
    "eps": float,
    "si_c": float,
    "si_epsilon": float,
    "clipgrad_norm": float,
    "optimizer": str,
}


def _unique(values: Iterable[Any]) -> List[Any]:
    seen: List[Any] = []
    for value in values:
        if isinstance(value, float):
            value = float(f"{value:.6g}")
        if value not in seen:
            seen.append(value)
    return seen


def _scale_float(base: float, factors: Iterable[float], minimum: float | None) -> List[float]:
    if base <= 0 and minimum is not None:
        base = minimum
    values = []
    for factor in factors:
        candidate = base * factor
        if minimum is not None:
            candidate = max(candidate, minimum)
        values.append(float(candidate))
    return _unique(values)


def _scale_int(base: int, factors: Iterable[float], minimum: int | None) -> List[int]:
    if base <= 0 and minimum is not None:
        base = max(minimum, 1)
    values = []
    for factor in factors:
        candidate = int(round(base * factor))
        if minimum is not None:
            candidate = max(candidate, minimum)
        if candidate <= 0:
            continue
        values.append(candidate)
    if not values and minimum is not None:
        values.append(max(minimum, 1))
    if not values:
        values.append(max(base, 1))
    return sorted(set(values))


def make_grid_factory(spec: GridSpec) -> Callable[[argparse.Namespace], Grid]:
    def factory(args: argparse.Namespace) -> Grid:
        grid: Grid = {}
        for key, meta in spec.items():
            if "values" in meta:
                grid[key] = list(meta["values"])
                continue

            base = getattr(args, key, None)
            if base in (None, 0) and "fallback" in meta:
                base = meta["fallback"]

            if base is None:
                continue

            kind = meta.get("kind", "float")
            factors = meta.get("factors", (0.5, 1.0, 2.0))
            minimum = meta.get("min")
            extra = meta.get("extra", [])

            if kind == "int":
                values = _scale_int(int(round(base)), factors, minimum)
                values.extend(int(v) for v in extra)
                grid[key] = sorted(set(values))
            else:
                values = _scale_float(float(base), factors, minimum)
                values.extend(float(v) for v in extra)
                grid[key] = _unique(values)
        return {k: v for k, v in grid.items() if v}

    return factory


TUNING_PRESETS: Dict[str, TuningPreset] = {
    "agem": TuningPreset(
        model_name="agem",
        description="Run grid or random search over AGEM hyperparameters.",
        default_output_root="logs/tuning/agem",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-3},
                "memory_strength": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.1, "fallback": 1.0},
                "n_memories": {"kind": "int", "factors": (0.5, 1.0, 2.0), "min": 64, "fallback": 256},
            }
        ),
    ),
    "anml": TuningPreset(
        model_name="anml",
        description="Run grid or random search over ANML hyperparameters.",
        default_output_root="logs/tuning/anml",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-3},
                "meta_lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 1e-3},
                "update_lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.01, "fallback": 0.1},
            }
        ),
    ),
    "bcl_dual": TuningPreset(
        model_name="bcl_dual",
        description="Run grid or random search over BCL-Dual hyperparameters.",
        default_output_root="logs/tuning/bcl_dual",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-3},
                "beta": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.1, "fallback": 1.0},
                "bcl_temperature": {"kind": "float", "factors": (0.5, 1.0, 1.5), "min": 0.5, "fallback": 2.0},
                "bcl_memory_strength": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.1, "fallback": 1.0},
                "bcl_n_memories": {"kind": "int", "factors": (0.5, 1.0, 2.0), "min": 128, "fallback": 2000},
            }
        ),
    ),
    "ctn": TuningPreset(
        model_name="ctn",
        description="Run grid or random search over CTN hyperparameters.",
        default_output_root="logs/tuning/ctn",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "ctn_lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 0.01},
                "ctn_beta": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 0.05},
                "ctn_n_meta": {"kind": "int", "factors": (1.0, 2.0, 3.0), "min": 1, "fallback": 2},
                "ctn_inner_steps": {"kind": "int", "factors": (1.0, 2.0, 3.0), "min": 1, "fallback": 2},
                "ctn_temperature": {"kind": "float", "factors": (0.5, 1.0, 1.5), "min": 1.0, "fallback": 5.0},
                "ctn_n_memories": {"kind": "int", "factors": (0.5, 1.0, 2.0), "min": 50, "fallback": 50},
                "ctn_memory_strength": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1.0, "fallback": 100.0},
                "ctn_alpha_init": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.01, "fallback": 0.1},
            }
        ),
    ),
    "eralg4": TuningPreset(
        model_name="eralg4",
        description="Run grid or random search over ER-Alg4 hyperparameters.",
        default_output_root="logs/tuning/eralg4",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-2},
                "memories": {"kind": "int", "factors": (0.5, 1.0, 2.0), "min": 100, "fallback": 400},
                "replay_batch_size": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 4.0, "fallback": 10.0},
                "opt_lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 0.1},
                "alpha_init": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.01, "fallback": 0.1},
            }
        ),
    ),
    "er_ring": TuningPreset(
        model_name="er_ring",
        description="Run grid or random search over ER-Ring hyperparameters.",
        default_output_root="logs/tuning/er_ring",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-2},
                "memories": {"kind": "int", "factors": (0.5, 1.0, 2.0), "min": 100, "fallback": 400},
                "replay_batch_size": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 4.0, "fallback": 10.0},
            }
        ),
    ),
    "ewc": TuningPreset(
        model_name="ewc",
        description="Run grid or random search over EWC hyperparameters.",
        default_output_root="logs/tuning/ewc",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-5, "fallback": 3e-2},
                "lamb": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-2, "fallback": 1.0},
                "clipgrad": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1.0, "fallback": 100.0},
                "momentum": {"kind": "float", "factors": (0.0, 1.0, 2.0), "extra": [0.9], "fallback": 0.0},
            }
        ),
    ),
    "gem": TuningPreset(
        model_name="gem",
        description="Run grid or random search over GEM hyperparameters.",
        default_output_root="logs/tuning/gem",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-2},
                "memory_strength": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.1, "fallback": 0.5},
                "n_memories": {"kind": "int", "factors": (0.5, 1.0, 2.0), "min": 32, "fallback": 256},
            }
        ),
    ),
    "icarl": TuningPreset(
        model_name="icarl",
        description="Run grid or random search over iCaRL hyperparameters.",
        default_output_root="logs/tuning/icarl",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-2},
                "memory_strength": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.1, "fallback": 0.5},
                "n_memories": {"kind": "int", "factors": (0.5, 1.0, 2.0), "min": 64, "fallback": 256},
            }
        ),
    ),
    "iid2": TuningPreset(
        model_name="iid2",
        description="Run grid or random search over IID2 hyperparameters.",
        default_output_root="logs/tuning/iid2",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {"lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-2}}
        ),
    ),
    "lamaml": TuningPreset(
        model_name="lamaml",
        description="Run grid or random search over La-MAML hyperparameters.",
        default_output_root="logs/tuning/lamaml",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-5, "fallback": 1e-3},
                "opt_lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 0.1},
                "opt_wt": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 0.1},
                "alpha_init": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.01, "fallback": 0.1},
            }
        ),
    ),
    "lamaml_cifar": TuningPreset(
        model_name="lamaml_cifar",
        description="Run grid or random search over La-MAML-CIFAR hyperparameters.",
        default_output_root="logs/tuning/lamaml_cifar",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-5, "fallback": 1e-3},
                "opt_lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 0.1},
                "opt_wt": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 0.1},
                "alpha_init": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.01, "fallback": 0.1},
            }
        ),
    ),
    "meralg1": TuningPreset(
        model_name="meralg1",
        description="Run grid or random search over MER-Alg1 hyperparameters.",
        default_output_root="logs/tuning/meralg1",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.3, 1.0, 3.0), "min": 1e-5, "fallback": 1e-3},
                "beta": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-3, "fallback": 0.1},
                "gamma": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-3, "fallback": 0.1},
                "memories": {"kind": "int", "factors": (0.5, 1.0, 2.0), "min": 100, "fallback": 400},
                "batches_per_example": {"kind": "float", "factors": (1.0, 2.0, 3.0), "min": 1.0, "fallback": 1.0},
            }
        ),
    ),
    "meta-bgd": TuningPreset(
        model_name="meta-bgd",
        description="Run grid or random search over Meta-BGD hyperparameters.",
        default_output_root="logs/tuning/meta_bgd",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "std_init": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 5e-2},
                "mean_eta": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.1, "fallback": 1.0},
                "train_mc_iters": {"kind": "int", "factors": (0.5, 1.0, 1.5), "min": 2, "fallback": 5},
                "bgd_optimizer": {"values": ["bgd", "adam", "sgd"]},
            }
        ),
    ),
    "rwalk": TuningPreset(
        model_name="rwalk",
        description="Run grid or random search over RWalk hyperparameters.",
        default_output_root="logs/tuning/rwalk",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-5, "fallback": 1e-3},
                "lamb": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-2, "fallback": 1.0},
                "alpha": {"kind": "float", "factors": (0.5, 1.0, 1.5), "min": 0.1, "fallback": 0.9},
                "eps": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-3, "fallback": 0.01},
                "clipgrad": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1.0, "fallback": 100.0},
            }
        ),
    ),
    "si": TuningPreset(
        model_name="si",
        description="Run grid or random search over Synaptic Intelligence hyperparameters.",
        default_output_root="logs/tuning/si",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-5, "fallback": 1e-3},
                "si_c": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.01, "fallback": 0.1},
                "si_epsilon": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 0.01},
                "clipgrad": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1.0, "fallback": 100.0},
            }
        ),
    ),
    "ucl": TuningPreset(
        model_name="ucl",
        description="Run grid or random search over UCL hyperparameters.",
        default_output_root="logs/tuning/ucl",
        type_hints=COMMON_TYPE_HINTS,
        grid_factory=make_grid_factory(
            {
                "lr": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-5, "fallback": 1e-3},
                "lr_rho": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-4, "fallback": 1e-2},
                "beta": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1e-5, "fallback": 2e-4},
                "alpha": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.05, "fallback": 0.3},
                "ratio": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 0.05, "fallback": 0.125},
                "clipgrad": {"kind": "float", "factors": (0.5, 1.0, 2.0), "min": 1.0, "fallback": 10.0},
            }
        ),
    ),
}


__all__ = ["COMMON_TYPE_HINTS", "TUNING_PRESETS", "make_grid_factory"]
