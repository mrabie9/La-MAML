#!/usr/bin/env python3
"""Hyperparameter tuning harness for the HAT learner."""

from __future__ import annotations
import sys
sys.path.append("tuning/")  # to import from parent directory
from tuning.hyperparam_tuner import TuningPreset, make_main

DEFAULT_GRID = {
    "lr": [1e-3, 1e-2, 1e-1],
    "gamma": [1.0, 1.3, 1.8],
    "smax": [25, 50, 100, 250],
    "batch_size": [128,256,512]
}

DEFAULT_TYPE_HINTS = {
    "lr": float,
    "gamma": float,
    "smax": float,
    "batch_size": int
}

PRESET = TuningPreset(
    model_name="hat",
    description="Run grid or random search over HAT hyperparameters.",
    default_output_root="logs/tuning/hat",
    default_grid=DEFAULT_GRID,
    type_hints=DEFAULT_TYPE_HINTS,
)

main = make_main(PRESET)

if __name__ == "__main__":
    main()
