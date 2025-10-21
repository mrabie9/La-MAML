#!/usr/bin/env python3
"""Hyperparameter tuning harness for the MER Algorithm 1 learner."""

from __future__ import annotations

from tuning.hyperparam_tuner import make_main
from tuning.presets import TUNING_PRESETS

main = make_main(TUNING_PRESETS["meralg1"])

if __name__ == "__main__":
    main()
