#!/usr/bin/env python3
"""Hyperparameter tuning harness for the RWalk learner."""

from __future__ import annotations
import sys
sys.path.append("tuning/")  # to import from parent directory
from tuning.hyperparam_tuner import make_main
from tuning.presets import TUNING_PRESETS

main = make_main(TUNING_PRESETS["rwalk"])

if __name__ == "__main__":
    main()
