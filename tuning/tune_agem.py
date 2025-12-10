#!/usr/bin/env python3
"""Hyperparameter tuning harness for the AGEM learner."""

from __future__ import annotations
import sys
sys.path.append("tuning/")  # to import from parent directory
from hyperparam_tuner import make_main
from presets import TUNING_PRESETS

main = make_main(TUNING_PRESETS["agem"])

if __name__ == "__main__":
    main()
