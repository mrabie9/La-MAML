#!/usr/bin/env python3
"""Hyperparameter tuning harness for the Secret Algo learner."""

from __future__ import annotations

from tuning.hyperparam_tuner import make_main
from tuning.presets import TUNING_PRESETS

main = make_main(TUNING_PRESETS["secret_algo"])

if __name__ == "__main__":
    main()
