#!/usr/bin/env python3
"""Hyperparameter tuning harness for the IID2 learner."""

from __future__ import annotations

from tuning.hyperparam_tuner import make_main
from tuning.presets import TUNING_PRESETS

main = make_main(TUNING_PRESETS["iid2"])

if __name__ == "__main__":
    main()
