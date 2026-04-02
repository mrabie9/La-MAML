#!/usr/bin/env python3
"""Hyperparameter tuning harness for the BCL-Dual learner."""

from __future__ import annotations
import sys
from pathlib import Path

try:
    # Works when the repo root is on PYTHONPATH (e.g. via `python -m` or
    # when running from the repo root).
    from tuning.hyperparam_tuner import make_main
    from tuning.presets import TUNING_PRESETS
except ModuleNotFoundError:
    # Ensure imports work regardless of current working directory.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from tuning.hyperparam_tuner import make_main
    from tuning.presets import TUNING_PRESETS

main = make_main(TUNING_PRESETS["bcl_dual"])

if __name__ == "__main__":
    main()
