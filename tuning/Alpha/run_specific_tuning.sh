#!/bin/bash
python3 tuning/Bravo/tune_lamaml.py --config configs/models/lamaml.yaml --tune-only "opt_lr, alpha_init"
python3 tuning/Alpha/tune_cmaml.py --config configs/models/cmaml.yaml --tune-only "opt_wt, alpha_init"
# python3 tuning/Bravo/tune_smaml.py --config configs/models/smaml.yaml --tune-only ""
