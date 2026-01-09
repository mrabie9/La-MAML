#!/bin/bash
python3 tuning/Bravo/tune_lamaml.py --tune-only "opt_lr, alpha_init"
python3 tuning/Alpha/tune_cmaml.py --tune-only "opt_wt, alpha_init"
# python3 tuning/Bravo/tune_smaml.py --tune-only ""