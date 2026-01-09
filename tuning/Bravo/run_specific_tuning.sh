#!/bin/bash
python3 tuning/Bravo/tune_lamaml.py --tune-only "opt_lr, alpha_init"
python3 tuning/Alpha/tune_cmaml.py --tune-only "opt_wt, alpha_init"
python3 tuning/Bravo/tune_hat.py --lr-first
python3 tuning/Bravo/tune_ewc.py --lr-first
python3 tuning/Bravo/tune_si.py --lr-first
python3 tuning/Bravo/tune_rwalk.py --lr-first



# python3 tuning/Bravo/tune_smaml.py --tune-only ""