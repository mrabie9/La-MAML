#!/bin/bash
# python3 tuning/Bravo/tune_lamaml.py --config configs/models/lamaml.yaml --tune-only "opt_lr, alpha_init" 
# python3 tuning/Bravo/tune_cmaml.py --config configs/models/cmaml.yaml --tune-only "opt_wt, alpha_init" 
# python3 tuning/Bravo/tune_packnet.py --config configs/models/packnet.yaml --lr-first
# python3 tuning/Bravo/tune_ewc.py --config configs/models/ewc.yaml --lr-first  
# python3 tuning/Bravo/tune_si.py --config configs/models/si.yaml --lr-first 
# python3 tuning/Bravo/tune_rwalk.py --config configs/models/rwalk.yaml --lr-first 

# python3 tuning/Bravo/tune_smaml.py --config configs/models/smaml.yaml --tune-only "opt_wt,opt_lr,alpha_init" 
# python3 tuning/Bravo/tune_la-er.py --config configs/models/la-er.yaml 
# python3 tuning/Bravo/tune_hat.py --config configs/models/hat.yaml --lr-first 

python3 tuning/Bravo/tune_ewc.py --config configs/models/ewc.yaml --lr-first #--dry-run
python3 tuning/Bravo/tune_si.py --config configs/models/si.yaml --lr-first #--dry-run
python3 tuning/Bravo/tune_bcl.py --config configs/models/bcl_dual.yaml --lr-first #--dry-run
python3 tuning/Bravo/tune_lwf.py --config configs/models/lwf.yaml --lr-first #--dry-run
python3 tuning/Bravo/tune_rwalk.py --config configs/models/rwalk.yaml --lr-first #--dry-run
