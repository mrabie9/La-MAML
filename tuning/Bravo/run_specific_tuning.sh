#!/bin/bash
# python3 tuning/Bravo/tune_lamaml.py --config configs/models/lamaml.yaml --tune-only "opt_lr, alpha_init" 
# python3 tuning/Bravo/tune_cmaml.py --config configs/models/cmaml.yaml --tune-only "opt_wt, alpha_init" 
# python3 tuning/Bravo/tune_packnet.py --config configs/models/packnet.yaml --lr-first
# python3 tuning/Bravo/tune_ewc.py --config configs/models/ewc.yaml --lr-first  
# python3 tuning/Bravo/tune_si.py --config configs/models/si.yaml --lr-first 
# python3 tuning/Bravo/tune_rwalk.py --config configs/models/rwalk.yaml --lr-first 

# python3 tuning/Bravo/tune_smaml.py --config configs/models/smaml.yaml --tune-only "opt_wt,opt_lr,alpha_init" 
# python3 tuning/Bravo/tune_hat.py --config configs/models/hat.yaml --lr-first 

# python3 tuning/Bravo/tune_ewc.py --config configs/models/ewc.yaml --hierarchical #--dry-run
# python3 tuning/Bravo/tune_si.py --config configs/models/si.yaml --hierarchical #--dry-run
# python3 tuning/Bravo/tune_bcl.py --config configs/models/bcl_dual.yaml --hierarchical #--dry-run
# python3 tuning/Bravo/tune_lwf.py --config configs/models/lwf.yaml --hierarchical #--dry-run
# python3 tuning/Bravo/tune_rwalk.py --config configs/models/rwalk.yaml --hierarchical #--dry-run
python3 tuning/Bravo/tune_la-er.py --config configs/models/la-er.yaml --hierarchical #--dry-run

python3 tuning/Bravo/tune_ctn.py --config configs/models/ctn.yaml --hierarchical
# python3 tuning/Bravo/tune_cmaml.py --config configs/models/cmaml.yaml --hierarchical
# python3 tuning/Bravo/tune_smaml.py --config configs/models/smaml.yaml --hierarchical
# python3 tuning/Bravo/tune_hat.py --config configs/models/hat.yaml --hierarchical
python3 tuning/Bravo/tune_packnet.py --config configs/models/packnet.yaml --hierarchical
# python3 tuning/Alpha/tune_agem.py --config configs/models/agem.yaml --hierarchical
# python3 tuning/Alpha/tune_gem.py --config configs/models/gem.yaml --hierarchical
python3 tuning/Bravo/tune_ucl.py --config configs/models/ucl.yaml --hierarchical


#--dry-run
