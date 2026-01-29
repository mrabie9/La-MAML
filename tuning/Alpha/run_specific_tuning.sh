#!/bin/bash
# python3 tuning/Alpha/tune_bcl.py --config configs/models/bcl_dual.yaml #--dry-run
# python3 tuning/Alpha/tune_ctn.py --config configs/models/ctn.yaml #--dry-run
python3 tuning/Alpha/tune_gem.py --config configs/models/gem.yaml #--dry-run
python3 tuning/Alpha/tune_icarl.py --config configs/models/icarl.yaml #--dry-run
# python3 tuning/Alpha/tune_lwf.py --config configs/models/lwf.yaml #--dry-run
# python3 tuning/Alpha/tune_ucl.py --config configs/models/ucl.yaml --lr-first #--dry-run
