#!/bin/bash
# python3 tuning/Alpha/tune_er_ring.py --config configs/models/er_ring.yaml --hierarchical --dry-run
# python3 tuning/Alpha/tune_eralg4.py --config configs/models/eralg4.yaml --hierarchical  --dry-run
python3 tuning/Alpha/tune_gem.py --config configs/models/gem.yaml --hierarchical --dry-run
python3 tuning/Alpha/tune_agem.py --config configs/models/agem.yaml --hierarchical --dry-run
python3 tuning/Alpha/tune_icarl.py --config configs/models/icarl.yaml --hierarchical --dry-run
python3 tuning/Alpha/tune_ucl.py --config configs/models/ucl.yaml --hierarchical --dry-run
# python3 tuning/Alpha/tune_lamaml.py --config configs/models/lamaml.yaml --hierarchical --dry-run

python3 tuning/Alpha/tune_bcl.py --config configs/models/bcl_dual.yaml --hierarchical --dry-run
python3 tuning/Alpha/tune_ewc.py --config configs/models/ewc.yaml --hierarchical --dry-run
python3 tuning/Alpha/tune_si.py --config configs/models/si.yaml --hierarchical --dry-run
python3 tuning/Alpha/tune_lwf.py --config configs/models/lwf.yaml --hierarchical --dry-run
python3 tuning/Alpha/tune_rwalk.py --config configs/models/rwalk.yaml --hierarchical --dry-run
