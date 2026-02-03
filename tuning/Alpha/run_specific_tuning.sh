#!/bin/bash
# python3 tuning/Alpha/tune_ctn.py --config configs/models/ctn.yaml #--dry-run
# python3 tuning/Alpha/tune_gem.py --config configs/models/gem.yaml #--dry-run
# python3 tuning/Alpha/tune_icarl.py --config configs/models/icarl.yaml #--dry-run
python3 tuning/Alpha/tune_er_ring.py --config configs/models/er_ring.yaml --lr-first #--dry-run
python3 tuning/Alpha/tune_eralg4.py --config configs/models/eralg4.yaml --lr-first #--dry-run
python3 tuning/Alpha/tune_gem.py --config configs/models/gem.yaml --lr-first #--dry-run
python3 tuning/Alpha/tune_agem.py --config configs/models/agem.yaml --lr-first #--dry-run
# python3 tuning/Alpha/tune_ucl.py --config configs/models/ucl.yaml --lr-first #--dry-run
