#!/bin/bash
python3 tuning/Bravo/tune_bcl.py 
python3 tuning/Alpha/tune_ctn.py
python3 tuning/Alpha/tune_gem.py
python3 tuning/Alpha/tune_icarl.py
python3 tuning/Alpha/tune_lwf.py
python3 tuning/Alpha/tune_ucl.py --lr-first
