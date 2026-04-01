#!/usr/bin/env bash
# Alpha suite: forwards to scripts/run_all_tuning.sh with the same concurrency
#
# Previously run by hand (examples):
#   # python3 tuning/Alpha/tune_er_ring.py --config configs/models/er_ring.yaml --hierarchical --dry-run
#   # python3 tuning/Alpha/tune_eralg4.py --config configs/models/eralg4.yaml --hierarchical  --dry-run
#   # python3 tuning/Alpha/tune_lamaml.py --config configs/models/lamaml.yaml --hierarchical --dry-run
# environment variables as scripts/full_experiments.sh:
#   CONCURRENCY_OPTION  (default 1): 0 = serial, 1 = up to MAX_JOBS parallel jobs
#   MAX_JOBS            (default 2)
#
# Examples:
#   CONCURRENCY_OPTION=0 ./tuning/Alpha/run_specific_tuning.sh
#   MAX_JOBS=2 ./tuning/Alpha/run_specific_tuning.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec "$REPO_ROOT/scripts/run_all_tuning.sh" \
    --scripts-root "$REPO_ROOT/tuning/Alpha" \
    --models "gem,agem,icarl,ucl,bcl,ewc,si,lwf,rwalk" \
    -- \
    --hierarchical \
    --dry-run
