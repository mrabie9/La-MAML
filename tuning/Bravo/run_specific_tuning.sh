#!/usr/bin/env bash
# Bravo suite: forwards to scripts/run_all_tuning.sh with the same concurrency
#
# Previously run by hand (examples, commented in older revision):
#   # tune_lamaml, tune_cmaml, tune_packnet, tune_ewc, tune_si, tune_rwalk, tune_smaml, tune_hat,
#   # tune_bcl, tune_lwf with various --lr-first / --hierarchical flags
# environment variables as scripts/full_experiments.sh:
#   CONCURRENCY_OPTION  (default 1): 0 = serial, 1 = up to MAX_JOBS parallel jobs
#   MAX_JOBS            (default 2)
#
# Examples:
#   CONCURRENCY_OPTION=0 ./tuning/Bravo/run_specific_tuning.sh
#   MAX_JOBS=2 ./tuning/Bravo/run_specific_tuning.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec "$REPO_ROOT/scripts/run_all_tuning.sh" \
    --scripts-root "$REPO_ROOT/tuning/Bravo" \
    --models "la-er,ctn,packnet,ucl" \
    -- \
    --hierarchical
