#!/usr/bin/env bash
# Alpha suite: delegates to scripts/run_all_tuning.sh.
#
# On lnx-elkk-1 / lnx-elkk-2, the same host schedule JSON as full_experiments.sh
# (SCHEDULE_JSON_PATH, default logs/eta_probe/full_experiments_host_schedule.json)
# restricts which models run on which server. Elsewhere, set HOST_KEY to mimic a host.
#
# Environment (see run_all_tuning.sh): CONCURRENCY_OPTION, MAX_JOBS, HOST_KEY,
# SCHEDULE_JSON_PATH, HOST_SCHEDULE_MODE=auto|on|off.
#
# Examples:
#   ./tuning/Alpha/run_specific_tuning.sh
#   HOST_SCHEDULE_MODE=off ./tuning/Alpha/run_specific_tuning.sh
#   CONCURRENCY_OPTION=0 ./tuning/Alpha/run_specific_tuning.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec "$REPO_ROOT/scripts/run_all_tuning.sh" \
    --scripts-root "$REPO_ROOT/tuning/Alpha" \
    --models "bcl_dual,ctn,ucl" \
    -- \
    --hierarchical \
    # --dry-run
