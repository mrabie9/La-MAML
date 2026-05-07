#!/bin/bash
# Run scripts/full_experiments.sh once per entry in TASK_ORDER_SEEDS, passing
# --task-order-seed for each run. Each iteration gets -d task_order_seed_<n> so
# logs land under a distinct logs/full_experiments/run_*_task_order_seed_<n>/
# folder.
#
# Usage:
#   ./scripts/full_experiments_task_order_seed_sweep.sh
#   ./scripts/full_experiments_task_order_seed_sweep.sh --mode cil
#
# full_experiments.sh activates la-maml_env when present (see AGENTS.md).
# On hosts whose short hostname is not lnx-elkk-1 or lnx-elkk-2, you may need
#   export HOST_KEY=lnx-elkk-1   # or lnx-elkk-2
# so the cached schedule JSON matches the intended machine.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Edit this list to choose which task-order seeds to run (order is preserved).
TASK_ORDER_SEEDS=(0 1 2)

overall_exit=0

for task_order_seed in "${TASK_ORDER_SEEDS[@]}"; do
  echo "[full_experiments_task_order_seed_sweep] === task_order_seed=${task_order_seed} ==="
  if ! "$SCRIPT_DIR/full_experiments.sh" \
    -d "task_order_seed_${task_order_seed}" \
    --task-order-seed "$task_order_seed" \
    "$@"; then
    overall_exit=1
  fi
done

exit "$overall_exit"
