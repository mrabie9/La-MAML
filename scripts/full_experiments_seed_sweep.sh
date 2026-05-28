#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_SHOT=0
FORWARDED_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --one-shot)
      ONE_SHOT=1
      shift
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

# Edit this list to choose which buffer sizes to run (order is preserved).
SEEDS=(55 39)

overall_exit=0

for n_mem in "${SEEDS[@]}"; do
  echo "[full_experiments_seed_sweep] === seed=${seed} ==="
  cmd=(
    "$SCRIPT_DIR/full_experiments.sh"
    -d "seed_${seed}"
    --seed "$seed"
  )
  if [ "$ONE_SHOT" -eq 1 ]; then
    cmd+=(--one-shot)
  fi
  cmd+=("${FORWARDED_ARGS[@]}")
  if ! "${cmd[@]}"; then
    overall_exit=1
  fi
done

exit "$overall_exit"
