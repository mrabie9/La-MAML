#!/bin/bash
# Run main.py with base config and each model config in configs/models/,
# excluding: anml, iid2, meralg1, meta-bgd, icarl.
# Stdout, stderr (including tracebacks), and run metadata are logged.

# set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Log file: stdout and stderr are appended here and also shown on the terminal
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/full_experiments_$(date +%Y%m%d_%H%M%S).log"

log_msg() {
    echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

log_msg "=== full_experiments.sh started ==="
log_msg "REPO_ROOT=$REPO_ROOT"
log_msg "LOG_FILE=$LOG_FILE"

# Activate environment (see AGENTS.md)
if [ -d "la-maml_env" ]; then
    source la-maml_env/bin/activate
    log_msg "Activated la-maml_env"
fi

EXCLUDED="anml iid2 meralg1 meta-bgd icarl"
BASE_CONFIG="configs/base.yaml"
MODELS_DIR="configs/models"

for model_yaml in "$MODELS_DIR"/*.yaml; do
    [ -f "$model_yaml" ] || continue
    basename="${model_yaml##*/}"
    name="${basename%.yaml}"
    skip=
    for ex in $EXCLUDED; do
        if [ "$name" = "$ex" ]; then
            skip=1
            break
        fi
    done
    if [ -n "$skip" ]; then
        log_msg "Skipping $name (excluded)"
        continue
    fi
    log_msg "--- Running: base + $name ---"
    python3 main.py --config "$BASE_CONFIG" --config "$model_yaml" 2>&1 | tee -a "$LOG_FILE"
    exit_code="${PIPESTATUS[0]}"
    if [ "$exit_code" -eq 0 ]; then
        log_msg "Completed: $name (exit 0)"
    else
        log_msg "ERROR: $name failed with exit code $exit_code"
    fi
done

log_msg "=== full_experiments.sh finished ==="
