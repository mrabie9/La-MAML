#!/bin/bash
# Run main.py with base config and each model config listed in INCLUDED.
# Edit INCLUDED below to choose which algorithms (config names in configs/models/) to run.
# Stdout, stderr (including tracebacks), and run metadata are logged.

# set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

# Algorithms to run: space-separated list of config names (without .yaml), e.g. agem bcl_dual cmaml
INCLUDED="ucl la-er lamaml smaml" #lwf packnet rwalk si 
# BETA= "lwf packnet rwalk si ucl la-er lamaml smaml"
# ALPHA= "ewc er_ring eralg4 agem gem bcl_dual cmaml ctn hat"

# Log file: stdout and stderr are appended here and also shown on the terminal
LOG_DIR="${REPO_ROOT}/logs/full_experiments"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/full_experiments_$(date +%Y%m%d_%H%M%S).log"

log_msg() {
    echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

log_msg "=== full_experiments.sh started ==="
log_msg "REPO_ROOT=$REPO_ROOT"
log_msg "LOG_FILE=$LOG_FILE"
log_msg "INCLUDED=$INCLUDED"

# Activate environment (see AGENTS.md)
if [ -d "la-maml_env" ]; then
    source la-maml_env/bin/activate
    log_msg "Activated la-maml_env"
fi

BASE_CONFIG="configs/base.yaml"
MODELS_DIR="configs/models"

for model_yaml in "$MODELS_DIR"/*.yaml; do
    [ -f "$model_yaml" ] || continue
    basename="${model_yaml##*/}"
    name="${basename%.yaml}"
    run=
    for inc in $INCLUDED; do
        if [ "$name" = "$inc" ]; then
            run=1
            break
        fi
    done
    if [ -z "$run" ]; then
        log_msg "Skipping $name (not in INCLUDED)"
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
