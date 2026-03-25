#!/bin/bash
# Run main.py with base config and each model config listed in INCLUDED.
# Edit INCLUDED below to choose which algorithms (config names in configs/models/) to run.
# Stdout, stderr (including tracebacks), and run metadata are logged.

# set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

EXPERIMENT_DESC=""
while [ $# -gt 0 ]; do
    case "$1" in
        -d|--desc|--description)
            EXPERIMENT_DESC="${2:-}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--desc/-d DESCRIPTION]"
            echo "  DESCRIPTION is used to label the logs folder (sanitized)."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--desc/-d DESCRIPTION]"
            exit 1
            ;;
    esac
done

# Host-based algorithm selection.
# Edit these lists to control which algorithms each machine runs.
INCLUDED_LNX_ELKK_1="ewc er_ring eralg4 agem gem bcl_dual cmaml ctn hat"
INCLUDED_LNX_ELKK_2="lwf packnet rwalk si ucl la-er lamaml smaml"

# CONCURRENCY_OPTION:
#   0 = run sequentially
#   1 = run python jobs in parallel (MAX_JOBS=2)
CONCURRENCY_OPTION="${CONCURRENCY_OPTION:-1}"
MAX_JOBS=2

HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
case "$HOST_SHORT" in
  lnx-elkk-2) INCLUDED="$INCLUDED_LNX_ELKK_2" ;;
  lnx-elkk-1) INCLUDED="$INCLUDED_LNX_ELKK_1" ;;
  *) INCLUDED="$INCLUDED_LNX_ELKK_2 $INCLUDED_LNX_ELKK_1" ;;
esac

# Log file: stdout and stderr are appended here and also shown on the terminal
LOG_DIR="${REPO_ROOT}/logs/full_experiments"
mkdir -p "$LOG_DIR"
# One timestamp per script run so summary + job logs stay grouped.
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Optional description suffix for the run folder.
DESC_SUFFIX=""
if [ -n "$EXPERIMENT_DESC" ]; then
    # Keep it filesystem-friendly: replace spaces, drop other weird characters.
    DESC_SANITIZED="$(echo "$EXPERIMENT_DESC" | tr ' ' '_' | tr -cd 'A-Za-z0-9._-')"
    if [ -n "$DESC_SANITIZED" ]; then
        DESC_SUFFIX="_${DESC_SANITIZED}"
    fi
fi

# Create a run-specific folder so all logs for this script execution are together.
RUN_LOG_DIR="${LOG_DIR}/run_${RUN_TIMESTAMP}${DESC_SUFFIX}"
mkdir -p "$RUN_LOG_DIR"

# Job logs live in a subfolder under the run directory.
JOB_LOG_DIR="${RUN_LOG_DIR}/job_logs"
mkdir -p "$JOB_LOG_DIR"

# Summary log lives alongside job logs inside the run directory.
LOG_FILE="${RUN_LOG_DIR}/full_experiments_${RUN_TIMESTAMP}.log"

log_msg() {
    echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

log_msg "=== full_experiments.sh started ==="
log_msg "REPO_ROOT=$REPO_ROOT"
log_msg "LOG_FILE=$LOG_FILE"
log_msg "INCLUDED=$INCLUDED"
if [ -n "$EXPERIMENT_DESC" ]; then
    log_msg "EXPERIMENT_DESC=$EXPERIMENT_DESC"
fi

# Activate environment (see AGENTS.md)
if [ -d "la-maml_env" ]; then
    source la-maml_env/bin/activate
    log_msg "Activated la-maml_env"
fi

BASE_CONFIG="configs/base.yaml"
MODELS_DIR="configs/models"

pids=()

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
    JOB_LOG_FILE="${JOB_LOG_DIR}/job_${name}_$(date +%Y%m%d_%H%M%S_%N).log"
    log_msg "--- Dispatching: base + $name (job log: $JOB_LOG_FILE) ---"

    if [ "$CONCURRENCY_OPTION" -eq 1 ]; then
        # Throttle to MAX_JOBS concurrent runs.
        while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 1
        done

        (
            echo "[$(date -Iseconds)] START $name" >>"$LOG_FILE"
            python3 main.py --config "$BASE_CONFIG" --config "$model_yaml" >"$JOB_LOG_FILE" 2>&1
            exit_code=$?
            if [ "$exit_code" -eq 0 ]; then
                echo "[$(date -Iseconds)] Completed: $name (exit 0)" >>"$LOG_FILE"
            else
                echo "[$(date -Iseconds)] ERROR: $name failed with exit code $exit_code" >>"$LOG_FILE"
            fi
            exit "$exit_code"
        ) &
        pids+=($!)
    else
        python3 main.py --config "$BASE_CONFIG" --config "$model_yaml" >"$JOB_LOG_FILE" 2>&1
        exit_code=$?
        if [ "$exit_code" -eq 0 ]; then
            log_msg "Completed: $name (exit 0)"
        else
            log_msg "ERROR: $name failed with exit code $exit_code"
        fi
    fi
done

if [ "$CONCURRENCY_OPTION" -eq 1 ]; then
    overall_exit=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            overall_exit=1
        fi
    done
    exit "$overall_exit"
fi

log_msg "=== full_experiments.sh finished ==="
