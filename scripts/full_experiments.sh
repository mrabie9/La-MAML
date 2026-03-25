#!/bin/bash
# Run main.py with base config.
# Edit INCLUDED_LNX_ELKK_1/INCLUDED_LNX_ELKK_2 to choose safety allowlist algorithms.
# Stdout, stderr (including tracebacks), and run metadata are logged.

# set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

EXPERIMENT_DESC=""
RERUN_PROBE=0
while [ $# -gt 0 ]; do
    case "$1" in
        -d|--desc|--description)
            EXPERIMENT_DESC="${2:-}"
            shift 2
            ;;
        --rerun-probe|--rerun-eta-probe|--reprobe)
            RERUN_PROBE=1
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [--desc/-d DESCRIPTION]"
            echo "  DESCRIPTION is used to label the logs folder (sanitized)."
            echo "  --rerun-probe   regenerate eta probe JSON + host schedule."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--desc/-d DESCRIPTION]"
            exit 1
            ;;
    esac
done

# Host-based algorithm selection (used only as a safety allowlist).
INCLUDED_LNX_ELKK_1="ewc er_ring eralg4 gem cmaml ctn hat" # bcl_dual agem
INCLUDED_LNX_ELKK_2="lwf packnet rwalk si ucl la-er lamaml smaml"
INCLUDED_ALL="$INCLUDED_LNX_ELKK_1 $INCLUDED_LNX_ELKK_2"

# CONCURRENCY_OPTION:
#   0 = run everything serially
#   1 = run pairs concurrently (2 jobs max by construction)
CONCURRENCY_OPTION="${CONCURRENCY_OPTION:-1}"
MAX_JOBS=2

HOST_SHORT_RAW="$(hostname -s 2>/dev/null || hostname)"
HOST_SHORT="$(echo "$HOST_SHORT_RAW" | tr '[:upper:]' '[:lower:]')"
HOST_KEY=""
case "$HOST_SHORT" in
  lnx-elkk-1) HOST_KEY="lnx-elkk-1" ;;
  lnx-elkk-2) HOST_KEY="lnx-elkk-2" ;;
  *)
    # Allow running on other hosts, but require an explicit HOST_KEY.
    HOST_KEY="${HOST_KEY:-}" ;;
esac
if [ -z "$HOST_KEY" ]; then
  echo "Unknown HOST_SHORT='$HOST_SHORT'. Expected lnx-elkk-1 or lnx-elkk-2." >&2
  exit 1
fi

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
log_msg "INCLUDED_ALL=$INCLUDED_ALL"
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

# Pair-aware host schedule (cached).
PROBE_JSON_PATH="${PROBE_JSON_PATH:-logs/eta_probe/eta_probe_elkk-1-algos_10-iter__avg-util-auto-pair.json}"
SCHEDULE_JSON_PATH="${SCHEDULE_JSON_PATH:-logs/eta_probe/full_experiments_host_schedule.json}"

log_msg "Using PROBE_JSON_PATH=$PROBE_JSON_PATH"
log_msg "Using SCHEDULE_JSON_PATH=$SCHEDULE_JSON_PATH"
log_msg "HOST_KEY=$HOST_KEY"

if [ "$RERUN_PROBE" -eq 1 ]; then
  log_msg "Rerun requested: regenerating probe + schedule."
  PROBE_MODELS=($INCLUDED_ALL)
  MAX_PAIRS=$(( ${#PROBE_MODELS[@]} / 2 ))
  if [ "$MAX_PAIRS" -lt 1 ]; then
    log_msg "ERROR: Could not infer max pairs from INCLUDED_ALL='$INCLUDED_ALL'." >&2
    exit 1
  fi
  log_msg "Running probe_concurrency_eta.py --auto-pair for models: ${PROBE_MODELS[*]}"
  python3 scripts/probe_concurrency_eta.py \
    --models "${PROBE_MODELS[@]}" \
    --auto-pair \
    --max-pairs "$MAX_PAIRS" \
    --wait-iters 10 \
    --n-epochs 1 \
    --base-config "$BASE_CONFIG" \
    --out-json "$PROBE_JSON_PATH" | tee -a "$LOG_FILE"
  log_msg "Probe regenerated: $PROBE_JSON_PATH"
fi

if [ ! -f "$SCHEDULE_JSON_PATH" ] || [ "$RERUN_PROBE" -eq 1 ]; then
  if [ ! -f "$PROBE_JSON_PATH" ]; then
    log_msg "ERROR: Probe JSON missing: $PROBE_JSON_PATH"
    exit 1
  fi
  log_msg "Generating schedule JSON (cached)..."
  python3 scripts/summarise_eta_probe.py \
    --generate-host-schedule \
    --probe-json "$PROBE_JSON_PATH" \
    --schedule-out-json "$SCHEDULE_JSON_PATH" \
    --host1 "lnx-elkk-1" \
    --host2 "lnx-elkk-2" | tee -a "$LOG_FILE"
else
  log_msg "Reusing existing schedule JSON."
fi

percent_time_saved="$(SCHEDULE_JSON_PATH="$SCHEDULE_JSON_PATH" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SCHEDULE_JSON_PATH"])
data = json.loads(path.read_text(encoding="utf-8"))
stats = data.get("stats") or {}
val = stats.get("percent_time_saved_parallel_vs_serial")
print(val if val is not None else "NA")
PY
)"
log_msg "Estimated time saved (parallel vs serial): ${percent_time_saved}%"

HOST_PAIRS=()
HOST_SINGLES=()

while IFS= read -r line; do
  kind="${line%% *}"
  rest="${line#* }"
  if [ "$kind" = "PAIR" ]; then
    HOST_PAIRS+=("$rest")
  elif [ "$kind" = "SINGLE" ]; then
    HOST_SINGLES+=("$rest")
  fi
done < <(HOST_KEY="$HOST_KEY" SCHEDULE_JSON_PATH="$SCHEDULE_JSON_PATH" python3 - <<'PY'
import json
import os
from pathlib import Path

host = os.environ["HOST_KEY"]
path = Path(os.environ["SCHEDULE_JSON_PATH"])
data = json.loads(path.read_text(encoding="utf-8"))
host_obj = data["hosts"][host]
pairs = host_obj.get("pairs") or []
singles = host_obj.get("singles") or []

for pair in pairs:
    a = pair.get("A")
    b = pair.get("B")
    if a and b:
        print(f"PAIR {a} {b}")
for s in singles:
    print(f"SINGLE {s}")
PY
)

run_job_sync() {
  local name="$1"
  local model_yaml="${MODELS_DIR}/${name}.yaml"
  if [ ! -f "$model_yaml" ]; then
    log_msg "ERROR: Missing model config: $model_yaml"
    return 1
  fi
  JOB_LOG_FILE="${JOB_LOG_DIR}/job_${name}_$(date +%Y%m%d_%H%M%S_%N).log"
  log_msg "--- Dispatching: base + $name (job log: $JOB_LOG_FILE) ---"
  echo "[$(date -Iseconds)] START $name" >>"$LOG_FILE"
  python3 main.py --config "$BASE_CONFIG" --config "$model_yaml" >"$JOB_LOG_FILE" 2>&1
  local exit_code=$?
  if [ "$exit_code" -eq 0 ]; then
    echo "[$(date -Iseconds)] Completed: $name (exit 0)" >>"$LOG_FILE"
  else
    echo "[$(date -Iseconds)] ERROR: $name failed with exit code $exit_code" >>"$LOG_FILE"
  fi
  return "$exit_code"
}

is_allowed() {
  local name="$1"
  for inc in $INCLUDED_ALL; do
    if [ "$name" = "$inc" ]; then
      return 0
    fi
  done
  return 1
}

overall_exit=0

log_msg "=== Pair phase on $HOST_KEY ==="
for pair in "${HOST_PAIRS[@]}"; do
  read -r a b <<<"$pair"

  allowed_a=1
  allowed_b=1
  if is_allowed "$a"; then allowed_a=0; fi
  if is_allowed "$b"; then allowed_b=0; fi

  if [ "$allowed_a" -ne 0 ] && [ "$allowed_b" -ne 0 ]; then
    log_msg "Skipping pair $a + $b (neither allowed by INCLUDED_ALL)"
    continue
  fi

  if [ "$CONCURRENCY_OPTION" -eq 0 ] || [ "$allowed_a" -ne 0 ] || [ "$allowed_b" -ne 0 ]; then
    # Serial fallback: run whatever side(s) are allowed.
    if [ "$allowed_a" -eq 0 ]; then
      run_job_sync "$a" || overall_exit=1
    fi
    if [ "$allowed_b" -eq 0 ]; then
      run_job_sync "$b" || overall_exit=1
    fi
    continue
  fi

  # Concurrent pair: run both in parallel and wait.
  JOB_LOG_FILE_A="${JOB_LOG_DIR}/job_${a}_$(date +%Y%m%d_%H%M%S_%N).log"
  JOB_LOG_FILE_B="${JOB_LOG_DIR}/job_${b}_$(date +%Y%m%d_%H%M%S_%N).log"

  echo "[$(date -Iseconds)] START $a" >>"$LOG_FILE"
  echo "[$(date -Iseconds)] START $b" >>"$LOG_FILE"

  (
    python3 main.py --config "$BASE_CONFIG" --config "${MODELS_DIR}/${a}.yaml" >"$JOB_LOG_FILE_A" 2>&1
    exit_code=$?
    echo "[$(date -Iseconds)] Completed: $a (exit $exit_code)" >>"$LOG_FILE"
    exit "$exit_code"
  ) &
  pid_a=$!

  (
    python3 main.py --config "$BASE_CONFIG" --config "${MODELS_DIR}/${b}.yaml" >"$JOB_LOG_FILE_B" 2>&1
    exit_code=$?
    echo "[$(date -Iseconds)] Completed: $b (exit $exit_code)" >>"$LOG_FILE"
    exit "$exit_code"
  ) &
  pid_b=$!

  wait "$pid_a" || overall_exit=1
  wait "$pid_b" || overall_exit=1
done

log_msg "=== Single phase on $HOST_KEY ==="
for name in "${HOST_SINGLES[@]}"; do
  if ! is_allowed "$name"; then
    log_msg "Skipping single $name (not in INCLUDED_ALL)"
    continue
  fi
  run_job_sync "$name" || overall_exit=1
done

exit "$overall_exit"
