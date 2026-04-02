#!/usr/bin/env bash
# Run tuning harnesses with the same concurrency and per-host algorithm split as
# scripts/full_experiments.sh (queue JSON: slot_high / slot_low).
#
# Environment (optional):
#   CONCURRENCY_OPTION   0 = serial; 1 = dual-slot queue (default 1) or flat pool if no host schedule
#   MAX_JOBS             ignored when host queue schedule is active (two slots, like full_experiments)
#   HOST_KEY             lnx-elkk-1 | lnx-elkk-2 (defaults from hostname; set to force split elsewhere)
#   SCHEDULE_JSON_PATH   defaults to $REPO_ROOT/logs/eta_probe/full_experiments_host_schedule.json
#   HOST_SCHEDULE_MODE   auto | on | off (default auto: on for elkk hosts when schedule file exists)

set -uo pipefail

usage() {
    cat <<'EOF'
Usage: run_all_tuning.sh [options] [-- extra args]

Options:
  --models "m1,m2"     Only run the listed model tuning scripts.
  --exclude "m3,m4"    Skip the listed models.
  --scripts-root DIR   Directory that contains tune_*.py files (default: tuning).
  --config FILE        Base config applied to every run (default: configs/tuning_defaults.yaml).
  --tune-only PARAM    Restrict grid to one or more hyperparameters (repeatable).
  --python PATH        Python interpreter to invoke (default: $PYTHON or python3).
  --log-dir DIR        Directory for aggregated log files (default: logs/tuning/suites).
  --schedule-json FILE JSON from summarise_eta_probe host schedule (default: see HOST_SCHEDULE).
  --use-host-schedule  Require per-host algorithm list from schedule JSON (same as full_experiments.sh).
  --no-host-schedule   Run all discovered/selected scripts (ignore host split).
  --dry-run            Print the commands without executing them.
  --keep-going         Continue running even if a tuning script fails.
  --list               List the discovered scripts and exit.
  -h, --help           Show this message.

Arguments after -- are forwarded verbatim to every tuning script.

Per-host split matches full_experiments.sh: jobs are taken from slot_high and slot_low for
$HOST_KEY. Experiment names in JSON (e.g. bcl_dual) map to tune_* models (bcl_dual or bcl).

Concurrency: with host schedule and CONCURRENCY_OPTION=1, uses two slots (high queue vs low
queue). Without host schedule, uses MAX_JOBS (default 2) as a flat job pool.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

if [[ -d "$REPO_ROOT/la-maml_env" ]]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/la-maml_env/bin/activate"
fi

CONCURRENCY_OPTION="${CONCURRENCY_OPTION:-1}"
MAX_JOBS="${MAX_JOBS:-2}"
HOST_SCHEDULE_MODE="${HOST_SCHEDULE_MODE:-auto}"
FORCE_HOST_SCHEDULE=""

PYTHON_BIN="${PYTHON:-python3}"
SCRIPTS_ROOT="../tuning"
CONFIG_FILE="../configs/tuning_defaults.yaml"
MODEL_CONFIG_DIR="../configs/models"
SCHEDULE_JSON_PATH="${SCHEDULE_JSON_PATH:-$REPO_ROOT/logs/eta_probe/full_experiments_host_schedule.json}"
KEEP_GOING=0
DRY_RUN=0
LIST_ONLY=0
LOG_ROOT="../logs/tuning/suites"
PYTHONPATH_OVERRIDE=""
declare -a MODE_FILTER=()
declare -a EXCLUDE_FILTER=()
declare -a EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            [[ $# -ge 2 ]] || { echo "Missing value for --models" >&2; exit 1; }
            tmp=${2//,/ }
            read -r -a parsed <<< "$tmp"
            MODE_FILTER+=("${parsed[@]}")
            shift 2
            ;;
        --exclude)
            [[ $# -ge 2 ]] || { echo "Missing value for --exclude" >&2; exit 1; }
            tmp=${2//,/ }
            read -r -a parsed <<< "$tmp"
            EXCLUDE_FILTER+=("${parsed[@]}")
            shift 2
            ;;
        --scripts-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --scripts-root" >&2; exit 1; }
            SCRIPTS_ROOT="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 1; }
            CONFIG_FILE="$2"
            shift 2
            ;;
        --schedule-json)
            [[ $# -ge 2 ]] || { echo "Missing value for --schedule-json" >&2; exit 1; }
            SCHEDULE_JSON_PATH="$2"
            shift 2
            ;;
        --use-host-schedule)
            FORCE_HOST_SCHEDULE="on"
            shift
            ;;
        --no-host-schedule)
            FORCE_HOST_SCHEDULE="off"
            shift
            ;;
        --tune-only)
            [[ $# -ge 2 ]] || { echo "Missing value for --tune-only" >&2; exit 1; }
            EXTRA_ARGS+=( "--tune-only" "$2" )
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "Missing value for --python" >&2; exit 1; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --log-dir" >&2; exit 1; }
            LOG_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --keep-going)
            KEEP_GOING=1
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "$MAX_JOBS" -lt 1 ]]; then
    echo "MAX_JOBS must be >= 1 (got $MAX_JOBS)." >&2
    exit 1
fi

if [[ ! -d "$SCRIPTS_ROOT" ]]; then
    echo "Scripts directory '$SCRIPTS_ROOT' does not exist." >&2
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file '$CONFIG_FILE' was not found." >&2
    exit 1
fi

HOST_SHORT_RAW="$(hostname -s 2>/dev/null || hostname)"
HOST_SHORT="$(echo "$HOST_SHORT_RAW" | tr '[:upper:]' '[:lower:]')"
DETECTED_HOST_KEY=""
case "$HOST_SHORT" in
    lnx-elkk-1) DETECTED_HOST_KEY="lnx-elkk-1" ;;
    lnx-elkk-2) DETECTED_HOST_KEY="lnx-elkk-2" ;;
esac
HOST_KEY="${HOST_KEY:-$DETECTED_HOST_KEY}"

if [[ -n "$FORCE_HOST_SCHEDULE" ]]; then
    HOST_SCHEDULE_MODE="$FORCE_HOST_SCHEDULE"
fi

USE_HOST_SCHEDULE=0
if [[ "$HOST_SCHEDULE_MODE" == "on" ]]; then
    USE_HOST_SCHEDULE=1
elif [[ "$HOST_SCHEDULE_MODE" == "off" ]]; then
    USE_HOST_SCHEDULE=0
else
    if [[ -n "$HOST_KEY" && -f "$SCHEDULE_JSON_PATH" ]]; then
        USE_HOST_SCHEDULE=1
    fi
fi

mapfile -t all_scripts < <(find "$SCRIPTS_ROOT" -type f -name 'tune_*.py' | sort)
if [[ ${#all_scripts[@]} -eq 0 ]]; then
    echo "No tuning scripts were found under '$SCRIPTS_ROOT'." >&2
    exit 1
fi

contains() {
    local needle=$1; shift
    for item in "$@"; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

declare -a SELECTED_SCRIPTS=()
declare -a SELECTED_MODELS=()
declare -A SEEN_MODEL=()
for script_path in "${all_scripts[@]}"; do
    name="$(basename "$script_path")"
    model="${name%.py}"
    model="${model#tune_}"
    [[ -n "$model" ]] || continue

    if [[ ${#MODE_FILTER[@]} -gt 0 ]] && ! contains "$model" "${MODE_FILTER[@]}"; then
        continue
    fi
    if [[ ${#EXCLUDE_FILTER[@]} -gt 0 ]] && contains "$model" "${EXCLUDE_FILTER[@]}"; then
        continue
    fi
    if [[ -n "${SEEN_MODEL[$model]:-}" ]]; then
        continue
    fi
    SEEN_MODEL["$model"]=1
    SELECTED_SCRIPTS+=("$script_path")
    SELECTED_MODELS+=("$model")
done

# ---- Host schedule (same JSON shape as full_experiments.sh) ----
SCHED_KIND="" # queue | legacy | ""
declare -a QUEUE_HIGH_IDX=()
declare -a QUEUE_LOW_IDX=()
declare -a LEGACY_PAIR_I0=()
declare -a LEGACY_PAIR_I1=()
declare -a LEGACY_SINGLE_IDX=()

index_of_tune_model() {
    local want="$1"
    local i
    for i in "${!SELECTED_MODELS[@]}"; do
        if [[ "${SELECTED_MODELS[$i]}" == "$want" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

index_for_exp_algo() {
    local exp="$1"
    local cand bcl_ix
    case "$exp" in
        bcl_dual)
            if bcl_ix=$(index_of_tune_model "bcl_dual"); then echo "$bcl_ix"; return 0; fi
            if bcl_ix=$(index_of_tune_model "bcl"); then echo "$bcl_ix"; return 0; fi
            ;;
        *)
            if cand=$(index_of_tune_model "$exp"); then echo "$cand"; return 0; fi
            ;;
    esac
    return 1
}

append_unique_idx() {
    local -n _arr=$1
    local ix="$2"
    local e
    for e in "${_arr[@]}"; do
        if [[ "$e" == "$ix" ]]; then
            return 0
        fi
    done
    _arr+=("$ix")
}

if [[ "$USE_HOST_SCHEDULE" -eq 1 ]]; then
    if [[ -z "$HOST_KEY" ]]; then
        echo "Host schedule requires HOST_KEY (lnx-elkk-1 or lnx-elkk-2); set env or run on those hosts." >&2
        exit 1
    fi
    if [[ ! -f "$SCHEDULE_JSON_PATH" ]]; then
        echo "Host schedule requested but file missing: $SCHEDULE_JSON_PATH" >&2
        exit 1
    fi

    declare -a RAW_HIGH=()
    declare -a RAW_LOW=()
    declare -a RAW_PAIR_A=()
    declare -a RAW_PAIR_B=()
    declare -a RAW_SINGLES=()

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        kind="${line%% *}"
        rest="${line#* }"
        if [[ "$kind" == "HIGH" ]]; then
            RAW_HIGH+=("$rest")
        elif [[ "$kind" == "LOW" ]]; then
            RAW_LOW+=("$rest")
        elif [[ "$kind" == "PAIR" ]]; then
            read -r pa pb <<<"$rest"
            RAW_PAIR_A+=("$pa")
            RAW_PAIR_B+=("$pb")
        elif [[ "$kind" == "SINGLE" ]]; then
            RAW_SINGLES+=("$rest")
        fi
    done < <(HOST_KEY="$HOST_KEY" SCHEDULE_JSON_PATH="$SCHEDULE_JSON_PATH" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

host = os.environ["HOST_KEY"]
path = Path(os.environ["SCHEDULE_JSON_PATH"])
data = json.loads(path.read_text(encoding="utf-8"))
host_obj = (data.get("hosts") or {}).get(host) or {}
if host_obj.get("slot_high") is not None or host_obj.get("slot_low") is not None:
    for m in host_obj.get("slot_high") or []:
        print(f"HIGH {m}")
    for m in host_obj.get("slot_low") or []:
        print(f"LOW {m}")
else:
    for pair in host_obj.get("pairs") or []:
        a = pair.get("A")
        b = pair.get("B")
        if a and b:
            print(f"PAIR {a} {b}")
    for s in host_obj.get("singles") or []:
        print(f"SINGLE {s}")
PY
)

    if [[ ${#RAW_HIGH[@]} -gt 0 || ${#RAW_LOW[@]} -gt 0 ]]; then
        SCHED_KIND="queue"
        for exp in "${RAW_HIGH[@]}"; do
            if idx=$(index_for_exp_algo "$exp"); then
                append_unique_idx QUEUE_HIGH_IDX "$idx"
            else
                echo "Note: schedule slot_high lists '$exp' but no matching tune script in selection; skipping." >&2
            fi
        done
        for exp in "${RAW_LOW[@]}"; do
            if idx=$(index_for_exp_algo "$exp"); then
                append_unique_idx QUEUE_LOW_IDX "$idx"
            else
                echo "Note: schedule slot_low lists '$exp' but no matching tune script in selection; skipping." >&2
            fi
        done
    elif [[ ${#RAW_PAIR_A[@]} -gt 0 || ${#RAW_SINGLES[@]} -gt 0 ]]; then
        SCHED_KIND="legacy"
    fi

    if [[ -z "$SCHED_KIND" ]]; then
        echo "No slot_high/slot_low or pairs/singles for host '$HOST_KEY' in $SCHEDULE_JSON_PATH." >&2
        exit 1
    fi
fi

if [[ "$USE_HOST_SCHEDULE" -eq 1 && "$SCHED_KIND" == "legacy" ]]; then
    LEGACY_PAIR_I0=()
    LEGACY_PAIR_I1=()
    LEGACY_SINGLE_IDX=()
    for pi in "${!RAW_PAIR_A[@]}"; do
        ia=""
        ib=""
        if idx=$(index_for_exp_algo "${RAW_PAIR_A[$pi]}"); then
            ia="$idx"
        else
            echo "Note: schedule pair A='${RAW_PAIR_A[$pi]}' has no tune script in selection; skipping pair." >&2
            continue
        fi
        if idx=$(index_for_exp_algo "${RAW_PAIR_B[$pi]}"); then
            ib="$idx"
        else
            echo "Note: schedule pair B='${RAW_PAIR_B[$pi]}' has no tune script in selection; skipping pair." >&2
            continue
        fi
        LEGACY_PAIR_I0+=("$ia")
        LEGACY_PAIR_I1+=("$ib")
    done
    for exp in "${RAW_SINGLES[@]}"; do
        if idx=$(index_for_exp_algo "$exp"); then
            LEGACY_SINGLE_IDX+=("$idx")
        else
            echo "Note: schedule single lists '$exp' but no matching tune script in selection; skipping." >&2
        fi
    done
fi

if [[ "$USE_HOST_SCHEDULE" -eq 1 && "$SCHED_KIND" == "queue" ]]; then
    declare -A ON_HOST
    for i in "${QUEUE_HIGH_IDX[@]}" "${QUEUE_LOW_IDX[@]}"; do
        ON_HOST["$i"]=1
    done
    for i in "${!SELECTED_MODELS[@]}"; do
        if [[ -z "${ON_HOST[$i]:-}" ]]; then
            echo "Note: skipping '${SELECTED_MODELS[$i]}' (not scheduled for host $HOST_KEY)." >&2
        fi
    done
fi

# Rebuild selected arrays for queue mode only (preserves indices for launch functions)
if [[ "$USE_HOST_SCHEDULE" -eq 1 && "$SCHED_KIND" == "queue" ]]; then
    declare -a NEW_SCRIPTS=()
    declare -a NEW_MODELS=()
    declare -A REMAP_IDX=()
    new_i=0
    for old_i in "${QUEUE_HIGH_IDX[@]}" "${QUEUE_LOW_IDX[@]}"; do
        REMAP_IDX["$old_i"]=$new_i
        NEW_SCRIPTS+=("${SELECTED_SCRIPTS[$old_i]}")
        NEW_MODELS+=("${SELECTED_MODELS[$old_i]}")
        new_i=$((new_i + 1))
    done
    SELECTED_SCRIPTS=("${NEW_SCRIPTS[@]}")
    SELECTED_MODELS=("${NEW_MODELS[@]}")
    declare -a QH_NEW=()
    declare -a QL_NEW=()
    for old_i in "${QUEUE_HIGH_IDX[@]}"; do
        QH_NEW+=("${REMAP_IDX[$old_i]}")
    done
    for old_i in "${QUEUE_LOW_IDX[@]}"; do
        QL_NEW+=("${REMAP_IDX[$old_i]}")
    done
    QUEUE_HIGH_IDX=("${QH_NEW[@]}")
    QUEUE_LOW_IDX=("${QL_NEW[@]}")
fi

if [[ "$USE_HOST_SCHEDULE" -eq 1 && "$SCHED_KIND" == "legacy" ]]; then
    declare -a NEW_SCRIPTS=()
    declare -a NEW_MODELS=()
    declare -A REMAP_IDX=()
    declare -a ORDER_OLD=()
    for pi in "${!LEGACY_PAIR_I0[@]}"; do
        ORDER_OLD+=("${LEGACY_PAIR_I0[$pi]}")
        ORDER_OLD+=("${LEGACY_PAIR_I1[$pi]}")
    done
    for old_i in "${LEGACY_SINGLE_IDX[@]}"; do
        ORDER_OLD+=("$old_i")
    done
    new_i=0
    for old_i in "${ORDER_OLD[@]}"; do
        if [[ -z "${REMAP_IDX[$old_i]:-}" ]]; then
            REMAP_IDX["$old_i"]=$new_i
            NEW_SCRIPTS+=("${SELECTED_SCRIPTS[$old_i]}")
            NEW_MODELS+=("${SELECTED_MODELS[$old_i]}")
            new_i=$((new_i + 1))
        fi
    done
    SELECTED_SCRIPTS=("${NEW_SCRIPTS[@]}")
    SELECTED_MODELS=("${NEW_MODELS[@]}")
    declare -a LPI0=()
    declare -a LPI1=()
    for pi in "${!LEGACY_PAIR_I0[@]}"; do
        o0="${LEGACY_PAIR_I0[$pi]}"
        o1="${LEGACY_PAIR_I1[$pi]}"
        LPI0+=("${REMAP_IDX[$o0]}")
        LPI1+=("${REMAP_IDX[$o1]}")
    done
    LEGACY_PAIR_I0=("${LPI0[@]}")
    LEGACY_PAIR_I1=("${LPI1[@]}")
    declare -a LS_NEW=()
    for old_i in "${LEGACY_SINGLE_IDX[@]}"; do
        LS_NEW+=("${REMAP_IDX[$old_i]}")
    done
    LEGACY_SINGLE_IDX=("${LS_NEW[@]}")
fi

if [[ ${#SELECTED_SCRIPTS[@]} -eq 0 ]]; then
    echo "No tuning scripts matched the provided filters (or host schedule removed all)." >&2
    exit 1
fi

if [[ $LIST_ONLY -eq 1 ]]; then
    echo "Tuning scripts to run (HOST_KEY=${HOST_KEY:-none}, use_host_schedule=$USE_HOST_SCHEDULE):"
    for idx in "${!SELECTED_SCRIPTS[@]}"; do
        printf "  %-12s %s\n" "${SELECTED_MODELS[$idx]}" "${SELECTED_SCRIPTS[$idx]}"
    done
    exit 0
fi

mkdir -p "$LOG_ROOT"
PYTHONPATH_OVERRIDE="$SCRIPT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
    PYTHONPATH_OVERRIDE="$PYTHONPATH_OVERRIDE:$PYTHONPATH"
fi
timestamp=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="${LOG_ROOT}/run_${timestamp}"
mkdir -p "$RUN_LOG_DIR"
LOG_FILE="${LOG_ROOT}/tuning_suite_${timestamp}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging tuning sweep to $LOG_FILE"
echo "Using config: $CONFIG_FILE"
echo "HOST_KEY=${HOST_KEY:-} USE_HOST_SCHEDULE=$USE_HOST_SCHEDULE SCHED_KIND=${SCHED_KIND:-none}"
echo "SCHEDULE_JSON_PATH=$SCHEDULE_JSON_PATH"
echo "CONCURRENCY_OPTION=$CONCURRENCY_OPTION MAX_JOBS=$MAX_JOBS"
echo "RUN_LOG_DIR=$RUN_LOG_DIR"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    printf "Forwarded args:%s\n" "$(printf ' %q' "${EXTRA_ARGS[@]}")"
fi

fill_cmd_for_idx() {
    local idx="$1"
    local script="${SELECTED_SCRIPTS[$idx]}"
    local model="${SELECTED_MODELS[$idx]}"
    cmd=( "$PYTHON_BIN" "$script" "--config" "$CONFIG_FILE" )
    local model_cfg="${MODEL_CONFIG_DIR}/${model}.yaml"
    if [[ ! -f "$model_cfg" && "$model" == "bcl" ]]; then
        model_cfg="${MODEL_CONFIG_DIR}/bcl_dual.yaml"
    fi
    if [[ -f "$model_cfg" ]]; then
        cmd+=( "--config" "$model_cfg" )
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=( "${EXTRA_ARGS[@]}" )
    fi
}

launch_job_at_index() {
    local idx="$1"
    local model="${SELECTED_MODELS[$idx]}"
    local script="${SELECTED_SCRIPTS[$idx]}"
    fill_cmd_for_idx "$idx"
    local job_stamp slot_tag
    job_stamp="$(date +"%Y%m%d_%H%M%S_%N")"
    slot_tag="${2:-slot}"
    local job_log="${JOB_LOG_DIR}/job_${model}_${slot_tag}_${job_stamp}.log"

    echo
    echo "Dispatching $model via $script ($slot_tag)"
    echo "  job log: $job_log"
    printf "Command:%s\n" "$(printf ' %q' "${cmd[@]}")"
    echo "[$(date -Iseconds)] START $model" >>"$job_log"

    (
        PYTHONPATH="$PYTHONPATH_OVERRIDE" "${cmd[@]}" >>"$job_log" 2>&1
        exit_code=$?
        if [[ "$exit_code" -eq 0 ]]; then
            echo "[$(date -Iseconds)] Completed: $model (exit 0)" >>"$job_log"
        else
            echo "[$(date -Iseconds)] ERROR: $model failed with exit code $exit_code" >>"$job_log"
        fi
        exit "$exit_code"
    ) &
    LAST_JOB_PID=$!
    echo "  background pid: $LAST_JOB_PID"
}

declare -a SUCCESSES=()
declare -a FAILURES=()
total=${#SELECTED_SCRIPTS[@]}
declare -a cmd=()
JOB_LOG_DIR="${RUN_LOG_DIR}/job_logs"
mkdir -p "$JOB_LOG_DIR"

reap_pid_record() {
    local pid="$1"
    local finished_model="$2"
    local wait_rc=0
    wait "$pid" || wait_rc=$?
    if [[ "$wait_rc" -ne 0 ]]; then
        FAILURES+=("$finished_model")
        echo "Run for $finished_model failed (job log under $JOB_LOG_DIR)."
        if [[ $KEEP_GOING -ne 1 ]]; then
            stop_launching=1
        fi
    else
        SUCCESSES+=("$finished_model")
        echo "Completed: $finished_model (exit 0)"
    fi
}

# -------------------- dry-run / serial flat --------------------
if [[ $DRY_RUN -eq 1 ]]; then
    for idx in "${!SELECTED_SCRIPTS[@]}"; do
        model="${SELECTED_MODELS[$idx]}"
        fill_cmd_for_idx "$idx"
        script="${SELECTED_SCRIPTS[$idx]}"
        echo
        echo "[ $((idx + 1)) / $total ] $model via $script"
        printf "Command:%s\n" "$(printf ' %q' "${cmd[@]}")"
        echo "Dry run enabled; skipping execution."
        SUCCESSES+=("$model")
    done
elif [[ "$USE_HOST_SCHEDULE" -eq 1 && "$SCHED_KIND" == "queue" && "$CONCURRENCY_OPTION" -eq 1 ]]; then
    echo "=== Dual-slot queue (slot_high / slot_low) for $HOST_KEY ==="
    stop_launching=0
    idx_h=0
    idx_l=0
    slot_high_pid=""
    slot_low_pid=""
    slot_high_model=""
    slot_low_model=""
    while [[ "$idx_h" -lt "${#QUEUE_HIGH_IDX[@]}" || "$idx_l" -lt "${#QUEUE_LOW_IDX[@]}" || -n "$slot_high_pid" || -n "$slot_low_pid" ]]; do
        if [[ -z "$slot_high_pid" && "$idx_h" -lt "${#QUEUE_HIGH_IDX[@]}" && "$stop_launching" -eq 0 ]]; then
            hi="${QUEUE_HIGH_IDX[$idx_h]}"
            launch_job_at_index "$hi" "high"
            slot_high_pid="$LAST_JOB_PID"
            slot_high_model="${SELECTED_MODELS[$hi]}"
            idx_h=$((idx_h + 1))
        fi
        if [[ -z "$slot_low_pid" && "$idx_l" -lt "${#QUEUE_LOW_IDX[@]}" && "$stop_launching" -eq 0 ]]; then
            lo="${QUEUE_LOW_IDX[$idx_l]}"
            launch_job_at_index "$lo" "low"
            slot_low_pid="$LAST_JOB_PID"
            slot_low_model="${SELECTED_MODELS[$lo]}"
            idx_l=$((idx_l + 1))
        fi
        if [[ -n "$slot_high_pid" ]]; then
            # `kill -0` is fragile: it stays true for zombies and can lead to a
            # deadlock if we never reap the finished job. Using `ps` lets us
            # reliably detect the zombie state.
            high_stat="$(ps -o stat= -p "$slot_high_pid" 2>/dev/null || true)"
            if [[ -z "$high_stat" ]] || [[ "$high_stat" == *Z* ]]; then
                reap_pid_record "$slot_high_pid" "$slot_high_model"
                slot_high_pid=""
                slot_high_model=""
                continue
            fi
        fi
        if [[ -n "$slot_low_pid" ]]; then
            low_stat="$(ps -o stat= -p "$slot_low_pid" 2>/dev/null || true)"
            if [[ -z "$low_stat" ]] || [[ "$low_stat" == *Z* ]]; then
                reap_pid_record "$slot_low_pid" "$slot_low_model"
                slot_low_pid=""
                slot_low_model=""
                continue
            fi
        fi
        sleep 1
    done
elif [[ "$USE_HOST_SCHEDULE" -eq 1 && "$SCHED_KIND" == "queue" ]]; then
    echo "=== Serial run (host queue order: high then low) ==="
    stop_launching=0
    for hi in "${QUEUE_HIGH_IDX[@]}"; do
        [[ "$stop_launching" -eq 1 ]] && break
        fill_cmd_for_idx "$hi"
        model="${SELECTED_MODELS[$hi]}"
        script="${SELECTED_SCRIPTS[$hi]}"
        echo
        echo "Running $model via $script"
        printf "Command:%s\n" "$(printf ' %q' "${cmd[@]}")"
        if PYTHONPATH="$PYTHONPATH_OVERRIDE" "${cmd[@]}"; then
            SUCCESSES+=("$model")
        else
            FAILURES+=("$model")
            echo "Run for $model failed."
            [[ $KEEP_GOING -eq 1 ]] || stop_launching=1
        fi
    done
    for lo in "${QUEUE_LOW_IDX[@]}"; do
        [[ "$stop_launching" -eq 1 ]] && break
        fill_cmd_for_idx "$lo"
        model="${SELECTED_MODELS[$lo]}"
        script="${SELECTED_SCRIPTS[$lo]}"
        echo
        echo "Running $model via $script"
        printf "Command:%s\n" "$(printf ' %q' "${cmd[@]}")"
        if PYTHONPATH="$PYTHONPATH_OVERRIDE" "${cmd[@]}"; then
            SUCCESSES+=("$model")
        else
            FAILURES+=("$model")
            echo "Run for $model failed."
            [[ $KEEP_GOING -eq 1 ]] || break
        fi
    done
elif [[ "$USE_HOST_SCHEDULE" -eq 1 && "$SCHED_KIND" == "legacy" ]]; then
    stop_launching=0
    for pi in "${!LEGACY_PAIR_I0[@]}"; do
        [[ "$stop_launching" -eq 1 ]] && break
        i0="${LEGACY_PAIR_I0[$pi]}"
        i1="${LEGACY_PAIR_I1[$pi]}"
        if [[ "$CONCURRENCY_OPTION" -eq 1 ]]; then
            launch_job_at_index "$i0" "pairA"
            pa_pid=$LAST_JOB_PID
            ma="${SELECTED_MODELS[$i0]}"
            launch_job_at_index "$i1" "pairB"
            pb_pid=$LAST_JOB_PID
            mb="${SELECTED_MODELS[$i1]}"
            wait_rc_a=0
            wait_rc_b=0
            wait "$pa_pid" || wait_rc_a=$?
            wait "$pb_pid" || wait_rc_b=$?
            if [[ "$wait_rc_a" -ne 0 ]]; then
                FAILURES+=("$ma")
                [[ $KEEP_GOING -eq 1 ]] || stop_launching=1
            else
                SUCCESSES+=("$ma")
            fi
            if [[ "$wait_rc_b" -ne 0 ]]; then
                FAILURES+=("$mb")
                [[ $KEEP_GOING -eq 1 ]] || stop_launching=1
            else
                SUCCESSES+=("$mb")
            fi
        else
            for ix in "$i0" "$i1"; do
                fill_cmd_for_idx "$ix"
                model="${SELECTED_MODELS[$ix]}"
                if PYTHONPATH="$PYTHONPATH_OVERRIDE" "${cmd[@]}"; then
                    SUCCESSES+=("$model")
                else
                    FAILURES+=("$model")
                    [[ $KEEP_GOING -eq 1 ]] || stop_launching=1
                    break
                fi
            done
        fi
    done
    for si in "${LEGACY_SINGLE_IDX[@]}"; do
        [[ "$stop_launching" -eq 1 ]] && break
        fill_cmd_for_idx "$si"
        model="${SELECTED_MODELS[$si]}"
        if PYTHONPATH="$PYTHONPATH_OVERRIDE" "${cmd[@]}"; then
            SUCCESSES+=("$model")
        else
            FAILURES+=("$model")
            [[ $KEEP_GOING -eq 1 ]] || break
        fi
    done
elif [[ "$CONCURRENCY_OPTION" -eq 0 ]]; then
    for idx in "${!SELECTED_SCRIPTS[@]}"; do
        model="${SELECTED_MODELS[$idx]}"
        fill_cmd_for_idx "$idx"
        script="${SELECTED_SCRIPTS[$idx]}"
        echo
        echo "[ $((idx + 1)) / $total ] Running $model via $script"
        printf "Command:%s\n" "$(printf ' %q' "${cmd[@]}")"
        if PYTHONPATH="$PYTHONPATH_OVERRIDE" "${cmd[@]}"; then
            SUCCESSES+=("$model")
        else
            FAILURES+=("$model")
            echo "Run for $model failed."
            if [[ $KEEP_GOING -ne 1 ]]; then
                break
            fi
        fi
    done
else
    echo "=== Flat pool (max $MAX_JOBS concurrent) ==="
    declare -a running_pids=()
    declare -a running_slot_idx=()
    stop_launching=0
    next_idx=0

    reap_finished_slots() {
        local new_pids=()
        local new_slots=()
        local i pid slot_idx wait_rc finished_model
        for i in "${!running_pids[@]}"; do
            pid="${running_pids[$i]}"
            slot_idx="${running_slot_idx[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
                new_slots+=("$slot_idx")
            else
                wait_rc=0
                wait "$pid" || wait_rc=$?
                finished_model="${SELECTED_MODELS[$slot_idx]}"
                if [[ "$wait_rc" -ne 0 ]]; then
                    FAILURES+=("$finished_model")
                    echo "Run for $finished_model failed (job log under $JOB_LOG_DIR)."
                    if [[ $KEEP_GOING -ne 1 ]]; then
                        stop_launching=1
                    fi
                else
                    SUCCESSES+=("$finished_model")
                    echo "Completed: $finished_model (exit 0)"
                fi
            fi
        done
        running_pids=("${new_pids[@]}")
        running_slot_idx=("${new_slots[@]}")
    }

    while [[ "$next_idx" -lt "$total" || ${#running_pids[@]} -gt 0 ]]; do
        reap_finished_slots
        while [[ "$stop_launching" -eq 0 && ${#running_pids[@]} -lt MAX_JOBS && "$next_idx" -lt "$total" ]]; do
            launch_job_at_index "$next_idx" "flat"
            running_pids+=($LAST_JOB_PID)
            running_slot_idx+=("$next_idx")
            next_idx=$((next_idx + 1))
        done
        if [[ ${#running_pids[@]} -gt 0 ]]; then
            sleep 1
        fi
    done
fi

echo
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run completed. No scripts were executed."
else
    if [[ ${#FAILURES[@]} -gt 0 ]]; then
        echo "Completed with failures."
        printf "Succeeded:%s\n" "$(printf ' %s' "${SUCCESSES[@]:-}")"
        printf "Failed:%s\n" "$(printf ' %s' "${FAILURES[@]}")"
        exit 1
    fi
    echo "All tuning scripts completed successfully."
fi
