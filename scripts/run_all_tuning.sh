#!/usr/bin/env bash
# Run every tuning harness sequentially while logging stdout/stderr.

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
  --dry-run            Print the commands without executing them.
  --keep-going         Continue running even if a tuning script fails.
  --list               List the discovered scripts and exit.
  -h, --help           Show this message.

Arguments after -- are forwarded verbatim to every tuning script.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
SCRIPTS_ROOT="../tuning"
CONFIG_FILE="../configs/tuning_defaults.yaml"
MODEL_CONFIG_DIR="../configs/models"
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

if [[ ! -d "$SCRIPTS_ROOT" ]]; then
    echo "Scripts directory '$SCRIPTS_ROOT' does not exist." >&2
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file '$CONFIG_FILE' was not found." >&2
    exit 1
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
    SELECTED_SCRIPTS+=("$script_path")
    SELECTED_MODELS+=("$model")
done

if [[ ${#SELECTED_SCRIPTS[@]} -eq 0 ]]; then
    echo "No tuning scripts matched the provided filters." >&2
    exit 1
fi

if [[ $LIST_ONLY -eq 1 ]]; then
    echo "Tuning scripts to run:"
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
LOG_FILE="${LOG_ROOT}/tuning_suite_${timestamp}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging tuning sweep to $LOG_FILE"
echo "Using config: $CONFIG_FILE"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    printf "Forwarded args:%s\n" "$(printf ' %q' "${EXTRA_ARGS[@]}")"
fi

declare -a SUCCESSES=()
declare -a FAILURES=()
total=${#SELECTED_SCRIPTS[@]}

for idx in "${!SELECTED_SCRIPTS[@]}"; do
    script="${SELECTED_SCRIPTS[$idx]}"
    model="${SELECTED_MODELS[$idx]}"

    cmd=( "$PYTHON_BIN" "$script" "--config" "$CONFIG_FILE" )
    model_cfg="${MODEL_CONFIG_DIR}/${model}.yaml"
    if [[ -f "$model_cfg" ]]; then
        cmd+=( "--config" "$model_cfg" )
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=( "${EXTRA_ARGS[@]}" )
    fi

    echo
    echo "[ $((idx + 1)) / $total ] Running $model via $script"
    printf "Command:%s\n" "$(printf ' %q' "${cmd[@]}")"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "Dry run enabled; skipping execution."
        SUCCESSES+=("$model")
        continue
    fi

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
