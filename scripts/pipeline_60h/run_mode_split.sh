#!/bin/bash
# Split the 60h pipeline across two servers, balanced by measured cpu-hours
# per (experiment, mode) bucket, with exp1 seed_ext widened from 3 new seeds
# to the full 6-seed sweep. exp5 (runtime pass) is included too, but kept out
# of the concurrent bucket queue: it's model-list-split in half (LPT-balanced
# from its own measured per-model timings) and run serially (MAX_JOBS=1,
# matching its GPU-contention-free intent) before the main queue on each
# server, so it neither races itself nor overlaps the GPU-heavy bucket jobs.
#
# Bucket cpu-hours (seed_ext doubled for 6 seeds):
#   seedext-til 83.9  seedext-cil 43.7  memsweep-til 30.4  memsweep-cil 10.7
#   taskorder-til 7.2 taskorder-cil 2.5 snr-til 82.1       snr-cil 43.4
#   runtime pass (10+10 models, serial): ~30.4 min per server, negligible
#
#   Server A (~148 cpu-h, ~37h wall @ MAX_JOBS=4, +~30min serial runtime pass):
#     seedext-til, memsweep-cil, taskorder-til, taskorder-cil, snr-cil
#   Server B (~156 cpu-h, ~39h wall @ MAX_JOBS=4, +~30min serial runtime pass):
#     seedext-cil, memsweep-til, snr-til
#
# Usage:
#   scripts/pipeline_60h/run_mode_split.sh A          # run server A's share
#   scripts/pipeline_60h/run_mode_split.sh B          # run server B's share
#   scripts/pipeline_60h/run_mode_split.sh A --list   # print server A's job specs only
#
# Each server should point PIPELINE_LOG_ROOT/DONE_FILE somewhere private to
# it (defaulted below to logs/pipeline_60h_split/server<A|B>) so the two
# runs' done-tracking and driver logs never collide, even if they share a
# filesystem. MAX_JOBS and FORCE are still read from the environment as
# usual (see lib_queue.sh).
#
# Run output (checkpoints/metrics) is separate from the above: every job's
# --log_dir is logs/<PIPELINE_DATE>_full-pipeline/<group>/saved_models,
# matching the logs/00_sync/<group>/saved_models layout directly. Pin
# PIPELINE_DATE to the same value on both servers (e.g. `export
# PIPELINE_DATE=20260922` before launching each) so their output lands
# under one shared date-stamped root instead of drifting if launched on
# different calendar days.

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVER="${1:-}"
LIST_ONLY=0
[ "${2:-}" = "--list" ] && LIST_ONLY=1

case "$SERVER" in
    A)
        BUCKETS=(
            "exp1_seed_extension.sh:til"
            "exp2_mem_sweep.sh:cil"
            "exp3_task_order.sh:til"
            "exp3_task_order.sh:cil"
            "exp4_snr_sweep.sh:cil"
        )
        RUNTIME_MODELS="ctn lamaml cmaml bcl_dual ucl agem icarl rwalk si ft"
        ;;
    B)
        BUCKETS=(
            "exp1_seed_extension.sh:cil"
            "exp2_mem_sweep.sh:til"
            "exp4_snr_sweep.sh:til"
        )
        RUNTIME_MODELS="iid2 gem smaml la-er packnet lwf er_ring eralg4 hat ewc"
        ;;
    *)
        echo "Usage: $0 {A|B} [--list]" >&2
        exit 1
        ;;
esac

# exp1_seed_extension.sh reads NEW_SEEDS from the environment; this is what
# widens it from the default 3 new seeds to the full 6-seed sweep.
export NEW_SEEDS="${NEW_SEEDS:-0,39,55,100,390,550}"

export PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-${PIPELINE_DIR}/../../logs/pipeline_60h_split/server${SERVER}}"
export DONE_FILE="${DONE_FILE:-${PIPELINE_LOG_ROOT}/done.txt}"

# shellcheck disable=SC1091
source "${PIPELINE_DIR}/lib_queue.sh"
# exp5 runs as a separate subprocess below (its own MAX_JOBS=1, its own
# done_runtime.txt); export PIPELINE_DATE so it resolves to the same
# date-stamped output root as this shell instead of recomputing "today"
# independently.
export PIPELINE_DATE

emit_fn_for_script() {
    case "$1" in
        exp1_seed_extension.sh) echo emit_seed_extension_jobs ;;
        exp2_mem_sweep.sh) echo emit_mem_sweep_jobs ;;
        exp3_task_order.sh) echo emit_task_order_jobs ;;
        exp4_snr_sweep.sh) echo emit_snr_sweep_jobs ;;
        *)
            echo "unknown bucket script: $1" >&2
            return 1
            ;;
    esac
}

# Prints this server's job specs, filtered to each bucket's mode, by
# sourcing each experiment script (for its emit_* function) and re-running
# only the specs whose mode field matches the bucket.
collect_bucket_specs() {
    local bucket script mode emit_fn spec spec_mode
    for bucket in "${BUCKETS[@]}"; do
        script="${bucket%%:*}"
        mode="${bucket##*:}"
        emit_fn="$(emit_fn_for_script "$script")" || return 1
        # shellcheck disable=SC1090
        source "${PIPELINE_DIR}/${script}"
        while IFS= read -r spec; do
            spec_mode="${spec#*|}"
            spec_mode="${spec_mode%%|*}"
            [ "$spec_mode" = "$mode" ] && echo "$spec"
        done < <("$emit_fn")
    done
}

if [ "$LIST_ONLY" = "1" ]; then
    collect_bucket_specs
    TIL_MODELS="$RUNTIME_MODELS" "${PIPELINE_DIR}/exp5_runtime_single_pass.sh" --list
    exit 0
fi

mkdir -p "$PIPELINE_LOG_ROOT"

# Runtime pass first, serially (its own MAX_JOBS=1 and done_runtime.txt,
# both set inside exp5_runtime_single_pass.sh), so its GPU-timing
# measurements aren't distorted by the concurrent bucket queue below.
TIL_MODELS="$RUNTIME_MODELS" "${PIPELINE_DIR}/exp5_runtime_single_pass.sh"

mapfile -t SPECS < <(collect_bucket_specs)
run_job_queue "${SPECS[@]}"
