#!/usr/bin/env bash
# Smoke-test observe() metric logits: one Tier-1 model, one Tier-2 model, one partial (la-er).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source la-maml_env/bin/activate

BASE_CONFIG="configs/observe_logits_smoke.yaml"
LOG_ROOT="logs/observe_logits_smoke_runs"
mkdir -p "$LOG_ROOT"

# name:yaml:tier_label
RUNS=(
  "agem:configs/models/til/agem.yaml:tier1_returns_logits"
  "cmaml:configs/models/til/cmaml.yaml:tier2_returns_none_fallback"
  "la-er:configs/models/til/la-er.yaml:tier2_partial_live_logits_if_3adc"
)

overall_exit=0
for entry in "${RUNS[@]}"; do
  IFS=: read -r name model_yaml tier <<<"$entry"
  job_log="${LOG_ROOT}/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== Running ${name} (${tier}) ===" | tee -a "${LOG_ROOT}/summary.log"
  if python3 main.py --config "$BASE_CONFIG" --config "$model_yaml" >"$job_log" 2>&1; then
    echo "PASS ${name}: exit 0 (log: $job_log)" | tee -a "${LOG_ROOT}/summary.log"
    if [ -f logs/observe_logits_smoke/metrics/task0.npz ]; then
      python3 - <<'PY' "$job_log"
import sys
from pathlib import Path
import numpy as np

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="replace")
npz = Path("logs/observe_logits_smoke/metrics/task0.npz")
data = np.load(npz)
keys = sorted(data.files)
print(f"  metrics keys: {keys}")
if "cls_tr_rec" in data:
    rec = data["cls_tr_rec"]
    print(f"  cls_tr_rec: shape={rec.shape} sample={rec.flat[:3]}")
# Heuristic: fallback path often follows observe with eval; Tier-1 should train without errors.
if "ERROR" in text or "Traceback" in text:
    print("  WARNING: log contains traceback")
PY
    fi
  else
    echo "FAIL ${name}: see $job_log" | tee -a "${LOG_ROOT}/summary.log"
    tail -n 40 "$job_log" | tee -a "${LOG_ROOT}/summary.log"
    overall_exit=1
  fi
  echo "" | tee -a "${LOG_ROOT}/summary.log"
done

exit "$overall_exit"
