#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="/home/yourslewis/lrm-launches/m2-newdata-src-20260530/results_v2/m2_newdata_baseline/m2_aux_light_hstu_newdata"
LAUNCH="$RUN_DIR/launch.json"
if [[ ! -f "$LAUNCH" ]]; then
  echo "launch.json not found: $LAUNCH" >&2
  exit 2
fi
PID=$(python3 - <<'PY'
import json
print(json.load(open('/home/yourslewis/lrm-launches/m2-newdata-src-20260530/results_v2/m2_newdata_baseline/m2_aux_light_hstu_newdata/launch.json'))['pid'])
PY
)
LOG=$(python3 - <<'PY'
import json
print(json.load(open('/home/yourslewis/lrm-launches/m2-newdata-src-20260530/results_v2/m2_newdata_baseline/m2_aux_light_hstu_newdata/launch.json'))['log'])
PY
)
echo "== process =="
ps -fp "$PID" || true
echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo "== compute apps =="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
echo "== log tail =="
tail -80 "$LOG" || true
