#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/yourslewis/lrm-launches/m2-newdata-src-20260530}
SEQ_LEN=${1:-${SEQ_LEN:-}}
if [ -z "$SEQ_LEN" ]; then
  echo "Usage: SEQ_LEN=400 $0  # or: $0 400" >&2
  exit 2
fi
case "$SEQ_LEN" in
  100|200|400|800) ;;
  *) echo "Unsupported SEQ_LEN=$SEQ_LEN (expected 100, 200, 400, or 800)" >&2; exit 2 ;;
esac

if [ "$SEQ_LEN" = "200" ]; then
  TRAIN_RUN=${TRAIN_RUN:-$ROOT/results_v2/m3_data_scale/m3_data_scale_50pct}
else
  TRAIN_RUN=${TRAIN_RUN:-$ROOT/results_v2/m4_seq_len_50pct/m4_seq_len_l${SEQ_LEN}_50pct}
fi
PROXY_BASE=/home/yourslewis/lrm_benchmarkv4/processed/m4_seq_len_50pct_l${SEQ_LEN}_proxy_500_seed1
SOURCE_PROXY_BASE=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1
PY=${PY:-/home/yourslewis/miniconda3/envs/hstu/bin/python}
DEVICE=${DEVICE:-cuda:1}
GPU_ID=${GPU_ID:-1}
LOCK=$PROXY_BASE/m4_seq_len_l${SEQ_LEN}_50pct_eval_500bank.lock
LOG_DIR=$PROXY_BASE/m4_seq_len_l${SEQ_LEN}_50pct_eval_500bank_logs
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/driver_$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"$LOG" 2>&1

echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ) seq_len=$SEQ_LEN data=50pct device=$DEVICE [500-BANK]"
if mkdir "$LOCK" 2>/dev/null; then
  trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
else
  echo "SKIP: another M4 seq-len L${SEQ_LEN} 50pct [500-BANK] eval is active ($LOCK)"
  exit 0
fi

if pgrep -u yourslewis -f "m4_seq_len_l${SEQ_LEN}_50pct_eval_500bank_.*(encode_target_queries|encode_documents|score_encoded_proxy)" >/dev/null; then
  echo "SKIP: existing M4 seq-len L${SEQ_LEN} 50pct [500-BANK] eval process active"
  exit 0
fi

GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ' || echo 999999)
echo "GPU${GPU_ID}_MEM_USED_MB=$GPU_USED"
if [ "${GPU_USED:-999999}" -gt 2000 ]; then echo "SKIP: GPU$GPU_ID not idle enough"; exit 0; fi

VAL=$TRAIN_RUN/validation_monitor.json
# Launchers write real artifacts under $TRAIN_RUN/<AZUREML_RUN_ID>/; resolve latest child if needed.
if [ ! -f "$VAL" ]; then
  CHILD_RUN=$(find "$TRAIN_RUN" -mindepth 1 -maxdepth 1 -type d -name "m4_seq_len_l${SEQ_LEN}_50pct_*" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2- || true)
  if [ -z "${CHILD_RUN:-}" ] && [ "$SEQ_LEN" = "200" ]; then
    CHILD_RUN=$(find "$TRAIN_RUN" -mindepth 1 -maxdepth 1 -type d -name "m3_data_scale_50pct_*" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d" " -f2- || true)
  fi
  if [ -n "${CHILD_RUN:-}" ] && [ -f "$CHILD_RUN/validation_monitor.json" ]; then
    TRAIN_RUN="$CHILD_RUN"
    VAL="$TRAIN_RUN/validation_monitor.json"
    echo "RESOLVED_CHILD_TRAIN_RUN=$TRAIN_RUN"
  fi
fi
if [ ! -f "$VAL" ]; then echo "SKIP: validation_monitor missing: $VAL"; exit 0; fi
BATCH=$($PY - <<PY
import json
j=json.load(open('$VAL'))
print(j.get('latest',{}).get('batch') or j.get('best',{}).get('batch') or '')
PY
)
if [ -z "$BATCH" ]; then echo "SKIP: cannot determine latest batch"; exit 0; fi
printf -v BATCH_PAD "%07d" "$BATCH"
CKPT=$TRAIN_RUN/ckpts/checkpoint_batch${BATCH_PAD}.pt
if [ ! -f "$CKPT" ]; then
  CKPT=$(ls -1t "$TRAIN_RUN"/ckpts/checkpoint_batch*.pt 2>/dev/null | head -1 || true)
  BATCH_PAD=$(basename "$CKPT" | sed -E 's/[^0-9]*([0-9]+).*/\1/')
fi
if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then echo "SKIP: checkpoint not found"; exit 0; fi

RUN_ID=m4_seq_len_l${SEQ_LEN}_50pct_eval_500bank_b${BATCH_PAD}
RUN_ROOT=$PROXY_BASE/${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)
DONE_MARKER=$PROXY_BASE/m4_seq_len_l${SEQ_LEN}_50pct_eval_500bank_done_batches.txt
if [ -f "$DONE_MARKER" ] && grep -qx "$BATCH_PAD" "$DONE_MARKER"; then echo "SKIP: batch $BATCH_PAD already evaluated [500-BANK]"; exit 0; fi
mkdir -p "$RUN_ROOT"
cd /home/yourslewis/lrm-scaling-all-events

TARGET_SIDECAR_GLOB='/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase7_compact_generation/full_sidecars/targets/*.parquet'
SELECTED_BANK_SUBSET=$SOURCE_PROXY_BASE/selected_banks_seed1_100_per_domain.json
HISTORY_PREFIX_SOURCE=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001
BANK_ROOT=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_history_slice_reproduction/package/artifacts/production_banked_candidates
BANK_GENERATOR=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_history_slice_reproduction/package/vendor/banked_candidate_generator_v001.py
HISTORY_READER=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_history_slice_reproduction/package/vendor/history_prefix_reader_v001.py
SOURCE_ROOT=$ROOT
GIN_CONFIG=$ROOT/proposed_2-mmoe_ple/config/generated_m4_seq_len_50pct/m4_aux_light_hstu_50pct_l${SEQ_LEN}.gin
EMBEDDING_ROOT=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve
CONTEXT_POLICY=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_history_slice_reproduction/package/context_policy.json
RAW_BANK_CACHE_DIR=$SOURCE_PROXY_BASE/raw_bank_cache_384fp16
MODEL_SUBMISSION_ID=m4_seq_len_l${SEQ_LEN}_50pct.${BATCH_PAD}.proxy500
PREDICTION_RUN_ID=$RUN_ID

if [ ! -f "$GIN_CONFIG" ]; then echo "SKIP: gin config missing: $GIN_CONFIG"; exit 0; fi

{
  echo "title=M4 seq-length L${SEQ_LEN} 50pct [500-BANK] proxy eval"
  echo "run_root=$RUN_ROOT"
  echo "checkpoint=$CKPT"
  echo "checkpoint_batch=$BATCH_PAD"
  echo "gin_config=$GIN_CONFIG"
  echo "source_root=$SOURCE_ROOT"
  echo "selected_bank_subset=$SELECTED_BANK_SUBSET"
  echo "bank_count=500"
  echo "seq_len=$SEQ_LEN"
  echo "device=$DEVICE"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "training_validation_monitor=$VAL"
  cat "$VAL"
} > "$RUN_ROOT/run_metadata.txt"

printf '{"stage":"encode_target_queries","seq_len":"%s","batch":"%s","bank_count":500,"at":"%s"}\n' "$SEQ_LEN" "$BATCH_PAD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/progress.jsonl"
$PY evaluation_harness/lrm_v001/bin/encode_target_queries.py \
  --target-sidecar-glob "$TARGET_SIDECAR_GLOB" \
  --selected-bank-subset "$SELECTED_BANK_SUBSET" \
  --history-prefix-source "$HISTORY_PREFIX_SOURCE" \
  --history-reader "$HISTORY_READER" \
  --source-root "$SOURCE_ROOT" \
  --gin-config-file "$GIN_CONFIG" \
  --checkpoint-path "$CKPT" \
  --embedding-root "$EMBEDDING_ROOT" \
  --context-policy "$CONTEXT_POLICY" \
  --output-query-cache-dir "$RUN_ROOT/query_cache" \
  --output-inference-log "$RUN_ROOT/query_encode_log.jsonl" \
  --query-dtype float32 \
  --target-batch-size 200000 \
  --encode-batch-size 4096 \
  --device "$DEVICE" \
  --max-sequence-length "$SEQ_LEN" \
  --context-checksum-mode none \
  --stdout-progress-every 10000

printf '{"stage":"encode_documents","seq_len":"%s","batch":"%s","bank_count":500,"at":"%s"}\n' "$SEQ_LEN" "$BATCH_PAD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/progress.jsonl"
$PY evaluation_harness/lrm_v001/bin/encode_documents.py \
  --bank-root "$BANK_ROOT" --bank-generator "$BANK_GENERATOR" \
  --selected-bank-subset "$SELECTED_BANK_SUBSET" \
  --source-root "$SOURCE_ROOT" --gin-config-file "$GIN_CONFIG" --checkpoint-path "$CKPT" \
  --embedding-root "$EMBEDDING_ROOT" --raw-bank-cache-dir "$RAW_BANK_CACHE_DIR" \
  --query-cache-dir "$RUN_ROOT/query_cache" --output-doc-cache-dir "$RUN_ROOT/doc_cache" \
  --output-inference-log "$RUN_ROOT/doc_encode_log.jsonl" --doc-dtype float16 --device "$DEVICE" --encode-batch-size 4096

printf '{"stage":"score_encoded_proxy","seq_len":"%s","batch":"%s","bank_count":500,"at":"%s"}\n' "$SEQ_LEN" "$BATCH_PAD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/progress.jsonl"
$PY evaluation_harness/lrm_v001/bin/score_encoded_proxy.py \
  --query-cache-dir "$RUN_ROOT/query_cache" --doc-cache-dir "$RUN_ROOT/doc_cache" \
  --bank-root "$BANK_ROOT" --bank-generator "$BANK_GENERATOR" \
  --selected-bank-subset "$SELECTED_BANK_SUBSET" \
  --model-submission-id "$MODEL_SUBMISSION_ID" --prediction-run-id "$PREDICTION_RUN_ID" \
  --output-compact "$RUN_ROOT/compact_predictions.jsonl" --output-metrics-json "$RUN_ROOT/compact_metrics.json" \
  --output-inference-log "$RUN_ROOT/score_log.jsonl" --compact-top-k 10 \
  --score-query-chunk-size 4096 --candidate-check-mode collisions --device "$DEVICE" \
  --log-flush-every 100 --output-flush-every 0 --stdout-progress-every 10000

printf '{"stage":"postprocess_longseq_gt1000","seq_len":"%s","batch":"%s","bank_count":500,"at":"%s"}\n' "$SEQ_LEN" "$BATCH_PAD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/progress.jsonl"
$PY "$ROOT/scripts/postprocess_m4_longseq_metrics.py" \
  --compact "$RUN_ROOT/compact_predictions.jsonl" \
  --output "$RUN_ROOT/compact_metrics_longseq_gt1000.json" \
  --context-bucket ctx_len_gt_1000 \
  --k 10

printf '{"stage":"done","seq_len":"%s","batch":"%s","bank_count":500,"at":"%s"}\n' "$SEQ_LEN" "$BATCH_PAD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/progress.jsonl"
echo "$BATCH_PAD" >> "$DONE_MARKER"
echo "$RUN_ROOT" > "$PROXY_BASE/latest_m4_seq_len_l${SEQ_LEN}_50pct_eval_500bank_run_dir.txt"
echo "DONE [500-BANK] seq_len=$SEQ_LEN run_root=$RUN_ROOT"
