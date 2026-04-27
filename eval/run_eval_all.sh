#!/bin/bash
# Run per-event-type evaluation for baseline / proposed1 / proposed2a checkpoints.
#
# Usage (defaults assume GPU server layout):
#   bash eval/run_eval_all.sh
#
# Optional overrides:
#   ROOT_DIR=/path/to/lrm-scaling-all-events \
#   DATA_ROOT=/path/to/processed \
#   RESULTS_DIR=/path/to/results_v2 \
#   CUDA_VISIBLE_DEVICES=0 \
#   bash eval/run_eval_all.sh

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/yourslewis/lrm-scaling-all-events}"
DATA_ROOT="${DATA_ROOT:-/home/yourslewis/lrm_benchmarkv4/processed}"
EMB_DIR="${EMB_DIR:-${DATA_ROOT}/semantic_embeddings}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results_v2}"
EVAL_SCRIPT="${ROOT_DIR}/eval/eval_per_event_type.py"
EVAL_BATCHES="${EVAL_BATCHES:-200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Per-model defaults (override with env vars when needed)
BASELINE_TRAIN_DIR="${BASELINE_TRAIN_DIR:-${ROOT_DIR}/baseline/train}"
BASELINE_GIN="${BASELINE_GIN:-${ROOT_DIR}/baseline/config/baseline_ads_only_v2.gin}"
BASELINE_DATA="${BASELINE_DATA:-${DATA_ROOT}/ads_only_v2}"
BASELINE_CKPT_DIR="${BASELINE_CKPT_DIR:-${RESULTS_DIR}/baseline_ads_only_100ep/None/ckpts}"
BASELINE_OUT="${BASELINE_OUT:-${RESULTS_DIR}/baseline_ads_only_100ep_eval_per_event.json}"

PROPOSED1_TRAIN_DIR="${PROPOSED1_TRAIN_DIR:-${ROOT_DIR}/proposed_1-all_events/train}"
PROPOSED1_GIN="${PROPOSED1_GIN:-${ROOT_DIR}/proposed_1-all_events/config/all_events_v2.gin}"
PROPOSED1_DATA="${PROPOSED1_DATA:-${DATA_ROOT}/all_events_v2}"
PROPOSED1_CKPT_DIR="${PROPOSED1_CKPT_DIR:-${RESULTS_DIR}/proposed1_all_events/None/ckpts}"
PROPOSED1_OUT="${PROPOSED1_OUT:-${RESULTS_DIR}/proposed1_all_events_eval_per_event.json}"

PROPOSED2A_TRAIN_DIR="${PROPOSED2A_TRAIN_DIR:-${ROOT_DIR}/proposed_2-mmoe_ple/train}"
PROPOSED2A_GIN="${PROPOSED2A_GIN:-${ROOT_DIR}/proposed_2-mmoe_ple/config/mmoe_all_events_v2.gin}"
PROPOSED2A_DATA="${PROPOSED2A_DATA:-${DATA_ROOT}/all_events_v2}"
PROPOSED2A_CKPT_DIR="${PROPOSED2A_CKPT_DIR:-${RESULTS_DIR}/proposed2a_mmoe/None/ckpts}"
PROPOSED2A_OUT="${PROPOSED2A_OUT:-${RESULTS_DIR}/proposed2a_mmoe_eval_per_event.json}"

EMB4=(
  --ads_semantic_embd_path "${EMB_DIR}/domain_0"
  --web_browsing_semantic_embd_path "${EMB_DIR}/domain_1"
  --shopping_semantic_embd_path "${EMB_DIR}/domain_2"
  --ads_pure_corpus_embd_path "${EMB_DIR}/domain_3"
)
EMB5=(
  "${EMB4[@]}"
  --other_semantic_embd_path "${EMB_DIR}/domain_4"
)

if [[ -f /home/yourslewis/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/yourslewis/miniconda3/etc/profile.d/conda.sh
  conda activate hstu
fi

export CUDA_VISIBLE_DEVICES

best_ckpt() {
  local ckpt_dir="$1"
  ls -1 "${ckpt_dir}"/checkpoint_batch*.pt 2>/dev/null | sort | tail -1
}

run_one() {
  local name="$1"
  local train_dir="$2"
  local gin_file="$3"
  local data_path="$4"
  local ckpt_dir="$5"
  local out_json="$6"
  local port="$7"
  shift 7
  local emb_args=("$@")

  echo ""
  echo "$(date) === Eval: ${name} ==="

  if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "[SKIP] Eval script not found: ${EVAL_SCRIPT}"
    return 0
  fi
  if [[ ! -d "${train_dir}" ]]; then
    echo "[SKIP] Train dir not found: ${train_dir}"
    return 0
  fi
  if [[ ! -f "${gin_file}" ]]; then
    echo "[SKIP] Gin config not found: ${gin_file}"
    return 0
  fi
  if [[ ! -d "${data_path}" ]]; then
    echo "[SKIP] Data path not found: ${data_path}"
    return 0
  fi

  local ckpt
  ckpt="$(best_ckpt "${ckpt_dir}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[SKIP] No checkpoint found under: ${ckpt_dir}"
    return 0
  fi

  echo "Using checkpoint: ${ckpt}"
  mkdir -p "$(dirname "${out_json}")"

  cd "${train_dir}"
  export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

  torchrun --nproc_per_node=1 --master_port="${port}" "${EVAL_SCRIPT}" \
    --gin_config_file="${gin_file}" \
    --data_path="${data_path}" \
    --ckpt_path="${ckpt}" \
    --mode=job \
    --eval_batches="${EVAL_BATCHES}" \
    --output_json="${out_json}" \
    "${emb_args[@]}"
}

run_one "baseline_ads_only_100ep" \
  "${BASELINE_TRAIN_DIR}" "${BASELINE_GIN}" "${BASELINE_DATA}" "${BASELINE_CKPT_DIR}" "${BASELINE_OUT}" 29510 \
  "${EMB4[@]}"

run_one "proposed1_all_events" \
  "${PROPOSED1_TRAIN_DIR}" "${PROPOSED1_GIN}" "${PROPOSED1_DATA}" "${PROPOSED1_CKPT_DIR}" "${PROPOSED1_OUT}" 29511 \
  "${EMB5[@]}"

run_one "proposed2a_mmoe" \
  "${PROPOSED2A_TRAIN_DIR}" "${PROPOSED2A_GIN}" "${PROPOSED2A_DATA}" "${PROPOSED2A_CKPT_DIR}" "${PROPOSED2A_OUT}" 29512 \
  "${EMB5[@]}"

echo ""
echo "$(date) === EVAL FINISHED ==="
echo "Outputs (if generated):"
echo "  ${BASELINE_OUT}"
echo "  ${PROPOSED1_OUT}"
echo "  ${PROPOSED2A_OUT}"
