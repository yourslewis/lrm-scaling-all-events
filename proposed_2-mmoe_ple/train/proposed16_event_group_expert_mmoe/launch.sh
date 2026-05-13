#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/yourslewis/lrm-scaling-all-events}
DATA=${DATA:-/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2}
EMB=${EMB:-/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings}
cd "$ROOT/proposed_2-mmoe_ple/train"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} AZUREML_RUN_ID=${AZUREML_RUN_ID:-proposed16_event_group_expert_mmoe_20260510} \
  torchrun --nproc_per_node=1 --master_port=${MASTER_PORT:-30250} main.py \
  --gin_config_file="$ROOT/proposed_2-mmoe_ple/config/proposed16_event_group_expert_mmoe.gin" \
  --output_path="$ROOT/results_v2/proposed16_event_group_expert_mmoe" \
  --data_path "$DATA" --mode=job \
  --ads_semantic_embd_path "$EMB/domain_0" \
  --web_browsing_semantic_embd_path "$EMB/domain_1" \
  --shopping_semantic_embd_path "$EMB/domain_2" \
  --ads_pure_corpus_embd_path "$EMB/domain_3" \
  --other_semantic_embd_path "$EMB/domain_4"
