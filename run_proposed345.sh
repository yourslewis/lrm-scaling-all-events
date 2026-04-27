#!/bin/bash
set -euo pipefail
source /home/yourslewis/miniconda3/etc/profile.d/conda.sh
conda activate hstu

EMB_DIR=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings
RESULTS=/home/yourslewis/lrm-scaling-all-events/results_v2
TRAIN_DIR=/home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/train
EMB_ARGS5="--ads_semantic_embd_path ${EMB_DIR}/domain_0 --web_browsing_semantic_embd_path ${EMB_DIR}/domain_1 --shopping_semantic_embd_path ${EMB_DIR}/domain_2 --ads_pure_corpus_embd_path ${EMB_DIR}/domain_3 --other_semantic_embd_path ${EMB_DIR}/domain_4"

# proposed3 on GPU0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29600 ${TRAIN_DIR}/main.py \
  --mode=job \
  --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/proposed3_heavy_mmoe.gin \
  --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
  --output_path=${RESULTS}/proposed3_heavy_mmoe \
  ${EMB_ARGS5}

# proposed4 on GPU1
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29601 ${TRAIN_DIR}/main.py \
  --mode=job \
  --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/proposed4_transformer_mmoe.gin \
  --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
  --output_path=${RESULTS}/proposed4_transformer_mmoe \
  ${EMB_ARGS5}

# proposed5 on GPU0 (after proposed3)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29602 ${TRAIN_DIR}/main.py \
  --mode=job \
  --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/proposed5_leak_eventtype.gin \
  --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
  --output_path=${RESULTS}/proposed5_leak_eventtype \
  ${EMB_ARGS5}
