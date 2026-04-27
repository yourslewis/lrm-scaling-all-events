#!/bin/bash
# GPU 0: Baseline then MMoE
set -e
source /home/yourslewis/miniconda3/etc/profile.d/conda.sh
conda activate hstu
export CUDA_VISIBLE_DEVICES=0

EMB_DIR=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings
RESULTS=/home/yourslewis/lrm-scaling-all-events/results_v2
EMB_ARGS="--ads_semantic_embd_path ${EMB_DIR}/domain_0 --web_browsing_semantic_embd_path ${EMB_DIR}/domain_1 --shopping_semantic_embd_path ${EMB_DIR}/domain_2 --ads_pure_corpus_embd_path ${EMB_DIR}/domain_3"

echo "$(date) === GPU0: Experiment 1: Baseline (ads-only) ==="
cd /home/yourslewis/lrm-scaling-all-events/baseline/train
torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --mode=job \
    --gin_config_file=/home/yourslewis/lrm-scaling-all-events/baseline/config/baseline_ads_only_v2.gin \
    --data_path=/home/yourslewis/lrm_benchmarkv4/processed/ads_only_v2 \
    --output_path=${RESULTS}/baseline_ads_only \
    ${EMB_ARGS}
echo "$(date) === GPU0: Experiment 1 Done ==="

EMB_ARGS5="--ads_semantic_embd_path ${EMB_DIR}/domain_0 --web_browsing_semantic_embd_path ${EMB_DIR}/domain_1 --shopping_semantic_embd_path ${EMB_DIR}/domain_2 --ads_pure_corpus_embd_path ${EMB_DIR}/domain_3 --other_semantic_embd_path ${EMB_DIR}/domain_4"

echo "$(date) === GPU0: Experiment 3: MMoE ==="
cd /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/train
torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --mode=job \
    --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/mmoe_all_events_v2.gin \
    --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
    --output_path=${RESULTS}/proposed2a_mmoe \
    ${EMB_ARGS5}
echo "$(date) === GPU0: Experiment 3 Done ==="

echo "$(date) === GPU0: ALL DONE ==="
