#!/bin/bash
# GPU 1: Proposed_1 then PLE
set -e
source /home/yourslewis/miniconda3/etc/profile.d/conda.sh
conda activate hstu
export CUDA_VISIBLE_DEVICES=1

EMB_DIR=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings
RESULTS=/home/yourslewis/lrm-scaling-all-events/results_v2
EMB_ARGS5="--ads_semantic_embd_path ${EMB_DIR}/domain_0 --web_browsing_semantic_embd_path ${EMB_DIR}/domain_1 --shopping_semantic_embd_path ${EMB_DIR}/domain_2 --ads_pure_corpus_embd_path ${EMB_DIR}/domain_3 --other_semantic_embd_path ${EMB_DIR}/domain_4"

echo "$(date) === GPU1: Experiment 2: Proposed 1 (all-events) ==="
cd /home/yourslewis/lrm-scaling-all-events/proposed_1-all_events/train
torchrun --nproc_per_node=1 --master_port=29501 main.py \
    --mode=job \
    --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_1-all_events/config/all_events_v2.gin \
    --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
    --output_path=${RESULTS}/proposed1_all_events \
    ${EMB_ARGS5}
echo "$(date) === GPU1: Experiment 2 Done ==="

echo "$(date) === GPU1: Experiment 4: PLE ==="
cd /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/train
torchrun --nproc_per_node=1 --master_port=29501 main.py \
    --mode=job \
    --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/ple_all_events_v2.gin \
    --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
    --output_path=${RESULTS}/proposed2b_ple \
    ${EMB_ARGS5}
echo "$(date) === GPU1: Experiment 4 Done ==="

echo "$(date) === GPU1: ALL DONE ==="
