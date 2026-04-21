#!/bin/bash
# Run per-event-type eval for all 3 models on GPU 0
set -e
source /home/yourslewis/miniconda3/etc/profile.d/conda.sh
conda activate hstu
export CUDA_VISIBLE_DEVICES=0

EMB_DIR=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings
RESULTS=/home/yourslewis/lrm-scaling-all-events/results_v2
EVAL_SCRIPT=/home/yourslewis/lrm-scaling-all-events/eval/eval_per_event_type.py
EMB4="--ads_semantic_embd_path ${EMB_DIR}/domain_0 --web_browsing_semantic_embd_path ${EMB_DIR}/domain_1 --shopping_semantic_embd_path ${EMB_DIR}/domain_2 --ads_pure_corpus_embd_path ${EMB_DIR}/domain_3"
EMB5="${EMB4} --other_semantic_embd_path ${EMB_DIR}/domain_4"

# Find best checkpoint (latest)
best_ckpt() {
    ls -1 "$1"/checkpoint_batch*.pt 2>/dev/null | sort | tail -1
}

echo "$(date) === Eval 1: Baseline (ads-only, 100ep) ==="
CKPT=$(best_ckpt ${RESULTS}/baseline_ads_only_100ep/None/ckpts)
echo "Using checkpoint: $CKPT"
cd /home/yourslewis/lrm-scaling-all-events/baseline/train
export PYTHONPATH=$(pwd):$PYTHONPATH
torchrun --nproc_per_node=1 --master_port=29510 $EVAL_SCRIPT \
    --gin_config_file=/home/yourslewis/lrm-scaling-all-events/baseline/config/baseline_ads_only_v2.gin \
    --data_path=/home/yourslewis/lrm_benchmarkv4/processed/ads_only_v2 \
    --ckpt_path=$CKPT \
    --mode=job \
    --eval_batches=200 \
    --output_json=${RESULTS}/baseline_ads_only_100ep_eval_per_event.json \
    ${EMB4}

echo ""
echo "$(date) === Eval 2: Proposed 1 (all-events vanilla HSTU) ==="
CKPT=$(best_ckpt ${RESULTS}/proposed1_all_events/None/ckpts)
echo "Using checkpoint: $CKPT"
cd /home/yourslewis/lrm-scaling-all-events/proposed_1-all_events/train
export PYTHONPATH=$(pwd):$PYTHONPATH
torchrun --nproc_per_node=1 --master_port=29510 $EVAL_SCRIPT \
    --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_1-all_events/config/all_events_v2.gin \
    --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
    --ckpt_path=$CKPT \
    --mode=job \
    --eval_batches=200 \
    --output_json=${RESULTS}/proposed1_all_events_eval_per_event.json \
    ${EMB5}

echo ""
echo "$(date) === Eval 3: Proposed 2a (MMoE) ==="
CKPT=$(best_ckpt ${RESULTS}/proposed2a_mmoe/None/ckpts)
echo "Using checkpoint: $CKPT"
cd /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/train
export PYTHONPATH=$(pwd):$PYTHONPATH
torchrun --nproc_per_node=1 --master_port=29510 $EVAL_SCRIPT \
    --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/mmoe_all_events_v2.gin \
    --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
    --ckpt_path=$CKPT \
    --mode=job \
    --eval_batches=200 \
    --output_json=${RESULTS}/proposed2a_mmoe_eval_per_event.json \
    ${EMB5}

echo ""
echo "$(date) === ALL EVAL DONE ==="
echo "Results:"
echo "  ${RESULTS}/baseline_ads_only_100ep_eval_per_event.json"
echo "  ${RESULTS}/proposed1_all_events_eval_per_event.json"
echo "  ${RESULTS}/proposed2a_mmoe_eval_per_event.json"
