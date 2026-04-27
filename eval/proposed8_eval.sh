#!/bin/bash
# Eval proposed8 balanced-weight MMoE — run immediately after training completes
set -e
source /home/yourslewis/miniconda3/etc/profile.d/conda.sh
conda activate hstu
export CUDA_VISIBLE_DEVICES=0

EMB_DIR=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings
RESULTS=/home/yourslewis/lrm-scaling-all-events/results_v2
EVAL_SCRIPT=/home/yourslewis/lrm-scaling-all-events/eval/eval_per_event_type.py
EMB5="--ads_semantic_embd_path ${EMB_DIR}/domain_0 --web_browsing_semantic_embd_path ${EMB_DIR}/domain_1 --shopping_semantic_embd_path ${EMB_DIR}/domain_2 --ads_pure_corpus_embd_path ${EMB_DIR}/domain_3 --other_semantic_embd_path ${EMB_DIR}/domain_4"

# Find best checkpoint (latest)
best_ckpt() {
    ls -1 "$1"/checkpoint_batch*.pt 2>/dev/null | sort | tail -1
}

echo "$(date) === Eval: Proposed8 (Balanced MMoE) ==="
CKPT=$(best_ckpt ${RESULTS}/proposed8_balanced_mmoe/None/ckpts)
echo "Using checkpoint: $CKPT"
cd /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/train
export PYTHONPATH=$(pwd):$PYTHONPATH
torchrun --nproc_per_node=1 --master_port=29510 $EVAL_SCRIPT \
    --gin_config_file=/home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/proposed8_balanced_mmoe.gin \
    --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
    --ckpt_path=$CKPT \
    --mode=job \
    --eval_batches=200 \
    --output_json=${RESULTS}/proposed8_balanced_mmoe_eval_per_event.json \
    ${EMB5}

echo ""
echo "$(date) === Proposed8 Eval DONE ==="
echo "Results: ${RESULTS}/proposed8_balanced_mmoe_eval_per_event.json"
echo ""
echo "=== Quick comparison ==="
echo "Proposed2a (weighted MMoE):"
python3 -c "import json; d=json.load(open('${RESULTS}/proposed2a_mmoe_eval_per_event.json')); print(json.dumps(d.get('overall',d), indent=2))" 2>/dev/null || echo "(not available)"
echo ""
echo "Proposed8 (balanced MMoE):"
python3 -c "import json; d=json.load(open('${RESULTS}/proposed8_balanced_mmoe_eval_per_event.json')); print(json.dumps(d.get('overall',d), indent=2))"
