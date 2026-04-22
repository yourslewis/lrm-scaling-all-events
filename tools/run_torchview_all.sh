#!/bin/bash
set -euo pipefail
source /home/yourslewis/miniconda3/etc/profile.d/conda.sh
conda activate hstu

TOOL=/home/yourslewis/lrm-scaling-all-events/tools/visualize_model.py
EMB=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings
DATA=/home/yourslewis/lrm_benchmarkv4/processed
OUTDIR=/home/yourslewis/lrm-scaling-all-events/results_v2/torchview_diagrams
mkdir -p "$OUTDIR"

EMB_ARGS="--ads_semantic_embd_path ${EMB}/domain_0 --web_browsing_semantic_embd_path ${EMB}/domain_1 --shopping_semantic_embd_path ${EMB}/domain_2 --ads_pure_corpus_embd_path ${EMB}/domain_3 --other_semantic_embd_path ${EMB}/domain_4"
BASE=/home/yourslewis/lrm-scaling-all-events

# Each experiment: name, gin_config, train_dir, data_path
declare -a EXPERIMENTS=(
    "baseline|${BASE}/baseline/config/baseline_ads_only_v2.gin|${BASE}/baseline/train|${DATA}/all_events_v2"
    "proposed1_all_events|${BASE}/proposed_1-all_events/config/all_events_v2.gin|${BASE}/proposed_1-all_events/train|${DATA}/all_events_v2"
    "proposed2a_mmoe|${BASE}/proposed_2-mmoe_ple/config/mmoe_all_events_v2.gin|${BASE}/proposed_2-mmoe_ple/train|${DATA}/all_events_v2"
    "proposed2b_ple|${BASE}/proposed_2-mmoe_ple/config/ple_all_events_v2.gin|${BASE}/proposed_2-mmoe_ple/train|${DATA}/all_events_v2"
    "proposed3_heavy_mmoe|${BASE}/proposed_2-mmoe_ple/config/proposed3_heavy_mmoe.gin|${BASE}/proposed_2-mmoe_ple/train|${DATA}/all_events_v2"
    "proposed4_transformer_mmoe|${BASE}/proposed_2-mmoe_ple/config/proposed4_transformer_mmoe.gin|${BASE}/proposed_2-mmoe_ple/train|${DATA}/all_events_v2"
    "proposed5_leak_eventtype|${BASE}/proposed_2-mmoe_ple/config/proposed5_leak_eventtype.gin|${BASE}/proposed_2-mmoe_ple/train|${DATA}/all_events_v2"
    "proposed6_hstu_mmoe|${BASE}/proposed_2-mmoe_ple/config/proposed6_hstu_mmoe.gin|${BASE}/proposed_2-mmoe_ple/train|${DATA}/all_events_v2"
)

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name gin_config train_dir data_path <<< "$entry"
    echo ""
    echo "=========================================="
    echo "  Generating torchview for: $name"
    echo "=========================================="
    python "$TOOL" \
        --gin_config_file "$gin_config" \
        --data_path "$data_path" \
        $EMB_ARGS \
        --output_dir "$OUTDIR" \
        --output_prefix "$name" \
        --train_dir "$train_dir" \
        --depths 1,2 \
        2>&1 || echo "[WARN] $name failed, continuing..."
    echo ""
done

echo "=========================================="
echo "All done. Output in: $OUTDIR"
ls -lh "$OUTDIR"/*.png 2>/dev/null || echo "No PNGs generated"
