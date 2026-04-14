#!/usr/bin/env bash
# ============================================================
# Baseline: ads-only HSTU on benchmark v4
# Run from baseline/train/ directory
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BASELINE_DIR")"
TRAIN_DIR="$BASELINE_DIR/train"

# ── Paths (edit for your machine) ──
DATA_INPUT="${DATA_INPUT:-/home/yourslewis/lrm_benchmarkv4/train/train_chunk_00.tsv}"
PROCESSED_DIR="${PROCESSED_DIR:-/home/yourslewis/lrm_benchmarkv4/processed/ads_only}"
GIN_CONFIG="$BASELINE_DIR/config/baseline_ads_only.gin"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/baseline_result}"
COMMIT_HASH=$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
RUN_DIR="$OUTPUT_DIR/$COMMIT_HASH-$(date +%Y%m%d-%H%M%S)"

echo "========================================"
echo "  Baseline: Ads-Only HSTU"
echo "  Commit:   $COMMIT_HASH"
echo "  Output:   $RUN_DIR"
echo "========================================"

# ── Step 1: Data preparation (if not already done) ──
if [ ! -d "$PROCESSED_DIR/train" ]; then
    echo "[1/3] Converting benchmark v4 → HSTU format (ads_only)..."
    python "$PROJECT_ROOT/data_prep/convert_benchmarkv4.py" \
        --input "$DATA_INPUT" \
        --output_dir "$PROCESSED_DIR" \
        --mode ads_only \
        --eval_users 500 \
        --min_ad_events_eval 5 \
        --seed 42 \
        --num_train_files 4 \
        --num_eval_files 1
else
    echo "[1/3] Processed data already exists at $PROCESSED_DIR, skipping."
fi

# ── Step 2: Training ──
echo "[2/3] Training baseline..."
mkdir -p "$RUN_DIR"

cd "$TRAIN_DIR"
torchrun --nproc_per_node=1 main.py \
    --gin_config_file="$GIN_CONFIG" \
    --data_path="$PROCESSED_DIR" \
    --ads_semantic_embd_path="$PROCESSED_DIR/embeddings/domain_0" \
    --output_path="$RUN_DIR" \
    --mode=local \
    2>&1 | tee "$RUN_DIR/train.log"

echo "[3/3] Done! Results in $RUN_DIR"
echo "  Checkpoints: $RUN_DIR/ckpts/"
echo "  Logs:        $RUN_DIR/train.log"
echo "  TensorBoard: tensorboard --logdir $RUN_DIR"
