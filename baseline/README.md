# Baseline: Ads-Only HSTU

Self-contained HSTU trained on ads events only (SearchClick, NativeClick).

## Architecture
- **Model:** HSTU (transformer-based sequential encoder)
- **Embeddings:** MultiDomainPrecomputed (random init, learned projection)
- **Loss:** Sampled Softmax (512 negatives, temp=0.05)
- **Eval:** Log perplexity + NDCG@10 per event group

## How to Run

```bash
# 1. Set data path (edit or export)
export DATA_INPUT=/home/yourslewis/lrm_benchmarkv4/train/train_chunk_00.tsv

# 2. Run the full pipeline (data prep + training)
bash run.sh
```

## Directory Layout
```
config/              # Gin configs
  baseline_ads_only.gin  # Default medium-scale config
data/                # Data loading code (dataset readers, collate, eval)
train/               # Training code (model, trainer, main.py)
infer/               # Inference and evaluation scripts
run.sh               # One-click run script
```

## Config: baseline_ads_only.gin
- Sequence length: 200
- Embedding dim: 128
- HSTU blocks: 16, heads: 4
- Batch size: 128, LR: 1e-3
- Epochs: 10
- Supervision: domain 0 only (ads)

## Metrics
Primary: **NDCG@10** (Ad events)
Secondary: HR@10, MRR, log_pplx
