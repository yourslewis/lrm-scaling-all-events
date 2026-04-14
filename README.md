# LRM Scaling & All-Events Training

## Project Goals
- **Goal A (Scaling):** Validate scaling laws — jointly increasing sequence length, model size, and training set size improves NDCG@10.
- **Goal B (All-Events):** Train on all event types and show non-ads signals improve both non-ads AND ads prediction.

## Structure
```
eval/                              # Approach-agnostic evaluation (per-group NDCG@10)
baseline/                          # Ads-only HSTU (self-contained)
proposed_1-all_events/             # All-events vanilla HSTU (self-contained)
proposed_2-mmoe_ple/               # All-events + MMoE/PLE (self-contained)
data_prep/                         # Data pipeline: split, subsample, truncate
*_result/                          # NOT in git — large files, on GPU server only
```

## Each experiment folder contains:
- `data/` — data loader and preprocessing
- `train/` — training code
- `infer/` — inference code (outputs eval-expected format)
- `config/` — experiment configs
- `README.md` — how to run

## Experiment Tracking
- Every training run starts with a git commit
- Result subfolder created with commit ID
- Model checkpoints, logs, eval results saved there

## Primary Metric: NDCG@10
## Secondary: HR@10, MRR, log_pplx, per-group NDCG@10

## Data
- Source: Benchmark v4 (`/home/yourslewis/lrm_benchmarkv4/`)
- Temporary: single chunk (5,121 users, 3.9M events)
- Full dataset TBD
