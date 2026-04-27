# Evaluation

Approach-agnostic evaluation scripts. All experiment folders produce output in the format expected by these scripts.

## Expected Input Format
TBD — will match the inference output format from each experiment.

## Metrics
- NDCG@10 (primary)
- HR@10
- MRR
- Per-group breakdown: Ad, Purchase, Browsing, Search, Others
- Per-event-type breakdown (NativeClick, SearchClick, UET, etc.)

## Scripts
- `eval_by_group.py` — JSONL-based, approach-agnostic group evaluation
- `eval_per_event_type.py` — checkpoint-based eval for train codepaths; outputs Overall + group + per-event-type metrics
- `run_eval_all.sh` — convenience runner for current model checkpoints (baseline / proposed1 / proposed2a)
