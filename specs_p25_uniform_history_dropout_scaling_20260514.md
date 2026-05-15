# P25 Uniform History-Dropout Sequence Scaling — 2026-05-14

## Goal

Measure the scaling trend with respect to **effective sequence length** while holding model size, data source, gates, optimizer, and training steps as fixed as practical.

P24 directly increased `max_sequence_length`, but that forced very small physical batches for long sequences and confounded the result with undertraining. P25 instead samples from a larger original history window and pads only to the post-drop effective length.

## Sampling policy

For each row/user sequence:

1. Keep the target/label event unchanged.
2. Take the last `W` original history events before the target.
3. Uniformly sample `ceil(W_observed * keep_rate)` history events without replacement.
4. Preserve chronological order.
5. Append the original target event.
6. Pad/truncate to `model_max_sequence_length + 1` including target.

No recency bias. No special treatment for recent events.

The sampling mask is deterministic by `user_id` and `history_sample_seed` so train/eval are reproducible.

## Grid

Fixed `keep_rate = 0.5`.

| Run | Original history window `W` | Keep rate | Expected effective history length | `get_reco_dataset.max_sequence_length` | Batch |
|---|---:|---:|---:|---:|---:|
| `p25_w0100_drop50_s10_p09_m01_o00` | 100 | 0.5 | ~50 | 50 | 16 |
| `p25_w0200_drop50_s10_p09_m01_o00` | 200 | 0.5 | ~100 | 100 | 16 |
| `p25_w0500_drop50_s10_p09_m01_o00` | 500 | 0.5 | ~250 | 250 | 16 |
| `p25_w1000_drop50_s10_p09_m01_o00` | 1000 | 0.5 | ~500 | 500 | 16 |

`max_sequence_length` is the model/effective history length, not the original sampling window.

## Fixed model/training settings

Use the P23 best gate setting as base:

```text
strong gate = 1.0
PageTitle gate = 0.9
MSN gate = 0.1
OutlookSenderDomain gate = 0.0
sigma = 300s
window = +/-10m
max_weight = 4.0
```

Other settings inherited from `p23_page_s10_p09_m01_o00`:

- same HSTU/MMoE model size
- same data source: `/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2`
- same optimizer/LR/warmup/loss
- same 26k max train steps
- same eval frequency: every 1000 steps

Physical batch is fixed at 16 across P25 variants to avoid the P24 confound where longer sequence runs used dramatically smaller batches.

## Metrics

Compare against P20/P23 baselines and across P25 variants:

- Overall HR@10
- Ads HR@10
- Overall NDCG@10
- Ads NDCG@10
- Score = `0.4 * Overall HR@10 + 0.6 * Ads HR@10`

Guardrails:

- Overall HR@10 should not collapse below 0.58 for a candidate to be considered balanced.
- Ads HR@10 should be compared to P23 best (~0.2323) and P20 baseline (~0.2121).

## Follow-up diagnostics

After candidates finish, run length-bucket diagnostics using the existing tertile evaluator if needed:

```text
eval/eval_by_sequence_length_tertile.py --quantiles 33,67,100
```

## Files

- Dataset implementation: `proposed_2-mmoe_ple/train/data/ads_datasets/semantic_next_event_prediction/semantic_next_event_prediction.py`
- Dataset config hook: `proposed_2-mmoe_ple/train/data/reco_dataset.py`
- Generated configs: `proposed_2-mmoe_ple/config/generated_p25_uniform_dropout_scaling/*.gin`
- Runner: `scripts/p25_uniform_dropout_scaling_autoresearch.py`
