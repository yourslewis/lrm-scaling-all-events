# P33-P37 — Hard-negative follow-up specs after P32

## Context
P28-P31 suggest that making global/domain-aware random negatives the dominant training signal is harmful versus the P23 in-batch reference. P29b showed useful Ads lift from a controlled hard-negative mix, but P30 rank-window sweeps and P31 denoising did not produce a robust Ads/overall tradeoff.

P32 tests a hybrid split with in-batch negatives restored as the largest component:
- 50% domain-aware in-batch random,
- 30% global domain-aware hard,
- 20% global domain-aware random,
- fallback: unfilled in-batch slots are filled from global hard negatives.

These follow-up specs keep the survey lessons explicit: hard negatives should be controlled, mixed with stable random/in-batch negatives, optionally introduced by curriculum, and filtered to reduce false negatives.

## Shared baselines
- P23 coordinate-search / in-batch reference.
- P28 n64 global random final: OHR .509, AHR .212, NDCG .432.
- P29b mixed hard global best: OHR .499, AHR .283, NDCG .426.
- P30/P31: negative evidence for heavier global-hard/global-random variants.
- P32: current hybrid test.

## P33 — In-batch first, tiny global hard supplement

### Hypothesis
P23-style in-batch negatives are the stable core. Global hard negatives should be a small regularizer, not a large share of the training distribution.

### Proposed split
For N=32:
- 80% normal in-batch negatives (`~26`)
- 10% domain-aware in-batch negatives if enough same-domain candidates exist (`~3`)
- 10% global hard negatives (`~3`)
- no global random negatives, unless needed as final fallback

### Config sketch
- `make_model.sampling_strategy = "HybridDomainInBatchHardGlobalNegativesSampler"` or a new in-batch-first variant.
- `make_model.num_negatives = 32`
- hard candidate pool = 1024
- hard rank window = 32-512
- strict fallback order: normal in-batch -> same-domain in-batch -> global hard -> global random only if necessary.

### Success criteria
- Preserve P23/P28-level OHR/NDCG.
- Improve AHR toward P29b without the P30/P31 global-heavy collapse.

## P34 — Curriculum hard negatives

### Hypothesis
Hard negatives are harmful early, before the query embedding is meaningful. Introduce them only after warmup.

### Schedule
For N=32:
- 0-5k: 100% in-batch
- 5k-15k: 90% in-batch, 10% global hard
- 15k+: 80% in-batch, 20% global hard

Optional variant:
- keep global random at 0-10% max as a regularizer.

### Implementation notes
- Add step-aware fractions to the sampler or runner/config.
- If step injection is awkward, launch chained continuations from checkpoints with different configs.

### Success criteria
- Avoid P30b-style early collapse.
- Match/improve P29b AHR while preserving overall NDCG.

## P35 — Ads-target-only hard negatives

### Hypothesis
Global hard negatives may be useful mainly for Ads targets. Applying them to all domains hurts broad representation learning and high-volume non-Ads domains.

### Proposed behavior
- For NativeClick/SearchClick targets:
  - use in-batch + global hard hybrid.
- For non-Ads targets:
  - use P23-style in-batch only, or at most tiny global supplement.

### Config sketch
- `hard_negative_apply_event_types = [1, 2]`
- `non_ads_negative_strategy = "InBatch"`
- Ads split candidate: 70% in-batch, 30% global hard.

### Success criteria
- Improve AHR versus P28/P31.
- Avoid broad OHR/NDCG regression.

## P36 — Popularity/degree-filtered global negatives

### Hypothesis
Uniform global negatives over huge domain pools are either too easy/head-biased or noisy. Filter candidate pools to mid-frequency items to avoid both trivial negatives and near-positive/duplicate false negatives.

### Proposed filters
- Exclude extreme head items.
- Exclude extreme tail/rare items if embeddings are sparse/noisy.
- Sample from mid-frequency / mid-degree bucket.
- Keep domain/event-type routing.

### Implementation notes
- Requires item frequency / degree statistics from training data or embedding metadata.
- Build per-domain filtered candidate id lists offline.
- Reuse existing hard-rank scoring within filtered candidate pools.

### Success criteria
- Higher positive-vs-negative margin stability.
- Better AHR than random global without OHR collapse.

## P37 — GNNO / graph-neighborhood medium-hard negatives

### Hypothesis
Transition-graph medium-overlap negatives are better controlled than raw embedding nearest-neighbor hard negatives.

### Proposed mining
- Build weighted item transition/co-occurrence graph from sequential data.
- For each target item/event, sample negatives with medium neighborhood overlap.
- Avoid maximum-overlap candidates to reduce false negatives.

### Implementation ladder
1. Offline graph construction by domain/event type.
2. Precompute medium-overlap candidate lists.
3. Training sampler draws from these lists with in-batch fallback.

### Success criteria
- Improve Top-K alignment / AHR without P30-style instability.
- Particularly watch Ads and Purchase cohorts.

## Recommendation order
1. Wait for P32 signal.
2. If P32 underperforms: run P33 (in-batch-first, tiny global hard).
3. If P32/P33 show any Ads lift but instability: run P34 curriculum.
4. If Ads remains the main gap: run P35 Ads-target-only hard negatives.
5. Longer build: P36 filtered pools, then P37 GNNO medium-overlap mining.

---

## Full-training-data retrain round — P14/P20/P23/P28/P29/P31/P32/P33B

### Goal

The old P14→P33B comparisons were trained on the smaller `all_events_v2` processed split and then re-evaluated on the frozen v3 validation slice. This round asks a different question:

> If the same model/negative-sampling recipes are retrained on the full v3-preserve training set, do the frozen-7 OHR/AHR transfer metrics improve?

This is not a new architecture search. It is a controlled data-scale retrain of the existing recipes.

### Data and evaluation protocol

Training data:

```text
/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve/train
```

Frozen validation target:

```text
/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_eval7raw_0_6/eval
```

The runner materializes a hybrid dataset path with the full-train split and frozen-7 eval split:

```text
/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_train_frozen7eval
```

Use v3 full-preserve semantic embeddings:

```text
/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve/domain_{0..4}
```

All final reported `FullTrain OHR` / `FullTrain AHR` values must come from full frozen-7 evaluation of the selected/best checkpoint, not from the small in-training validation window.

### Model recipes to retrain sequentially

Retrain one model at a time, preserving each original recipe:

| Table id | Recipe to preserve | Config source |
|---|---|---|
| `P14_latest` | stabilized target-event-group residual baseline | `proposed14_stabilized_group_residual.gin` |
| `P20_s300_page10_best` | P20 ad-anchor grid, sigma=300, PageTitle gate=1.0 | `generated_p20_grid/p20_s300_page10.gin` |
| `P23_best` | coordinate-search selected ad-anchor gates, PageTitle=0.9 | `generated_p23_coordinate_search/p23_page_s10_p09_m01_o00.gin` |
| `P28n64_best` | domain-aware global random train negatives, n=64 | `generated_p28_domain_random_negatives/p28_domain_rand_n64.gin` |
| `P29b_best` | mixed hard/global negatives, hard fraction=0.25, n=32 | `generated_p29_mixed_hard_global_negatives/p29b_hardmix_f025_n32.gin` |
| `P31b_best` | denoised hard/global negatives, hard fraction=0.35, n=32 | `generated_p31_denoised_hard_negatives/p31b_denoised_hardmix_f035_n32.gin` |
| `P32a_best` | hybrid inbatch50/hard30/global20, n=32 | `generated_p32_hybrid_inbatch_hard_global/p32a_hybrid_inbatch50_hard30_global20_n32.gin` |
| `P33B_best` | broad/P23-style inbatch90 + hard10, n=32 | `generated_p33_inbatch_hard_followups/p33b_p23_inbatch90_hard10_n32.gin` |

### Monitoring and early stopping

Run sequentially, not as a parallel sweep. For each model:

1. Train with validation every 1,000 batches.
2. Track the transfer-oriented validation score:

```text
score = 0.4 * Overall_HR@10 + 0.6 * Ads_HR@10
```

3. Keep the validation-monitor best checkpoint.
4. Stop early once the score has peaked and failed to improve for the patience window:

```text
min_batch = 12,000
patience_batches = 8,000
max_batch = 70,000
```

5. After stop/completion, run full frozen-7 evaluation on the selected best checkpoint and write table-compatible rows to:

```text
results_v2/fulltrain_retrains_20260522/lrm_v3_fulltrain_eval_metrics.json
```

The local comparison report can mirror those rows into:

```text
/Users/yourslewis/.openclaw/workspace-rex/tmp/lrm_v3_fulltrain_eval_metrics.json
```

### Reporting columns

The hourly comparison table should keep the historical old-vs-frozen columns and add:

- `FullTrain OHR`: full-training-data retrain, frozen-7 Overall HR@10.
- `FullTrain AHR`: full-training-data retrain, frozen-7 Ads HR@10.

Until a model finishes retrain + final frozen-7 eval, render these as `—` rather than substituting old or baseline metrics.
