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
