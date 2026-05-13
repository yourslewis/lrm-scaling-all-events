# P19 — Sharp Bidirectional Ad-Anchor Weighting

## Motivation

P18 tested whether non-ad target events close to a future Ads anchor should receive larger loss weight. The result was directionally useful for Ads but too expensive for the rest of the model:

- P18 best checkpoint improved Ads vs P14:
  - Ads NDCG@10: P14 best ~0.0976 → P18 best ~0.1249
  - Ads HR@10: P14 best ~0.1457 → P18 best ~0.1744
- But P18 hurt the high-volume groups and overall quality:
  - Overall NDCG@10: P14 best ~0.5993 → P18 best ~0.4508
  - Search/Browsing/Purchase all degraded.

Root cause hypothesis: P18’s decay was far too broad:

```text
weight = 1 + 3 * exp(-dt / 7200s)
lookahead = 24h
```

This gives very large boosts to events that are not truly local to the ad click. Even events 10–20 minutes away remain close to max weight, and events ~1–2 hours away still get a large boost. Spot checks also show many same-minute and post-click events around Ads anchors, so P19 should use both sides of the anchor and a much sharper local window.

## Data facts from spot check

Processed dataset:

```text
/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2
```

Training/eval parquet schema only contains:

```text
user_id
encoded_ids
types
timestamps_unix
```

There is no explicit commercial intent score in the processed training data. Any intent gating must be derived from event type, embeddings, or empirical co-occurrence statistics.

Event types observed in all_events_v2:

```text
EdgePageTitle
MSN
UET
OrganicSearchQuery
EdgeSearchQuery
OutlookSenderDomain
EdgeShoppingCart
SearchClick
AbandonCart
UETShoppingView
NativeClick
ChromePageTitle
UETShoppingCart
EdgeShoppingPurchase
```

Ads anchors:

```text
NativeClick
SearchClick
```

Important ordering note: event arrays are not always chronological. Weighting must use timestamps, not array position. Same-timestamp events are common around ad anchors; tie handling should be explicit.

## Shared P19 design

All four settings are based on P14 architecture:

```text
P14 stabilized group residual + sharp bidirectional ad-anchor weighting
```

Shared behavior:

- Ads anchors: `NativeClick`, `SearchClick`
- Apply boost to non-ad target events only.
- Use both pre-anchor and post-anchor events.
- Use timestamp distance to nearest ad anchor, not sequence index distance.
- Use a sharp local decay so events outside ~10 minutes are near base weight.
- Base weight remains `1.0`.
- Max weight is configurable as `X`, default `4.0`.

For a target event `e_i` with timestamp `t_i`, find the nearest ad anchor on enabled sides:

```text
pre anchors:  t_ad <= t_i and t_i - t_ad <= pre_window_seconds
post anchors: t_ad >= t_i and t_ad - t_i <= post_window_seconds

dt_i = min(|t_i - t_ad| over eligible anchors)
```

Then compute proximity using Gaussian decay:

```text
proximity_i = exp(-0.5 * (dt_i / sigma_seconds)^2)
```

Default:

```text
max_weight X = 4.0
sigma_seconds = 180.0   # 3 minutes
pre_window_seconds = 600.0   # 10 minutes
post_window_seconds = 600.0  # 10 minutes
```

With `X=4.0`, `sigma=180s`:

```text
0 min   → weight ≈ 4.00
1 min   → weight ≈ 3.84
3 min   → weight ≈ 2.82
5 min   → weight ≈ 1.75
10 min  → weight ≈ 1.01
>10 min → weight = 1.00 / no anchor
```

General formula:

```text
weight_i = 1.0 + (X - 1.0) * proximity_i * side_multiplier_i * gate_i
```

Where:

- `side_multiplier_i` can differ for pre vs post anchor.
- `gate_i` is the setting-specific event gate.
- If no eligible anchor is found, `weight_i = 1.0`.
- Ads targets themselves are not boosted by this mechanism.

Recommended side multipliers for first pass:

```text
pre_side_multiplier = 1.0
post_side_multiplier = 1.0
```

A later ablation can make post-click weaker, e.g. `post_side_multiplier = 0.5`.

## Validation requirements

Do not judge by final checkpoint. Use validation monitor and per-event audit.

Record at least:

- Overall NDCG@10 / HR@10 / MRR
- Ads group NDCG@10 / HR@10 / MRR
- NativeClick and SearchClick metrics separately
- Browsing, Search, Purchase, Others group metrics
- Macro group NDCG@10
- Best checkpoint batch and final checkpoint metrics

Add diagnostics for the weight distribution:

```text
fraction weighted > 1.0
fraction weighted >= 1.5
fraction weighted >= 2.0
fraction weighted >= 3.0
fraction weighted >= 3.5
mean weight by event type
mean weight by group
pre-anchor vs post-anchor counts
```

This is mandatory because P18 likely reweighted too much of the dataset.

## P19A — Sharp bidirectional proximity only

### Goal

Test only the two core fixes:

1. Sharp local decay: events outside ~10 minutes should be near base weight.
2. Bidirectional anchor context: include both pre-ad and post-ad events.

No event-type filtering. Any non-ad target event within ±10 minutes of an Ads anchor can be boosted.

### Config

```gin
make_model.enable_ad_anchor_proximity_weighting = True

make_model.ad_anchor_types = ["NativeClick", "SearchClick"]
make_model.ad_anchor_apply_to_non_ad_only = True

make_model.ad_anchor_use_pre = True
make_model.ad_anchor_use_post = True

make_model.ad_anchor_pre_window_seconds = 600.0
make_model.ad_anchor_post_window_seconds = 600.0

make_model.ad_anchor_max_weight = 4.0
make_model.ad_anchor_sigma_seconds = 180.0

make_model.ad_anchor_pre_side_multiplier = 1.0
make_model.ad_anchor_post_side_multiplier = 1.0

make_model.ad_anchor_event_gate_mode = "none"
```

### Formula

```text
gate_i = 1.0
weight_i = 1.0 + (4.0 - 1.0) * proximity_i
```

### Expected outcome

P19A should tell us whether P18 failed mostly because of overly broad attribution. If Ads improves with less damage to Search/Browsing/Purchase, the sharp local window is working.

### Risk

High-volume weak events such as `MSN` and `EdgePageTitle` can still receive large same-minute boosts.

## P19B — Sharp bidirectional + event-type gate

### Goal

Add a hand-defined event-type gate so commercial/search/shopping events receive full boost, while weak/noisy event types receive little or no boost.

This is the recommended next run if only one setting can be trained.

### Config

```gin
make_model.enable_ad_anchor_proximity_weighting = True

make_model.ad_anchor_types = ["NativeClick", "SearchClick"]
make_model.ad_anchor_apply_to_non_ad_only = True

make_model.ad_anchor_use_pre = True
make_model.ad_anchor_use_post = True

make_model.ad_anchor_pre_window_seconds = 600.0
make_model.ad_anchor_post_window_seconds = 600.0

make_model.ad_anchor_max_weight = 4.0
make_model.ad_anchor_sigma_seconds = 180.0

make_model.ad_anchor_pre_side_multiplier = 1.0
make_model.ad_anchor_post_side_multiplier = 1.0

make_model.ad_anchor_event_gate_mode = "event_type"
```

### Event-type gate

```python
event_type_gate = {
    # Strong commercial/search/shopping intent
    "OrganicSearchQuery": 1.0,
    "EdgeSearchQuery": 1.0,
    "UET": 1.0,
    "UETShoppingView": 1.0,
    "UETShoppingCart": 1.0,
    "EdgeShoppingCart": 1.0,
    "EdgeShoppingPurchase": 1.0,
    "AbandonCart": 1.0,

    # Page/title context: useful but very high volume and noisy
    "EdgePageTitle": 0.4,
    "ChromePageTitle": 0.4,

    # Weak/noisy for ad intent
    "MSN": 0.1,
    "OutlookSenderDomain": 0.0,
}
```

### Formula

```text
gate_i = event_type_gate[type_i]
weight_i = 1.0 + (4.0 - 1.0) * proximity_i * gate_i
```

At `dt=0` with `X=4`:

```text
Search/UET/Shopping  → weight 4.0
EdgePageTitle        → weight 2.2
MSN                  → weight 1.3
OutlookSenderDomain  → weight 1.0
```

At `dt=10m`, all event types are approximately base weight.

### Expected outcome

P19B should preserve P18’s Ads gain while reducing collateral damage from high-volume weak events.

## P19C — Sharp bidirectional + semantic similarity gate

### Goal

Use semantic similarity between the target event embedding and the nearest Ads anchor embedding as a soft gate. This tests whether ad-relevant context can be identified semantically instead of manually by event type.

### Config

```gin
make_model.enable_ad_anchor_proximity_weighting = True

make_model.ad_anchor_types = ["NativeClick", "SearchClick"]
make_model.ad_anchor_apply_to_non_ad_only = True

make_model.ad_anchor_use_pre = True
make_model.ad_anchor_use_post = True

make_model.ad_anchor_pre_window_seconds = 600.0
make_model.ad_anchor_post_window_seconds = 600.0

make_model.ad_anchor_max_weight = 4.0
make_model.ad_anchor_sigma_seconds = 180.0

make_model.ad_anchor_pre_side_multiplier = 1.0
make_model.ad_anchor_post_side_multiplier = 1.0

make_model.ad_anchor_event_gate_mode = "semantic_similarity"

make_model.ad_anchor_semantic_sim_min = 0.20
make_model.ad_anchor_semantic_sim_max = 0.65
make_model.ad_anchor_semantic_gate_power = 1.0
```

### Formula

Compute cosine similarity:

```text
sim_i = cosine(embedding_i, embedding_nearest_anchor)
```

Map it to a gate:

```text
gate_i = clip(
    (sim_i - semantic_sim_min) / (semantic_sim_max - semantic_sim_min),
    0,
    1
) ^ semantic_gate_power
```

Then:

```text
weight_i = 1.0 + (4.0 - 1.0) * proximity_i * gate_i
```

### Optional safety variant

If pure semantic gating is too noisy, use semantic gate only for an allowlist:

```python
semantic_allowlist = {
    "OrganicSearchQuery",
    "EdgeSearchQuery",
    "UET",
    "UETShoppingView",
    "UETShoppingCart",
    "EdgeShoppingCart",
    "EdgeShoppingPurchase",
    "AbandonCart",
    "EdgePageTitle",
    "ChromePageTitle",
}
```

Exclude or zero out:

```python
"MSN"
"OutlookSenderDomain"
```

### Expected outcome

P19C may discover useful ad-adjacent context beyond manual event-type priors.

### Risks

- Ad anchor embeddings may not be semantically meaningful enough.
- Same-domain ID embeddings may not compare well across all event types.
- More expensive than P19B.

## P19D — Sharp bidirectional + empirical ad-near probability gate

### Goal

Learn the gate from train-split co-occurrence statistics:

```text
P(ad anchor within ±10m | event)
```

This tests whether data-driven ad-nearness priors outperform hand-set event-type gates.

### Empirical gate variants

#### D1: Event-type empirical probability

Robust and simple:

```text
p_type = P(ad anchor within ±10m | event_type)
```

#### D2: Encoded-ID empirical probability with event-type fallback

More specific but requires smoothing:

```text
p_id = P(ad anchor within ±10m | encoded_id)
```

Use train split only. Do not compute this on eval/test.

### Smoothing

For each encoded ID:

```text
smoothed_p_id =
    (near_ad_count[id] + alpha * p_type[type])
    / (total_count[id] + alpha)
```

Default:

```text
alpha = 100.0
min_count_for_id_gate = 20
fallback = event-type empirical gate
```

### Gate normalization

Normalize probabilities into `[0, 1]`:

```text
gate_i = clip(
    (p_i - p_low) / (p_high - p_low),
    0,
    1
) ^ empirical_gate_power
```

Recommended:

```text
p_low = global near-ad background rate
p_high = 95th percentile over eligible event IDs/types
empirical_gate_power = 1.0
```

### Config

```gin
make_model.enable_ad_anchor_proximity_weighting = True

make_model.ad_anchor_types = ["NativeClick", "SearchClick"]
make_model.ad_anchor_apply_to_non_ad_only = True

make_model.ad_anchor_use_pre = True
make_model.ad_anchor_use_post = True

make_model.ad_anchor_pre_window_seconds = 600.0
make_model.ad_anchor_post_window_seconds = 600.0

make_model.ad_anchor_max_weight = 4.0
make_model.ad_anchor_sigma_seconds = 180.0

make_model.ad_anchor_pre_side_multiplier = 1.0
make_model.ad_anchor_post_side_multiplier = 1.0

make_model.ad_anchor_event_gate_mode = "empirical_prob"

make_model.ad_anchor_empirical_prob_path = "/home/yourslewis/lrm-scaling-all-events/data/ad_anchor_empirical_prob_10m_train_only.json"
make_model.ad_anchor_empirical_scope = "encoded_id_with_type_fallback"
make_model.ad_anchor_empirical_alpha = 100.0
make_model.ad_anchor_empirical_min_count = 20
make_model.ad_anchor_empirical_gate_power = 1.0
make_model.ad_anchor_empirical_gate_clip_min = 0.0
make_model.ad_anchor_empirical_gate_clip_max = 1.0
```

### Formula

```text
gate_i = empirical_gate(encoded_id_i, type_i)
weight_i = 1.0 + (4.0 - 1.0) * proximity_i * gate_i
```

### Expected outcome

P19D can learn that some event IDs/types have strong empirical ad-near association while others do not.

### Risks

- Encoded-ID gate can overfit common IDs or ad-heavy users.
- Must smooth aggressively.
- Must compute on train split only to avoid eval leakage.

## Recommended run order

If running all four:

```text
1. P19A — sharp bidirectional proximity sanity check
2. P19B — hand event-type gate; likely best practical setting
3. P19D — empirical probability gate if B works
4. P19C — semantic gate if embedding similarity appears meaningful
```

If running only one next:

```text
Run P19B.
```

It directly addresses P18’s biggest failure mode: large boosts for high-volume weak event types such as `MSN` and noisy `EdgePageTitle`.

## Implementation notes

- Existing P18 implementation should not be reused as-is because it is future-only and uses broad exponential decay.
- Implement timestamp-based nearest-anchor distance across both pre and post windows.
- Use `max`/nearest attribution rather than summing multiple anchors.
- Ensure same-timestamp behavior is deterministic:
  - timestamp distance is zero;
  - if multiple anchors exist at the same timestamp, use one nearest anchor but do not sum boosts.
- Keep Ads targets unmodified in the first version.
- Add diagnostic logging for event weights from the first training run.
