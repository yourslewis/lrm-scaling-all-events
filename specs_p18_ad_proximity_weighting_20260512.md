# P18 — Ad-Proximity Weighted All-Events Training

## Motivation

Previous MMoE / multi-head experiments improved high-volume non-ad traffic but often collapsed Ads. We want non-ad events to still contribute to Ads, but not all non-ad events are equally useful for Ads. A non-ad event shortly before a future Ads event is likely more ad-relevant than a random non-ad event far away from any Ads event.

Example chronological pattern:

```text
Search → UET view → PageTitle → NativeClick ad
```

The pre-ad non-ad events should receive increasing emphasis as they get closer in wall-clock time to the future Ads anchor:

```text
Search:     medium boost
UET view:   large boost
PageTitle:  largest boost
Ad:         normal/explicit ad supervision
```

This keeps the all-events next-event objective but reweights supervision so ad-adjacent non-ad events matter more.

## Core idea

For each target event `e_i`, look forward in wall-clock time for future Ads anchors:

- Ads anchors: `NativeClick`, `SearchClick`
- Future condition: `t_ad > t_i`
- Lookahead condition: `t_ad - t_i <= lookahead_seconds`

Select the strongest future Ads anchor by time-decayed proximity:

```text
proximity_i = max_{future Ads a_j} exp(-(t_ad_j - t_i) / tau_seconds)
```

This `max` is nearest/strongest future-ad attribution. It prevents ad-dense sessions from repeatedly summing boosts into very large weights.

Then for non-ad targets:

```text
weight_i = base_nonad_weight + boost_scale * proximity_i
```

Default base remains `1.0`, so normal non-ad events still train the shared HSTU. `boost_scale > 1` makes ad-adjacent non-ad events significantly stronger than unrelated non-ad events, which matters because non-ad events are much more numerous.

For Ads targets, keep the normal domain/ad supervision weight unless explicitly configured otherwise.

## Recommended configs

### P18a conservative 2h

Good first run.

```text
base_nonad_weight = 1.0
boost_scale = 3.0
tau_seconds = 7200          # 2h
lookahead_seconds = 86400   # 24h max attribution window
aggregation = max / nearest future ad
apply_to_non_ad_only = true
```

### P18b short-session 30m

Tests whether only very tight pre-ad behavior matters.

```text
base_nonad_weight = 1.0
boost_scale = 4.0
tau_seconds = 1800          # 30m
lookahead_seconds = 7200    # 2h
aggregation = max
apply_to_non_ad_only = true
```

### P18c long-intent 24h

Tests longer pre-ad intent trails.

```text
base_nonad_weight = 1.0
boost_scale = 2.0
tau_seconds = 86400         # 24h
lookahead_seconds = 259200  # 3d
aggregation = max
apply_to_non_ad_only = true
```

## Initial run choice

Start with P18a because it is strong enough to matter but not too aggressive:

- unrelated non-ad: weight ≈ 1.0
- non-ad 2h before Ads: weight ≈ 1.0 + 3.0/e ≈ 2.1
- non-ad 30min before Ads: weight ≈ 1.0 + 3.0*0.78 ≈ 3.3
- non-ad immediately before Ads: weight ≈ 4.0

## Validation

Do not select the final epoch. Use validation monitor and per-event audit.

Primary monitored metric for now:

```text
ndcg_10
```

But final judgment should be group-aware:

- Overall NDCG@10
- Ads NDCG@10
- Purchase NDCG@10
- macro group NDCG@10
- no Ads collapse vs P14/P1/baseline

## Implementation notes

- Compute proximity weights inside the model forward pass from target `next_type_ids` and target `label_timestamps`.
- Use timestamps directly so tensor order does not matter.
- Ads event type IDs: 1 (`NativeClick`), 2 (`SearchClick`).
- Only apply proximity boost to non-ad targets in the first version.
