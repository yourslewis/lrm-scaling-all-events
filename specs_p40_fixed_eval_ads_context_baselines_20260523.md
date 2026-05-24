# P40 — Fixed-target Ads-context baselines and evaluation contract

## Why this round exists

The previous P39 sweep mixed several confounders: the historical Ads-only baseline used an Ads-projected sequence, while P39 evaluated only final-label Ads events in an all-domain sequence. That made AHR denominators depend on model/evaluator constraints.

P40 locks the evaluation contract first, then trains only the baselines needed to answer whether non-Ads context/supervision helps Ads retrieval.

## Non-negotiable evaluation contract

All reported metrics must use the same frozen source rows:

```text
/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_eval7raw_0_6/eval
```

Unsupported model/slice cells are reported as `N/A`. Do **not** substitute a different evaluation set, target denominator, or projection just because a model cannot consume a slice.

## Metrics table

Report eight real metrics per model:

| Slice | Definition | Metrics |
|---|---|---|
| Cold Ads | first `NativeClick`/`SearchClick` per user with non-empty full prefix | micro AHR@10, macro AHR@10 |
| Warm Ads | Ads targets after the first Ads event | micro AHR@10, macro AHR@10 |
| All Ads | all valid Ads target positions | micro AHR@10, macro AHR@10 |
| All-domain | all valid target positions across all event types | micro OHR@10, macro OHR@10 |

Frozen-7 target/user counts:

| Slice | Targets | Users |
|---|---:|---:|
| Cold Ads | 38,010 | 38,010 |
| Warm Ads | 322,482 | 25,528 |
| All Ads | 360,492 | 39,469 |
| All-domain | 35,167,392 | 40,106 |

## Model rows

| ID | Name | Purpose |
|---|---|---|
| M0 | Historical Ads-only HSTU | Existing checkpoint; valid only for Warm Ads under fixed-target contract. Cold/AllAds/AllDom are `N/A`. |
| M0b | M0 all-event prefix shim | Same weights as M0, feed all-event prefix as a diagnostic/OOD test. No retraining. |
| M1 | Clean all-event Ads-loss HSTU | New clean baseline: plain HSTU, full all-event sequence, loss only on Ads target positions. |
| M2 | Aux-light all-event HSTU | Same as M1, add selected light non-Ads auxiliary supervision. |
| M3 | Aux-medium all-event HSTU | Same as M1, stronger non-Ads auxiliary supervision. |
| D1 | Existing P39A/P39B diagnostics | Re-evaluate only as diagnostics; architecture/negative sampler are confounded. |

## Training profiles

All new trainable profiles use the same all-event train source and frozen eval rows:

```text
train: /home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve/train
eval:  /home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_eval7raw_0_6/eval
```

For a clean baseline, P40 removes P39's MMoE, event residual head, ad-anchor weighting, and hard-global negative sampler. It uses plain HSTU with all-domain embeddings and full all-event prefixes.

### M1 — Ads-loss only

```gin
make_model.multi_task_module_type = "none"
make_model.enable_event_type_residual_conditioning = False
make_model.enable_ad_anchor_proximity_weighting = False
make_model.sampling_strategy = "InBatch"
make_model.supervision_train_domains = [0]
make_model.supervision_domain_weights = {0: 1.0}
```

### M2 — light auxiliary

```gin
make_model.supervision_train_domains = [0, 1, 2, 3, 4]
make_model.supervision_domain_weights = {0: 16.0, 1: 0.05, 2: 0.10, 3: 1.0, 4: 0.05}
```

### M3 — medium auxiliary

```gin
make_model.supervision_train_domains = [0, 1, 2, 3, 4]
make_model.supervision_domain_weights = {0: 16.0, 1: 0.20, 2: 0.40, 3: 2.0, 4: 0.10}
```

## Reporting guardrails

1. Fill only the fixed 6x8 anchor table.
2. Keep target/user counts as metadata rows, not metrics.
3. For M0 unsupported cells, report `N/A`, never `0.0`.
4. If any evaluator uses a cap/sampling for speed, mark the table as sampled and do not compare it to full metrics.
5. Final PR/training reports must include the exact checkpoint, config, data path, evaluator command, and target counts.
