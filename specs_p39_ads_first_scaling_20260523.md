# P39 — Ads-first P29B scaling recipe for full v3 data

## Goal

The full-data retrain sweep improved OHR but did not reliably improve AHR. P39 changes the question from “does more data help the old balanced recipe?” to:

1. Can adding non-Ads data improve **Ads HR@10** over an Ads-first recipe?
2. Once the recipe is Ads-protective, does more training data improve **Ads HR@10** further?

The immediate target is to beat the Ads-only HSTU baseline on frozen-7:

```text
Frozen-7 Ads-only baseline AHR = 0.3066437
```

## Starting point

Use P29B as the base recipe because it was the best prior hard-negative direction for Ads:

- mixed hard/global negatives;
- `hard_negative_fraction = 0.25`;
- `num_negatives = 32`;
- rank window `32-512`, candidate pool `1024`;
- P23/P29 ad-anchor and stabilized group-residual architecture.

## Key change: Ads-first objective

Previous full-data runs used balanced/equal domain weights, which let the much larger non-Ads volume dominate shared capacity. P39 keeps non-Ads only as auxiliary signal and selects checkpoints by AHR.

Checkpoint selection metric:

```text
primary = Ads HR@10
```

Guardrails reported, not primary-selected:

```text
Overall HR@10
Overall NDCG@10
Ads NDCG@10
```

## Profiles

### P39A — P29B Ads-only supervision

Purpose: establish the Ads-first P29B baseline with full v3 data.

```gin
make_model.supervision_train_domains = [0]
make_model.supervision_domain_weights = {0: 1.0}
Trainer.validation_metric_name = "ads_hr_10"
```

This still uses the full sequence/context rows, but only Ads-domain targets contribute loss.

### P39B — light non-Ads auxiliary

Purpose: test whether adding non-Ads auxiliary targets helps AHR while keeping Ads dominant.

```gin
make_model.supervision_train_domains = [0, 1, 2, 3, 4]
make_model.supervision_domain_weights = {0: 16.0, 1: 0.05, 2: 0.10, 3: 2.0, 4: 0.05}
Trainer.validation_metric_name = "ads_hr_10"
```

### P39C — medium non-Ads auxiliary

Purpose: test whether stronger non-Ads auxiliary supervision helps or starts stealing Ads capacity.

```gin
make_model.supervision_train_domains = [0, 1, 2, 3, 4]
make_model.supervision_domain_weights = {0: 16.0, 1: 0.20, 2: 0.40, 3: 2.0, 4: 0.10}
Trainer.validation_metric_name = "ads_hr_10"
```

## Data-scale tests

Use the same frozen-7 eval set for every run:

```text
/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_eval7raw_0_6/eval
```

Train with v3-preserve subsets built from sorted train parquet parts:

| Scale | Purpose |
|---:|---|
| 20% | cheap signal and baseline recipe comparison |
| 50% | intermediate scaling point |
| 100% | final target, must beat AHR 0.3066437 |

Evidence criteria:

1. Non-Ads helps if `P39B` or `P39C` beats `P39A` at the same train scale.
2. More data helps if the winning Ads-first auxiliary profile improves from 20% → 50% → 100%.
3. A recipe is promising only if frozen-7 final AHR approaches or beats `0.3066437`; OHR is secondary.

## Runner

```bash
python -u scripts/p39_ads_first_scaling.py \
  --profiles p39a_full p39b_20 p39b_50 p39b_full p39c_full \
  --gpu 0 \
  --poll-seconds 300 \
  --min-batch 12000 \
  --patience-batches 8000
```

The runner writes final frozen-7 rows to:

```text
results_v2/p39_ads_first_scaling/p39_ads_first_scaling_metrics.json
```
