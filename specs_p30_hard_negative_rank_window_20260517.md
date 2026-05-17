# P30 — Hard-negative rank-window sweep

## Motivation
P29 showed that a mixed sampler (mostly event-type-aware global random negatives plus a controlled hard-negative fraction) can improve Ads metrics, especially `p29b_hardmix_f025_n32`, but the final/best tradeoff suggests hardness may be too unstable across training.

Before moving to a heavier negative-cache/ANN miner, P30 tests the main hardness knob directly: the rank window used after scoring a random candidate pool with the current query embedding.

## Base
Stacked on P29 / P28:
- P28: event-type-aware global random training negatives.
- P29: `MixedHardGlobalNegativesSampler`, n=32, hard fraction 0.25, candidate pool 1024, rank window 32–512.

## Hypothesis
- A safer medium-hard window should reduce false-negative / too-hard risk and stabilize late training.
- A more aggressive window should improve top-K if P29 was still too easy, but may hurt stability or Ads.

## Runs
Both use `make_model.num_negatives = 32`, `hard_negative_fraction = 0.25`, candidate pool 2048, and unchanged eval sampling.

| Run | Rank window | Intent |
|---|---:|---|
| `p30a_hardmix_safe_r128_1024_n32` | 128–1024 | safer medium-hard negatives |
| `p30b_hardmix_aggr_r8_256_n32` | 8–256 | more aggressive hard negatives |

## Success criteria
Compare against P29b best and P28 n64:
- P29b best: OHR/HR@10 0.4994 @23k, AHR/Ads HR@10 0.2828 @26k, NDCG@10 0.4263 @23k.
- P28 n64 final: OHR 0.509, AHR 0.212, NDCG 0.432.

Primary: improve Ads HR@10 while preserving NDCG@10 close to P28/P29b.
Secondary: avoid late collapse and keep throughput acceptable.

## Decision rule
- If safer window keeps Ads high and reduces late degradation, use it as next baseline for false-negative filters/cache.
- If aggressive window wins, test a curriculum that starts safe and moves toward aggressive.
- If neither wins, move to P30b/P31 negative cache with false-negative filters.
