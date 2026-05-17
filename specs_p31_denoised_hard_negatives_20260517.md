# P31 — Denoised / false-negative-filtered hard negatives

## Motivation
P29b was the strongest hard-negative run so far, while P30 rank-window sweeps showed that simply making the mined negatives safer or harder did not improve the tradeoff. This suggests the bottleneck is negative quality / false negatives rather than hardness alone.

## Base
P29b mixed hard global negatives:
- 75% event-type-aware global random negatives.
- 25% online medium-hard negatives.
- n=32 negatives.
- candidate pool 1024.
- rank window 32-512.
- eval sampler unchanged for comparability.

## Change
Add batch-local false-negative filtering to `MixedHardGlobalNegativesSampler`:
- cache all valid supervision positives in the same sequence row via `process_batch()`;
- exclude exact current target and same-row positives from hard candidate scores;
- resample uniform negatives that collide with same-row positives.

This approximates user-history positive filtering without a separate offline index.

## Runs
| Run | Hard fraction | Filter | Intent |
|---|---:|---|---|
| `p31a_denoised_hardmix_f025_n32` | 0.25 | batch positives | direct P29b + denoising |
| `p31b_denoised_hardmix_f035_n32` | 0.35 | batch positives | test if denoising permits a stronger hard mix |

## Baselines
- P29b best: OHR/HR@10 0.4994 @23k, AHR/Ads HR@10 0.2828 @26k, NDCG@10 0.4263 @23k.
- P28 n64 final: OHR 0.509, AHR 0.212, NDCG 0.432.
- P30a best: OHR 0.4019, AHR 0.2323, NDCG 0.3165; P30b was stopped early due severe collapse.

## Success criteria
Primary: improve or match P29b Ads HR while preserving NDCG close to P29b/P28.
Secondary: reduce late degradation vs P29/P30.
