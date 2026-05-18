# P32 — Hybrid in-batch + global hard + global random negatives

## Motivation
P28-P31 suggest that making global/domain-aware random negatives the main training signal hurts the P23-style setup. P23 used pure in-batch negatives and remains the strongest reference. P32 restores in-batch negatives as the majority signal while keeping a controlled amount of global hard and global random negatives.

## Sampler split
For `N = make_model.num_negatives`:
- Type 1: `n1 = 50% * N` domain-aware in-batch random negatives.
- Type 2: `n2 = 30% * N` global domain-aware hard negatives.
- Type 3: `n3 = N - n1 - n2` global domain-aware random negatives.

For N=32 this is approximately:
- n1=16 in-batch domain-aware negatives,
- n2=10 global hard negatives,
- n3=6 global random negatives.

If the same-domain in-batch pool cannot provide enough unique candidates for n1, missing in-batch slots are filled from the global hard-negative pool.

## Run
| Run | Split | Hard window | Intent |
|---|---|---|---|
| `p32a_hybrid_inbatch50_hard30_global20_n32` | 50/30/20 | 32-512, pool 1024 | Restore in-batch as primary, global as controlled supplement |

## Baselines
- P23/P23-coordinate in-batch reference.
- P28 n64 final: OHR .509, AHR .212, NDCG .432.
- P29b best: OHR .499, AHR .283, NDCG .426.
- P30/P31 global-heavy variants for negative evidence.

## Success criteria
Primary: match or improve P29b Ads HR while keeping overall/NDCG near P23/P28/P29b.
Secondary: avoid the P30 collapse pattern and reduce dependence on global random negatives.
