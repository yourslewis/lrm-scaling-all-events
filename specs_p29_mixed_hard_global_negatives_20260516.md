# P29 — Mixed Hard Global Negatives

Date: 2026-05-16
Branch: `feature/p29-hard-global-negatives-20260516`
Base: stacked on PR #12 (`feature/p28-domain-random-negatives-20260516094630`) because P29 depends on P28's event-type-aware negative routing and deterministic seeding.

## Motivation

P28 tested event-type-aware global random negatives. This is safer than in-batch negatives but likely easier: most global items are unrelated, while in-batch positives are drawn from the same traffic distribution and can be semantically/user-interest similar.

We want to recover some of the discriminative pressure of in-batch negatives without the same false-negative risk. The proposed approach is to mix mostly uniform global negatives with a controlled fraction of medium-hard global negatives.

## Literature basis

- Shi et al., WWW 2023, **On the Theories Behind Hard Negative Sampling for Recommendation** (`arXiv:2302.03472`): hardness should be controllable; harder negatives align better with smaller Top-K objectives but can be risky.
- Fan et al., SIGIR 2023, **Neighborhood-based Hard Negative Mining for Sequential Recommendation** (`arXiv:2306.10047`): mines hard negatives from behavior-graph neighborhood overlap and uses curriculum from easy to hard.
- Google, **Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations**: mixes batch and uniform negatives to reduce selection bias in implicit feedback.
- Chennu et al., LargeRecSys/RecSys 2024, **Evaluating Performance and Bias of Negative Sampling in Large-Scale Sequential Recommendation Models** (`arXiv:2410.17276`): negative strategy changes head/mid/tail and aggregate tradeoffs; segmented metrics matter.
- Lindgren et al., NeurIPS 2021, **Efficient Training of Retrieval Models using Negative Cache**: cached item embeddings allow large negative pools efficiently.
- Zhan et al., SIGIR 2021, **Optimizing Dense Retrieval Model Training with Hard Negatives** (`arXiv:2104.08051`): static hard negatives can destabilize; mix random negatives for stability and use dynamic negatives when possible.

## Design

Add a new sampler in a separate file/class:

```text
proposed_2-mmoe_ple/train/modeling/sequential/hard_negative_sampler.py
  class MixedHardGlobalNegativesSampler
```

The sampler is train-only. Evaluation keeps the existing rotate/global eval sampler so P23/P27/P28 metric comparability is preserved.

For each positive target:

1. Route by target event type to the 5-domain embedding store:
   - Ads: NativeClick, SearchClick -> 0
   - Browsing/Web: EdgePageTitle, UET, UETShoppingView, ChromePageTitle, MSN -> 1
   - SearchQuery: EdgeSearchQuery, OrganicSearchQuery -> 2
   - PurchaseCart: UETShoppingCart, AbandonCart, EdgeShoppingCart, EdgeShoppingPurchase -> 3
   - OutlookSenderDomain -> 4
2. Draw `K * (1 - hard_fraction)` uniform random negatives from that domain's current shard.
3. Draw `K * hard_fraction` hard negatives by:
   - sampling a candidate pool from the same domain shard;
   - scoring candidates by dot product with the current query embedding;
   - sampling from a rank window, not the absolute top candidates.

The rank window is a false-negative guard. For example, `rank_start=32, rank_end=512` avoids the closest candidates that may be true alternatives/near-duplicates.

## Config knobs

Added gin-configurable `make_model` args:

```python
make_model.hard_negative_fraction
make_model.hard_negative_candidate_pool_size
make_model.hard_negative_rank_start
make_model.hard_negative_rank_end
```

## Initial experiment grid

Use P28 n32 as the base and keep eval unchanged.

- **P29A**: hard fraction 0.10, candidate pool 1024, rank 32-512
- **P29B**: hard fraction 0.25, candidate pool 1024, rank 32-512

Both use:

```text
make_model.sampling_strategy = "MixedHardGlobalNegativesSampler"
make_model.num_negatives = 32
Trainer.eval_batch_size = 32
Trainer.eval_max_batches = 50
create_data_loader.num_workers = 0
```

## Success criteria

Compare against P28 n32 and P23 target:

- P23 target (`p23_page_s10_p09_m01_o00`): Overall HR@10 0.6062, Ads HR@10 0.2323.
- P28 n32 final: Overall HR@10 ~0.5019, Ads HR@10 ~0.1919.

Good signs:

- improves P28 n32 overall without hurting Ads;
- or improves Ads toward P23 while preserving most of P28 n32 overall;
- segmented metrics do not show severe head/mid/tail or domain collapse.

Stop/rollback signs:

- early eval collapses like P28 n48+;
- Ads HR drops below P28 n32 while overall also degrades;
- hard-negative sampler creates instability/OOM.

## Guardrails

- Do not change eval logic.
- Keep train hard-negative logic in the separate `hard_negative_sampler.py` class.
- Keep P28 random/global sampler behavior intact.
- Use exact absolute paths in cron/table reporting.
