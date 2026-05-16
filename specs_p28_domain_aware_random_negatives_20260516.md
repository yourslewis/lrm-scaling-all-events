# P28 — Domain-aware random training negatives

## Goal

Test whether the Ads/overall tradeoff improves when training negatives better match the domain-aware global-negative retrieval evaluation.

Prior diagnosis showed the severe Ads HR collapse in the P27A repro was primarily caused by reducing training batch size, which weakened the in-batch negative pool. P28 keeps the P23 architecture/objective and replaces pure in-batch train negatives with domain-aware random negatives.

## Baseline

Use the P23 coordinate-search setting as the base:

- HSTU max sequence length: 200
- MMoE: 4 experts, expert hidden dim 128
- Stabilized target-event-group residual heads
- Equal domain weights `{0:1.0, 1:1.0, 2:1.0, 3:1.0, 4:1.0}`
- Sharp ad-anchor weighting:
  - sigma: 300s
  - pre/post window: ±600s
  - max weight: 4.0
  - PageTitle gate: 0.9
  - MSN gate: 0.1
  - Outlook gate: 0.0
- Training batch size: 32
- Validation: `eval_batch_size=32`, `eval_max_batches=50`, metric `ndcg_10`

Reference P23 results are noisy but roughly:

- Overall HR@10: ~0.60–0.63
- Ads HR@10: ~0.17–0.23 depending eval slice / seeded run

## Change

Switch training sampler from:

```gin
make_model.sampling_strategy = "InBatch"
```

to:

```gin
make_model.sampling_strategy = "RotateInDomainGlobalNegativesSampler"
```

and sweep:

```gin
make_model.num_negatives = 32
make_model.num_negatives = 48
make_model.num_negatives = 64
make_model.num_negatives = 96
```

The rotate sampler is domain-aware via `positive_ids // domain_offset`:

- Ads positives sample from Ads / AdsCorpus pools
- WebBrowsing positives sample from the WebBrowsing pool
- Shopping/Search positives sample from the Shopping pool

This aligns training negatives more closely with the validation retrieval setup than pure in-batch negatives.

## Required code support

`RotateInDomainGlobalNegativesSampler` maintains shard pools and needs `rotate()` called before the first training step. Evaluation already rotates the eval sampler before validation. P28 adds a train-time initialization in `Trainer.setup()`:

```python
train_sampler = getattr(self._model_unwrapped, "negatives_sampler", {}).get("train")
if hasattr(train_sampler, "rotate") and not getattr(train_sampler, "pools", None):
    logging.info("initializing train negatives sampler via rotate()")
    train_sampler.rotate()
```

For `InBatch`, this is a no-op.

## Configs

Full-structure configs:

- `proposed_2-mmoe_ple/config/generated_p28_domain_random_negatives/p28_domain_rand_n32.gin`
- `proposed_2-mmoe_ple/config/generated_p28_domain_random_negatives/p28_domain_rand_n48.gin`
- `proposed_2-mmoe_ple/config/generated_p28_domain_random_negatives/p28_domain_rand_n64.gin`
- `proposed_2-mmoe_ple/config/generated_p28_domain_random_negatives/p28_domain_rand_n96.gin`

Fallback config, only if full n96 fails due CUDA OOM:

- `proposed_2-mmoe_ple/config/generated_p28_domain_random_negatives/p28s_domain_rand_n96_small_fallback.gin`

The fallback is explicitly **not architecture-equivalent**:

- `hstu_encoder.num_blocks = 8`
- `make_model.expert_hidden_dim = 96`

Do not compare it directly to full-structure P28 rows without highlighting this caveat.

## Launch

```bash
cd /home/yourslewis/lrm-scaling-all-events
python3 scripts/p28_domain_random_negatives_autoresearch.py \
  --profiles all \
  --max-batch 26000 \
  --poll-seconds 120 \
  --allow-small-n96-fallback
```

The script writes state to:

```text
results_v2/p28_domain_random_negatives_autoresearch/state.json
```

and per-run summaries to:

```text
results_v2/p28_domain_random_negatives_autoresearch/summary.jsonl
```

## Success criteria

Primary:

- Improve Ads HR@10 vs P23 while keeping Overall HR@10 close to P23.

Guardrails:

- If Ads HR improves only by ~1–2 points, verify with fixed eval negative pools and/or repeated eval because Ads eval sample counts are small.
- If Overall HR drops materially, the random negatives may be too hard or may over-regularize non-Ads domains.
- If n96 requires fallback, report it separately as a smaller-model stress test, not as the n96 full model result.

## Live-run note

The first live P28 run was launched on the GPU server with the full P23 seeded architecture and a local train-sampler rotate patch. The live server also had the full deterministic seeding patch from PR #11 applied. This P28 PR is the experiment PR; for exact reproducibility, merge/apply PR #11 as well.
