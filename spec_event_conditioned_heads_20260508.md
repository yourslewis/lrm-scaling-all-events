# LRM Event-Conditioned Scoring / Heads — Training Spec

**Date:** 2026-05-08
**Status:** Approved to implement/start training by Wenhao in chat.
**Repo:** `/home/yourslewis/lrm-scaling-all-events` on GPU server
**Policy:** No GitHub push. Specs first, then training.

## Goal

Improve both Ads and non-Ads event prediction by using the known target event type at prediction time.

We prefer using target-event information only near the output/scoring layer rather than injecting it deeply into the sequence encoder, because prior leakage/conditioning variants were fragile and may have destabilized the shared representation.

## Constraints

- Preserve strong shared HSTU/MMoE trunk behavior.
- Avoid replacing the shared score with a private head from scratch.
- All specialization must be residual/identity initialized so the model starts as the baseline and learns corrections.
- Future summaries must include overall + per-event evaluation every time.
- Do not push code to GitHub.

## Experiments to Start

### P11 — Event-Type Residual Scorer

**Purpose:** Test whether leaking the target event type at output time improves both Ads and non-Ads without multiple heads.

Architecture:

```text
base_user = trunk(history)
condition = embedding(target_event_type)
residual = MLP([base_user, condition])
user_t = base_user + residual_scale * residual
score = dot(user_t, item)
```

Implementation details:

- Condition granularity: per event type.
- Residual MLP final layer is zero initialized, so initial model equals the base model.
- Target event type is `next_type_ids`, aligned to labels.
- Use P8-style MMoE trunk with equal domain weights.
- No load-balance/z-loss for this first test.

Config intent:

```text
make_model.multi_task_module_type = "mmoe"
make_model.enable_event_type_residual_conditioning = True
make_model.event_type_residual_granularity = "event"
make_model.event_type_residual_hidden_dim = 128
make_model.event_type_residual_scale = 1.0
```

### P12 — Event-Group Residual Heads

**Purpose:** Safer multi-head style model with one residual scorer per event group instead of per event.

Groups:

- Ad: NativeClick, SearchClick
- Browsing: EdgePageTitle, UET, UETShoppingView
- Search: EdgeSearchQuery, OrganicSearchQuery
- Purchase: UETShoppingCart, AbandonCart, EdgeShoppingCart, EdgeShoppingPurchase
- Others: OutlookSenderDomain

Architecture:

```text
base_user = trunk(history)
group = group(target_event_type)
residual = MLP([base_user, group_embedding])
user_g = base_user + residual_scale * residual
score = dot(user_g, item)
```

Implementation details:

- Condition granularity: event group.
- Zero-initialized final residual layer.
- Use P8-style MMoE trunk with equal domain weights.

Config intent:

```text
make_model.multi_task_module_type = "mmoe"
make_model.enable_event_type_residual_conditioning = True
make_model.event_type_residual_granularity = "group"
make_model.event_type_residual_hidden_dim = 128
make_model.event_type_residual_scale = 1.0
```

## Training Plan

Start two GPU jobs:

- GPU0: P11 event-type residual scorer.
- GPU1: P12 event-group residual heads.

Shared settings:

- Dataset: `benchmarkv4_all_events_v2`
- Max sequence length: 200
- Item embedding dim: 128
- HSTU: 16 blocks, 4 heads, dqk=32, dv=32
- Loss: `SampledSoftmaxLoss`, temperature 0.05, 256 negatives
- Sampling: `InBatch`
- Domain weights: `{0:1,1:1,2:1,3:1,4:1}`
- Train domains: `[0,1,2,3,4]`
- Epochs: 100, but select checkpoint by proper per-event eval, not final epoch.

## Evaluation Requirement

For every result report:

- Overall NDCG@10, HR@10, MRR
- Event-group NDCG@10, HR@10, MRR
- Per-event NDCG@10, HR@10, MRR
- Checkpoint-selection table across early/mid/final checkpoints

Important: evaluation must use target-event conditioning in retrieval scoring when evaluating P11/P12. If an eval script bypasses the wrapper and encodes only the raw base model, it is invalid for these experiments and must be fixed before final judgment.

## Success Criteria

Primary success:

- Better non-Ad event/group metrics than P8v1 without severe Ads degradation.

Practical candidate preference:

- If P11 improves both Ads and non-Ads, prefer P11 because it avoids multiple heads.
- If P12 improves balance more reliably, use P12 as safer group-head design.

Reference checkpoints:

- P8v1: strongest completed model by prior proper eval.
- P8v3@14k: best overall/search in audit.
- P10@12k: best balanced Ads/Purchase/Others in audit.
