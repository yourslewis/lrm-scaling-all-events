# LRM Stabilized Event Conditioning — P13/P14 Spec

**Date:** 2026-05-08
**Status:** Approved by Wenhao to spec, implement, PR, and kick off training.
**Base:** P11/P12 event-conditioned residual experiments.
**Policy:** PR required; no merge without review. GPU training can start after spec/code/PR.

## Motivation

P11/P12 confirmed target-event conditioning can help, but collapse diagnostics show the residual path can become numerically ill.

P11 debug finding:

- At batch 14k, using the correct target event type made high-volume non-ad event losses pathological:
  - OrganicSearchQuery: correct-type loss 45.3 vs zero-type loss 3.6
  - UET: correct-type loss 41.5 vs zero-type loss 5.2
  - EdgePageTitle: correct-type loss 31.9 vs zero-type loss 4.3
- Residual/base norm reached roughly 5x for those event types.
- P12 collapsed harder, likely because group-level residuals amplify one unstable direction across broad groups.

Hypothesis:

> The target-event residual is added after the normalized trunk/MMoE representation without post-condition normalization or residual scale control. Under temperature 0.05, the model can exploit query norm / score scale instead of learning stable ranking geometry.

## Goals

1. Keep the benefit of target-event conditioning.
2. Prevent residual/query norm explosion.
3. Detect per-event instability early.
4. Continue reporting overall + per-event metrics for all checkpoint audits.

## P13 — Stabilized Event-Type Residual Scorer

Condition on the exact target event type, but stabilize the residual path.

Architecture:

```text
base = task_specific_user_embedding
residual = adapter([base, event_type_emb])
alpha_t = max_scale * sigmoid(raw_alpha_t)
q = L2Norm(base + alpha_t * residual)
score = dot(q, item)
```

Key changes versus P11:

- post-condition L2 normalization
- small learnable event-specific residual gate
- alpha initialized small, not fixed 1.0
- metrics: residual norm, base norm, residual/base ratio, alpha mean/max

Config intent:

```text
make_model.enable_event_type_residual_conditioning = True
make_model.event_type_residual_granularity = "event"
make_model.event_type_residual_stabilized = True
make_model.event_type_residual_max_scale = 0.05
make_model.event_type_residual_l2_normalize = True
```

## P14 — Stabilized Event-Group Residual Heads

Same stabilization as P13, but condition on event group.

Groups:

- Ad: NativeClick, SearchClick
- Browsing: EdgePageTitle, UET, UETShoppingView
- Search: EdgeSearchQuery, OrganicSearchQuery
- Purchase: UETShoppingCart, AbandonCart, EdgeShoppingCart, EdgeShoppingPurchase
- Others: OutlookSenderDomain

Architecture:

```text
base = task_specific_user_embedding
residual = adapter([base, group_emb(target_event_type)])
alpha_g = max_scale * sigmoid(raw_alpha_g)
q = L2Norm(base + alpha_g * residual)
score = dot(q, item)
```

P14 is lower priority than P13 because P12 collapsed more severely, but it is useful to test whether stabilization rescues group heads.

## Training Plan

Run two jobs:

- GPU0: P13 stabilized event residual
- GPU1: P14 stabilized group residual

Shared settings:

- dataset: `benchmarkv4_all_events_v2`
- base architecture: P8-style HSTU + MMoE
- equal domain weights `{0:1,1:1,2:1,3:1,4:1}`
- SampledSoftmaxLoss, temperature 0.05
- InBatch negatives
- max epochs 100, but checkpoint selection must be based on proper per-event eval, not final epoch

## Required Diagnostics

Training logs should expose:

- task positive/negative similarity + gap
- gate entropy
- residual/base ratio
- residual norm
- base norm
- alpha mean/max

Audit reports must include:

- overall NDCG@10, HR@10, MRR
- event-group NDCG@10, HR@10, MRR
- per-event NDCG@10, HR@10, MRR
- checkpoint comparison table

## Success Criteria

P13/P14 are promising if they avoid the P11/P12 collapse pattern and beat either:

- P8v1/P8v3 for overall/search; or
- P10@12k for balanced Ads + Purchase + Others.

Early warning signs:

- residual/base ratio > 1.0 for high-volume event types
- alpha saturating near max scale
- per-event loss spikes for UET, EdgePageTitle, OrganicSearchQuery
- gate entropy approaching zero
- similarity scale exploding while NDCG/HR collapse
