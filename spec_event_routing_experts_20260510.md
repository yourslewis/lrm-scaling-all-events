# Spec: P15/P16/P17 Event-Aware Routing and Expert Specialization

Date: 2026-05-10 UTC
Dataset: `benchmarkv4_all_events_v2`
Base: P8-style HSTU + MMoE, equal domain weights `{0:1,1:1,2:1,3:1,4:1}`, SampledSoftmaxLoss, temperature `0.05`, in-batch/global rotated negatives, max epochs 100.

## Motivation
P13/P14 stabilized output residual conditioning fixed the most severe P11 collapse for high-volume non-ad events, especially in P14, but Ads and Purchase remain weak and residual magnitudes are very large. Current P13/P14 use target event type only after the domain/task MMoE output. They do **not** route different event types to different MMoE experts.

This spec tests whether target event type/group should influence expert routing itself, not only post-MMoE residual scoring.

## Shared evaluation requirements
Every checkpoint summary must include:
- overall NDCG@10 / HR@10 / MRR;
- event-group metrics: Ad, Browsing, Search, Purchase, Others;
- per-event metrics for all observed event types;
- residual/routing diagnostics when available.

Do not judge from final checkpoint only. Audit early/mid/latest checkpoints.

## P15 — Event-Type-Gated MMoE
Hypothesis: letting target event type condition the MMoE router can specialize expert mixtures without introducing separate full heads.

Architecture:
```text
history -> shared HSTU -> shared experts
(target event type embedding + HSTU state) -> per-domain/task gates
weighted shared experts -> task/domain embedding -> retrieval score
```

Key properties:
- same shared expert pool as P8/P13;
- event type changes gate logits, not expert parameters;
- no post-residual conditioner by default;
- diagnostics: gate entropy per domain/task.

Success criteria:
- improve Ads/Purchase over P14 without collapsing high-volume browsing/search events;
- avoid low-entropy router collapse across all tasks.

## P16 — Event-Group-Specific Expert MMoE
Hypothesis: exact event-type routing is too granular; five event groups should get separate expert capacity while sharing a common base.

Architecture:
```text
history -> shared HSTU
shared experts + selected event-group experts
(target event group + task) gate selects among shared + group experts
-> retrieval score
```

Key properties:
- shared experts are always available;
- only the target event group expert bank is active per example/position;
- group-specific capacity targets Ads/Purchase without fragmenting into rare exact event types.

Success criteria:
- improve Ad/Purchase group metrics vs P14;
- preserve P14-like UET / OrganicSearchQuery / EdgePageTitle performance;
- no group expert collapse or massive gate entropy drop.

## P17 — Hierarchical Domain MMoE + Event-Group Expert Residual
Hypothesis: the safest hierarchy is domain/task routing first, then a small event-group-specific expert residual after domain embedding.

Architecture:
```text
history -> shared HSTU -> domain/task MMoE
selected event-group residual adapter(domain embedding)
q = L2Norm(domain_embedding + alpha_group * group_residual)
```

Key properties:
- preserves domain/task MMoE as first-stage router;
- uses separate residual adapters per event group, unlike P14's single condition-embedding adapter;
- alpha bounded by `max_scale=0.05`, post-condition L2 norm enabled.

Success criteria:
- match/exceed P14 overall while reducing residual dominance;
- improve weak Ads/Purchase without damaging UET/Search/Browsing;
- alpha should not saturate immediately for all groups, and residual/base should be lower than P14.

## Launch plan
GPU capacity currently permits one new job on GPU0 because P14 is still running on GPU1. Start P15 first on GPU0; keep P16/P17 launch-ready for later or when a GPU frees up.
