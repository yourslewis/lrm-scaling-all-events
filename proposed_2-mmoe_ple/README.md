# Proposed 2: All-Events + MMoE / PLE

Self-contained HSTU with MMoE or PLE gating, trained on all 14 event types.
Tests whether per-group gating prevents negative transfer from browsing dominance.

## Variants (controlled via config)
- **MMoE**: Shared experts + per-group gate networks
- **PLE**: Task-specific experts + shared experts + progressive extraction

## Hypothesis
Group-specific gating allows the model to learn specialized representations for ads vs browsing, preventing negative transfer while still benefiting from cross-signal information.

## Expected Outcome
+10-20% Ad NDCG@10 over Proposed 1 if negative transfer exists.
