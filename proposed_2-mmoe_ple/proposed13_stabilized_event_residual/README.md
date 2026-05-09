# Proposed13 — Stabilized Event-Type Residual Scorer

Fixes P11's residual/query scale instability.

```text
residual = adapter([base, event_type_emb])
alpha_t = max_scale * sigmoid(raw_alpha_t)
q = L2Norm(base + alpha_t * residual)
score = dot(q, item)
```

- Exact target event type conditioning.
- Small learnable alpha, initialized near zero.
- Post-condition L2 normalization.
- Logs residual/base/query norm and alpha diagnostics.
