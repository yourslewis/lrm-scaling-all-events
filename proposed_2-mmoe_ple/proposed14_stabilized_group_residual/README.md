# Proposed14 — Stabilized Event-Group Residual Heads

Fixes P12's group-residual scale instability using the same stabilization as P13.

```text
residual = adapter([base, group_emb(target_event_type)])
alpha_g = max_scale * sigmoid(raw_alpha_g)
q = L2Norm(base + alpha_g * residual)
score = dot(q, item)
```

- Event group conditioning: Ad, Browsing, Search, Purchase, Others.
- Small learnable alpha, initialized near zero.
- Post-condition L2 normalization.
- Logs residual/base/query norm and alpha diagnostics.
