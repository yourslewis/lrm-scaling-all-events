# Proposed11 — Event-Type Residual Scorer

Target event type is known at prediction time and conditions only the output/scoring representation.

```text
user_t = user + residual_scale * MLP([user, event_type_emb])
score = dot(user_t, item)
```

The final residual layer is zero-initialized, so training starts exactly as the base P8-style MMoE trunk.

- Config: `config.gin` or `../config/proposed11_event_residual.gin`
- Module: `../train/proposed11_event_residual/event_type_residual.py`
- Current run output: `../../results_v2/proposed11_event_residual/p11_event_residual_20260508`
- Log: `/tmp/gpu0_proposed11_event_residual.log`
