# Proposed12 — Event-Group Residual Heads

Target event type is mapped into a group and conditions only the output/scoring representation.

Groups: Ad, Browsing, Search, Purchase, Others.

```text
user_g = user + residual_scale * MLP([user, group_emb(target_event_type)])
score = dot(user_g, item)
```

The final residual layer is zero-initialized, so training starts exactly as the base P8-style MMoE trunk.

- Config: `config.gin` or `../config/proposed12_group_residual.gin`
- Residual module: `../train/proposed11_event_residual/event_type_residual.py`
- Group mapping: `../train/proposed12_group_residual/event_groups.py`
- Current run output: `../../results_v2/proposed12_group_residual/p12_group_residual_20260508`
- Log: `/tmp/gpu1_proposed12_group_residual.log`
