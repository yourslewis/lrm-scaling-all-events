# P41 — M0 Contract Isolation Eval

## Purpose

Explain why historical M0 Ads-only baseline has high native Ads AHR (~0.3066 on frozen-7 Ads-only projection) while M0b under the v001 500-bank proxy is much lower (~0.0816 all-Ads AHR).

This is an **evaluation-only** experiment. Do not train or retrain any model.

## Core hypothesis

The M0 drop is a mixture of:

1. **Target denominator shift** — old native eval is warm-ish Ads-only projection; v001 proxy reports cold/warm/all Ads slices.
2. **History/position contract shift** — old M0 uses consecutive Ads-projected sequences; M0b/current proxy anchors targets at full all-event timeline positions.
3. **Candidate protocol shift** — old Ads-only eval uses domain-0 Ads projection candidate setup; v001 proxy uses banked 10k candidates.
4. **Adapter OOD shift** — M0b feeds M0 weights through an all-event-prefix shim that M0 was not trained to consume.

P41 isolates these factors one at a time.

## Fixed inputs

Existing checkpoint only:

```text
/home/yourslewis/lrm-scaling-all-events/results_v2/baseline_ads_only_100ep/None/ckpts/checkpoint_batch0070000.pt
```

Old native frozen-7 reference:

```text
/home/yourslewis/lrm-scaling-all-events/results_v2/v3_eval7raw_ads_only_baseline_20260521/baseline_ads_only_100ep_frozen7_ads_only_projection_eval.json
AHR@10 = 0.3066437
count = 25,528
```

Current v001 proxy reference:

```text
M0b all Ads AHR@10 = 0.0816 micro / 0.0685 macro_user
M0b warm Ads AHR@10 = 0.0850 micro / 0.0736 macro_user
M0b cold Ads AHR@10 = 0.0295 micro / 0.0292 macro_user
```

Proxy data:

```text
/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1
selected_banks_seed1_100_per_domain.json
```

## Experiment rows

### P41A — M0 native projection on proxy Ads targets, old candidate setup

Question: how much of the drop is only the target set / denominator?

- Use the same Ads target IDs/positions as the v001 proxy Ads slices.
- Reconstruct each target's **Ads-only projected history**.
- Use M0's native Ads-only evaluator/candidate setup.
- Report only Ads AHR@10: cold, warm, all Ads if reconstructable; otherwise warm/all only with explicit denominator.

Interpretation:

```text
If P41A is near 0.30: target denominator is not the main issue.
If P41A is much lower: proxy target set itself is harder/different.
```

### P41B — M0 native Ads-projected history + v001 banked candidates

Question: how much of the drop is the candidate protocol?

- Use M0 native Ads-projected histories with consecutive projected positions.
- Score the same v001 500-bank candidate sets used by M0b/M2.
- Do not use full all-event prefixes.
- Report cold/warm/all Ads AHR@10 under v001 proxy metrics.

Interpretation:

```text
If P41B ≈ P41A but >> M0b: adapter/full-prefix mismatch is the main issue.
If P41B drops close to M0b: v001 candidate protocol is the main issue.
```

### P41C — M0b fixed-position adapter under v001 candidates

Question: how much of M0b's drop is from position/timeline mismatch rather than candidate protocol?

- Start from the current M0b all-prefix shim.
- For the M0 input path, project to Ads-only history before the target.
- Re-index positions consecutively in Ads-projected order.
- Preserve true timestamps only if M0 native training used them; otherwise avoid leaking full-timeline gaps into position IDs.
- Score v001 500-bank candidate sets.

Interpretation:

```text
P41C > M0b and near P41B: position/history contract fix works.
P41C ≈ M0b: the issue is not just position IDs; likely candidate/domain/objective mismatch.
```

## Required table columns

Add P41 rows to the same proxy comparison table only after each row passes sanity checks.

Report:

| Slice | Metrics |
|---|---|
| Cold Ads | micro AHR@10, macro_user AHR@10, target count, user count |
| Warm Ads | micro AHR@10, macro_user AHR@10, target count, user count |
| All Ads | micro AHR@10, macro_user AHR@10, target count, user count |
| All-domain | `N/A` for native M0 variants unless the adapter truly supports all-domain scoring |

Do not report all-domain OHR for native M0 unless the evaluator has a valid all-domain item universe and target contract.

## Sanity checks

For each P41 row:

1. No NaN/Inf positive scores.
2. `top_k` is non-empty.
3. Rank-1 rate is plausible, not all targets at rank 1.
4. Candidate count matches expected protocol.
5. Slice target counts are printed and compared against current proxy counts.
6. Output explicitly states whether histories are Ads-projected or all-event.

## Expected outcomes

Likely result bands:

```text
P41A: near old native AHR if target set shift is small.
P41B: isolates candidate-bank difficulty; may land between 0.08 and 0.30.
P41C: should tell whether fixing projected/consecutive positions recovers M0b.
```

Do not assume M0 will reach 0.30 under v001 candidates until P41B/P41C are measured.

## Recommendation

Run P41B first if implementation cost is manageable, because it directly tests the most decision-relevant question:

```text
Can M0 still do well when evaluated natively, but against the same v001 banked candidates as M2?
```

If P41B remains low, M2's advantage is likely not just adapter/position mismatch. If P41B is high, M0b is unfairly depressed by the current shim and should not be used as the M0 proxy row.
