# P41 — P23 adapter smoke for `lrm_benchmark_v001`

## Purpose

Bridge the legacy `p23_page_s10_p09_m01_o00` checkpoint into the new `lrm_benchmark_v001` evaluation framework without changing the checkpoint or official evaluator semantics.

The adapter produces `lrm_prediction_record_v001` JSONL that can be scored by the sidecar-aware official evaluator (`official_evaluator_v001.0.3`).

## What changed

Added:

- `scripts/p23_v001_infer.py` — smoke-only inference adapter for P23.
- `model_submissions/p23_v001_smoke/context_policy.json` — declared context transforms and compatibility shims.
- `model_submissions/p23_v001_smoke/model_submission_manifest.json` — smoke package identity/digests.

Not included in this PR:

- production sidecars;
- target ID samples;
- prediction JSONL;
- evaluation outputs;
- scorer-private labels/candidate truth.

Those artifacts remain in the local Phase 8 validation package and should not be copied into the model-code repo.

## Adapter flow

```text
v001 target row
  -> HistoryPrefixReader [T1, target_ts)
  -> last 200 context events for legacy P23
  -> regenerate banked 10k candidate set
  -> score candidates with P23 checkpoint
  -> emit lrm_prediction_record_v001 JSONL
  -> official sidecar-aware evaluator scores predictions
```

## Smoke result already run locally

The local Phase 8 package contains the bounded smoke evidence:

```text
workspace-rex/specs/lrm-experiment-framework/phase8_reproduction/p23_adapter_smoke/
```

Observed result:

```text
run_status: passed
validation.status: passed
evaluator_impl_version: official_evaluator_v001.0.3
selected_target_count: 4
prediction_records: 4
candidate_scores_per_target: 10,001
```

## Why sample size was 4

Yes, the sample size of `4` was intentional for a **first plumbing smoke test**, not for metric interpretation.

It was chosen to cover the smallest useful cross-section while keeping the run fast and easy to inspect:

- 2 non-Ads/Browsing targets;
- 2 Ads targets;
- both Ads examples showed different context/history characteristics;
- one sample included unsupported newer event-type IDs, exercising the compatibility mapping path;
- each target still requires scoring all 10,001 official banked candidates, so even a tiny sample validates candidate regeneration and prediction coverage.

The metric output from this smoke is not meaningful for ranking because all slices are low-support/invalid. It only proves:

- the old checkpoint can load;
- official history prefix input can be converted into legacy tensors;
- official banked candidate sets can be scored;
- prediction JSONL validates;
- the official evaluator accepts the model-backed predictions.

## Known limitations before full reproduction

- Unsupported v001 `event_type_id` values above the legacy P23 embedding range `0..13` are mapped to `UNK=0`.
- Zero-context targets are excluded by the smoke adapter.
- The adapter uses the legacy base HSTU query path with v001 static embeddings.
- A larger fixed-sample validation is required before full reproduction.

## Next validation step

Run a larger deterministic fixed sample before broad reproduction:

```text
P23 adapter -> larger sample official evaluator smoke -> compare sanity metrics/support -> then small panel P23/P28/P29B/P31B/P40M2
```
