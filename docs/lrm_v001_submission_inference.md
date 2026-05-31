# LRM benchmark v001 submission inference for legacy P23

This note documents the repo-side inference runner added for the current
`lrm_benchmark_v001` data/submission contract. It intentionally does **not**
change benchmark data, target/candidate manifests, prediction schema, or the
official evaluator.

## Entrypoints

- `proposed_2-mmoe_ple/infer/lrm_v001/sequential_submission_infer.py`
  - emits `lrm_prediction_record_v001` JSONL in `--output-mode full` (default);
  - can emit derived compact records in `--output-mode compact` or both formats in `--output-mode both`;
  - reads frozen target sidecars or a bounded JSONL sample;
  - materializes history through the official v001 history reader;
  - generates frozen banked candidate sets through the official bank generator;
  - scores every candidate in each 10,001-candidate set in memory before deriving either output;
  - writes a separate inference log with context policy/debug information.
- `proposed_2-mmoe_ple/infer/lrm_v001/compact_metrics.py`
  - computes exact pessimistic positive rank after all candidates have been scored;
  - writes optional topK candidates plus a digest of the full score order instead of 10,001 predictions per target;
  - streams exact HR/AHR/OHR, NDCG@K, and MRR aggregates using only target rank statistics and per-user accumulators;
  - is execution-side only and does not modify the official v001 data/candidate/prediction contract.
- `proposed_2-mmoe_ple/infer/lrm_v001/compact_streaming_evaluate.py`
  - recomputes aggregate metrics from an existing compact JSONL without materializing full prediction arrays.
- `proposed_2-mmoe_ple/infer/lrm_v001/run_full_submission_with_safety_gate.py`
  - runs a bounded burn-in;
  - estimates full-set runtime and JSONL storage for full, compact, or both outputs;
  - writes `<run_id>.safety_gate.json`;
  - only starts full-set inference when `--auto-proceed-if-sane` is provided and
    thresholds pass;
  - optionally runs the official evaluator after full-set inference.

## Compact/streaming output mode

`--output-mode compact` is not an official submission JSONL. It is a derived
execution/evaluation artifact for tractable full-set measurement:

- the runner still regenerates and scores all 10,001 official banked candidates
  for each target;
- it computes the official pessimistic positive rank exactly:
  `1 + count(score > positive_score) + count(score == positive_score and candidate != positive)`;
- it persists `rank_stats`, optional `top_k`, `candidate_count`, candidate-set
  digest, and a SHA-256 digest over the full sorted candidate/score order;
- it streams exact aggregate metrics from positive ranks, matching the official
  evaluator formulas for HR/AHR/OHR, NDCG@K, MRR, and macro-by-user aggregation;
- it avoids writing the full 18M × 10,001 prediction array.

Use `--output-mode both` on small samples to compare compact metrics against the
full official evaluator output. Use `--output-mode compact` for full-set runs only
after the burn-in gate is sane.

## Submission contract compliance

Each prediction record contains only schema-approved fields:

- `schema_version = lrm_prediction_record_v001`
- `benchmark_version = lrm_benchmark_v001`
- `model_submission_id`
- `prediction_run_id`
- `target_id`
- `candidate_protocol_label`
- `candidate_set_id`
- full ranked `predictions` coverage for the generated candidate set
- `inference_metadata` with:
  - `generated_at`
  - `entrypoint_name`
  - `model_artifact_digest`
  - `context_policy_digest`
  - `seed`
  - `context_policy_mode = declared_transforms_over_full_available_prefix`
  - `notes`

No labels, raw user history, raw context, or metric fields are written to the
prediction JSONL. Debug-only fields such as context length and positive rank are
restricted to the sidecar inference log.

## Sequential reuse policy

The runner groups targets by the canonical history reference embedded in
`context_reader_ref`:

```text
canonical_row_array_v001:eval/<part_file>:source_row_index=<row>:target_event_id=<event>
```

That key is the v001 row-array user/history row. For each referenced canonical
Parquet part, the runner scans the part sequentially once and materializes only
needed histories. This avoids per-target random history lookups and does not
write a reusable full-prefix cache.

For each user/history row:

1. materialize every target prefix through the official reader contract;
2. split targets by whether the legacy P23 checkpoint can represent the full
   prefix (`raw_context_event_count <= 200` by default);
3. for all short/full-feasible prefixes, run **one causal HSTU forward** over the
   longest prefix and extract the hidden state at each target prefix position;
4. score that target's 10,001 candidates from the extracted query state;
5. for zero-context targets, emit a deterministic zero-query fallback ranking
   (all scores tied, tie-broken by candidate id) because no causal sequential
   state exists;
6. for long prefixes, use the latest-200 window because P23 was trained with
   max sequence length 200 and cannot encode the full available prefix.

The short/full-feasible path is true sequential one-pass reuse. The long path is
not full-history reuse; it is explicitly labeled in the inference log.

## Context policy labels

### Zero context (`raw_context_event_count == 0`)

- Policy label in inference log: `zero_context_no_history_fallback`
- No causal sequential state exists. The runner emits schema-valid full-candidate
  coverage with a zero query; scores tie and are deterministically ordered by
  candidate id.

## Short-history vs long-history policy

### Short history (`raw_context_event_count <= 200`)

- Policy label in inference log: `full_available_history_for_p23`
- P23 sees every available event in `[T1, target_ts)`.
- Multiple targets on the same user/history row can share a single causal HSTU
  pass; target query states are selected by position.

### Long history (`raw_context_event_count > 200`)

- Policy label in inference log:
  `latest_200_due_legacy_p23_max_sequence_length`
- P23 sees only the latest 200 events before the target timestamp.
- This is necessary for the legacy checkpoint and must not be described as
  full-history evaluation.
- Long-history metrics must be interpreted separately from short-history
  full-available-history metrics.

## Example bounded validation command

```bash
RUN_ID=p23_short_history60_sequential_$(date -u +%Y%m%dT%H%M%SZ)
BASE=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_sequential_inference_reproduction
PKG=$BASE/package
RUN=$BASE/runs/$RUN_ID
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yourslewis/miniconda3/envs/hstu/bin/python \
  /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/infer/lrm_v001/sequential_submission_infer.py \
  --target-jsonl "$PKG/artifacts/short_history_sample_60_context_sorted.jsonl" \
  --history-prefix-source /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001 \
  --bank-root "$PKG/artifacts/production_banked_candidates" \
  --bank-generator "$PKG/vendor/banked_candidate_generator_v001.py" \
  --history-reader "$PKG/vendor/history_prefix_reader_v001.py" \
  --source-root /home/yourslewis/lrm-scaling-all-events \
  --gin-config-file /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/generated_p23_coordinate_search/p23_page_s10_p09_m01_o00.gin \
  --checkpoint-path /home/yourslewis/lrm-scaling-all-events/results_v2/p23_page_s10_p09_m01_o00/p23_page_s10_p09_m01_o00_20260514/ckpts/best_checkpoint_ndcg_10.pt \
  --model-submission-id p23_page_s10_p09_m01_o00.v001_short_history60_sequential \
  --prediction-run-id "$RUN_ID" \
  --context-policy "$PKG/context_policy.json" \
  --output-mode both \
  --output-predictions "$RUN/predictions.jsonl" \
  --output-compact "$RUN/compact_predictions.jsonl" \
  --output-metrics-json "$RUN/compact_metrics.json" \
  --output-inference-log "$RUN/inference_log.jsonl" \
  --output-target-ids "$RUN/prediction_target_ids.txt" \
  --device cuda:0 \
  --equivalence-check-targets 5
```

## Full-set safety-gated command

```bash
RUN_ID=p23_full_v001_sequential_$(date -u +%Y%m%dT%H%M%SZ)
/home/yourslewis/miniconda3/envs/hstu/bin/python \
  /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/infer/lrm_v001/run_full_submission_with_safety_gate.py \
  --target-sidecar-glob '/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase7_compact_generation/full_sidecars/targets/*.parquet' \
  --output-root /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_full_sequential_runs \
  --run-id "$RUN_ID" \
  --history-prefix-source /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001 \
  --bank-root /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_history_slice_reproduction/package/artifacts/production_banked_candidates \
  --bank-generator /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_history_slice_reproduction/package/vendor/banked_candidate_generator_v001.py \
  --history-reader /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_history_slice_reproduction/package/vendor/history_prefix_reader_v001.py \
  --source-root /home/yourslewis/lrm-scaling-all-events \
  --gin-config-file /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/config/generated_p23_coordinate_search/p23_page_s10_p09_m01_o00.gin \
  --checkpoint-path /home/yourslewis/lrm-scaling-all-events/results_v2/p23_page_s10_p09_m01_o00/p23_page_s10_p09_m01_o00_20260514/ckpts/best_checkpoint_ndcg_10.pt \
  --embedding-root /home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve \
  --model-submission-id p23_page_s10_p09_m01_o00.v001_full_sequential \
  --context-policy /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase8_history_slice_reproduction/package/context_policy.json \
  --device cuda:0 \
  --output-mode compact \
  --compact-top-k 10 \
  --burn-in-targets 100 \
  --max-estimated-hours 24 \
  --max-estimated-jsonl-gb 200 \
  --auto-proceed-if-sane
```

If the burn-in estimate exceeds thresholds, the runner writes a blocker report
and exits without starting unbounded inference.

## Known risks

- The scorer preserves the previous P23 adapter acceptance choice: base HSTU
  query plus learned item projection. It does not add a benchmark-side feature or
  change evaluator semantics.
- Long-history P23 metrics are latest-200 metrics, not full-history metrics.
- Full-set official JSONL can be enormous because every target writes all 10,001
  ranked candidates. Use compact mode plus the safety gate for tractable
  full-set metrics, and reserve full/both mode for bounded equivalence samples.
