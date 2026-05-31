# LRM v001 500-Bank Proxy Eval + Raw Bank Cache Runbook

## Purpose

This runbook is the handoff for the short-term **LRM v001 fast 500-bank proxy eval** and the parked full-target follow-up.

Current plan (**Option A**) is a fast proxy evaluation:

- Use the existing v001 candidate bank specs.
- Select a fixed subset of 500 banks: 100 banks/domain × 5 domains.
- Evaluate only targets whose existing `(target_canonical_domain_id, negative_bank_id)` is in that subset.
- Use a pre-cooked raw 384-d fp16 bank embedding cache resident on GPU.
- Keep model projection/checkpoint behavior unchanged.
- Emit compact prediction/metric artifacts, not full 10TB-scale prediction JSONL.

Performance target for the fast path:

```text
~1.8M selected targets in ~20 minutes
~= 1,500 targets/second end-to-end
```

Anything materially below this target should be treated as a performance regression or fallback to a non-fast path until proven otherwise.

Future plan (**Option B**) is parked for later:

- Evaluate 100% of targets with all 5,000 original banks.
- Use all-bank CPU-hosted raw cache/staging or all-bank projected 128-d cache.
- Avoid target-major random mmap lookup into huge raw embedding shards.

## What does not change in the benchmark framework

Option A is an execution-side proxy. It must not redefine v001.

Unchanged:

- Existing v001 bank specs remain **ID-only** and are not rebuilt.
- Existing target sidecars/execution shards remain the target source.
- Existing `(target_canonical_domain_id, negative_bank_id)` assignments remain authoritative.
- Candidate sets are not remapped to easier/different banks.
- Positive item handling and positive-bank collision/replacement semantics remain unchanged.
- Model gin/checkpoint loading and the trainable projection layer remain model-owned.
- Official v001 prediction schema/evaluator are not changed.
- Compact output is a derived execution artifact for this proxy; it is not a new official benchmark contract.

The proxy result must be labeled as **500-bank seed-1 10% proxy**, not as official full v001.

## Current embedding family

For current P20/P23/v001 runs, the shared frozen raw embedding artifact is:

```text
embedding root: /home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve
encoder: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
raw dim: 384
raw dtype: fp16
embedding module: MultiDomainPrecomputed
```

The model-owned trainable projection is checkpoint-specific:

```text
384 -> 1024 -> 128 -> HSTU@128
```

Do **not** use this raw cache for the older/alternate PinSage config family:

```text
PinSage frozen 64 -> trainable MLP -> 256 -> HSTU@256
```

That family needs a separate cache/pre-cook if evaluated.

## Ownership split

### ML engineer / model owner owns

- Model checkpoint and gin config.
- Verifying the checkpoint uses the current MiniLM-384 `MultiDomainPrecomputed` embedding family before reusing the raw cache.
- Verifying the trainable projection layer remains active and checkpoint-specific.
- Integrating or reviewing the fast bank-major scorer path.
- Preserving ranking correctness: exact positive score/rank, tie handling, candidate counts, positive collision replacement, and compact topK.
- Running scorer/unit equivalence gates before comparing models.

### Test engineer / eval owner owns

- Using the same selected-bank manifest and same raw bank cache for every baseline/treatment comparison.
- Running the proxy command with required flags and compact output only.
- Labeling results as the 500-bank seed-1 proxy, not official full v001.
- Collecting required artifacts and digests.
- Checking cache/runtime health while the run is active.
- Blocking or rerunning if throughput/cache gates fail.

### Benchmark/data infra owns

- Candidate bank specs, ID-only:
  - `bank_id`
  - `start_id`
  - `step`
  - `count=10000`
  - digests / generation metadata
- Target sidecars / execution shards containing original target metadata:
  - `target_canonical_domain_id`
  - `negative_bank_id`
  - `candidate_set_digest`
  - `context_reader_ref`
  - positive item metadata
- The raw embedding root used to build the model-independent selected-bank raw cache.

## Required fixed artifacts for current Option A

Selected-bank manifest:

```text
/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/selected_banks_seed1_100_per_domain.json
```

Raw bank cache:

```text
/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/raw_bank_cache_384fp16
```

The raw cache is model-independent for current MiniLM-384 runs and is shared across baseline/treatments using the same raw embedding root.

Observed cache size:

```text
raw_bank_cache_384fp16: ~3.7G on disk
resident_raw_bytes: 3.84GB on GPU
```

For each run, preserve:

- selected-bank manifest and digest;
- raw bank cache `manifest.json` and digest;
- git SHA or local diff identifier for inference code;
- model gin config path and checkpoint path;
- checkpoint digest / model digest;
- exact command or command file;
- `compact_predictions.jsonl`;
- `compact_metrics.json`;
- `inference_log.jsonl`;
- `target_ids.txt` if emitted;
- final throughput/cache summary.

## Artifact build commands

These commands exist in the repo and can be used to rebuild the fixed proxy artifacts when needed.

Create the selected-bank manifest:

```bash
python proposed_2-mmoe_ple/infer/lrm_v001/make_selected_bank_subset.py \
  --output /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/selected_banks_seed1_100_per_domain.json \
  --banks-per-domain 100 \
  --domains 0,1,2,3,4 \
  --total-banks-per-domain 1000 \
  --seed 1 \
  --strategy uniform_random_per_domain
```


`$BANK_GENERATOR` and `$HISTORY_READER` must point to the existing vendor files from the v001 reproduction package, for example the Phase 8 package vendor files:

```text
vendor/banked_candidate_generator_v001.py
vendor/history_prefix_reader_v001.py
```

They are inputs to these commands, not files in `proposed_2-mmoe_ple/infer/lrm_v001/`.

Build the raw MiniLM-384 selected-bank cache:

```bash
python proposed_2-mmoe_ple/infer/lrm_v001/build_raw_candidate_bank_cache.py \
  --bank-root "$BANK_ROOT" \
  --bank-generator "$BANK_GENERATOR" \
  --embedding-root /home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve \
  --selected-bank-subset /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/selected_banks_seed1_100_per_domain.json \
  --output-dir /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/raw_bank_cache_384fp16 \
  --dtype float16
```

## Engineer-facing CLI contract

The evaluation framework should have a clean boundary between model-owned encoding code and testing-owned metric/scoring code. Testing engineers should not need to edit model internals to get numbers, and ML engineers should be able to change model code while preserving a stable embedding-output contract.

### Testing engineer CLIs: metric production

These CLIs are owned by the evaluation/testing side and should be runnable without changing model code.

1. **Stream metrics from compact predictions** — implemented:

```bash
python proposed_2-mmoe_ple/infer/lrm_v001/compact_streaming_evaluate.py \
  --compact-jsonl "$RUN_ROOT/compact_predictions.jsonl" \
  --output-json "$RUN_ROOT/compact_metrics.json" \
  --k 10 \
  --candidate-protocol-label fixed_500_bank_seed1_10pct_proxy
```

Expected output includes headline slices:

```text
all_domain: OHR@10
all_ads:    AHR@10
cold_ads:   AHR@10
warm_ads:   AHR@10
```

Each headline metric must include both:

```text
micro_*@10       # target-weighted average over targets
macro_user_*@10  # average of per-user metrics
```

2. **End-to-end fixed 500-bank proxy scoring** — implemented:

```bash
python proposed_2-mmoe_ple/infer/lrm_v001/fast_proxy_eval_runner.py \
  --target-sidecar-glob "$TARGET_SIDECAR_GLOB" \
  --selected-bank-subset /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/selected_banks_seed1_100_per_domain.json \
  --target-batch-size 200000 \
  --history-prefix-source "$HISTORY_PREFIX_SOURCE" \
  --bank-root "$BANK_ROOT" \
  --bank-generator "$BANK_GENERATOR" \
  --history-reader "$HISTORY_READER" \
  --source-root /home/yourslewis/lrm-scaling-all-events \
  --gin-config-file "$GIN_CONFIG" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --embedding-root /home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve \
  --context-policy "$CONTEXT_POLICY" \
  --raw-bank-cache-dir /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/raw_bank_cache_384fp16 \
  --raw-bank-cache-placement gpu \
  --candidate-cache-max-banks 500 \
  --model-submission-id "$MODEL_SUBMISSION_ID" \
  --prediction-run-id "$PREDICTION_RUN_ID" \
  --output-compact "$RUN_ROOT/compact_predictions.jsonl" \
  --output-metrics-json "$RUN_ROOT/compact_metrics.json" \
  --output-inference-log "$RUN_ROOT/inference_log.jsonl" \
  --query-cache-dir "$RUN_ROOT/query_cache" \
  --device cuda:0 \
  --compact-top-k 10 \
  --query-cache-batch-size 4096 \
  --query-cache-dtype float32 \
  --score-query-chunk-size 4096 \
  --extra-embedding-chunk-size 4096 \
  --context-checksum-mode none \
  --candidate-check-mode collisions \
  --log-flush-every 100 \
  --output-flush-every 0 \
  --stdout-progress-every 10000
```

This runner currently does both model query encoding and bank-major scoring. It is useful for current experiments, but the long-term testing contract should separate the model-owned encoders below.

### ML engineer CLIs: model-owned embedding production

These are the CLIs that should be provided by the ML engineer / model owner to testing engineers. They should be stable across model architecture changes. Testing engineers should only consume their artifacts.

#### 1. Sequential target/query encoder CLI — required contract

Purpose: encode every selected target into the model query vector used for retrieval scoring.

Input requirements:

```text
- target source: target sidecar parquet glob or target JSONL
- selected-bank subset manifest: fixed 500-bank seed-1 manifest
- history prefix source: canonical row-array root
- history reader module: v001 vendor history_prefix_reader_v001.py
- model source root + gin config + checkpoint
- embedding root compatible with the checkpoint
- context policy JSON
```

Recommended CLI name:

```text
proposed_2-mmoe_ple/infer/lrm_v001/encode_target_queries.py
```

Required command shape:

```bash
python proposed_2-mmoe_ple/infer/lrm_v001/encode_target_queries.py \
  --target-sidecar-glob "$TARGET_SIDECAR_GLOB" \
  --selected-bank-subset "$SELECTED_BANK_SUBSET" \
  --history-prefix-source "$HISTORY_PREFIX_SOURCE" \
  --history-reader "$HISTORY_READER" \
  --source-root "$SOURCE_ROOT" \
  --gin-config-file "$GIN_CONFIG" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --embedding-root "$EMBEDDING_ROOT" \
  --context-policy "$CONTEXT_POLICY" \
  --output-query-cache-dir "$RUN_ROOT/query_cache" \
  --output-inference-log "$RUN_ROOT/query_encode_log.jsonl" \
  --query-dtype float32 \
  --target-batch-size 200000 \
  --encode-batch-size 4096 \
  --device cuda:0 \
  --max-sequence-length 200
```

Required output contract:

```text
query_cache/manifest.json
query_cache/groups/domain=<domain_id>/bank=<negative_bank_id>/part-*.pt
```

Each group part must contain:

```text
queries: float tensor [num_targets, query_dim]
targets: list of minimal target records with target_id, target_event_id, user_id,
         target_canonical_domain_id, negative_bank_id, positive_item_id,
         candidate_set_digest, headline/slice metadata
```

Manifest must include:

```text
model_digest
checkpoint_path/checksum
context_policy_digest
query_dim
query_dtype
target_count
group_count
selected_bank_subset_digest
created_at
```

Correctness gates:

- Same target count as selected-bank-filtered target sidecars.
- Stable `(target_id, user_id, target_event_id)` ordering within each group part.
- Query vectors are the exact vectors used by the model's retrieval scorer.
- Re-running with the same checkpoint/artifacts produces the same manifest digest or an explicitly explained nondeterminism note.

Current status: **implemented as standalone `encode_target_queries.py`**. The same query-cache builder is still used internally by `fast_proxy_eval_runner.py` for the combined runner.

#### 2. Document/item encoder CLI — required contract

Purpose: encode candidate documents/items into the model document vector space used by retrieval scoring.

Input requirements:

```text
- bank root: production banked candidate artifacts
- bank generator module: v001 vendor banked_candidate_generator_v001.py
- selected-bank subset manifest: same manifest used for query encoding
- model source root + gin config + checkpoint
- embedding root compatible with the checkpoint
- raw bank cache: model-independent MiniLM-384 selected-bank cache
- optional query cache: required when testing scorer must handle positive/replacement extras without loading the model
```

Recommended CLI name:

```text
proposed_2-mmoe_ple/infer/lrm_v001/encode_documents.py
```

Required command shape for selected-bank eval:

```bash
python proposed_2-mmoe_ple/infer/lrm_v001/encode_documents.py \
  --bank-root "$BANK_ROOT" \
  --bank-generator "$BANK_GENERATOR" \
  --selected-bank-subset "$SELECTED_BANK_SUBSET" \
  --source-root "$SOURCE_ROOT" \
  --gin-config-file "$GIN_CONFIG" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --embedding-root "$EMBEDDING_ROOT" \
  --raw-bank-cache-dir "$RAW_BANK_CACHE_DIR" \
  --query-cache-dir "$RUN_ROOT/query_cache" \
  --output-doc-cache-dir "$RUN_ROOT/doc_cache" \
  --output-inference-log "$RUN_ROOT/doc_encode_log.jsonl" \
  --doc-dtype float16 \
  --device cuda:0 \
  --encode-batch-size 4096
```

Required output contract:

```text
doc_cache/manifest.json
doc_cache/domain=<domain_id>/bank=<bank_id>.pt
```

Each bank file must contain:

```text
candidate_ids: list[str] length 10000
embeddings: tensor [10000, doc_dim]
extra_ids: list[str] for positive/replacement items needed by the query cache but absent from the base bank
extra_embeddings: tensor [len(extra_ids), doc_dim]
```

Manifest must include:

```text
model_digest
checkpoint_path/checksum
embedding_root/checksum or artifact id
raw_bank_cache_digest
selected_bank_subset_digest
doc_dim
doc_dtype
bank_count
per-bank candidate_count and extra_count
created_at
```

Correctness gates:

- Candidate IDs exactly match `banked_candidate_generator_v001.materialize_bank(...)` for each bank.
- Doc embeddings are normalized exactly as the scoring path expects.
- Positive-bank collision handling remains a scoring concern; the doc encoder should not mutate bank membership.
- For MiniLM-384 `MultiDomainPrecomputed`, raw 384-d bank cache is model-independent, but projected doc embeddings are checkpoint-specific because the projection layer is trained.

Current status: **implemented as standalone `encode_documents.py`**. Current combined scoring code can still project bank embeddings internally from `raw_bank_cache_384fp16`, but testing engineers now also have a model-owned projected-doc artifact interface.

### Testing engineer consumer CLI after encoder split

Once the two ML-owned encoder CLIs exist, testing should run a pure consumer scorer that does not load model internals.

Input requirements:

```text
- query cache produced by encode_target_queries.py
- doc cache produced by encode_documents.py for the same checkpoint/selected-bank manifest
- bank root + bank generator for candidate collision/replacement contract checks
- model/prediction run labels
```

Output requirements:

```text
- compact_predictions.jsonl: one compact record per selected target
- compact_metrics.json: headline and detailed micro/macro_user metrics
- inference_log.jsonl: run progress, group timings, failure diagnostics
```

CLI syntax:

```bash
python proposed_2-mmoe_ple/infer/lrm_v001/score_encoded_proxy.py \
  --query-cache-dir "$RUN_ROOT/query_cache" \
  --doc-cache-dir "$RUN_ROOT/doc_cache" \
  --bank-root "$BANK_ROOT" \
  --bank-generator "$BANK_GENERATOR" \
  --selected-bank-subset "$SELECTED_BANK_SUBSET" \
  --model-submission-id "$MODEL_SUBMISSION_ID" \
  --prediction-run-id "$PREDICTION_RUN_ID" \
  --output-compact "$RUN_ROOT/compact_predictions.jsonl" \
  --output-metrics-json "$RUN_ROOT/compact_metrics.json" \
  --output-inference-log "$RUN_ROOT/inference_log.jsonl" \
  --compact-top-k 10 \
  --score-query-chunk-size 4096
```

Current status: **implemented as `score_encoded_proxy.py`**. `fast_proxy_eval_runner.py` remains the working combined runner for convenience and backwards-compatible production runs.

## Runner command status

### Implemented: safe sequential proxy runner with raw cache

`sequential_submission_infer.py` has selected-bank/raw-cache/compact-output flags. This is runnable and preserves correctness, but it is target-major. Use it as the safe fallback and for equivalence/debug runs.

```bash
python proposed_2-mmoe_ple/infer/lrm_v001/sequential_submission_infer.py \
  --target-sidecar-glob "$TARGET_SIDECAR_GLOB" \
  --selected-bank-subset /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/selected_banks_seed1_100_per_domain.json \
  --history-prefix-source "$HISTORY_PREFIX_SOURCE" \
  --bank-root "$BANK_ROOT" \
  --bank-generator "$BANK_GENERATOR" \
  --history-reader "$HISTORY_READER" \
  --source-root /home/yourslewis/lrm-scaling-all-events \
  --gin-config-file "$GIN_CONFIG" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --embedding-root /home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve \
  --model-submission-id "$MODEL_SUBMISSION_ID" \
  --prediction-run-id "$PREDICTION_RUN_ID" \
  --context-policy "$CONTEXT_POLICY" \
  --output-mode compact \
  --output-compact "$RUN_ROOT/compact_predictions.jsonl" \
  --output-metrics-json "$RUN_ROOT/compact_metrics.json" \
  --compact-no-score-order-digest \
  --output-inference-log "$RUN_ROOT/inference_log.jsonl" \
  --output-target-ids "$RUN_ROOT/target_ids.txt" \
  --device cuda:0 \
  --candidate-cache-max-banks 500 \
  --raw-bank-cache-dir /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/raw_bank_cache_384fp16 \
  --raw-bank-cache-placement gpu \
  --log-flush-every 0 \
  --output-flush-every 0 \
  --stdout-progress-every 10000
```

### Implemented: fast bank-major combined runner

`fast_proxy_eval_runner.py` is the current production-speed runner for the fixed 500-bank / 10% proxy. It builds/reuses a query cache, projects/loads selected banks, scores by bank-major matrix multiply, emits compact predictions, and writes compact metrics.

Use `--debug-equivalence-targets N` with `N <= 1000` for smoke/debug only. Production runs should normally disable it and rely on unit/equivalence gates plus aggregate metric checks.

Required implementation property: group selected targets by `(target_canonical_domain_id, negative_bank_id)`, project/load each selected bank once, score a query batch against that bank matrix, and emit compact records through the same compact metrics path. Do not reintroduce target-major raw mmap lookup.

## Required flags and settings

For the 500-bank proxy, these are required for any runner/scorer path:

```text
--selected-bank-subset <selected_banks_seed1_100_per_domain.json>
--raw-bank-cache-dir <raw_bank_cache_384fp16>
--raw-bank-cache-placement gpu
--candidate-cache-max-banks 500
--output-compact <compact_predictions.jsonl>
--output-metrics-json <compact_metrics.json>
```

Sequential runner only:

```text
--output-mode compact
--compact-no-score-order-digest
```

Strongly recommended for the fast proxy:

```text
--target-batch-size 200000
--query-cache-batch-size 4096
--score-query-chunk-size 4096
--context-checksum-mode none
--candidate-check-mode collisions
--log-flush-every 100
--output-flush-every 0
--stdout-progress-every 10000
```

Important comparison rule:

```text
Use the same selected-bank manifest for every baseline/treatment comparison.
Do not sample a different 500-bank subset per model.
```

## Expected target coverage

Full v001 target set:

```text
total targets: 18,024,232
Ads targets:     177,007
```

Current 500-bank seed-1 proxy keeps roughly 10%:

```text
total targets: ~1.8M
Ads targets:   ~17.7K
```

This preserves the original candidate sets for selected targets. It does **not** remap targets to different banks.

## Correctness gates

Before comparing models:

1. Verify selected-bank manifest:
   - schema is `lrm_v001_selected_bank_subset_v001`;
   - `banks_per_domain=100`;
   - 5 domains × 100 banks = 500 selected banks;
   - same manifest digest across all compared models.
2. Verify raw cache manifest:
   - schema is `lrm_v001_raw_candidate_bank_cache_v001`;
   - `total_banks=500`;
   - raw dim is 384;
   - dtype is fp16;
   - embedding root is `semantic_embeddings_v3_full_preserve`;
   - selected-bank digest matches the selected-bank manifest.
3. Verify model family:
   - `MultiDomainPrecomputed` MiniLM-384 path only;
   - not PinSage 64→256;
   - trainable projection is still loaded from the checkpoint.
4. Run scorer/compact unit gates before relying on fast runner changes:

   ```bash
   python proposed_2-mmoe_ple/infer/lrm_v001/tests/test_fast_proxy_bank_scorer.py
   python proposed_2-mmoe_ple/infer/lrm_v001/tests/test_compact_metrics.py
   python proposed_2-mmoe_ple/infer/lrm_v001/tests/test_phase9_option_a_integration.py
   ```

5. For any new fast runner CLI, run a small A/B equivalence slice against the implemented sequential raw-cache runner:
   - same selected-bank manifest;
   - same targets;
   - same checkpoint;
   - same compact topK;
   - exact match for candidate count, positive score tolerance, pessimistic rank, hit@10/NDCG@10, and compact topK ordering.
6. Confirm compact metrics are produced from the selected target population only, and run metadata labels the result as the 500-bank seed-1 proxy.

## Runtime and performance gates

Healthy Option A run should show:

```text
evictions = 0
raw_transfer_s = 0.0              # for GPU placement
resident_raw_bytes ~= 3.84GB
candidate scoring p50 ~= milliseconds, not seconds
cache hit rate climbs toward high 90% after warmup
end-to-end throughput ~= 1,500 targets/s for the true fast runner
~1.8M targets finish in ~20 minutes
```

Smoke test observed for candidate scoring path:

```text
1000 selected targets
p50 total candidate scoring: ~0.0038s/target
p90:                         ~0.0049s/target
p99:                         ~0.0051s/target
raw_transfer_s:              0.0
evictions:                   0
```

The smoke timing above is not enough by itself. The production gate is end-to-end throughput across history loading, query encoding, bank scoring, compact output, and metrics aggregation.

## Common failure modes

Check these first when a run is slow, wrong, or incomparable:

1. Missing `--selected-bank-subset`, causing the run to include all targets/banks.
2. Missing or wrong `--raw-bank-cache-dir`, falling back to raw global mmap lookup.
3. `--raw-bank-cache-placement cpu` used accidentally for the GPU-resident proxy.
4. `--candidate-cache-max-banks` lower than selected bank count, causing evictions.
5. Wrong embedding family, e.g. PinSage 64 path using MiniLM-384 cache.
6. Different selected-bank manifests across baseline/treatment models.
7. `--output-mode full` or `both`, producing huge full JSONL and unnecessary work.
8. Full score-order digest enabled for compact proxy, causing avoidable sort/digest overhead.
9. Future fast runner not actually grouping by bank, so it silently behaves like target-major scoring.
10. Compact metrics labeled as official full v001 instead of 10% proxy.
11. Stale driver/process state: progress file says running but output line counts stop growing.
12. GPU memory pressure forcing raw cache eviction or CPU transfer.

## Current reproduction run

Current P23 proxy reproduction run was launched at:

```text
/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_phase9_option_a_proxy_500_seed1/p23_proxy500_seed1_10pct_20260527T221552Z
```

It uses:

```text
model: p23_page_s10_p09_m01_o00
checkpoint: results_v2/p23_page_s10_p09_m01_o00/p23_page_s10_p09_m01_o00_20260514/ckpts/best_checkpoint_ndcg_10.pt
selected banks: seed1, 100/domain
raw cache placement: gpu
output mode: compact
```

## Parked future Option B: 100% targets

For full official-style 100% target evaluation, do **not** remap targets to 500 banks.

Use all original targets and all original bank IDs:

```text
18,024,232 targets
5,000 banks = 1,000/domain × 5 domains
```

Candidate approaches:

1. CPU-hosted raw 384-d fp16 all-bank store with bank staging:

```text
--raw-bank-cache-placement cpu
```

This requires avoiding target-major random bank churn. Prefer bank-grouped scoring or a large enough GPU projected-bank cache.

2. Model-specific projected 128-d fp16 all-bank cache:

```text
5000 banks × 10000 candidates × 128 dims × fp16 ~= 11.9 GiB
```

This may fit on one 24GB GPU with careful overhead management, but it is checkpoint-specific because the projection layer is trained.

Do not return to the original target-major raw mmap lookup over huge global embedding shards; profiling showed that path is the bottleneck.
