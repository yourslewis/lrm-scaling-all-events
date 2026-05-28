# LRM v001 Evaluation Harness

This folder is the engineer-facing entry point for the LRM v001 fixed-bank evaluation framework.

It splits the framework into clear ownership boundaries:

- **ML engineers** provide model-owned embedding artifacts:
  - target/user query embeddings at each target
  - projected document/item embeddings for candidate banks
- **Testing engineers** consume those artifacts to score candidates and compute metrics without loading model internals.

The current fixed proxy is the **500-bank seed-1 / ~10% target proxy**. Results must be labeled as proxy results, not official full v001, unless the full 18M-target contract is explicitly run.

## CLI summary

### 1. ML-owned target/query encoder

```bash
python evaluation_harness/lrm_v001/bin/encode_target_queries.py \
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
  --device cuda:0
```

Output:

```text
query_cache/manifest.json
query_cache/groups/domain_<domain_id>/bank_<bank_id>/batch_*.pt
```

### 2. ML-owned document/item encoder

```bash
python evaluation_harness/lrm_v001/bin/encode_documents.py \
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
  --device cuda:0
```

Output:

```text
doc_cache/manifest.json
doc_cache/domain_<domain_id>/bank_<bank_id>.pt
```

Each bank contains `candidate_ids`, `embeddings`, and optional `extra_ids` / `extra_embeddings` needed for positives or replacement candidates absent from the base bank.

### 3. Testing-owned encoded scorer + metrics

```bash
python evaluation_harness/lrm_v001/bin/score_encoded_proxy.py \
  --query-cache-dir "$RUN_ROOT/query_cache" \
  --doc-cache-dir "$RUN_ROOT/doc_cache" \
  --bank-root "$BANK_ROOT" \
  --bank-generator "$BANK_GENERATOR" \
  --selected-bank-subset "$SELECTED_BANK_SUBSET" \
  --model-submission-id "$MODEL_SUBMISSION_ID" \
  --prediction-run-id "$PREDICTION_RUN_ID" \
  --output-compact "$RUN_ROOT/compact_predictions.jsonl" \
  --output-metrics-json "$RUN_ROOT/compact_metrics.json" \
  --output-inference-log "$RUN_ROOT/score_log.jsonl" \
  --compact-top-k 10 \
  --score-query-chunk-size 4096 \
  --device cuda:0
```

Output:

```text
compact_predictions.jsonl
compact_metrics.json
score_log.jsonl
```

## Required headline metrics

For table reporting, use HR@10 metrics in both micro and macro-user forms:

- `all_domain`: `micro_OHR@10`, `macro_user_OHR@10`
- `all_ads`: `micro_AHR@10`, `macro_user_AHR@10`
- `cold_ads`: `micro_AHR@10`, `macro_user_AHR@10`
- `warm_ads`: `micro_AHR@10`, `macro_user_AHR@10`

Definitions:

- `micro`: target-weighted average over targets
- `macro_user`: average over per-user metrics

## Full runbook

See [`RUNBOOK.md`](RUNBOOK.md) for input requirements, output artifact contracts, correctness gates, performance gates, and common failure modes.
