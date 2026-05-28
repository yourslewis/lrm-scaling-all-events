# LRM benchmark v001 sequence-native execution shards

`proposed_2-mmoe_ple/infer/lrm_v001/build_execution_shards.py` creates an execution-only target ordering artifact for v001 inference. It preserves the frozen benchmark contract: target rows are reordered but not modified, candidate sets are untouched, labels/metrics are not added, and official manifests/evaluator inputs remain authoritative.

## Ordering

Rows are sorted by:

1. `context_reader_ref` split;
2. `context_reader_ref` canonical parquet part file;
3. `context_reader_ref` `source_row_index`;
4. `target_ts`;
5. `target_id`.

This makes all targets for the same canonical user/history row contiguous where possible, which lets `sequential_submission_infer.py` scan each referenced history part and reuse one causal sequence pass for multiple target prefixes.

## Build command

```bash
python3 proposed_2-mmoe_ple/infer/lrm_v001/build_execution_shards.py \
  --target-sidecar-glob '/path/to/full_sidecars/targets/*.parquet' \
  --output-dir /path/to/execution_shards/p23_full_v001 \
  --targets-per-shard 50000
```

For a bounded smoke/sample build:

```bash
python3 proposed_2-mmoe_ple/infer/lrm_v001/build_execution_shards.py \
  --target-jsonl /path/to/short_history_sample_60_context_sorted.jsonl \
  --output-dir /tmp/p23_short60_execution_shards \
  --targets-per-shard 20 \
  --overwrite
```

The builder writes:

```text
<output-dir>/
  execution_shard_manifest.json
  shards/
    shard_00000.targets.jsonl
    shard_00001.targets.jsonl
    ...
```

The manifest includes row-preservation digests, shard file digests, group/count metadata, and read-back validation status.

## Sequential runner integration

Run one inference job per shard JSONL:

```bash
SHARD=/path/to/execution_shards/p23_full_v001/shards/shard_00000.targets.jsonl
RUN_ID=p23_full_v001_seq_shard_00000
RUN=/path/to/runs/$RUN_ID
mkdir -p "$RUN"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/yourslewis/miniconda3/envs/hstu/bin/python \
  /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/infer/lrm_v001/sequential_submission_infer.py \
  --target-jsonl "$SHARD" \
  --history-prefix-source /home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001 \
  --bank-root /path/to/production_banked_candidates \
  --bank-generator /path/to/banked_candidate_generator_v001.py \
  --history-reader /path/to/history_prefix_reader_v001.py \
  --source-root /home/yourslewis/lrm-scaling-all-events \
  --gin-config-file /path/to/p23_page_s10_p09_m01_o00.gin \
  --checkpoint-path /path/to/best_checkpoint_ndcg_10.pt \
  --model-submission-id p23_page_s10_p09_m01_o00.v001_full_sequential \
  --prediction-run-id "$RUN_ID" \
  --context-policy /path/to/context_policy.json \
  --output-mode compact \
  --output-compact "$RUN/compact_predictions.jsonl" \
  --output-metrics-json "$RUN/compact_metrics.json" \
  --output-inference-log "$RUN/inference_log.jsonl" \
  --output-target-ids "$RUN/prediction_target_ids.txt" \
  --device cuda:0
```

Use a unique `prediction_run_id` and output directory per shard. Resume a failed shard by rerunning the same command with `--resume`. Concatenate outputs only after every shard succeeds; do not treat the execution-shard manifest as a replacement for the official target/candidate manifests.

## Validation guarantees

Default read-back validation checks:

- input and output target counts match;
- input and output row-multiset digests match;
- input and output target-id multiset digests match;
- distinct history group counts match;
- output order is nondecreasing by the sequence-native sort key;
- each history group appears contiguously.

If a single history group is larger than `--targets-per-shard`, that one shard is allowed to exceed the requested count rather than splitting the group.
