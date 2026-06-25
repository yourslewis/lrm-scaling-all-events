# Spec: Parallel AML Data Prep for LRM-L800 pconv v3

Date: 2026-06-24
Branch: `feat/lrm-l800-msan`
Workspace: `pconv-aml-offline`

## Motivation

The current pconv v3 prep pipeline is too monolithic for the L800 data scale:

- Relay originally depended on stopped compute instance `cjpvm`; fixed by moving to AmlCompute.
- Relay then OOMed even on Linux CPU clusters because one AML job copied all 110 train + 40 val shards.
- Windows-backed `WIN-*` clusters cannot run the Linux AML environment image.
- Vocab v3 is memory-bounded on one node via hash buckets, but spill + reduce still run inside one job.
- Parquet conversion is naturally shard-parallel but currently runs as one command and aggregates validation into one output.
- Encode uses multiple GPUs on one node but final dense per-domain materialization remains a bottleneck.

The target is a distributed AML v2 pipeline where each expensive unit is retryable independently.

## Available Computes

Use only Linux-compatible computes for Linux Docker AML images:

| Compute | Size | Use |
| --- | --- | --- |
| `CPU-D2ADSV4` | `Standard_D2as_v4` | high-fan-out light CPU tasks; relay workers if shard memory is small |
| `CPU-ECONVDDA` | `Standard_DS11_v2` | heavier CPU reducers/converters; fallback relay workers |
| `CPU-D2AV4`, `CPU-A1V2`, `CPU-D1V2`, `CPU-D2ADSV5` | small CPU | optional light manifest/probe tasks |

Avoid for this pipeline:

- `WIN-*` clusters (`WIN-ECONVDDA`, `WIN-D2ADSV4`, `WIN-E8aV4`, `WIN-E32aV4`, `WIN-D2ADSV5`) because the AML prep image is Linux and Docker pull fails with `image operating system "linux" cannot be used on this platform`.
- `cjpvm` / `lrm-smoke-submitter` because compute instances must be manually running before submission.

## Proposed DAG

```text
discover_raw_shards
  -> relay_parallel
  -> raw_manifest

raw_manifest
  -> vocab_spill_parallel
  -> vocab_bucket_reduce_parallel
  -> vocab_prefix_sum
  -> vocab_finalize_parallel
  -> vocab_meta

raw_manifest + vocab_meta
  -> parquet_train_parallel
  -> parquet_val_parallel
  -> seqview_manifest

vocab_meta
  -> encode_bucket_parallel
  -> encode_domain_merge_parallel
  -> embedding_meta

seqview_manifest + embedding_meta
  -> train
```

## Data Layout Contracts

All output roots must include both a data version and a layout version so AML cache and manual debugging are unambiguous:

```text
azureml://datastores/workspaceblobstore/paths/derived/lrm_v4_pconv_v3/
  {data_version}/
    layout_v1/
      raw/
      vocab_spill/
      vocab_reduced/
      vocab/
      seqview/
      embeddings/
```

### Raw manifest and relay output

```text
raw/
  manifest.jsonl
  _SUCCESS
  train/train_chunk_0000.tsv
  train/train_chunk_0001.tsv
  val/val_chunk_0000.tsv
```

Each `manifest.jsonl` row:

```json
{"split":"train","shard_index":0,"source_uri":"...","dest_relpath":"train/train_chunk_0000.tsv","size_bytes":null,"etag":null}
```

### Vocab spill

```text
vocab_spill/
  domain_0/bucket_0000/part_train_0000.txt
  domain_0/bucket_0000/part_train_0001.txt
  domain_1/bucket_0342/part_val_0012.txt
```

Rules:

- No shared append across AML workers.
- One mapper writes unique `part_{split}_{shard}.txt` files.
- Text lines are normalized exactly once using the shared normalization function.

### Reduced vocab

```text
vocab_reduced/
  domain_0/bucket_0000/texts.txt
  domain_0/bucket_0000/count.json
```

Rules:

- Reducer owns exactly one `(domain, bucket)`.
- Deduplicate by exact full normalized text, not by hash.
- Sort lexicographically for deterministic ID assignment.

### Final vocab

```text
vocab/
  vocab_meta.json
  domain_0_text2id/manifest.json
  domain_0_text2id/bucket_0000.pkl
  domain_0_id2text/manifest.json
  domain_0_id2text/bucket_0000.pkl
```

No monolithic `domain_D_text2id.pkl` or `domain_D_id2text.pkl` for this scale.

### Seqview parquet

```text
seqview/
  metadata.json
  train/part_train_0000.parquet
  eval/part_val_0000.parquet
```

Validation must be per-shard, not accumulated into one dataframe or one parquet part.

### Embeddings

Compatibility layout:

```text
embeddings/
  embedding_meta.json
  domain_0/shard_0.npy
  domain_0/_parts/bucket_0000.npy
  domain_0/_parts/bucket_0000.ids.npy
```

Long-term layout should let training consume `_parts/` directly and remove dense `shard_0.npy` merges.

## Stage Designs

### 1. `discover_raw_shards`

Command job on a small Linux CPU compute.

Responsibilities:

- List `train` and `val` source shards from the Cosmos/ADLS datastore path.
- Write `raw_source_manifest.jsonl` with one row per shard.
- Avoid copying data.

### 2. `relay_parallel`

AML v2 `parallel` job or generated fan-out command jobs over `raw_source_manifest.jsonl`.

Parallel unit: one manifest row / one TSV shard.

Responsibilities:

- Stream one source shard to one deterministic raw output path.
- Use small read/write chunks.
- Write a per-shard done marker or rely on atomic temp-path rename where possible.

Compute:

- Prefer `CPU-D2ADSV4` for high fan-out if one shard fits.
- Use `CPU-ECONVDDA` if individual shards still need larger memory.

Retry behavior:

- Retry failed shard only.
- Do not rerun all 150 shards after one failure.

### 3. `vocab_spill_parallel`

Parallel unit: one raw TSV shard.

Responsibilities:

- Parse events.
- Normalize text with shared `extract_text_normalized`.
- Map event type to domain.
- Compute `bucket = stable_hash(text) % num_buckets`.
- Write text to unique spill files under `(domain, bucket, split, shard)`.

Recommended `num_buckets`: start with `4096`. `512` is too likely to leave skewed reducers at L800 scale.

### 4. `vocab_bucket_reduce_parallel`

Parallel unit: one `(domain, bucket)`.

Responsibilities:

- Read all spill parts for that `(domain, bucket)`.
- Deduplicate by full text.
- Sort text.
- Write `texts.txt` and `count.json`.

Implementation notes:

- Use local disk for temporary files.
- If in-memory `set` still OOMs, switch to external sort: concatenate local parts, Unix `sort -u`, then stream count.

### 5. `vocab_prefix_sum`

Small serial command job.

Responsibilities:

- Read all `count.json` files.
- Compute deterministic `start_id` for every `(domain, bucket)`:

```text
start_id[domain,bucket] = MIN_ITEM_ID + sum(count[domain,b] for b < bucket)
```

- Write `vocab_offsets.json` and `vocab_meta.json` skeleton.

### 6. `vocab_finalize_parallel`

Parallel unit: one `(domain, bucket)`.

Responsibilities:

- Read `texts.txt` and offset.
- Assign IDs deterministically.
- Write bucket `text2id` and `id2text` pickles.
- Write/update per-bucket manifest rows.

### 7. `parquet_*_parallel`

Parallel unit: one raw TSV shard.

Responsibilities:

- Read sharded vocab lazily by hashing lookup text to bucket.
- Convert one raw TSV shard to one parquet part.
- Write per-part stats: row count, miss count, max sequence length.

Validation uses exactly the same per-shard path as train.

### 8. `seqview_manifest`

Small serial command job.

Responsibilities:

- Aggregate per-part stats.
- Write final `metadata.json`.

### 9. `encode_bucket_parallel`

Parallel unit: one `(domain, bucket)` or a small group of buckets.

Responsibilities:

- Read bucket `id2text`.
- Encode texts on GPU.
- Write bucket embedding array and IDs array.

### 10. `encode_domain_merge_parallel`

Compatibility stage, one job per domain.

Responsibilities:

- Merge bucket arrays into dense `domain_D/shard_0.npy` for current training code.

Long-term:

- Remove this stage by teaching training to read sharded embedding manifests.

## AML Implementation Strategy

Preferred implementation:

- Use AML v2 `parallel` jobs for row-based fan-out over manifests.
- If `parallel` job input/output semantics are too restrictive, generate explicit command jobs from manifests for the first implementation.

New/changed files expected:

```text
aml_dataprep/parallel/
  discover_raw_shards.py
  relay_one_shard.py
  vocab_spill_one_shard.py
  vocab_reduce_bucket.py
  vocab_prefix_sum.py
  vocab_finalize_bucket.py
  parquet_one_shard.py
  aggregate_seqview_manifest.py
  encode_bucket.py
  merge_domain_embeddings.py

aml_dataprep/pipeline_v3_parallel_pconv.yml
```

Shared normalization/domain code should be factored from current `step1_collect_vocab_v3.py` / `step3_v3.py` to avoid byte drift:

```text
data_prep/vocab_common_v3.py
```

## Determinism Requirements

- Use `hashlib.blake2b(..., digest_size=8)` or the current stable hash; never use Python `hash()`.
- Normalize text identically in vocab, parquet, and encode.
- Sort reducer output lexicographically before assigning IDs.
- Bucket and domain iteration order must be numeric ascending.
- Record `num_buckets`, `MIN_ITEM_ID`, normalizer version, code git SHA, and data version in `vocab_meta.json`.

## Failure and Cache Semantics

- Every parallel task writes unique deterministic output paths.
- Avoid shared append outputs.
- Prefer temp output path + marker/rename for retry safety.
- Include `data_version` and `layout_version` in every output root.
- Bump `layout_version` when changing bucket count, normalization, output schema, embedding dtype/model, or parquet schema.

## Rollout Plan

### Phase 1: CPU prep scalability

1. Add shared vocab/common utilities.
2. Add raw discovery + per-shard relay.
3. Split vocab into spill/reduce/prefix/finalize modes.
4. Add per-shard parquet conversion and seqview manifest aggregation.
5. Add `pipeline_v3_parallel_pconv.yml` through CPU prep; keep existing encode/train wiring initially if needed.

### Phase 2: embedding scalability

1. Add bucket-parallel encode.
2. Add per-domain dense merge for compatibility.
3. Validate embeddings match current shape and training starts.

### Phase 3: remove dense bottlenecks

1. Update training loader to consume sharded embeddings.
2. Delete or bypass `shard_0.npy` merge.

## Acceptance Criteria

- Relay can retry an individual shard without restarting all copies.
- Vocab reducers run independently by `(domain, bucket)` with deterministic IDs.
- Val parquet is emitted as multiple parts and does not accumulate all rows in memory.
- Pipeline YAML uses only Linux-compatible computes for Linux AML images.
- Existing train command can run with compatibility embedding layout.
- A small subset smoke run completes with a low shard count and reduced bucket count before launching full L800.
