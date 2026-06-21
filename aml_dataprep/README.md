# AML Data-Prep Pipeline (regenerate LRM training data on Azure ML)

Regenerates the LRM benchmark-v4 training data **entirely on AML / Singularity**,
reading the raw TSV shards directly from cosmos (ADLS Gen1) — so no 168 GB upload
from the gpu-trainer is needed. The output is functionally equivalent to the
frozen L800 dataset (same deterministic vocab -> embeddings -> parquet recipe),
so a model trained on it reproduces recall at a similar level.

## Why this exists
Uploading the derived 168 GB embeddings to cosmos kept failing. The raw shards,
however, are already on cosmos. Re-running the (scripted, deterministic) data
pipeline on 8x A100 is faster than the original 2x 4090 run and avoids the upload.

## Data flow

```
cosmos ADLS  (datastore: bingads_algo_pipelines_c08)
  local/User/wenhlu/LRM_benchmark_v4/train/train_chunk_000..109.tsv   (110 shards)
  local/User/wenhlu/LRM_benchmark_v4/val/val_chunk_0..39.tsv          (40 shards)
        |
        v   Job 1: data_prep/step1_collect_vocab_v2.py  (CPU, deterministic)
  workspaceblobstore: derived/lrm_v4/vocab_v3/
        domain_<d>_text2id.pkl, domain_<d>_id2text.pkl, vocab_meta.json
        |
        +--> Job 2: aml_dataprep/encode_embeddings_multigpu.py  (8x A100)
        |        workspaceblobstore: derived/lrm_v4/semantic_embeddings_v3/
        |            domain_<d>/shard_0.npy   (float16, indexed by item_id)
        |
        +--> Job 3: data_prep/step3_v2.py  (CPU)
                 workspaceblobstore: derived/lrm_v4/seqview_v001/
                     train/part_XXXX.parquet (one per chunk) + eval/part_0000.parquet
        |
        v   Job 4 (existing): L800 training
  reads seqview_v001 (data_path) + semantic_embeddings_v3 (domain_0..4)
```

## Jobs

| # | Spec | Compute | Reads | Writes |
|---|------|---------|-------|--------|
| 1 | `jobs/1_vocab.yml`   | 1 node (CPU use) | cosmos train+val | `vocab_v3` |
| 2 | `jobs/2_encode.yml`  | 8x A100-40       | `vocab_v3`       | `semantic_embeddings_v3` |
| 3 | `jobs/3_parquet.yml` | 1 node (CPU use) | cosmos + `vocab_v3` | `seqview_v001` |

All target Singularity VC `ads-shared-nd40`, `job_tier: Standard`.

## Submit (from the submitter compute instance, az ml v2 CLI)

```bash
RG=wb-aml; WS=pconv-aml-offline
az ml job create -f aml_dataprep/jobs/1_vocab.yml   -g $RG -w $WS   # wait -> Completed
az ml job create -f aml_dataprep/jobs/2_encode.yml  -g $RG -w $WS   # 8x A100 encode
az ml job create -f aml_dataprep/jobs/3_parquet.yml -g $RG -w $WS
```

(Jobs are submitted sequentially because 2 and 3 depend on 1's vocab output. A
future revision can fold these into a single `az ml` *pipeline* with explicit
data dependencies; kept as 3 command jobs here for transparency / debuggability.)

## Determinism / fidelity notes
- `step1` sorts the shard glob and assigns ids sequentially from `MIN_ITEM_ID=20`,
  so vocab ids are reproducible.
- `encode_embeddings_multigpu.py` only parallelizes; each id is encoded by exactly
  one worker and placed at row `item_id`. `embedding(text)` is a pure function of
  text + model, so the merged `shard_0.npy` is independent of GPU split.
- `step3` re-derives `encoded_id = domain*1e9 + item_id` from the same vocab, so
  parquet ids line up with the embedding rows.
- `MiniLM` model `paraphrase-multilingual-MiniLM-L12-v2` (emb_dim 384, fp16).

## Known gaps / TODO
- **val shards 0,1**: at time of writing cosmos had val 2..39 (38/40). Train is
  complete (110/110). Re-run Job 1 + Job 3 after the 2 missing val shards land to
  get the full 40-chunk eval set.
- Output currently lands on `workspaceblobstore` (fast, in-region). Copy to cosmos
  later if a durable cosmos copy is wanted.
- Consider converting to an `az ml` pipeline job for one-shot submission.
