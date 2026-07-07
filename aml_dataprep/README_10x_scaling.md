# PConv Fullgraph 10x_v2 Scaling Pipeline

This note documents the corrected `10x_v2` path for the pconv/fullgraph AML run. It replaces the flawed `10x_v1` approach that only changed data and output paths while leaving CPU prep and GPU train/eval effectively single-instance.

## Files

- `generate_pconv_10x_pipeline.py`: materializes a concrete AML pipeline with CPU fan-out jobs and fan-in barriers.
- `pipeline_pconv_fullgraph_10x_v2.yml`: generated 10x_v2 pipeline using the 10x Cosmos root and isolated 10x_v2 output root.
- `submit_pconv_fullgraph_10x_v2.py`: optional submit helper. It is not run by validation.
- `validate_pconv_10x_pipeline.py`: local static validation plus `azure.ai.ml.load_job` when the SDK is installed.
- `parallel/relay_partition.py`, `vocab_spill_partition.py`, `parquet_partition.py`: shard-aware CPU wrappers using explicit `--shard_index` and `--num_shards`.
- `parallel/merge_partition_dirs.py`, `check_ready_manifests.py`: fan-in helpers that require ready manifests before downstream jobs run.
- `parallel/vocab_finalize_all_buckets.py`: finalizes vocab buckets after the reduce fan-in and prefix-sum step.

## Generate

Default corrected 10x_v2 generation:

```bash
python aml_dataprep/generate_pconv_10x_pipeline.py \
  --output aml_dataprep/pipeline_pconv_fullgraph_10x_v2.yml \
  --cpu-compute azureml:CPU-D2ADSV4 \
  --cpu-shards 10 \
  --num-buckets 64 \
  --gpu-instance-count 1 \
  --eval-batches 100 \
  --epochs 3
```

Key knobs:

- `--source-root`: Cosmos/AML datastore root. Defaults to `local/User/wenhlu/LRM_benchmark_v4_10x`.
- `--output-root`: isolated blob output root. Defaults to `derived/lrm_v4_pconv_v3/full_graph_10x_v2`.
- `--cpu-compute`: CPU target. Use multi-instance pools such as `azureml:CPU-D2ADSV4` or `azureml:CPU-E8aV4` rather than single-instance `CPU-E32SA`.
- `--cpu-shards`: number of relay/spill/parquet fan-out jobs. The generated default is 10 for 10x CPU scaling; increase toward available pool capacity for wider fan-out.
- `--num-buckets`: vocab hash buckets. This controls reduce parallelism as `5 * num_buckets` reduce jobs.
- `--gpu-instance-count`: exposed pipeline parameter. Default is `1`.
- `--eval-batches`: passed to per-event-type eval.
- `--epochs`: training epochs written into the temporary gin config.

## Submit

Do not submit during local validation. When ready to launch from an authenticated environment:

```bash
python aml_dataprep/submit_pconv_fullgraph_10x_v2.py \
  --pipeline aml_dataprep/pipeline_pconv_fullgraph_10x_v2.yml \
  --dry-run
```

Remove `--dry-run` only when intentionally submitting to AML.

## Scaling Semantics

CPU work is explicitly partitioned by manifest ordinal modulo `cpu_shards`. Each relay, vocab-spill, and parquet wrapper receives a fixed `--shard_index` and `--num_shards`, writes only its owned rows, and emits `_ready/*.json` sidecars. Merge jobs require one ready sidecar per input directory and fail on non-identical path collisions. Downstream vocab, parquet, train, and eval jobs reference merge/check outputs so AML enforces fan-in dependencies.

GPU `instance_count` is parameterized but still gated. Current train and eval commands use single-node `torchrun --nproc_per_node=8` and eval uses `torchrun --standalone --nproc_per_node=1`; generated jobs fail fast if `gpu_instance_count != 1`. Use `--allow-multinode-gpu` only to generate an experimental YAML, and update the torch distributed launch before running it with multiple GPU nodes.

## Local Validation

```bash
python -m unittest tests.aml_dataprep.test_parallel_partitioning tests.aml_dataprep.test_10x_pipeline_generator
python aml_dataprep/validate_pconv_10x_pipeline.py aml_dataprep/pipeline_pconv_fullgraph_10x_v2.yml
```

`validate_pconv_10x_pipeline.py` runs static dependency checks everywhere. If `azure.ai.ml` is installed, it also calls `azure.ai.ml.load_job` without submitting the job.
