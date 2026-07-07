#!/usr/bin/env python3
"""Generate the pconv/fullgraph 10x_v2 AML pipeline with explicit CPU fan-out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

GPU_COMPUTE = "/subscriptions/72a1fe05-772c-4836-869f-761a5805fcd4/resourceGroups/Ads-Singularity-RG-01/providers/Microsoft.MachineLearningServices/virtualClusters/ads-shared-nd40"
GPU_INSTANCE_TYPE = "Singularity.ND96rs_v4"


def emit(lines: list[str], text: str = "") -> None:
    lines.append(text)


def command_block(lines: list[str], command: str, indent: int = 4) -> None:
    pad = " " * indent
    emit(lines, f"{pad}command: >-")
    for line in command.strip().splitlines():
        emit(lines, f"{pad}  {line.rstrip()}")


def cpu_env(lines: list[str], indent: int = 4, parquet: bool = False) -> None:
    pad = " " * indent
    emit(lines, f"{pad}environment:")
    emit(lines, f"{pad}  image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest")
    emit(lines, f"{pad}  conda_file:")
    emit(lines, f"{pad}    name: pconv-10x-v2-cpu")
    emit(lines, f"{pad}    channels: [conda-forge]")
    emit(lines, f"{pad}    dependencies:")
    emit(lines, f"{pad}      - python=3.10")
    if parquet:
        emit(lines, f"{pad}      - pip")
        emit(lines, f"{pad}      - pip:")
        emit(lines, f"{pad}          - numpy")
        emit(lines, f"{pad}          - pandas")
        emit(lines, f"{pad}          - pyarrow")


def add_output(lines: list[str], name: str, root: str, suffix: str) -> None:
    emit(lines, f"  {name}:")
    emit(lines, "    type: uri_folder")
    emit(lines, "    mode: rw_mount")
    emit(lines, f"    path: {root.rstrip('/')}/{suffix}")


def output_value() -> str:
    return "{type: uri_folder, mode: rw_mount}"


def add_command_job(lines: list[str], name: str, display: str, compute: str, inputs: dict[str, str], outputs: dict[str, str], command: str, *, env: str | None = None, parquet_env: bool = False, gpu: bool = False, instance_count: str | None = None, identity: bool = False) -> None:
    emit(lines, f"  {name}:")
    emit(lines, "    type: command")
    emit(lines, f"    display_name: {display}")
    emit(lines, f"    compute: {compute}")
    if gpu:
        emit(lines, "    resources:")
        emit(lines, f"      instance_type: {GPU_INSTANCE_TYPE}")
        emit(lines, f"      instance_count: {instance_count or 1}")
        emit(lines, "      properties:")
        emit(lines, "        s_vc: ads-relevance")
        emit(lines, "        sla_tier: Premium")
        emit(lines, "    environment: azureml:lrm-gpu-env:1")
    elif env:
        emit(lines, f"    environment: {env}")
    else:
        cpu_env(lines, parquet=parquet_env)
    if identity:
        emit(lines, "    identity:")
        emit(lines, "      type: user_identity")
    emit(lines, "    code: ..")
    if inputs:
        emit(lines, "    inputs:")
        for key, value in inputs.items():
            emit(lines, f"      {key}: {value}")
    if outputs:
        emit(lines, "    outputs:")
        for key, value in outputs.items():
            emit(lines, f"      {key}: {value}")
    command_block(lines, command.replace("{gpu_guard()}", gpu_guard()))
    emit(lines)


def gpu_guard() -> str:
    return "if [ \"${{inputs.gpu_instance_count}}\" != \"1\" ]; then echo 'gpu_instance_count > 1 is gated: current torchrun commands are single-node only' >&2; exit 64; fi &&"


def validate_args(args: argparse.Namespace) -> None:
    if args.cpu_shards < 1:
        raise SystemExit("--cpu-shards must be >= 1")
    if args.num_buckets < 1:
        raise SystemExit("--num-buckets must be >= 1")
    if args.gpu_instance_count != 1 and not args.allow_multinode_gpu:
        raise SystemExit("gpu_instance_count > 1 requires --allow-multinode-gpu because train/eval still use single-node torchrun --standalone/--nproc_per_node commands")


def build_yaml(args: argparse.Namespace) -> str:
    root = args.output_root.rstrip("/")
    lines: list[str] = []
    emit(lines, "$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json")
    emit(lines, "type: pipeline")
    emit(lines, "display_name: lrm-l800-pconv-fullgraph-10x-v2")
    emit(lines, "experiment_name: lrm-l800-pconv-fullgraph-10x")
    emit(lines, "description: >-")
    emit(lines, "  Generated 10x_v2 pconv/fullgraph pipeline with explicit CPU fan-out, fan-in")
    emit(lines, "  readiness checks, and gated GPU instance_count parameterization.")
    emit(lines)
    emit(lines, "settings:")
    emit(lines, f"  default_compute: {args.cpu_compute}")
    emit(lines, "  default_datastore: azureml:workspaceblobstore")
    emit(lines, "  continue_on_step_failure: false")
    emit(lines)
    emit(lines, "inputs:")
    emit(lines, f"  source_root: {args.source_root}")
    emit(lines, f"  output_root: {root}")
    emit(lines, f"  data_version: {args.data_version}")
    emit(lines, "  layout_version: layout_v1")
    emit(lines, f"  cpu_compute: {args.cpu_compute}")
    emit(lines, f"  cpu_shards: {args.cpu_shards}")
    emit(lines, f"  num_buckets: {args.num_buckets}")
    emit(lines, f"  gpu_instance_count: {args.gpu_instance_count}")
    emit(lines, f"  num_epochs: {args.epochs}")
    emit(lines, f"  eval_batches: {args.eval_batches}")
    emit(lines)
    emit(lines, "outputs:")
    for name in ["discovered", "raw", "vocab_spill", "vocab_reduced", "vocab", "seqview", "seqview_metadata", "embeddings", "train_output", "eval_output"]:
        add_output(lines, name, "${{parent.inputs.output_root}}", name)
    emit(lines)
    emit(lines, "jobs:")
    add_command_job(lines, "discover_raw_shards", "discover-raw-shards-10x-v2", args.cpu_compute, {
        "source_root": "${{parent.inputs.source_root}}",
        "data_version": "${{parent.inputs.data_version}}",
    }, {"discovered": "${{parent.outputs.discovered}}"}, """
python aml_dataprep/parallel/discover_raw_shards.py
--source_root ${{inputs.source_root}}
--data_version ${{inputs.data_version}}
--output_dir ${{outputs.discovered}}
""", env="azureml:lrm-relay-env:2", identity=True)

    relay_outputs = []
    for shard in range(args.cpu_shards):
        name = f"relay_shard_{shard:04d}"
        relay_outputs.append(f"${{{{parent.jobs.{name}.outputs.raw}}}}")
        add_command_job(lines, name, f"relay-shard-{shard:04d}", args.cpu_compute, {
            "discovered": "${{parent.jobs.discover_raw_shards.outputs.discovered}}",
        }, {"raw": output_value()}, f"""
python aml_dataprep/parallel/relay_partition.py
--manifest ${{{{inputs.discovered}}}}/raw_source_manifest.jsonl
--shard_index {shard}
--num_shards {args.cpu_shards}
--output_dir ${{{{outputs.raw}}}}
""", env="azureml:lrm-relay-env:2", identity=True)

    merge_inputs = {f"raw_{i:04d}": value for i, value in enumerate(relay_outputs)}
    merge_cmd_inputs = " ".join(f"${{{{inputs.raw_{i:04d}}}}}" for i in range(args.cpu_shards))
    add_command_job(lines, "merge_relay_raw", "merge-relay-raw", args.cpu_compute, merge_inputs, {"raw": "${{parent.outputs.raw}}"}, f"""
python aml_dataprep/parallel/merge_partition_dirs.py
--input_dirs {merge_cmd_inputs}
--output_dir ${{{{outputs.raw}}}}
--stage relay
--expected_count {args.cpu_shards}
""")

    spill_outputs = []
    for shard in range(args.cpu_shards):
        name = f"vocab_spill_shard_{shard:04d}"
        spill_outputs.append(f"${{{{parent.jobs.{name}.outputs.spill}}}}")
        add_command_job(lines, name, f"vocab-spill-shard-{shard:04d}", args.cpu_compute, {
            "discovered": "${{parent.jobs.discover_raw_shards.outputs.discovered}}",
            "raw": "${{parent.jobs.merge_relay_raw.outputs.raw}}",
        }, {"spill": output_value()}, f"""
python aml_dataprep/parallel/vocab_spill_partition.py
--raw_manifest ${{{{inputs.discovered}}}}/raw_source_manifest.jsonl
--raw_root ${{{{inputs.raw}}}}
--shard_index {shard}
--num_shards {args.cpu_shards}
--output_dir ${{{{outputs.spill}}}}
--num_buckets {args.num_buckets}
""")

    spill_merge_inputs = {f"spill_{i:04d}": value for i, value in enumerate(spill_outputs)}
    spill_cmd_inputs = " ".join(f"${{{{inputs.spill_{i:04d}}}}}" for i in range(args.cpu_shards))
    add_command_job(lines, "merge_vocab_spill", "merge-vocab-spill", args.cpu_compute, spill_merge_inputs, {"spill": "${{parent.outputs.vocab_spill}}"}, f"""
python aml_dataprep/parallel/merge_partition_dirs.py
--input_dirs {spill_cmd_inputs}
--output_dir ${{{{outputs.spill}}}}
--stage vocab_spill
--expected_count {args.cpu_shards}
""")

    reduce_outputs = []
    for domain in range(5):
        for bucket in range(args.num_buckets):
            name = f"vocab_reduce_d{domain}_b{bucket:04d}"
            reduce_outputs.append((name, f"${{{{parent.jobs.{name}.outputs.reduced}}}}"))
            add_command_job(lines, name, f"vocab-reduce-d{domain}-b{bucket:04d}", args.cpu_compute, {
                "spill": "${{parent.jobs.merge_vocab_spill.outputs.spill}}",
            }, {"reduced": output_value()}, f"""
python aml_dataprep/parallel/vocab_reduce_bucket.py
--spill_root ${{{{inputs.spill}}}}
--domain {domain}
--bucket {bucket}
--output_dir ${{{{outputs.reduced}}}}
""")

    reduce_inputs = {name: value for name, value in reduce_outputs}
    reduce_cmd_inputs = " ".join(f"${{{{inputs.{name}}}}}" for name, _ in reduce_outputs)
    add_command_job(lines, "merge_vocab_reduced", "merge-vocab-reduced", args.cpu_compute, reduce_inputs, {"reduced": "${{parent.outputs.vocab_reduced}}"}, f"""
python aml_dataprep/parallel/merge_partition_dirs.py
--input_dirs {reduce_cmd_inputs}
--output_dir ${{{{outputs.reduced}}}}
--stage vocab_reduce
--expected_count {len(reduce_outputs)}
""")

    add_command_job(lines, "vocab_prefix_sum", "vocab-prefix-sum", args.cpu_compute, {
        "reduced": "${{parent.jobs.merge_vocab_reduced.outputs.reduced}}",
        "data_version": "${{parent.inputs.data_version}}",
        "layout_version": "${{parent.inputs.layout_version}}",
    }, {"offsets": output_value()}, f"""
python aml_dataprep/parallel/vocab_prefix_sum.py
--reduced_root ${{{{inputs.reduced}}}}
--output_dir ${{{{outputs.offsets}}}}
--num_buckets {args.num_buckets}
--data_version ${{{{inputs.data_version}}}}
--layout_version ${{{{inputs.layout_version}}}}
""")

    add_command_job(lines, "vocab_finalize", "vocab-finalize-all-buckets", args.cpu_compute, {
        "reduced": "${{parent.jobs.merge_vocab_reduced.outputs.reduced}}",
        "offsets": "${{parent.jobs.vocab_prefix_sum.outputs.offsets}}",
    }, {"vocab": "${{parent.outputs.vocab}}"}, f"""
python aml_dataprep/parallel/vocab_finalize_all_buckets.py
--reduced_root ${{{{inputs.reduced}}}}
--offsets_root ${{{{inputs.offsets}}}}
--vocab_root ${{{{outputs.vocab}}}}
--num_buckets {args.num_buckets}
""")

    parquet_outputs = []
    for shard in range(args.cpu_shards):
        name = f"parquet_shard_{shard:04d}"
        parquet_outputs.append(f"${{{{parent.jobs.{name}.outputs.seqview}}}}")
        add_command_job(lines, name, f"parquet-shard-{shard:04d}", args.cpu_compute, {
            "discovered": "${{parent.jobs.discover_raw_shards.outputs.discovered}}",
            "raw": "${{parent.jobs.merge_relay_raw.outputs.raw}}",
            "vocab": "${{parent.jobs.vocab_finalize.outputs.vocab}}",
        }, {"seqview": output_value()}, f"""
python aml_dataprep/parallel/parquet_partition.py
--raw_manifest ${{{{inputs.discovered}}}}/raw_source_manifest.jsonl
--raw_root ${{{{inputs.raw}}}}
--vocab_dir ${{{{inputs.vocab}}}}
--shard_index {shard}
--num_shards {args.cpu_shards}
--output_dir ${{{{outputs.seqview}}}}
--mode all_events
""", parquet_env=True)

    parquet_inputs = {f"seqview_{i:04d}": value for i, value in enumerate(parquet_outputs)}
    parquet_cmd_inputs = " ".join(f"${{{{inputs.seqview_{i:04d}}}}}" for i in range(args.cpu_shards))
    add_command_job(lines, "merge_seqview", "merge-seqview", args.cpu_compute, parquet_inputs, {"seqview": "${{parent.outputs.seqview}}"}, f"""
python aml_dataprep/parallel/merge_partition_dirs.py
--input_dirs {parquet_cmd_inputs}
--output_dir ${{{{outputs.seqview}}}}
--stage parquet
--expected_count {args.cpu_shards}
""")

    add_command_job(lines, "check_parquet_ready", "check-parquet-ready", args.cpu_compute, {
        "seqview": "${{parent.jobs.merge_seqview.outputs.seqview}}",
    }, {"ready": output_value()}, f"""
python aml_dataprep/parallel/check_ready_manifests.py
--ready_dir ${{{{inputs.seqview}}}}/_ready
--pattern parquet_merged.json
--num_shards 1
--output_dir ${{{{outputs.ready}}}}
--stage parquet_fanin
""")

    add_command_job(lines, "aggregate_seqview_manifest", "aggregate-seqview-manifest", args.cpu_compute, {
        "seqview": "${{parent.jobs.merge_seqview.outputs.seqview}}",
        "vocab": "${{parent.jobs.vocab_finalize.outputs.vocab}}",
        "parquet_ready": "${{parent.jobs.check_parquet_ready.outputs.ready}}",
    }, {"metadata": "${{parent.outputs.seqview_metadata}}"}, """
python aml_dataprep/parallel/aggregate_seqview_manifest.py
--seqview_dir ${{inputs.seqview}}
--vocab_dir ${{inputs.vocab}}
--output_dir ${{outputs.metadata}}
--mode all_events
""")

    add_command_job(lines, "encode_embeddings", "encode-embeddings-10x-v2", GPU_COMPUTE, {
        "vocab": "${{parent.jobs.vocab_finalize.outputs.vocab}}",
        "metadata": "${{parent.jobs.aggregate_seqview_manifest.outputs.metadata}}",
        "gpu_instance_count": "${{parent.inputs.gpu_instance_count}}",
    }, {"embeddings": "${{parent.outputs.embeddings}}"}, """
set -e &&
{gpu_guard()}
pip install -q "sentence-transformers==5.4.1" &&
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader &&
python aml_dataprep/encode_embeddings_multigpu_v3.py
--vocab_dir ${{inputs.vocab}}
--output_dir ${{outputs.embeddings}}
--domains 0,1,2,3,4
--batch_size 1024
--model_name sentence-transformers/all-MiniLM-L6-v2
--emb_dim 384
""", gpu=True, instance_count="${{parent.inputs.gpu_instance_count}}")

    add_command_job(lines, "train", "train-10x-v2", GPU_COMPUTE, {
        "seqview": "${{parent.jobs.merge_seqview.outputs.seqview}}",
        "metadata": "${{parent.jobs.aggregate_seqview_manifest.outputs.metadata}}",
        "embeddings": "${{parent.jobs.encode_embeddings.outputs.embeddings}}",
        "num_epochs": "${{parent.inputs.num_epochs}}",
        "gpu_instance_count": "${{parent.inputs.gpu_instance_count}}",
    }, {"model": "${{parent.outputs.train_output}}"}, """
set -e &&
{gpu_guard()}
(pip install -q -r proposed_2-mmoe_ple/requirements.txt 2>/dev/null || true) &&
cp proposed_2-mmoe_ple/config/generated_m4_seq_len_50pct/m4_aux_light_hstu_50pct_l800.gin /tmp/train_10x_v2.gin &&
sed -i "s/Trainer.num_epochs = [0-9][0-9]*/Trainer.num_epochs = ${{inputs.num_epochs}}/" /tmp/train_10x_v2.gin &&
cd proposed_2-mmoe_ple/train &&
torchrun --nproc_per_node=8 main.py
--gin_config_file=/tmp/train_10x_v2.gin
--data_path=${{inputs.seqview}}/train
--ads_semantic_embd_path=${{inputs.embeddings}}/domain_0
--web_browsing_semantic_embd_path=${{inputs.embeddings}}/domain_1
--shopping_semantic_embd_path=${{inputs.embeddings}}/domain_2
--ads_pure_corpus_embd_path=${{inputs.embeddings}}/domain_3
--other_semantic_embd_path=${{inputs.embeddings}}/domain_4
--output_path=${{outputs.model}}
--mode=job
--run_id=pconv_fullgraph_10x_v2
""", gpu=True, instance_count="${{parent.inputs.gpu_instance_count}}")

    add_command_job(lines, "evaluate", "eval-10x-v2", GPU_COMPUTE, {
        "seqview": "${{parent.jobs.merge_seqview.outputs.seqview}}",
        "metadata": "${{parent.jobs.aggregate_seqview_manifest.outputs.metadata}}",
        "embeddings": "${{parent.jobs.encode_embeddings.outputs.embeddings}}",
        "model": "${{parent.jobs.train.outputs.model}}",
        "eval_batches": "${{parent.inputs.eval_batches}}",
        "gpu_instance_count": "${{parent.inputs.gpu_instance_count}}",
    }, {"eval": "${{parent.outputs.eval_output}}"}, """
set -e &&
{gpu_guard()}
mkdir -p ${{outputs.eval}} &&
(pip install -q -r proposed_2-mmoe_ple/requirements.txt 2>/dev/null || true) &&
CKPT=$(find ${{inputs.model}} -path "*ckpts*" -name "best_checkpoint_*.pt" -print | sort | tail -1) &&
if [ -z "$CKPT" ]; then CKPT=$(find ${{inputs.model}} -path "*ckpts*" -name "checkpoint_batch*.pt" -print | sort | tail -1); fi &&
if [ -z "$CKPT" ]; then echo "No checkpoint found under ${{inputs.model}}" >&2; find ${{inputs.model}} -maxdepth 5 -type f >&2; exit 1; fi &&
echo "$CKPT" > ${{outputs.eval}}/checkpoint_path.txt &&
torchrun --standalone --nproc_per_node=1 eval/eval_per_event_type.py
--gin_config_file=proposed_2-mmoe_ple/config/generated_m4_seq_len_50pct/m4_aux_light_hstu_50pct_l800.gin
--data_path=${{inputs.seqview}}/eval
--ckpt_path="$CKPT"
--ads_semantic_embd_path=${{inputs.embeddings}}/domain_0
--web_browsing_semantic_embd_path=${{inputs.embeddings}}/domain_1
--shopping_semantic_embd_path=${{inputs.embeddings}}/domain_2
--ads_pure_corpus_embd_path=${{inputs.embeddings}}/domain_3
--other_semantic_embd_path=${{inputs.embeddings}}/domain_4
--eval_batches=${{inputs.eval_batches}}
--output_json=${{outputs.eval}}/eval_per_event_type.json
--mode=job
""", gpu=True, instance_count="${{parent.inputs.gpu_instance_count}}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--source-root", default="azureml://subscriptions/72a0fe10-0a76-4898-9b7b-640e6e236fdc/resourcegroups/wb-aml/workspaces/pconv-aml-offline/datastores/bingads_algo_pipelines_c08/paths/local/User/wenhlu/LRM_benchmark_v4_10x")
    p.add_argument("--data-version", default="v3-20260707-pconv-fullgraph-10x-v2")
    p.add_argument("--output-root", default="azureml://datastores/workspaceblobstore/paths/derived/lrm_v4_pconv_v3/full_graph_10x_v2")
    p.add_argument("--cpu-compute", default="azureml:CPU-D2ADSV4")
    p.add_argument("--cpu-shards", type=int, default=10)
    p.add_argument("--num-buckets", type=int, default=4096)
    p.add_argument("--gpu-instance-count", type=int, default=1)
    p.add_argument("--allow-multinode-gpu", action="store_true")
    p.add_argument("--eval-batches", type=int, default=100)
    p.add_argument("--epochs", type=int, default=3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    validate_args(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_yaml(args), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
