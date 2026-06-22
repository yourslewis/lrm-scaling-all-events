#!/usr/bin/env python3
"""
Relay step (pipeline step 0): stream raw TSV shards from cosmos (ADLS Gen1) to
the pipeline's blob output. Runs on a cosmos-capable compute (the compute
instance / a wbsrvMSI CPU compute) because Singularity VC nodes cannot read the
cosmos datastore. Streams one shard at a time so it fits on a small disk.

A `--data_version` arg is part of the step's cache key: bump it to force the
relay (and only the relay) to re-run when the cosmos data changed. When it is
unchanged AML reuses this step's prior output.

Output layout (under --output_dir):
    train/train_chunk_*.tsv
    val/val_chunk_*.tsv
"""
import argparse
import os
import time

from azureml.fsspec import AzureMachineLearningFileSystem


def relay(fs, rel_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in fs.ls(rel_dir) if f.endswith(".tsv"))
    print(f"[{rel_dir}] {len(files)} shards to relay", flush=True)
    total = 0
    for i, f in enumerate(files):
        name = f.split("/")[-1]
        dst = os.path.join(out_dir, name)
        relpath = f.split("/paths/")[-1] if "/paths/" in f else f
        t0 = time.time()
        with fs.open(relpath, "rb") as src, open(dst, "wb") as out:
            while True:
                chunk = src.read(16 << 20)
                if not chunk:
                    break
                out.write(chunk)
        sz = os.path.getsize(dst)
        total += sz
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  {i+1}/{len(files)} {name} {sz/1e9:.2f}GB "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[{rel_dir}] done, {total/1e9:.1f} GB total", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_version", required=True,
                   help="Cache-busting token; bump to force a re-relay.")
    p.add_argument("--cosmos_sub", default="72a0fe10-0a76-4898-9b7b-640e6e236fdc")
    p.add_argument("--cosmos_rg", default="wb-aml")
    p.add_argument("--cosmos_ws", default="pconv-aml-offline")
    p.add_argument("--cosmos_datastore", default="bingads_algo_pipelines_c08")
    p.add_argument("--cosmos_root",
                   default="local/User/wenhlu/LRM_benchmark_v4")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    print(f"=== relay (data_version={args.data_version}) ===", flush=True)
    base = (f"azureml://subscriptions/{args.cosmos_sub}/resourcegroups/"
            f"{args.cosmos_rg}/workspaces/{args.cosmos_ws}/datastores/"
            f"{args.cosmos_datastore}/paths/{args.cosmos_root}")
    fs = AzureMachineLearningFileSystem(base)
    relay(fs, f"{args.cosmos_root}/train", os.path.join(args.output_dir, "train"))
    relay(fs, f"{args.cosmos_root}/val", os.path.join(args.output_dir, "val"))
    # marker so downstream + humans can see which version produced this
    with open(os.path.join(args.output_dir, "_data_version.txt"), "w") as f:
        f.write(args.data_version + "\n")
    print("=== relay complete ===", flush=True)


if __name__ == "__main__":
    main()
