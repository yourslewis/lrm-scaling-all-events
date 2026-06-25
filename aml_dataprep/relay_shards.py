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

# Workflow notes:
# Monolithic relay used by the non-generated CPU-prep pipeline: copy all raw TSV
# shards from cosmos/ADLS into the workspace blob output before vocab/parquet.
# Performance tricks:
# - Stream shard-by-shard to fit small CI disks and avoid holding TSVs in memory.
# - Keep --data_version in the command line so AML cache invalidation is explicit.

import argparse
import gc
import os
import resource
import subprocess
import sys
import time

from azureml.fsspec import AzureMachineLearningFileSystem


def _rss_gb():
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def copy_one(base, src_relpath, dst):
    """Copy one shard in this process.

    AzureML/fsspec and/or output mounts can retain memory across opened files.
    The top-level relay launches this function in a fresh subprocess for each
    shard, keeping peak memory bounded to a single file copy.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fs = AzureMachineLearningFileSystem(base)
    t0 = time.time()
    total = 0
    with fs.open(src_relpath, "rb") as src, open(dst, "wb") as out:
        while True:
            # Keep chunks deliberately small. Large chunks plus SDK/mount
            # buffering caused SIGKILL/OOM on single-node AML CPU relays.
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
        out.flush()
        os.fsync(out.fileno())
    print(f"copied {os.path.basename(dst)} {total/1e9:.2f}GB "
          f"in {time.time()-t0:.0f}s rss_max={_rss_gb():.2f}GB",
          flush=True)


def relay(base, fs, rel_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in fs.ls(rel_dir) if f.endswith(".tsv"))
    print(f"[{rel_dir}] {len(files)} shards to relay", flush=True)
    total = 0
    for i, f in enumerate(files):
        name = f.split("/")[-1]
        dst = os.path.join(out_dir, name)
        relpath = f.split("/paths/")[-1] if "/paths/" in f else f
        t0 = time.time()
        cmd = [sys.executable, __file__, "--copy_one",
               "--data_version", "copy-one",
               "--cosmos_sub", "unused",
               "--cosmos_rg", "unused",
               "--cosmos_ws", "unused",
               "--cosmos_datastore", "unused",
               "--cosmos_root", "unused",
               "--base", base,
               "--src_relpath", relpath,
               "--dst", dst]
        subprocess.run(cmd, check=True)
        sz = os.path.getsize(dst)
        total += sz
        gc.collect()
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  {i+1}/{len(files)} {name} {sz/1e9:.2f}GB "
                  f"({time.time()-t0:.0f}s) parent_rss_max={_rss_gb():.2f}GB",
                  flush=True)
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
    p.add_argument("--output_dir")
    p.add_argument("--copy_one", action="store_true")
    p.add_argument("--base")
    p.add_argument("--src_relpath")
    p.add_argument("--dst")
    args = p.parse_args()

    base = (f"azureml://subscriptions/{args.cosmos_sub}/resourcegroups/"
            f"{args.cosmos_rg}/workspaces/{args.cosmos_ws}/datastores/"
            f"{args.cosmos_datastore}/paths/{args.cosmos_root}")
    if args.copy_one:
        copy_one(args.base, args.src_relpath, args.dst)
        return

    if not args.output_dir:
        raise ValueError("--output_dir is required unless --copy_one is set")

    print(f"=== relay (data_version={args.data_version}) ===", flush=True)
    fs = AzureMachineLearningFileSystem(base)
    relay(base, fs, f"{args.cosmos_root}/train", os.path.join(args.output_dir, "train"))
    relay(base, fs, f"{args.cosmos_root}/val", os.path.join(args.output_dir, "val"))
    # marker so downstream + humans can see which version produced this
    with open(os.path.join(args.output_dir, "_data_version.txt"), "w") as f:
        f.write(args.data_version + "\n")
    print("=== relay complete ===", flush=True)


if __name__ == "__main__":
    main()
