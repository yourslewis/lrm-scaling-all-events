#!/usr/bin/env python3
"""Run a tiny real subset through the Phase 1 parallel-prep worker scripts.

This is intentionally a smoke-test bridge for the static AML pipeline scaffold:
it executes a few deterministic shard/bucket units inside each boundary job so
AML outputs are non-empty and validate IO/contracts before generated fan-out is
used for the full run.
"""

# Workflow notes:
# The smoke runner exercises the same worker scripts as the real fan-out plan,
# but on a tiny deterministic subset. It validates AML mounts, manifests, vocab,
# parquet, and metadata contracts before spending quota on the full job matrix.
# Performance tricks:
# - Limit shard/bucket counts while preserving the full step ordering.
# - Reuse local temporary roots so smoke checks do not create large blob outputs.

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.vocab_common_v3 import NUM_DOMAINS
from aml_dataprep.parallel.relay_one_shard import _open_source


def _script(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def _rows(manifest):
    with open(manifest, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _selected(manifest, max_train, max_val):
    limits = {"train": max_train, "val": max_val}
    seen = {"train": 0, "val": 0}
    out = []
    for row in _rows(manifest):
        split = row["split"]
        if split in limits and seen[split] < limits[split]:
            out.append(row)
            seen[split] += 1
    if not out:
        raise ValueError(f"no rows selected from {manifest}")
    return out


def _run(cmd):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)))


def _relay_sample(args):
    total_bytes = 0
    for row in _selected(args.manifest, args.max_train_shards, args.max_val_shards):
        dst = os.path.join(args.raw_out, row["dest_relpath"])
        tmp = dst + ".tmp"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        lines = 0
        with _open_source(row["source_uri"]) as src, open(tmp, "wb") as out:
            # Keep header plus a small deterministic prefix of data rows.
            for raw_line in src:
                out.write(raw_line)
                total_bytes += len(raw_line)
                lines += 1
                if lines >= args.max_lines + 1:
                    break
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, dst)
        with open(dst + ".done", "w", encoding="utf-8") as f:
            json.dump({"source_uri": row["source_uri"], "sample_lines": max(0, lines - 1)}, f, sort_keys=True)
        print(f"sample relayed {row['split']}:{row['shard_index']} lines={max(0, lines-1)} -> {dst}", flush=True)
    os.makedirs(args.raw_out, exist_ok=True)
    with open(os.path.join(args.raw_out, "_SMOKE_SUCCESS"), "w", encoding="utf-8") as f:
        f.write(f"bytes={total_bytes}\n")


def _spill(args):
    for row in _selected(args.manifest, args.max_train_shards, args.max_val_shards):
        _run([sys.executable, _script("vocab_spill_one_shard.py"),
              "--raw_manifest", args.manifest,
              "--raw_root", args.raw_root,
              "--split", row["split"],
              "--shard_index", row["shard_index"],
              "--output_dir", args.spill_out,
              "--num_buckets", args.num_buckets])


def _buckets_with_spill(spill_root, num_buckets):
    buckets = []
    for domain in range(NUM_DOMAINS):
        for bucket in range(num_buckets):
            pattern = os.path.join(spill_root, f"domain_{domain}", f"bucket_{bucket:04d}", "part_*.txt")
            if glob.glob(pattern):
                buckets.append((domain, bucket))
    return buckets


def _reduce(args):
    buckets = _buckets_with_spill(args.spill_root, args.num_buckets)
    print(f"reducing {len(buckets)} non-empty buckets", flush=True)
    for domain, bucket in buckets:
        _run([sys.executable, _script("vocab_reduce_bucket.py"),
              "--spill_root", args.spill_root,
              "--domain", domain,
              "--bucket", bucket,
              "--output_dir", args.reduced_out])
    os.makedirs(args.reduced_out, exist_ok=True)
    with open(os.path.join(args.reduced_out, "_SMOKE_BUCKETS.json"), "w", encoding="utf-8") as f:
        json.dump([{"domain": d, "bucket": b} for d, b in buckets], f, sort_keys=True)


def _copy_offsets_to_vocab(offsets_root, vocab_root):
    if not offsets_root:
        return
    os.makedirs(vocab_root, exist_ok=True)
    for name in ("vocab_offsets.json", "vocab_meta.json"):
        src = os.path.join(offsets_root, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(vocab_root, name))
    for entry in os.listdir(offsets_root):
        if entry.startswith("domain_") and os.path.isdir(os.path.join(offsets_root, entry)):
            dst = os.path.join(vocab_root, entry)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(os.path.join(offsets_root, entry), dst)


def _finalize(args):
    _copy_offsets_to_vocab(args.offsets_root, args.vocab_root)
    buckets = []
    meta_path = os.path.join(args.reduced_root, "_SMOKE_BUCKETS.json")
    if os.path.exists(meta_path):
        buckets = [(int(x["domain"]), int(x["bucket"])) for x in json.load(open(meta_path, encoding="utf-8"))]
    else:
        buckets = _buckets_with_spill(args.reduced_root, args.num_buckets)
    print(f"finalizing {len(buckets)} non-empty buckets", flush=True)
    for domain, bucket in buckets:
        _run([sys.executable, _script("vocab_finalize_bucket.py"),
              "--reduced_root", args.reduced_root,
              "--vocab_root", args.vocab_root,
              "--domain", domain,
              "--bucket", bucket])
    os.makedirs(args.markers_out, exist_ok=True)
    with open(os.path.join(args.markers_out, "_SMOKE_FINALIZED.json"), "w", encoding="utf-8") as f:
        json.dump({"num_buckets_finalized": len(buckets)}, f, sort_keys=True)


def _parquet(args):
    for row in _selected(args.manifest, args.max_train_shards, args.max_val_shards):
        _run([sys.executable, _script("parquet_one_shard.py"),
              "--raw_manifest", args.manifest,
              "--raw_root", args.raw_root,
              "--vocab_dir", args.vocab_dir,
              "--split", row["split"],
              "--shard_index", row["shard_index"],
              "--output_dir", args.seqview_out,
              "--mode", args.mode])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["relay", "spill", "reduce", "finalize", "parquet"])
    p.add_argument("--manifest")
    p.add_argument("--raw_root")
    p.add_argument("--raw_out")
    p.add_argument("--spill_root")
    p.add_argument("--spill_out")
    p.add_argument("--reduced_root")
    p.add_argument("--reduced_out")
    p.add_argument("--vocab_root")
    p.add_argument("--vocab_dir")
    p.add_argument("--offsets_root")
    p.add_argument("--markers_out")
    p.add_argument("--seqview_out")
    p.add_argument("--num_buckets", type=int, default=64)
    p.add_argument("--max_train_shards", type=int, default=1)
    p.add_argument("--max_val_shards", type=int, default=1)
    p.add_argument("--max_lines", type=int, default=1000)
    p.add_argument("--mode", choices=["ads_only", "all_events"], default="all_events")
    args = p.parse_args()

    if args.stage == "relay":
        _relay_sample(args)
    elif args.stage == "spill":
        _spill(args)
    elif args.stage == "reduce":
        _reduce(args)
    elif args.stage == "finalize":
        _finalize(args)
    elif args.stage == "parquet":
        _parquet(args)


if __name__ == "__main__":
    main()
