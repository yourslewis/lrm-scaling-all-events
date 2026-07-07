#!/usr/bin/env python3
"""Run vocab spill for a deterministic partition of raw manifest rows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aml_dataprep.parallel.partition import add_partition_args, load_manifest_rows, partition_rows, write_ready_manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_manifest", required=True)
    p.add_argument("--raw_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_buckets", type=int, default=4096)
    p.add_argument("--max_open_handles", type=int, default=128)
    add_partition_args(p)
    args = p.parse_args()

    script = Path(__file__).resolve().parent / "vocab_spill_one_shard.py"
    selected = list(partition_rows(load_manifest_rows(args.raw_manifest), args.shard_index, args.num_shards))
    for _, row in selected:
        subprocess.check_call([
            sys.executable,
            str(script),
            "--raw_manifest",
            args.raw_manifest,
            "--raw_root",
            args.raw_root,
            "--split",
            row["split"],
            "--shard_index",
            str(row["shard_index"]),
            "--output_dir",
            args.output_dir,
            "--num_buckets",
            str(args.num_buckets),
            "--max_open_handles",
            str(args.max_open_handles),
        ])
    ready_path = write_ready_manifest(Path(args.output_dir) / "_ready", f"vocab_spill_shard_{args.shard_index:04d}.json", {
        "stage": "vocab_spill",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_ordinals": [ordinal for ordinal, _ in selected],
        "selected": [f"{row['split']}:{row['shard_index']}" for _, row in selected],
        "count": len(selected),
    })
    print(f"vocab spill partition {args.shard_index}/{args.num_shards}: {len(selected)} rows; ready={ready_path}", flush=True)


if __name__ == "__main__":
    main()
