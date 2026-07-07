#!/usr/bin/env python3
"""Relay a deterministic partition of source manifest rows into raw layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aml_dataprep.parallel.partition import add_partition_args, load_manifest_rows, partition_rows, write_ready_manifest
from aml_dataprep.parallel.relay_one_shard import _open_source


def _copy_row(row: dict, output_dir: str, chunk_bytes: int) -> None:
    dst = os.path.join(output_dir, row["dest_relpath"])
    tmp = dst + ".tmp"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with _open_source(row["source_uri"]) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, length=chunk_bytes)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)
    with open(dst + ".done", "w", encoding="utf-8") as f:
        json.dump({"source_uri": row["source_uri"], "dest_relpath": row["dest_relpath"]}, f, sort_keys=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--chunk_bytes", type=int, default=1 << 20)
    p.add_argument("--dry_run", action="store_true")
    add_partition_args(p)
    args = p.parse_args()

    rows = load_manifest_rows(args.manifest)
    selected = list(partition_rows(rows, args.shard_index, args.num_shards))
    for _, row in selected:
        if not args.dry_run:
            _copy_row(row, args.output_dir, args.chunk_bytes)
    ready_path = write_ready_manifest(os.path.join(args.output_dir, "_ready"), f"relay_shard_{args.shard_index:04d}.json", {
        "stage": "relay",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_ordinals": [ordinal for ordinal, _ in selected],
        "selected": [f"{row['split']}:{row['shard_index']}" for _, row in selected],
        "count": len(selected),
        "dry_run": args.dry_run,
    })
    print(f"relay partition {args.shard_index}/{args.num_shards}: {len(selected)} rows; ready={ready_path}", flush=True)


if __name__ == "__main__":
    main()
