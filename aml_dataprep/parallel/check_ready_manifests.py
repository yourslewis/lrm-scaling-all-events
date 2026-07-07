#!/usr/bin/env python3
"""Validate ready manifests before allowing downstream AML fan-in steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aml_dataprep.parallel.partition import write_ready_manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ready_dir", required=True)
    p.add_argument("--pattern", required=True, help="Python format string using shard_index, e.g. stage_shard_{shard_index:04d}.json")
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stage", required=True)
    args = p.parse_args()

    ready_dir = Path(args.ready_dir)
    seen = []
    for shard_index in range(args.num_shards):
        path = ready_dir / args.pattern.format(shard_index=shard_index)
        if not path.exists():
            print(f"missing ready manifest: {path}", file=sys.stderr)
            return 2
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if int(payload.get("shard_index", shard_index)) != shard_index:
            print(f"ready manifest shard mismatch in {path}", file=sys.stderr)
            return 3
        if int(payload.get("num_shards", args.num_shards)) != args.num_shards:
            print(f"ready manifest num_shards mismatch in {path}", file=sys.stderr)
            return 4
        seen.append(str(path))
    out = write_ready_manifest(args.output_dir, f"{args.stage}_ready.json", {
        "stage": args.stage,
        "num_shards": args.num_shards,
        "ready_manifests": seen,
    })
    print(f"validated {len(seen)} ready manifests for {args.stage}; ready={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
