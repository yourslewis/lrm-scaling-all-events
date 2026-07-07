#!/usr/bin/env python3
"""Merge unique fan-out output directories into one downstream layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aml_dataprep.parallel.partition import write_ready_manifest


def _copy_tree_contents(src: Path, dst: Path) -> int:
    copied = 0
    for root, _, files in os.walk(src):
        root_path = Path(root)
        rel_root = root_path.relative_to(src)
        for name in files:
            src_file = root_path / name
            dst_file = dst / rel_root / name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_file.exists():
                if src_file.read_bytes() == dst_file.read_bytes():
                    continue
                raise FileExistsError(f"merge collision: {dst_file}")
            shutil.copy2(src_file, dst_file)
            copied += 1
    return copied


def _load_ready_items(src: Path, expected_stage: str | None) -> list[dict]:
    ready = sorted((src / "_ready").glob("*.json"))
    if not ready:
        raise FileNotFoundError(f"missing ready manifest under {src / '_ready'}")
    items = []
    for path in ready:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if expected_stage and payload.get("stage") != expected_stage:
            raise ValueError(f"ready stage mismatch in {path}: expected {expected_stage}, got {payload.get('stage')}")
        items.append({"path": str(path), "payload": payload})
    return items


def _validate_shards(items: list[dict], expected_num_shards: int) -> None:
    shard_indexes = []
    for item in items:
        payload = item["payload"]
        path = item["path"]
        if "shard_index" not in payload:
            raise ValueError(f"missing shard_index in ready manifest: {path}")
        if int(payload.get("num_shards", expected_num_shards)) != expected_num_shards:
            raise ValueError(f"ready num_shards mismatch in {path}: expected {expected_num_shards}, got {payload.get('num_shards')}")
        shard_indexes.append(int(payload["shard_index"]))
    duplicates = sorted({idx for idx in shard_indexes if shard_indexes.count(idx) > 1})
    if duplicates:
        raise ValueError(f"duplicate ready shard_index values: {duplicates}")
    expected = set(range(expected_num_shards))
    actual = set(shard_indexes)
    if actual != expected:
        raise ValueError(f"ready shard coverage mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--expected_count", type=int, required=True)
    p.add_argument("--expected_stage")
    p.add_argument("--expect_shards", action="store_true")
    p.add_argument("--expected_num_shards", type=int)
    args = p.parse_args()

    if len(args.input_dirs) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} input dirs, got {len(args.input_dirs)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    ready_items = []
    for raw in args.input_dirs:
        src = Path(raw)
        if not src.exists():
            raise FileNotFoundError(f"missing input dir: {src}")
        ready_items.extend(_load_ready_items(src, args.expected_stage))
        copied_files += _copy_tree_contents(src, output_dir)
    if args.expect_shards:
        _validate_shards(ready_items, args.expected_num_shards or args.expected_count)
    ready_path = write_ready_manifest(output_dir / "_ready", f"{args.stage}_merged.json", {
        "stage": args.stage,
        "expected_count": args.expected_count,
        "input_dirs": args.input_dirs,
        "ready_files": [item["path"] for item in ready_items],
        "copied_files": copied_files,
    })
    print(f"merged {len(args.input_dirs)} dirs for {args.stage}; files={copied_files}; ready={ready_path}", flush=True)


if __name__ == "__main__":
    main()
