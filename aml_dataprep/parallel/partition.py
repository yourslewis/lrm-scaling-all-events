#!/usr/bin/env python3
"""Shared deterministic partition helpers for AML fan-out workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Union


def validate_partition(shard_index: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")


def owns_ordinal(ordinal: int, shard_index: int, num_shards: int) -> bool:
    validate_partition(shard_index, num_shards)
    if ordinal < 0:
        raise ValueError(f"ordinal must be non-negative, got {ordinal}")
    return ordinal % num_shards == shard_index


def partition_rows(rows: Iterable[dict], shard_index: int, num_shards: int) -> Iterator[tuple[int, dict]]:
    validate_partition(shard_index, num_shards)
    for ordinal, row in enumerate(rows):
        if ordinal % num_shards == shard_index:
            yield ordinal, row


PathLike = Union[str, os.PathLike]


def load_manifest_rows(manifest: PathLike) -> list[dict]:
    with open(manifest, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_ready_manifest(output_dir: PathLike, name: str, payload: dict) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def add_partition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard_index", required=True, type=int)
    parser.add_argument("--num_shards", required=True, type=int)
