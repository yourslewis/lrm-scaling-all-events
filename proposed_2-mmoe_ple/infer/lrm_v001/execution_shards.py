#!/usr/bin/env python3
"""Build sequence-native execution shards for LRM-v001 target rows.

The shard files are execution artifacts only. They reorder target rows so rows
sharing a canonical history reference are contiguous, which lets inference scan
history parts sequentially and reuse one encoded user sequence where possible.
They do not modify official target/candidate manifests or schemas.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

CONTEXT_REF_RE = re.compile(
    r"^canonical_row_array_v001:(?P<split>[^/]+)/(?P<file>[^:]+):source_row_index=(?P<source_row_index>\d+):"
)
SHARD_SCHEMA_VERSION = "lrm_v001_execution_shard_manifest_v001"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_sidecar_rows(paths: Sequence[str | Path], *, columns: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional production dependency
        raise RuntimeError("pyarrow is required for --target-sidecar-glob") from exc
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=65536, columns=list(columns) if columns else None):
            for row in batch.to_pylist():
                yield dict(row)


def parse_context_ref(row: Mapping[str, Any]) -> tuple[str, str, int]:
    ref = str(row.get("context_reader_ref") or "")
    m = CONTEXT_REF_RE.match(ref)
    if not m:
        raise ValueError(f"target {row.get('target_id')} lacks parsable canonical context_reader_ref: {ref!r}")
    return m.group("split"), m.group("file"), int(m.group("source_row_index"))


def target_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    split, part_file, source_row_index = parse_context_ref(row)
    return (split, part_file, source_row_index, str(row.get("target_ts") or ""), str(row.get("target_id") or ""))


def context_group_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    split, part_file, source_row_index = parse_context_ref(row)
    return (split, part_file, source_row_index)


def load_targets(*, target_jsonl: str | None, target_sidecar_glob: str | None, max_targets: int | None = None) -> list[dict[str, Any]]:
    if bool(target_jsonl) == bool(target_sidecar_glob):
        raise ValueError("provide exactly one of target_jsonl or target_sidecar_glob")
    rows: list[dict[str, Any]]
    if target_jsonl:
        rows = read_jsonl(target_jsonl)
    else:
        paths = sorted(glob.glob(str(target_sidecar_glob)))
        if not paths:
            raise FileNotFoundError(f"no target sidecars matched {target_sidecar_glob!r}")
        rows = list(iter_sidecar_rows(paths))
    if max_targets is not None:
        rows = rows[: max(0, int(max_targets))]
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def build_execution_shards(
    rows: Sequence[dict[str, Any]],
    *,
    output_dir: str | Path,
    shard_size: int = 2048,
    shard_prefix: str = "shard",
) -> dict[str, Any]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=target_sort_key)

    groups: list[tuple[tuple[str, str, int], list[dict[str, Any]]]] = []
    for key, group_iter in itertools_groupby(sorted_rows, key=context_group_key):
        groups.append((key, list(group_iter)))

    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    oversized_context_groups = 0
    for _, group_rows in groups:
        if len(group_rows) > shard_size:
            oversized_context_groups += 1
            if current:
                shards.append(current)
                current = []
            for start in range(0, len(group_rows), shard_size):
                shards.append(group_rows[start : start + shard_size])
            continue
        if current and len(current) + len(group_rows) > shard_size:
            shards.append(current)
            current = []
        current.extend(group_rows)
    if current:
        shards.append(current)

    part_counts: collections.Counter[str] = collections.Counter()
    group_counts: collections.Counter[str] = collections.Counter()
    shard_entries: list[dict[str, Any]] = []
    for idx, shard_rows in enumerate(shards):
        rel = f"{shard_prefix}_{idx:05d}.targets.jsonl"
        path = out_dir / rel
        write_jsonl(path, shard_rows)
        shard_group_keys = [context_group_key(row) for row in shard_rows]
        unique_groups = list(dict.fromkeys(shard_group_keys))
        for _, part_file, _ in unique_groups:
            part_counts[part_file] += 1
        for key in unique_groups:
            group_counts[canonical_json(key)] += 1
        entry = {
            "shard_index": idx,
            "path": rel,
            "target_count": len(shard_rows),
            "context_group_count": len(unique_groups),
            "first_sort_key": list(target_sort_key(shard_rows[0])) if shard_rows else None,
            "last_sort_key": list(target_sort_key(shard_rows[-1])) if shard_rows else None,
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
        }
        shard_entries.append(entry)

    split_group_counts = {k: v for k, v in group_counts.items() if v > 1}
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "created_at": utc_now(),
        "target_count": len(sorted_rows),
        "shard_count": len(shard_entries),
        "shard_size": shard_size,
        "sort_key": ["context_reader_ref.split", "context_reader_ref.part_file", "context_reader_ref.source_row_index", "target_ts", "target_id"],
        "grouping_policy": "keep_same_context_reader_ref_contiguous_and_avoid_cross_shard_split_when_group_size_allows",
        "context_group_count": len(groups),
        "oversized_context_groups_split": oversized_context_groups,
        "context_groups_split_across_shards": len(split_group_counts),
        "part_file_shard_touch_count": dict(sorted(part_counts.items())),
        "shards": shard_entries,
        "official_contract_change": False,
    }
    manifest_path = out_dir / "execution_shards.manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_payload_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def itertools_groupby(rows: Sequence[dict[str, Any]], *, key):
    # Tiny local wrapper to keep imports minimal and type-friendly.
    import itertools

    return itertools.groupby(rows, key=key)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LRM-v001 sequence-native execution target shards")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--target-jsonl")
    src.add_argument("--target-sidecar-glob")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--max-targets", type=int)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_targets(target_jsonl=args.target_jsonl, target_sidecar_glob=args.target_sidecar_glob, max_targets=args.max_targets)
    manifest = build_execution_shards(rows, output_dir=args.output_dir, shard_size=args.shard_size)
    print(json.dumps({"manifest_path": manifest["manifest_path"], "target_count": manifest["target_count"], "shard_count": manifest["shard_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
