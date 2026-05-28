#!/usr/bin/env python3
"""Build sequence-native execution shards for LRM benchmark v001 targets.

This is an execution artifact builder: it does not modify target rows, candidate
sets, labels, manifests, or the official evaluator contract. It reorders frozen
v001 target rows into shard JSONL files that are friendlier to sequential
inference runners:

* primary grouping: canonical history row from ``context_reader_ref``;
* intra-history order: ``target_ts``, then ``target_id``;
* shard boundaries avoid splitting a history row group unless a single group is
  larger than the requested shard size.

The output shards can be passed directly to
``sequential_submission_infer.py --target-jsonl <shard>``. The manifest records
row-preservation digests and read-back validation results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from typing import Any, Iterable, Mapping, Sequence


BENCHMARK_VERSION = "lrm_benchmark_v001"
MANIFEST_SCHEMA_VERSION = "lrm_v001_execution_shard_manifest_v001"
SHARD_SCHEMA_VERSION = "lrm_v001_target_execution_shard_v001"
_CONTEXT_REF_RE = re.compile(
    r"^canonical_row_array_v001:(?P<split>[^/]+)/(?P<file>[^:]+):source_row_index=(?P<source_row_index>\d+):"
)


class ShardBuildError(RuntimeError):
    """Raised for contract or validation failures."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def parse_context_ref(row: Mapping[str, Any], *, strict: bool = True) -> tuple[str, str, int, str]:
    ref = str(row.get("context_reader_ref") or "")
    m = _CONTEXT_REF_RE.match(ref)
    if not m:
        if strict:
            raise ShardBuildError(
                f"target {row.get('target_id')} lacks parsable canonical context_reader_ref: {ref!r}"
            )
        fallback = ref or f"missing_context_ref:{row.get('target_id')}"
        return "", fallback, -1, fallback
    split = m.group("split")
    part_file = m.group("file")
    source_row_index = int(m.group("source_row_index"))
    group_key = f"{split}/{part_file}/source_row_index={source_row_index}"
    return split, part_file, source_row_index, group_key


def target_sort_key(row: Mapping[str, Any], *, strict_context_ref: bool = True) -> tuple[str, str, int, str, str, str]:
    split, part_file, source_row_index, group_key = parse_context_ref(row, strict=strict_context_ref)
    target_ts = str(row.get("target_ts") or row.get("target_time") or "")
    target_id = str(row.get("target_id") or "")
    if not target_ts:
        raise ShardBuildError(f"target {target_id or row!r} lacks target_ts/target_time")
    if not target_id:
        raise ShardBuildError(f"target row lacks target_id: {row!r}")
    return split, part_file, source_row_index, target_ts, target_id, group_key


def iter_jsonl_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ShardBuildError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ShardBuildError(f"target row at {path}:{line_number} is not a JSON object")
            yield row


def iter_parquet_rows(paths: Sequence[str | Path], *, batch_size: int) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - local smoke tests use JSONL
        raise ShardBuildError("--target-sidecar-glob requires pyarrow") from exc

    for path in paths:
        pf = pq.ParquetFile(path)
        for record_batch in pf.iter_batches(batch_size=batch_size):
            for row in record_batch.to_pylist():
                if not isinstance(row, dict):
                    raise ShardBuildError(f"parquet row from {path} is not a mapping")
                yield dict(row)


def iter_input_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.target_jsonl:
        yield from iter_jsonl_rows(args.target_jsonl)
        return
    paths = sorted(glob.glob(args.target_sidecar_glob))
    if not paths:
        raise ShardBuildError(f"no target sidecars matched {args.target_sidecar_glob!r}")
    yield from iter_parquet_rows(paths, batch_size=args.read_batch_size)


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-200000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS targets;
        DROP TABLE IF EXISTS groups;
        DROP TABLE IF EXISTS written;
        CREATE TABLE targets (
            seq INTEGER PRIMARY KEY,
            target_id TEXT NOT NULL,
            context_reader_ref TEXT NOT NULL,
            group_key TEXT NOT NULL,
            sort_split TEXT NOT NULL,
            sort_file TEXT NOT NULL,
            sort_row INTEGER NOT NULL,
            target_ts TEXT NOT NULL,
            row_json TEXT NOT NULL,
            row_digest TEXT NOT NULL
        );
        """
    )
    conn.commit()


def validate_benchmark_version(row: Mapping[str, Any], expected: str) -> None:
    observed = row.get("benchmark_id") or row.get("benchmark_version")
    if observed is not None and str(observed) != expected:
        raise ShardBuildError(
            f"target {row.get('target_id')} benchmark mismatch: observed={observed!r} expected={expected!r}"
        )


def ingest_targets(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    inserted = 0
    first_target_id = None
    last_target_id = None
    first_context_ref = None
    last_context_ref = None
    batch: list[tuple[Any, ...]] = []

    for row in iter_input_rows(args):
        validate_benchmark_version(row, args.benchmark_version)
        sort_split, sort_file, sort_row, target_ts, target_id, group_key = target_sort_key(
            row, strict_context_ref=not args.allow_unparsed_context_ref
        )
        row_json = canonical_json(row)
        row_digest = sha256_text(row_json)
        context_ref = str(row.get("context_reader_ref") or "")
        if first_target_id is None:
            first_target_id = target_id
            first_context_ref = context_ref
        last_target_id = target_id
        last_context_ref = context_ref
        batch.append(
            (
                inserted,
                target_id,
                context_ref,
                group_key,
                sort_split,
                sort_file,
                sort_row,
                target_ts,
                row_json,
                row_digest,
            )
        )
        inserted += 1
        if args.max_targets is not None and inserted >= args.max_targets:
            break
        if len(batch) >= args.sqlite_insert_batch_size:
            conn.executemany(
                "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()
            batch.clear()
    if batch:
        conn.executemany("INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
        conn.commit()
    if inserted == 0:
        raise ShardBuildError("input target source produced zero rows")

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_targets_order
          ON targets(sort_split, sort_file, sort_row, target_ts, target_id);
        CREATE INDEX IF NOT EXISTS idx_targets_digest ON targets(row_digest);
        CREATE INDEX IF NOT EXISTS idx_targets_target_id ON targets(target_id);
        CREATE INDEX IF NOT EXISTS idx_targets_group ON targets(group_key);
        CREATE TABLE groups AS
          SELECT group_key, COUNT(*) AS target_count
          FROM targets
          GROUP BY group_key;
        CREATE INDEX IF NOT EXISTS idx_groups_key ON groups(group_key);
        """
    )
    conn.commit()
    duplicate_count = conn.execute(
        "SELECT COUNT(*) FROM (SELECT target_id FROM targets GROUP BY target_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    if duplicate_count and not args.allow_duplicate_target_ids:
        examples = conn.execute(
            "SELECT target_id, COUNT(*) FROM targets GROUP BY target_id HAVING COUNT(*) > 1 LIMIT 10"
        ).fetchall()
        raise ShardBuildError(f"duplicate target_id values found: {examples}")
    group_count = conn.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
    max_group_size = conn.execute("SELECT MAX(target_count) FROM groups").fetchone()[0]
    return {
        "target_count": inserted,
        "history_group_count": int(group_count),
        "max_history_group_size": int(max_group_size),
        "duplicate_target_id_group_count": int(duplicate_count),
        "first_input_target_id": first_target_id,
        "last_input_target_id": last_target_id,
        "first_input_context_reader_ref": first_context_ref,
        "last_input_context_reader_ref": last_context_ref,
    }


def digest_query(conn: sqlite3.Connection, sql: str, column: int = 0) -> str:
    h = hashlib.sha256()
    for (value,) in conn.execute(sql):
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    manifest_path = output_dir / "execution_shard_manifest.json"
    if shard_dir.exists() and any(shard_dir.iterdir()):
        if not overwrite:
            raise ShardBuildError(f"output shard directory is not empty: {shard_dir} (use --overwrite)")
        for path in shard_dir.glob("shard_*.targets.jsonl"):
            path.unlink()
    shard_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not overwrite:
        raise ShardBuildError(f"manifest already exists: {manifest_path} (use --overwrite)")
    return shard_dir, manifest_path


def write_shards(conn: sqlite3.Connection, args: argparse.Namespace, shard_dir: Path) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    targets_per_shard = max(int(args.targets_per_shard), 1)
    query = """
      SELECT t.row_json,
             t.target_id,
             t.context_reader_ref,
             t.group_key,
             t.row_digest,
             g.target_count
      FROM targets t
      JOIN groups g ON g.group_key = t.group_key
      ORDER BY t.sort_split, t.sort_file, t.sort_row, t.target_ts, t.target_id
    """

    current_f = None
    current_path: Path | None = None
    current_hash = hashlib.sha256()
    current_target_id_hash = hashlib.sha256()
    current_count = 0
    current_groups = 0
    current_first_target_id = None
    current_last_target_id = None
    current_first_context_ref = None
    current_last_context_ref = None
    current_group_key = None
    global_order_hash = hashlib.sha256()
    output_index = 0

    def close_current() -> None:
        nonlocal current_f, current_path, current_hash, current_target_id_hash
        nonlocal current_count, current_groups, current_first_target_id, current_last_target_id
        nonlocal current_first_context_ref, current_last_context_ref
        if current_f is None or current_path is None:
            return
        current_f.close()
        shards.append(
            {
                "shard_index": len(shards),
                "path": str(current_path),
                "target_count": current_count,
                "history_group_count": current_groups,
                "first_target_id": current_first_target_id,
                "last_target_id": current_last_target_id,
                "first_context_reader_ref": current_first_context_ref,
                "last_context_reader_ref": current_last_context_ref,
                "sha256": sha256_file(current_path),
                "ordered_target_id_digest": "sha256:" + current_target_id_hash.hexdigest(),
            }
        )
        current_f = None
        current_path = None
        current_hash = hashlib.sha256()
        current_target_id_hash = hashlib.sha256()
        current_count = 0
        current_groups = 0
        current_first_target_id = None
        current_last_target_id = None
        current_first_context_ref = None
        current_last_context_ref = None

    def open_current() -> None:
        nonlocal current_f, current_path
        if current_f is not None:
            return
        current_path = shard_dir / f"shard_{len(shards):05d}.targets.jsonl"
        current_f = open(current_path, "w", encoding="utf-8")

    for row_json, target_id, context_ref, group_key, _row_digest, group_size in conn.execute(query):
        new_group = group_key != current_group_key
        if new_group:
            if current_count > 0 and current_count + int(group_size) > targets_per_shard:
                close_current()
            current_group_key = group_key
            current_groups += 1
        open_current()
        assert current_f is not None
        if current_count == 0:
            current_first_target_id = target_id
            current_first_context_ref = context_ref
        current_last_target_id = target_id
        current_last_context_ref = context_ref
        current_f.write(row_json + "\n")
        encoded_target_id = str(target_id).encode("utf-8")
        current_target_id_hash.update(encoded_target_id + b"\n")
        global_order_hash.update(str(target_id).encode("utf-8") + b"\t" + str(context_ref).encode("utf-8") + b"\n")
        current_count += 1
        output_index += 1
    close_current()
    if not shards:
        raise ShardBuildError("no shards were written")
    # Stash the global order digest on the list object via a sentinel dict? Keep API simple: caller recomputes from shards.
    return shards


def validate_shards(
    conn: sqlite3.Connection,
    shards: Sequence[Mapping[str, Any]],
    *,
    strict_context_ref: bool,
) -> dict[str, Any]:
    conn.executescript(
        """
        DROP TABLE IF EXISTS written;
        CREATE TABLE written (
            output_index INTEGER PRIMARY KEY,
            target_id TEXT NOT NULL,
            group_key TEXT NOT NULL,
            sort_split TEXT NOT NULL,
            sort_file TEXT NOT NULL,
            sort_row INTEGER NOT NULL,
            target_ts TEXT NOT NULL,
            row_digest TEXT NOT NULL
        );
        """
    )
    output_index = 0
    batch: list[tuple[Any, ...]] = []
    previous_sort_key: tuple[str, str, int, str, str] | None = None
    current_group_key = None
    closed_groups: set[str] = set()
    order_errors: list[str] = []
    contiguity_errors: list[str] = []

    for shard in shards:
        path = Path(str(shard["path"]))
        if not path.exists():
            raise ShardBuildError(f"missing shard file during validation: {path}")
        expected_sha = shard.get("sha256")
        observed_sha = sha256_file(path)
        if expected_sha != observed_sha:
            raise ShardBuildError(f"shard digest mismatch for {path}: {observed_sha} != {expected_sha}")
        for row in iter_jsonl_rows(path):
            sort_split, sort_file, sort_row, target_ts, target_id, group_key = target_sort_key(
                row, strict_context_ref=strict_context_ref
            )
            sort_key = (sort_split, sort_file, sort_row, target_ts, target_id)
            if previous_sort_key is not None and sort_key < previous_sort_key and len(order_errors) < 10:
                order_errors.append(f"output index {output_index}: {sort_key!r} < {previous_sort_key!r}")
            if group_key != current_group_key:
                if current_group_key is not None:
                    closed_groups.add(current_group_key)
                if group_key in closed_groups and len(contiguity_errors) < 10:
                    contiguity_errors.append(f"group {group_key!r} reappeared at output index {output_index}")
                current_group_key = group_key
            previous_sort_key = sort_key
            row_digest = sha256_text(canonical_json(row))
            batch.append((output_index, target_id, group_key, sort_split, sort_file, sort_row, target_ts, row_digest))
            output_index += 1
            if len(batch) >= 10_000:
                conn.executemany("INSERT INTO written VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                conn.commit()
                batch.clear()
    if batch:
        conn.executemany("INSERT INTO written VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        conn.commit()

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_written_digest ON written(row_digest);
        CREATE INDEX IF NOT EXISTS idx_written_target_id ON written(target_id);
        CREATE INDEX IF NOT EXISTS idx_written_group ON written(group_key);
        """
    )
    conn.commit()
    input_count = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    output_count = conn.execute("SELECT COUNT(*) FROM written").fetchone()[0]
    input_row_multiset_digest = digest_query(conn, "SELECT row_digest FROM targets ORDER BY row_digest")
    output_row_multiset_digest = digest_query(conn, "SELECT row_digest FROM written ORDER BY row_digest")
    input_target_id_digest = digest_query(conn, "SELECT target_id FROM targets ORDER BY target_id")
    output_target_id_digest = digest_query(conn, "SELECT target_id FROM written ORDER BY target_id")
    input_group_count = conn.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
    output_group_count = conn.execute("SELECT COUNT(DISTINCT group_key) FROM written").fetchone()[0]
    status = "passed"
    failures = []
    if input_count != output_count:
        failures.append(f"count mismatch: input={input_count} output={output_count}")
    if input_row_multiset_digest != output_row_multiset_digest:
        failures.append("row multiset digest mismatch")
    if input_target_id_digest != output_target_id_digest:
        failures.append("target id multiset digest mismatch")
    if input_group_count != output_group_count:
        failures.append(f"group count mismatch: input={input_group_count} output={output_group_count}")
    if order_errors:
        failures.append("execution order is not nondecreasing")
    if contiguity_errors:
        failures.append("history groups are not contiguous")
    if failures:
        status = "failed"
    return {
        "status": status,
        "failures": failures,
        "input_target_count": int(input_count),
        "output_target_count": int(output_count),
        "input_history_group_count": int(input_group_count),
        "output_history_group_count": int(output_group_count),
        "input_row_multiset_digest": input_row_multiset_digest,
        "output_row_multiset_digest": output_row_multiset_digest,
        "input_target_id_multiset_digest": input_target_id_digest,
        "output_target_id_multiset_digest": output_target_id_digest,
        "order_errors_sample": order_errors,
        "contiguity_errors_sample": contiguity_errors,
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    source_stats: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    source: dict[str, Any]
    if args.target_jsonl:
        source = {"kind": "target_jsonl", "path": args.target_jsonl}
    else:
        source = {"kind": "target_sidecar_glob", "glob": args.target_sidecar_glob}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "shard_schema_version": SHARD_SCHEMA_VERSION,
        "benchmark_version": args.benchmark_version,
        "created_at": utc_now(),
        "artifact_purpose": "execution_only_sequence_native_target_shards_not_authoritative_benchmark_contract",
        "contract_preservation": {
            "target_rows_reordered_only": True,
            "target_rows_modified": False,
            "candidate_sets_modified": False,
            "labels_or_metrics_included": False,
            "official_manifest_modified": False,
        },
        "source": source,
        "sort_order": [
            "context_reader_ref.split",
            "context_reader_ref.part_file",
            "context_reader_ref.source_row_index",
            "target_ts",
            "target_id",
        ],
        "sharding_policy": {
            "targets_per_shard_requested": args.targets_per_shard,
            "do_not_split_history_groups": True,
            "oversized_history_group_policy": "allow_single_shard_to_exceed_requested_target_count",
        },
        "source_stats": dict(source_stats),
        "shard_count": len(shards),
        "shards": list(shards),
        "validation": dict(validation),
        "integration": {
            "sequential_runner_arg": "--target-jsonl <manifest.shards[i].path>",
            "run_id_recommendation": "use one unique prediction_run_id/output directory per shard, then concatenate prediction/compact outputs only after every shard succeeds",
            "resume_recommendation": "resume individual shard runs with sequential_submission_infer.py --resume using the same shard JSONL and output paths",
        },
        "work_db_path": str(db_path) if args.keep_work_db else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--target-jsonl", help="target rows as JSONL")
    src.add_argument("--target-sidecar-glob", help="glob for frozen target sidecar parquet files")
    ap.add_argument("--output-dir", required=True, help="directory for execution_shard_manifest.json and shards/")
    ap.add_argument("--benchmark-version", default=BENCHMARK_VERSION)
    ap.add_argument("--targets-per-shard", type=int, default=50_000)
    ap.add_argument("--max-targets", type=int, help="optional bounded build for smoke/debug")
    ap.add_argument("--read-batch-size", type=int, default=8192)
    ap.add_argument("--sqlite-insert-batch-size", type=int, default=10_000)
    ap.add_argument("--work-dir", help="optional directory for sqlite spool DB; defaults under output-dir/_work")
    ap.add_argument("--keep-work-db", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-duplicate-target-ids", action="store_true")
    ap.add_argument("--allow-unparsed-context-ref", action="store_true")
    ap.add_argument("--skip-readback-validation", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.targets_per_shard <= 0:
        raise SystemExit("--targets-per-shard must be positive")
    if args.max_targets is not None and args.max_targets <= 0:
        raise SystemExit("--max-targets must be positive when provided")

    output_dir = Path(args.output_dir)
    shard_dir, manifest_path = ensure_output_dir(output_dir, overwrite=args.overwrite)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "execution_shards.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = connect_db(db_path)
    try:
        init_db(conn)
        source_stats = ingest_targets(conn, args)
        shards = write_shards(conn, args, shard_dir)
        if args.skip_readback_validation:
            validation = {"status": "skipped", "reason": "--skip-readback-validation"}
        else:
            validation = validate_shards(conn, shards, strict_context_ref=not args.allow_unparsed_context_ref)
        manifest = build_manifest(
            args=args,
            source_stats=source_stats,
            shards=shards,
            validation=validation,
            db_path=db_path,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "manifest": str(manifest_path),
            "shard_count": len(shards),
            "target_count": source_stats["target_count"],
            "validation_status": validation.get("status"),
        }, indent=2, sort_keys=True))
        if validation.get("status") == "failed":
            return 2
        return 0
    except ShardBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        conn.close()
        if not args.keep_work_db:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
