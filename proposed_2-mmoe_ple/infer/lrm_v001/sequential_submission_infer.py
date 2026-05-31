#!/usr/bin/env python3
"""P23/LRM-v001 submission inference with sample-scoped history loading and sequential reuse.

This runner emits `lrm_prediction_record_v001` JSONL without modifying the
benchmark data/eval contract. It is designed for the current v001 production
contract:

* target metadata is read from frozen target sidecars or a JSONL sample;
* candidate ids are generated from frozen banked candidate artifacts;
* raw context is materialized through the official history reader;
* predictions cover every candidate in the target candidate set and contain only
  schema-allowed inference metadata.

Sequential reuse policy:

* If all target prefixes for a sampled user/history row fit inside the legacy
  P23 max sequence length (default 200), run one causal HSTU forward over the
  longest prefix and extract each target-position query state.
* If a target needs more than 200 events, P23 cannot encode the full available
  history. The runner uses the latest-200 policy for that target and labels it
  in the sidecar inference log. This keeps the prediction JSONL schema-valid
  while preserving the policy distinction for audit/debug.
"""
from __future__ import annotations

import argparse
import collections
from contextlib import ExitStack
import datetime as dt
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_metrics import (  # noqa: E402
    StreamingMetricAggregator,
    make_compact_record,
    pessimistic_rank_from_ranked,
)
from candidate_embedding_cache import (  # noqa: E402
    CandidateEmbeddingCache,
    score_candidate_ids_online,
    score_candidate_set_from_query,
)
from raw_candidate_bank_cache import RawCandidateBankCache  # noqa: E402


_CONTEXT_REF_RE = re.compile(
    r"^canonical_row_array_v001:(?P<split>[^/]+)/(?P<file>[^:]+):source_row_index=(?P<source_row_index>\d+):"
)

SCHEMA_VERSION = "lrm_prediction_record_v001"
BENCHMARK_VERSION = "lrm_benchmark_v001"
ENTRYPOINT = "proposed_2-mmoe_ple/infer/lrm_v001/sequential_submission_infer.py"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def load_module(name: str, path: str | Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_parquet_rows(paths: Sequence[str | Path], *, columns: Sequence[str] | None = None):
    import pyarrow.parquet as pq  # type: ignore

    for path in paths:
        table = pq.read_table(path, columns=list(columns) if columns else None)
        for row in table.to_pylist():
            yield row


TARGET_COLUMNS = [
    "benchmark_id",
    "target_id",
    "target_event_id",
    "user_id",
    "target_ts",
    "target_canonical_domain_id",
    "target_domain",
    "target_event_type",
    "candidate_protocol_label",
    "candidate_set_id",
    "candidate_set_digest",
    "negative_bank_id",
    "bank_selection_seed_material_digest",
    "positive_item_id",
    "headline_slices",
    "diagnostic_buckets",
    "raw_context_event_count",
    "context_reader_ref",
]


def load_selected_bank_subset(path: str | None) -> dict[int, set[int]] | None:
    if not path:
        return None
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "lrm_v001_selected_bank_subset_v001":
        raise ValueError(f"unsupported selected bank subset schema: {manifest.get('schema')}")
    return {int(domain_id): {int(x) for x in bank_ids} for domain_id, bank_ids in manifest["domains"].items()}


def _target_filters(args):
    wanted = None
    if args.target_id_file:
        wanted = {line.strip() for line in open(args.target_id_file, "r", encoding="utf-8") if line.strip()}
    done = None
    resume_path = None
    if args.resume:
        for candidate_path in (getattr(args, "output_predictions", None), getattr(args, "output_compact", None)):
            if candidate_path and Path(candidate_path).exists():
                resume_path = candidate_path
                break
    if resume_path:
        done = {str(row["target_id"]) for row in read_jsonl(resume_path)}
    selected_banks = getattr(args, "_selected_bank_filter", None)
    return wanted, done, selected_banks


def _keep_target(target: Mapping[str, Any], wanted: set[str] | None, done: set[str] | None, selected_banks: dict[int, set[int]] | None = None) -> bool:
    tid = str(target["target_id"])
    if wanted is not None and tid not in wanted:
        return False
    if done is not None and tid in done:
        return False
    if selected_banks is not None:
        try:
            domain_id = int(target.get("target_canonical_domain_id"))
            bank_id = int(target.get("negative_bank_id"))
        except Exception:
            return False
        if bank_id not in selected_banks.get(domain_id, set()):
            return False
    return True


def iter_target_batches(args):
    """Yield bounded target batches without loading the full production set into memory."""
    wanted, done, selected_banks = _target_filters(args)
    yielded = 0
    limit = args.max_targets
    batch_size = max(int(args.target_batch_size), 1)

    def maybe_emit(batch):
        nonlocal yielded
        if not batch:
            return None
        if limit is not None:
            remaining = limit - yielded
            if remaining <= 0:
                return None
            batch = batch[:remaining]
        yielded += len(batch)
        return batch

    if args.target_jsonl:
        batch: list[dict[str, Any]] = []
        with open(args.target_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not _keep_target(row, wanted, done, selected_banks):
                    continue
                batch.append(row)
                if len(batch) >= batch_size:
                    out = maybe_emit(batch)
                    if out:
                        yield out
                    batch = []
                    if limit is not None and yielded >= limit:
                        return
        out = maybe_emit(batch)
        if out:
            yield out
        return

    paths = sorted(glob.glob(args.target_sidecar_glob))
    if not paths:
        raise FileNotFoundError(f"no target sidecars matched {args.target_sidecar_glob!r}")
    import pyarrow.parquet as pq  # type: ignore

    # Accumulate *kept* targets across parquet record batches. The selected-bank
    # proxy keeps roughly 10% of rows; emitting once per raw record batch turns a
    # requested 200k batch into ~20k selected targets and forces query extraction
    # to rescan the same history parquet part many more times. Accumulating kept
    # rows lets --target-batch-size control the selected-target batch size.
    batch: list[dict[str, Any]] = []
    for path in paths:
        pf = pq.ParquetFile(path)
        for record_batch in pf.iter_batches(batch_size=batch_size, columns=TARGET_COLUMNS):
            for row in record_batch.to_pylist():
                row = dict(row)
                if not _keep_target(row, wanted, done, selected_banks):
                    continue
                batch.append(row)
                emit_threshold = batch_size
                if limit is not None:
                    emit_threshold = min(emit_threshold, max(1, limit - yielded))
                if len(batch) >= emit_threshold:
                    out = maybe_emit(batch)
                    if out:
                        yield out
                    batch = []
                    if limit is not None and yielded >= limit:
                        return
    out = maybe_emit(batch)
    if out:
        yield out


def parse_context_ref(target: Mapping[str, Any]) -> tuple[str, int]:
    ref = str(target.get("context_reader_ref") or "")
    m = _CONTEXT_REF_RE.match(ref)
    if not m:
        raise ValueError(f"target {target.get('target_id')} lacks parsable canonical context_reader_ref: {ref!r}")
    if m.group("split") != "eval":
        raise ValueError(f"target {target.get('target_id')} is not an eval context ref: {ref!r}")
    return m.group("file"), int(m.group("source_row_index"))


def grouped_targets(targets: Iterable[dict[str, Any]]):
    by_part: dict[str, dict[int, list[dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for target in targets:
        part_file, source_row_index = parse_context_ref(target)
        by_part[part_file][source_row_index].append(target)
    return by_part


def event_type_id_for_context(ev: Mapping[str, Any], max_supported: int = 13) -> int:
    """Map v001 event_type_id to the legacy P23 embedding range."""
    try:
        value = int(ev.get("event_type_id") or 0)
    except Exception:
        value = 0
    if value < 0 or value > max_supported:
        return 0
    return value


def build_model(args):
    import gin  # type: ignore
    import torch  # type: ignore
    import fbgemm_gpu  # noqa: F401  # type: ignore

    source_train = Path(args.source_root) / "proposed_2-mmoe_ple" / "train"
    sys.path.insert(0, str(source_train))
    from data.reco_dataset import RecoDataset  # type: ignore
    from trainer.util import make_model  # type: ignore
    from trainer.train import Trainer  # noqa: F401  # type: ignore  # registers gin
    from trainer.data_loader import create_data_loader  # noqa: F401  # type: ignore  # registers gin

    gin.clear_config()
    gin.parse_config_file(args.gin_config_file)

    shard_sizes = {
        0: 4_489_403,
        1: 176_044_856,
        2: 47_742_266,
        3: 4_692_056,
        4: 945_621,
    }
    dataset = RecoDataset(
        dataset_name="lrm_benchmark_v001_v3_full_preserve",
        max_sequence_length=int(args.max_sequence_length),
        positional_sampling_ratio=1.0,
        train_dataset=None,
        eval_dataset=None,
        domain_to_item_id_range={d: (20, shard_sizes[d] - 1) for d in shard_sizes},
        embd_dim=384,
        domain_offset=1_000_000_000,
        shard_size=1_000_000_000,
        shard_counts={d: 1 for d in range(5)},
        min_item_id=20,
        max_item_id=4_000_945_620,
        num_ratings=0,
        # P23 checkpoint event-type embedding has rows 0..13.
        num_event_types=13,
    )
    embedding_dirs = {d: str(Path(args.embedding_root) / f"domain_{d}") for d in range(5)}
    model = make_model(dataset=dataset, precomputed_embeddings_domain_to_dir=embedding_dirs)
    snapshot = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(snapshot["MODEL_STATE"], strict=True)
    model.to(args.device)
    model.eval()
    return model


def events_to_tensors(events: list[dict[str, Any]], *, max_sequence_length: int, device: str):
    import torch  # type: ignore

    if not events:
        raise ValueError("P23 adapter cannot score zero-length context targets")
    if len(events) > max_sequence_length:
        raise ValueError(f"internal error: {len(events)} events exceed max_sequence_length={max_sequence_length}")
    input_ids = [int(ev["encoded_id"]) for ev in events]
    timestamps = [int(ev.get("event_time_unix_s") or 0) for ev in events]
    raw_type_ids: list[int] = []
    type_ids: list[int] = []
    mapped_unknown = 0
    for ev in events:
        raw = int(ev.get("event_type_id") or 0)
        mapped = event_type_id_for_context(ev)
        raw_type_ids.append(raw)
        type_ids.append(mapped)
        if raw != mapped:
            mapped_unknown += 1
    actual_len = len(input_ids)
    pad_n = max_sequence_length - actual_len
    return {
        "input_ids": torch.tensor([input_ids + [0] * pad_n], dtype=torch.long, device=device),
        "timestamps": torch.tensor([timestamps + [0] * pad_n], dtype=torch.long, device=device),
        "type_ids": torch.tensor([type_ids + [0] * pad_n], dtype=torch.long, device=device),
        "lengths": torch.tensor([actual_len], dtype=torch.long, device=device),
        "ratings": torch.zeros((1,), dtype=torch.long, device=device),
        "mapped_unsupported_event_type_count": mapped_unknown,
        "raw_type_ids_tail": raw_type_ids[-10:],
        "type_ids_tail": type_ids[-10:],
    }


def _encode_sequence_batch(model, tensors):
    import torch  # type: ignore

    with torch.inference_mode():
        raw_input_embeddings = model.model._embedding_module.get_raw_item_embeddings(tensors["input_ids"])
        past_embeddings = model.model._embedding_module(raw_input_embeddings.to(dtype=torch.float32))
        return model.model(
            past_lengths=tensors["lengths"],
            past_ids=tensors["input_ids"],
            past_embeddings=past_embeddings,
            past_payloads={
                "timestamps": tensors["timestamps"],
                "ratings": tensors["ratings"],
                "type_ids": tensors["type_ids"],
            },
        )


def encode_positions_one_pass(model, tensors, positions: Mapping[str, int]) -> dict[str, Any]:
    """Run base HSTU once and return normalized hidden state at requested 0-based positions."""
    import torch.nn.functional as F  # type: ignore

    encoded_seq = _encode_sequence_batch(model, tensors)
    result = {}
    length = int(tensors["lengths"].item())
    for key, pos in positions.items():
        if pos < 0 or pos >= length:
            raise ValueError(f"requested position {pos} outside encoded length {length}")
        result[key] = F.normalize(encoded_seq[0, pos, :].float(), p=2, dim=-1).unsqueeze(0)
    return result


def events_to_batched_tensors(events_batch: Sequence[Sequence[dict[str, Any]]], *, max_sequence_length: int, device: str):
    """Pad multiple context windows for one batched HSTU forward."""
    import torch  # type: ignore

    if not events_batch:
        raise ValueError("cannot encode an empty context batch")
    input_rows: list[list[int]] = []
    timestamp_rows: list[list[int]] = []
    type_rows: list[list[int]] = []
    lengths: list[int] = []
    mapped_unknown = 0
    for events in events_batch:
        cur = list(events)
        if not cur:
            raise ValueError("batched encoder requires non-empty context windows")
        if len(cur) > max_sequence_length:
            raise ValueError(f"internal error: {len(cur)} events exceed max_sequence_length={max_sequence_length}")
        input_ids = [int(ev["encoded_id"]) for ev in cur]
        timestamps = [int(ev.get("event_time_unix_s") or 0) for ev in cur]
        type_ids = []
        for ev in cur:
            raw = int(ev.get("event_type_id") or 0)
            mapped = event_type_id_for_context(ev)
            type_ids.append(mapped)
            if raw != mapped:
                mapped_unknown += 1
        pad_n = max_sequence_length - len(cur)
        input_rows.append(input_ids + [0] * pad_n)
        timestamp_rows.append(timestamps + [0] * pad_n)
        type_rows.append(type_ids + [0] * pad_n)
        lengths.append(len(cur))
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "timestamps": torch.tensor(timestamp_rows, dtype=torch.long, device=device),
        "type_ids": torch.tensor(type_rows, dtype=torch.long, device=device),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        "ratings": torch.zeros((len(events_batch),), dtype=torch.long, device=device),
        "mapped_unsupported_event_type_count": mapped_unknown,
    }


def encode_positions_batched(model, tensors, positions: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    """Return normalized states for multiple batch rows/positions from one HSTU forward."""
    import torch.nn.functional as F  # type: ignore

    encoded_seq = _encode_sequence_batch(model, tensors)
    result = {}
    lengths = [int(x) for x in tensors["lengths"].detach().cpu().tolist()]
    for key, (batch_idx, pos) in positions.items():
        if batch_idx < 0 or batch_idx >= len(lengths):
            raise ValueError(f"requested batch index {batch_idx} outside batch size {len(lengths)}")
        if pos < 0 or pos >= lengths[batch_idx]:
            raise ValueError(f"requested position {pos} outside encoded length {lengths[batch_idx]}")
        result[key] = F.normalize(encoded_seq[batch_idx, pos, :].float(), p=2, dim=-1).unsqueeze(0)
    return result


def encode_final_legacy(model, tensors):
    """Legacy final-state encode path, used only for optional equivalence checks."""
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    with torch.inference_mode():
        raw_input_embeddings = model.model._embedding_module.get_raw_item_embeddings(tensors["input_ids"])
        past_embeddings = model.model._embedding_module(raw_input_embeddings.to(dtype=torch.float32))
        query = model.model.encode(
            past_lengths=tensors["lengths"],
            past_ids=tensors["input_ids"],
            past_embeddings=past_embeddings,
            past_payloads={
                "timestamps": tensors["timestamps"],
                "ratings": tensors["ratings"],
                "type_ids": tensors["type_ids"],
            },
        )
        return F.normalize(query.float(), p=2, dim=-1)


def zero_context_query(model, *, device: str):
    """Cold-start fallback for targets with no valid context events.

    The v001 prediction schema requires full candidate ranking even when a legacy
    sequential checkpoint cannot form a user state. A zero query yields tied
    scores; `score_candidates_from_query` then deterministically orders ties by
    candidate id.
    """
    import torch  # type: ignore

    dim = int(getattr(model, "item_embedding_dim", getattr(model.model, "_item_embedding_dim", 128)))
    return torch.zeros((1, dim), dtype=torch.float32, device=device)


def score_candidates_from_query(model, query, candidate_ids: list[str], *, chunk_size: int, device: str):
    pairs, _ = score_candidate_ids_online(model, query, candidate_ids, chunk_size=chunk_size, device=device)
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs


def _build_raw_context(reader_mod, reader, history, target: Mapping[str, Any], target_dt, events: list[dict[str, Any]]):
    supplied_count = target.get("raw_context_event_count")
    if supplied_count is not None and int(supplied_count) != len(events):
        raise ValueError(
            f"raw_context_event_count mismatch for {target.get('target_id')}: "
            f"target={supplied_count} materialized={len(events)}"
        )
    checksum_payload = {
        "schema_version": reader_mod.SCHEMA_VERSION,
        "benchmark_id": reader_mod.BENCHMARK_ID,
        "dataset_manifest_id": reader.dataset_manifest_id,
        "canonical_root": str(reader.canonical_root),
        "split": reader.split,
        "user_id": str(target.get("user_id")),
        "target_time": target_dt,
        "events": events,
    }
    ctx = reader_mod.RawContext(
        benchmark_id=reader_mod.BENCHMARK_ID,
        schema_version=reader_mod.SCHEMA_VERSION,
        dataset_manifest_id=reader.dataset_manifest_id,
        dataset_version=history.dataset_version,
        canonical_root=str(reader.canonical_root),
        split=reader.split,
        user_id=str(target.get("user_id")),
        target_time=target_dt,
        context_start_time=reader_mod.T1,
        context_end_time_exclusive=target_dt,
        events=events,
        checksum=reader_mod._stable_json_sha256(checksum_payload),
    )
    ctx.to_dict()  # contract check, including no scorer-private/raw fields
    return ctx


def _target_datetime(reader_mod, history, target: Mapping[str, Any]):
    user_id = str(target.get("user_id"))
    if user_id != history.user_id:
        raise ValueError(f"target/user mismatch: target={target.get('target_id')} user={user_id} history={history.user_id}")
    target_time = target.get("target_ts", target.get("target_time"))
    if target_time is None:
        raise ValueError(f"target {target.get('target_id')} lacks target_ts/target_time")
    target_dt = reader_mod._parse_datetime(target_time)
    if not (reader_mod.T1_PLUS_W < target_dt <= reader_mod.T2):
        raise ValueError(f"target {target.get('target_id')} time outside v001 target interval: {reader_mod._iso(target_dt)}")
    return target_dt


def materialize_context_from_history(reader_mod, reader, history, target: Mapping[str, Any]):
    target_dt = _target_datetime(reader_mod, history, target)
    events: list[dict[str, Any]] = []
    for ev in history.events:
        if ev.fields.get("timestamp_quality_status") != "valid":
            continue
        if reader_mod.T1 <= ev.event_time < target_dt:
            events.append(ev.model_visible_dict())
    return _build_raw_context(reader_mod, reader, history, target, target_dt, events)


def materialize_contexts_from_history_one_pass(reader_mod, reader, history, targets: Sequence[Mapping[str, Any]]):
    """Materialize multiple target prefixes from one chronological history scan.

    The previous path rescanned `history.events` once per target. For proxy evals
    with many targets on the same canonical history row, that made context
    materialization the top CPU cost. This keeps the exact RawContext/checksum
    contract but advances a prefix pointer across targets sorted by target time.
    If a history is unexpectedly not chronological, it falls back to the audited
    single-target implementation rather than changing event order semantics.
    """
    if not targets:
        return []
    target_dts = [(target, _target_datetime(reader_mod, history, target)) for target in targets]
    valid_events = [
        (ev.event_time, ev.model_visible_dict())
        for ev in history.events
        if ev.fields.get("timestamp_quality_status") == "valid" and reader_mod.T1 <= ev.event_time
    ]
    if any(valid_events[i][0] > valid_events[i + 1][0] for i in range(len(valid_events) - 1)):
        return [(target, materialize_context_from_history(reader_mod, reader, history, target)) for target in targets]

    out_by_id: dict[str, Any] = {}
    prefix: list[dict[str, Any]] = []
    event_idx = 0
    for target, target_dt in sorted(target_dts, key=lambda item: (item[1], str(item[0].get("target_id")))):
        while event_idx < len(valid_events) and valid_events[event_idx][0] < target_dt:
            prefix.append(valid_events[event_idx][1])
            event_idx += 1
        # RawContext owns a target-specific immutable snapshot; copy the prefix
        # list but do not rescan the history for each target.
        out_by_id[str(target.get("target_id"))] = _build_raw_context(reader_mod, reader, history, target, target_dt, list(prefix))
    return [(target, out_by_id[str(target.get("target_id"))]) for target in targets]


def selected_histories_for_part(reader, part_file: str, needed: Mapping[int, list[dict[str, Any]]], *, batch_size: int, log_f):
    import pyarrow.parquet as pq  # type: ignore

    part_path = Path(reader.split_path) / part_file
    if not part_path.exists():
        raise FileNotFoundError(f"missing canonical part referenced by target sidecar: {part_path}")
    needed_indices = set(needed)
    columns = [
        "benchmark_id",
        "dataset_version",
        "split",
        "user_id",
        "source_file",
        "source_row_index",
        "canonical_order",
        "legacy_order",
        "valid_event_count",
        "events",
        "row_checksum",
    ]
    schema_names = set(reader._dataset.schema.names)
    if "source_file_sha256" in schema_names:
        columns.insert(6, "source_file_sha256")
    log_json(log_f, {
        "progress": "part_scan_start",
        "part_file": part_file,
        "needed_source_row_indices": len(needed_indices),
        "strategy": "sequential_parquet_part_scan_no_reusable_prefix_cache",
        "at": utc_now(),
    })
    found: dict[int, Any] = {}
    pf = pq.ParquetFile(part_path)
    rows_seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            rows_seen += 1
            idx = int(row["source_row_index"])
            if idx in needed_indices:
                history = reader._row_to_history(row)
                expected_users = {str(t["user_id"]) for t in needed[idx]}
                if history.user_id not in expected_users:
                    raise ValueError(
                        f"context_reader_ref collision in {part_file} source_row_index={idx}: "
                        f"history user {history.user_id} not in sample users {sorted(expected_users)}"
                    )
                found[idx] = history
        if len(found) == len(needed_indices):
            break
    missing = sorted(needed_indices - set(found))
    if missing:
        raise KeyError(f"missing source_row_index values in {part_path}: {missing[:10]} total={len(missing)}")
    log_json(log_f, {
        "progress": "part_scan_done",
        "part_file": part_file,
        "rows_seen": rows_seen,
        "histories_loaded": len(found),
        "at": utc_now(),
    })
    return found


_LOG_FLUSH_EVERY = 1
_LOG_WRITE_COUNT = 0
_OUTPUT_FLUSH_EVERY = 1
_OUTPUT_WRITE_COUNT = 0


def configure_log_flush(every: int) -> None:
    """Configure JSONL sidecar flushing cadence.

    every=1 preserves the old safe/debuggable behavior. every=0 leaves flushing
    to Python/file close, which is materially faster for million-target proxy
    evals where per-target fsync-ish flush overhead dominates.
    """
    global _LOG_FLUSH_EVERY, _LOG_WRITE_COUNT
    _LOG_FLUSH_EVERY = max(0, int(every))
    _LOG_WRITE_COUNT = 0


def log_json(log_f, payload: Mapping[str, Any]) -> None:
    global _LOG_WRITE_COUNT
    log_f.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    _LOG_WRITE_COUNT += 1
    if _LOG_FLUSH_EVERY == 1 or (_LOG_FLUSH_EVERY > 1 and _LOG_WRITE_COUNT % _LOG_FLUSH_EVERY == 0):
        log_f.flush()


def configure_output_flush(every: int) -> None:
    global _OUTPUT_FLUSH_EVERY, _OUTPUT_WRITE_COUNT
    _OUTPUT_FLUSH_EVERY = max(0, int(every))
    _OUTPUT_WRITE_COUNT = 0


def maybe_flush_output(file_obj) -> None:
    global _OUTPUT_WRITE_COUNT
    _OUTPUT_WRITE_COUNT += 1
    if _OUTPUT_FLUSH_EVERY == 1 or (_OUTPUT_FLUSH_EVERY > 1 and _OUTPUT_WRITE_COUNT % _OUTPUT_FLUSH_EVERY == 0):
        file_obj.flush()


def maybe_print_progress(args, *, target_counter: int, target_id: str) -> None:
    every = int(getattr(args, "stdout_progress_every", 1) or 0)
    if every > 0 and target_counter % every == 0:
        print(json.dumps({"progress": "target_done", "target_index": target_counter, "target_id": target_id}), flush=True)


def inference_metadata(args, *, generated_at: str, model_digest: str, context_policy_digest: str) -> dict[str, Any]:
    # Keep this object strictly within prediction_schema.json additionalProperties=false.
    return {
        "generated_at": generated_at,
        "entrypoint_name": ENTRYPOINT,
        "model_artifact_digest": model_digest,
        "context_policy_digest": context_policy_digest,
        "seed": int(args.seed),
        "context_policy_mode": "declared_transforms_over_full_available_prefix",
        "notes": "P23 v001 submission inference. Short histories use full available prefix with one causal sequence pass per user/history where possible; long histories use latest-200 due to the legacy checkpoint max sequence length. Unsupported event_type_id values above checkpoint range are mapped to UNK=0.",
    }


def score_target(
    args,
    generator,
    bank_cache,
    candidate_embedding_cache,
    model,
    query,
    target,
    *,
    generated_at: str,
    model_digest: str,
    context_policy_digest: str,
    emit_full_record: bool,
):
    domain_id = int(target["target_canonical_domain_id"])
    bank_path = Path(args.bank_root) / "banks" / f"domain_{domain_id}_banks.production.json"
    if domain_id not in bank_cache:
        bank_cache[domain_id] = generator.load_bank_artifact(bank_path)
    bank = bank_cache[domain_id]
    cand_result = generator.generate_candidates_for_target(target, bank)
    cand_errors = generator.validate_generated_candidates(cand_result)
    if cand_errors:
        raise RuntimeError(f"candidate generation failed for {target['target_id']}: {cand_errors}")
    if cand_result.candidate_set_digest != target.get("candidate_set_digest"):
        raise RuntimeError(f"candidate digest mismatch for {target['target_id']}")
    ranked, score_metrics = score_candidate_set_from_query(
        model,
        query,
        cand_result,
        chunk_size=args.chunk_size,
        device=args.device,
        candidate_cache=candidate_embedding_cache,
        generator_mod=generator,
        bank_artifact=bank,
        timing_sync_cuda=args.candidate_cache_timing_sync_cuda,
    )
    record = None
    if emit_full_record:
        record = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_version": BENCHMARK_VERSION,
            "model_submission_id": args.model_submission_id,
            "prediction_run_id": args.prediction_run_id,
            "target_id": target["target_id"],
            "candidate_protocol_label": target["candidate_protocol_label"],
            "candidate_set_id": target["candidate_set_id"],
            "predictions": [
                {"candidate_id": cid, "rank": rank, "score": score}
                for rank, (cid, score) in enumerate(ranked, start=1)
            ],
            "inference_metadata": inference_metadata(
                args,
                generated_at=generated_at,
                model_digest=model_digest,
                context_policy_digest=context_policy_digest,
            ),
        }
    return record, cand_result, ranked, score_metrics


def emit_target_outputs(
    args,
    *,
    pred_f,
    compact_f,
    compact_aggregator: StreamingMetricAggregator,
    record: dict[str, Any] | None,
    cand_result,
    ranked: Sequence[tuple[str, float]],
    target: Mapping[str, Any],
    ctx,
    generated_at: str,
    model_digest: str,
    context_policy_digest: str,
    context_policy_label: str,
    model_inference_policy: str,
) -> dict[str, Any]:
    rank_stats = pessimistic_rank_from_ranked(
        ranked,
        str(target.get("positive_item_id")),
        k=int(args.compact_top_k),
    )
    if pred_f is not None:
        if record is None:
            raise RuntimeError("internal error: full output requested but prediction record was not built")
        pred_f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        maybe_flush_output(pred_f)
    if compact_f is not None:
        compact_record = make_compact_record(
            target=target,
            ranked=ranked,
            rank_stats=rank_stats,
            top_k=int(args.compact_top_k),
            model_submission_id=args.model_submission_id,
            prediction_run_id=args.prediction_run_id,
            generated_at=generated_at,
            model_digest=model_digest,
            context_policy_digest=context_policy_digest,
            candidate_count=len(cand_result.candidate_item_ids),
            candidate_set_digest=cand_result.candidate_set_digest,
            context_checksum=getattr(ctx, "checksum", None),
            context_policy_label=context_policy_label,
            model_inference_policy=model_inference_policy,
            include_full_score_order_digest=not bool(getattr(args, "compact_no_score_order_digest", False)),
        )
        compact_f.write(json.dumps(compact_record, sort_keys=True, separators=(",", ":")) + "\n")
        maybe_flush_output(compact_f)
        compact_aggregator.add_compact_record(compact_record)
    return rank_stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-version", default=BENCHMARK_VERSION)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-jsonl")
    group.add_argument("--target-sidecar-glob")
    ap.add_argument("--target-id-file")
    ap.add_argument("--selected-bank-subset", help="selected-bank subset manifest; filters targets to matching (domain_id, negative_bank_id)")
    ap.add_argument("--max-targets", type=int)
    ap.add_argument("--target-batch-size", type=int, default=2048, help="targets per planning batch; avoids loading full production set into memory")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--history-prefix-source", required=True, help="canonical row-array root")
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--bank-generator", required=True)
    ap.add_argument("--history-reader", required=True)
    ap.add_argument("--source-root", default="/home/yourslewis/lrm-scaling-all-events")
    ap.add_argument("--gin-config-file", required=True)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--embedding-root", default="/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve")
    ap.add_argument("--model-submission-id", required=True)
    ap.add_argument("--prediction-run-id", required=True)
    ap.add_argument("--context-policy", required=True)
    ap.add_argument("--output-mode", choices=["full", "compact", "both"], default="full")
    ap.add_argument("--output-predictions", help="official full lrm_prediction_record_v001 JSONL; required for --output-mode full/both")
    ap.add_argument("--output-compact", help="derived compact JSONL with positive rank, topK, and digests; required for --output-mode compact/both")
    ap.add_argument("--output-metrics-json", help="derived compact streaming metrics JSON; required for --output-mode compact/both")
    ap.add_argument("--compact-top-k", type=int, default=10)
    ap.add_argument("--compact-no-score-order-digest", action="store_true", help="omit expensive full_score_order_digest from derived compact records")
    ap.add_argument("--log-flush-every", type=int, default=1, help="flush inference log every N writes; 0 flushes only at close for fast proxy evals")
    ap.add_argument("--output-flush-every", type=int, default=1, help="flush prediction/compact outputs every N writes; 0 flushes only at close")
    ap.add_argument("--stdout-progress-every", type=int, default=1, help="print target_done progress every N targets; 0 disables per-target stdout")
    ap.add_argument("--output-inference-log", required=True)
    ap.add_argument("--output-target-ids")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-sequence-length", type=int, default=200)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--candidate-cache-max-banks", type=int, default=0, help="enable projected negative-bank embedding cache with this many resident banks")
    ap.add_argument("--candidate-cache-dir", help="optional directory for model-digest-scoped projected bank embedding .npy files")
    ap.add_argument("--candidate-cache-disk-dtype", choices=["float32"], default="float32", help="persist exact float32 projected embeddings; lower precision would change scores")
    ap.add_argument("--candidate-cache-timing-sync-cuda", action="store_true", help="synchronize CUDA around scoring timers for accurate perf logs")
    ap.add_argument("--raw-bank-cache-dir", help="model-independent raw selected-bank embedding cache from build_raw_candidate_bank_cache.py")
    ap.add_argument("--raw-bank-cache-placement", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--history-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--equivalence-check-targets", type=int, default=0)
    args = ap.parse_args()
    configure_log_flush(args.log_flush_every)
    configure_output_flush(args.output_flush_every)

    if args.benchmark_version != BENCHMARK_VERSION:
        raise SystemExit("unsupported benchmark version")
    write_full = args.output_mode in {"full", "both"}
    write_compact = args.output_mode in {"compact", "both"}
    if write_full and not args.output_predictions:
        raise SystemExit("--output-predictions is required for --output-mode full/both")
    if write_compact and not args.output_compact:
        raise SystemExit("--output-compact is required for --output-mode compact/both")
    if write_compact and not args.output_metrics_json:
        raise SystemExit("--output-metrics-json is required for --output-mode compact/both")
    if args.compact_top_k <= 0:
        raise SystemExit("--compact-top-k must be positive")
    args._selected_bank_filter = load_selected_bank_subset(args.selected_bank_subset)

    import torch  # type: ignore

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1] if ":" in args.device else 0))

    generator = load_module("banked_candidate_generator_v001", args.bank_generator)
    reader_mod = load_module("history_prefix_reader_v001", args.history_reader)
    model = build_model(args)
    history_reader = reader_mod.HistoryPrefixReader.open(
        canonical_root=args.history_prefix_source,
        split="eval",
        mode="eval_inference",
    )

    model_digest = sha256_file(args.checkpoint_path)
    context_policy_digest = sha256_file(args.context_policy)
    generated_at = utc_now()
    bank_cache: dict[int, Any] = {}
    if args.raw_bank_cache_dir:
        candidate_embedding_cache = RawCandidateBankCache(
            cache_dir=args.raw_bank_cache_dir,
            model_digest=model_digest,
            max_banks=args.candidate_cache_max_banks,
            device=args.device,
            placement=args.raw_bank_cache_placement,
            timing_sync_cuda=args.candidate_cache_timing_sync_cuda,
        )
    else:
        candidate_embedding_cache = CandidateEmbeddingCache(
            model_digest=model_digest,
            max_banks=args.candidate_cache_max_banks,
            device=args.device,
            chunk_size=args.chunk_size,
            disk_dir=args.candidate_cache_dir,
            disk_dtype=args.candidate_cache_disk_dtype,
            timing_sync_cuda=args.candidate_cache_timing_sync_cuda,
        )
    if args.output_predictions:
        Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    if args.output_compact:
        Path(args.output_compact).parent.mkdir(parents=True, exist_ok=True)
    if args.output_metrics_json:
        Path(args.output_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_inference_log).parent.mkdir(parents=True, exist_ok=True)
    if args.output_target_ids:
        Path(args.output_target_ids).parent.mkdir(parents=True, exist_ok=True)

    full_mode = "a" if args.resume and args.output_predictions and Path(args.output_predictions).exists() else "w"
    compact_mode = "a" if args.resume and args.output_compact and Path(args.output_compact).exists() else "w"
    target_counter = 0
    loaded_histories = 0
    encoder_passes = 0
    short_full_available_targets = 0
    long_latest_200_targets = 0
    started = time.time()
    compact_aggregator = StreamingMetricAggregator(k=args.compact_top_k)
    if write_compact and args.resume and args.output_compact and Path(args.output_compact).exists():
        for previous in read_jsonl(args.output_compact):
            compact_aggregator.add_compact_record(previous)

    with ExitStack() as stack:
        pred_f = stack.enter_context(open(args.output_predictions, full_mode, encoding="utf-8")) if write_full else None
        compact_f = stack.enter_context(open(args.output_compact, compact_mode, encoding="utf-8")) if write_compact else None
        log_f = stack.enter_context(open(args.output_inference_log, "a", encoding="utf-8"))
        log_json(log_f, {
            "progress": "run_start",
            "target_source": "jsonl" if args.target_jsonl else "sidecar_glob",
            "max_targets": args.max_targets,
            "target_batch_size": args.target_batch_size,
            "adapter": Path(__file__).name,
            "history_loading_policy": "group_by_context_reader_ref_part_and_source_row_index_scan_each_referenced_part_once",
            "model_inference_policy": "one_causal_hstu_forward_per_user_history_when_full_prefix_fits_else_latest_200_per_target",
            "output_schema": SCHEMA_VERSION,
            "output_mode": args.output_mode,
            "compact_top_k": args.compact_top_k if write_compact else None,
            "compact_score_order_digest": (not args.compact_no_score_order_digest) if write_compact else None,
            "log_flush_every": args.log_flush_every,
            "output_flush_every": args.output_flush_every,
            "stdout_progress_every": args.stdout_progress_every,
            "candidate_coverage": "all_generated_candidates_per_target_scored_in_memory",
            "candidate_embedding_cache": candidate_embedding_cache.snapshot(),
            "selected_bank_subset": args.selected_bank_subset,
            "selected_bank_count": (sum(len(v) for v in args._selected_bank_filter.values()) if args._selected_bank_filter else None),
            "raw_bank_cache_dir": args.raw_bank_cache_dir,
            "at": generated_at,
        })
        batch_counter = 0
        for targets in iter_target_batches(args):
            batch_counter += 1
            by_part = grouped_targets(targets)
            log_json(log_f, {
                "progress": "target_batch_start",
                "batch_index": batch_counter,
                "batch_target_count": len(targets),
                "batch_part_count": len(by_part),
                "at": utc_now(),
            })
            for part_file in sorted(by_part):
                found = selected_histories_for_part(
                    history_reader,
                    part_file,
                    by_part[part_file],
                    batch_size=args.history_batch_size,
                    log_f=log_f,
                )
                loaded_histories += len(found)
                for source_row_index in sorted(by_part[part_file]):
                    history = found[source_row_index]
                    user_targets = sorted(by_part[part_file][source_row_index], key=lambda t: (str(t.get("target_ts")), str(t["target_id"])))
                    target_contexts = materialize_contexts_from_history_one_pass(reader_mod, history_reader, history, user_targets)
    
                    zero_contexts = [(t, c) for t, c in target_contexts if c.event_count == 0]
                    short_contexts = [(t, c) for t, c in target_contexts if 0 < c.event_count <= args.max_sequence_length]
                    long_contexts = [(t, c) for t, c in target_contexts if c.event_count > args.max_sequence_length]
    
                    if short_contexts:
                        max_ctx = max(short_contexts, key=lambda tc: tc[1].event_count)[1]
                        tensors = events_to_tensors(list(max_ctx.events), max_sequence_length=args.max_sequence_length, device=args.device)
                        positions = {str(t["target_id"]): c.event_count - 1 for t, c in short_contexts}
                        queries = encode_positions_one_pass(model, tensors, positions)
                        encoder_passes += 1
                        log_json(log_f, {
                            "progress": "history_encoded",
                            "context_policy_label": "full_available_history_for_p23",
                            "part_file": part_file,
                            "source_row_index": source_row_index,
                            "user_id": history.user_id,
                            "targets_for_history": len(short_contexts),
                            "encoder_passes_for_history": 1,
                            "max_prefix_len": max_ctx.event_count,
                            "positions_extracted": sorted(positions.values()),
                            "at": utc_now(),
                        })
                        for target, ctx in short_contexts:
                            target_counter += 1
                            short_full_available_targets += 1
                            eq = None
                            if target_counter <= args.equivalence_check_targets:
                                legacy_tensors = events_to_tensors(list(ctx.events), max_sequence_length=args.max_sequence_length, device=args.device)
                                legacy_query = encode_final_legacy(model, legacy_tensors)
                                eq = float((queries[str(target["target_id"])] - legacy_query).abs().max().detach().cpu().item())
                            record, cand_result, ranked, score_metrics = score_target(
                                args, generator, bank_cache, candidate_embedding_cache, model, queries[str(target["target_id"])], target,
                                generated_at=generated_at, model_digest=model_digest, context_policy_digest=context_policy_digest,
                                emit_full_record=write_full,
                            )
                            rank_stats = emit_target_outputs(
                                args,
                                pred_f=pred_f,
                                compact_f=compact_f,
                                compact_aggregator=compact_aggregator,
                                record=record,
                                cand_result=cand_result,
                                ranked=ranked,
                                target=target,
                                ctx=ctx,
                                generated_at=generated_at,
                                model_digest=model_digest,
                                context_policy_digest=context_policy_digest,
                                context_policy_label="full_available_history_for_p23",
                                model_inference_policy="one_causal_hstu_forward_per_history_extract_position",
                            )
                            if args.output_target_ids:
                                with open(args.output_target_ids, "a" if args.resume or target_counter > 1 else "w", encoding="utf-8") as ids_f:
                                    ids_f.write(str(target["target_id"]) + "\n")
                            positive_rank = rank_stats["pessimistic_rank"]
                            log_json(log_f, {
                                "progress": "target_done",
                                "target_id": target["target_id"],
                                "target_index": target_counter,
                                "part_file": part_file,
                                "source_row_index": source_row_index,
                                "user_id": history.user_id,
                                "candidate_count": len(cand_result.candidate_item_ids),
                                "candidate_set_digest": cand_result.candidate_set_digest,
                                "context_event_count_available": ctx.event_count,
                                "context_event_count_used": ctx.event_count,
                                "context_policy_label": "full_available_history_for_p23",
                                "model_inference_policy": "one_causal_hstu_forward_per_history_extract_position",
                                "encoder_position_extracted": ctx.event_count - 1,
                                "targets_for_history": len(short_contexts),
                                "equivalence_max_abs_diff_vs_legacy_encode": eq,
                                "context_checksum": ctx.checksum,
                                "positive_item_rank_debug_only": positive_rank,
                                "candidate_scoring": score_metrics,
                                "candidate_embedding_cache": candidate_embedding_cache.snapshot(),
                                "at": utc_now(),
                            })
                            maybe_print_progress(args, target_counter=target_counter, target_id=str(target["target_id"]))
    
                    for target, ctx in zero_contexts:
                        target_counter += 1
                        query = zero_context_query(model, device=args.device)
                        record, cand_result, ranked, score_metrics = score_target(
                            args, generator, bank_cache, candidate_embedding_cache, model, query, target,
                            generated_at=generated_at, model_digest=model_digest, context_policy_digest=context_policy_digest,
                            emit_full_record=write_full,
                        )
                        rank_stats = emit_target_outputs(
                            args,
                            pred_f=pred_f,
                            compact_f=compact_f,
                            compact_aggregator=compact_aggregator,
                            record=record,
                            cand_result=cand_result,
                            ranked=ranked,
                            target=target,
                            ctx=ctx,
                            generated_at=generated_at,
                            model_digest=model_digest,
                            context_policy_digest=context_policy_digest,
                            context_policy_label="zero_context_no_history_fallback",
                            model_inference_policy="zero_query_tie_break_by_candidate_id",
                        )
                        if args.output_target_ids:
                            with open(args.output_target_ids, "a" if args.resume or target_counter > 1 else "w", encoding="utf-8") as ids_f:
                                ids_f.write(str(target["target_id"]) + "\n")
                        positive_rank = rank_stats["pessimistic_rank"]
                        log_json(log_f, {
                            "progress": "target_done",
                            "target_id": target["target_id"],
                            "target_index": target_counter,
                            "part_file": part_file,
                            "source_row_index": source_row_index,
                            "user_id": history.user_id,
                            "candidate_count": len(cand_result.candidate_item_ids),
                            "candidate_set_digest": cand_result.candidate_set_digest,
                            "context_event_count_available": 0,
                            "context_event_count_used": 0,
                            "context_policy_label": "zero_context_no_history_fallback",
                            "model_inference_policy": "zero_query_tie_break_by_candidate_id",
                            "encoder_position_extracted": None,
                            "context_checksum": ctx.checksum,
                            "positive_item_rank_debug_only": positive_rank,
                            "candidate_scoring": score_metrics,
                            "candidate_embedding_cache": candidate_embedding_cache.snapshot(),
                            "at": utc_now(),
                        })
                        maybe_print_progress(args, target_counter=target_counter, target_id=str(target["target_id"]))

                    if long_contexts:
                        # P23 cannot represent the full prefix. Score latest-200 windows and make the
                        # policy explicit in logs/docs; the prediction JSONL remains contract-clean.
                        long_windows = [list(ctx.events)[-args.max_sequence_length :] for _, ctx in long_contexts]
                        tensors = events_to_batched_tensors(long_windows, max_sequence_length=args.max_sequence_length, device=args.device)
                        positions = {
                            str(target["target_id"]): (idx, len(long_windows[idx]) - 1)
                            for idx, (target, _) in enumerate(long_contexts)
                        }
                        queries = encode_positions_batched(model, tensors, positions)
                        encoder_passes += 1
                        log_json(log_f, {
                            "progress": "history_encoded",
                            "context_policy_label": "latest_200_due_legacy_p23_max_sequence_length",
                            "part_file": part_file,
                            "source_row_index": source_row_index,
                            "user_id": history.user_id,
                            "targets_for_history": len(long_contexts),
                            "encoder_passes_for_history": 1,
                            "batched_windows": len(long_contexts),
                            "at": utc_now(),
                        })
                        for target, ctx in long_contexts:
                            target_counter += 1
                            long_latest_200_targets += 1
                            query = queries[str(target["target_id"])]
                            window_len = min(ctx.event_count, args.max_sequence_length)
                            record, cand_result, ranked, score_metrics = score_target(
                                args, generator, bank_cache, candidate_embedding_cache, model, query, target,
                                generated_at=generated_at, model_digest=model_digest, context_policy_digest=context_policy_digest,
                                emit_full_record=write_full,
                            )
                            rank_stats = emit_target_outputs(
                                args,
                                pred_f=pred_f,
                                compact_f=compact_f,
                                compact_aggregator=compact_aggregator,
                                record=record,
                                cand_result=cand_result,
                                ranked=ranked,
                                target=target,
                                ctx=ctx,
                                generated_at=generated_at,
                                model_digest=model_digest,
                                context_policy_digest=context_policy_digest,
                                context_policy_label="latest_200_due_legacy_p23_max_sequence_length",
                                model_inference_policy="latest_200_window_batched_causal_hstu_forward",
                            )
                            if args.output_target_ids:
                                with open(args.output_target_ids, "a" if args.resume or target_counter > 1 else "w", encoding="utf-8") as ids_f:
                                    ids_f.write(str(target["target_id"]) + "\n")
                            positive_rank = rank_stats["pessimistic_rank"]
                            log_json(log_f, {
                                "progress": "target_done",
                                "target_id": target["target_id"],
                                "target_index": target_counter,
                                "part_file": part_file,
                                "source_row_index": source_row_index,
                                "user_id": history.user_id,
                                "candidate_count": len(cand_result.candidate_item_ids),
                                "candidate_set_digest": cand_result.candidate_set_digest,
                                "context_event_count_available": ctx.event_count,
                                "context_event_count_used": window_len,
                                "context_policy_label": "latest_200_due_legacy_p23_max_sequence_length",
                                "model_inference_policy": "latest_200_window_batched_causal_hstu_forward",
                                "encoder_position_extracted": window_len - 1,
                                "context_checksum": ctx.checksum,
                                "positive_item_rank_debug_only": positive_rank,
                                "candidate_scoring": score_metrics,
                                "candidate_embedding_cache": candidate_embedding_cache.snapshot(),
                                "at": utc_now(),
                            })
                            maybe_print_progress(args, target_counter=target_counter, target_id=str(target["target_id"]))
    
        elapsed_s = time.time() - started
        if target_counter == 0:
            raise RuntimeError("no targets to score after filtering/resume")
        if write_compact:
            metrics_payload = compact_aggregator.result(
                created_at=utc_now(),
                inputs={
                    "target_source": args.target_jsonl or args.target_sidecar_glob,
                    "output_compact": args.output_compact,
                    "output_predictions": args.output_predictions if write_full else None,
                    "model_submission_id": args.model_submission_id,
                    "prediction_run_id": args.prediction_run_id,
                },
            )
            Path(args.output_metrics_json).write_text(
                json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        log_json(log_f, {
            "progress": "run_done",
            "targets_scored": target_counter,
            "histories_loaded": loaded_histories,
            "target_batches": batch_counter,
            "encoder_passes": encoder_passes,
            "short_full_available_targets": short_full_available_targets,
            "long_latest_200_targets": long_latest_200_targets,
            "compact_metrics_json": args.output_metrics_json if write_compact else None,
            "candidate_embedding_cache": candidate_embedding_cache.snapshot(),
            "elapsed_s": elapsed_s,
            "targets_per_hour": (target_counter / elapsed_s * 3600.0) if elapsed_s > 0 else None,
            "at": utc_now(),
        })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
