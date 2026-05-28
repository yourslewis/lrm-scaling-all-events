#!/usr/bin/env python3
"""End-to-end bank-major fast proxy evaluator for LRM-v001 selected banks.

This runner is execution-side only. It keeps the frozen v001 target/candidate
contract intact, but avoids the old target-major 10k-candidate scoring loop:

1. stream/filter target sidecars by a selected bank subset;
2. materialize contexts and encode query vectors once;
3. spool compact query batches grouped by (target domain, negative bank);
4. load the raw 384-d selected-bank cache, project one bank at a time;
5. score each group with large query @ bank.T matmuls;
6. write compact_predictions.jsonl and compact_metrics.json.

It intentionally does not emit official full prediction JSONL. Use the existing
sequential runner when the full lrm_prediction_record_v001 audit artifact is
required.
"""
from __future__ import annotations

import argparse
import collections
from contextlib import ExitStack
from dataclasses import dataclass
import datetime as dt
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_embedding_cache import score_candidate_set_from_query  # noqa: E402
from compact_metrics import (  # noqa: E402
    StreamingMetricAggregator,
    make_compact_record,
    pessimistic_rank_from_ranked,
)
from fast_proxy_bank_scorer import BankMajorTarget, score_bank_major_compact_records  # noqa: E402
from raw_candidate_bank_cache import RawCandidateBankCache  # noqa: E402
import sequential_submission_infer as seq  # noqa: E402


QUERY_CACHE_SCHEMA = "lrm_v001_fast_proxy_query_cache_v001"
QUERY_BATCH_SCHEMA = "lrm_v001_fast_proxy_query_batch_v001"
ENTRYPOINT = "proposed_2-mmoe_ple/infer/lrm_v001/fast_proxy_eval_runner.py"


@dataclass(frozen=True)
class FastContext:
    events: list[dict[str, Any]]
    checksum: str | None

    @property
    def event_count(self) -> int:
        return len(self.events)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dump_line(f, obj: Mapping[str, Any]) -> None:
    f.write(json.dumps(dict(obj), sort_keys=True, separators=(",", ":")) + "\n")


def _target_group(target: Mapping[str, Any]) -> tuple[int, int]:
    return int(target["target_canonical_domain_id"]), int(target["negative_bank_id"])


def _target_minimal(target: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only fields needed for compact metrics/scoring and collision repair."""
    return {
        "benchmark_id": target.get("benchmark_id"),
        "target_id": str(target["target_id"]),
        "target_event_id": str(target.get("target_event_id", target["target_id"])),
        "user_id": str(target["user_id"]),
        "target_ts": target.get("target_ts", target.get("target_time")),
        "target_canonical_domain_id": int(target["target_canonical_domain_id"]),
        "target_domain": target.get("target_domain"),
        "target_event_type": target.get("target_event_type"),
        "candidate_protocol_label": target["candidate_protocol_label"],
        "candidate_set_id": target["candidate_set_id"],
        "candidate_set_digest": target.get("candidate_set_digest"),
        "negative_bank_id": int(target["negative_bank_id"]),
        "bank_selection_seed_material_digest": target.get("bank_selection_seed_material_digest"),
        "positive_item_id": str(target["positive_item_id"]),
        "headline_slices": list(target.get("headline_slices") or []),
        "diagnostic_buckets": dict(target.get("diagnostic_buckets") or {}),
        "raw_context_event_count": target.get("raw_context_event_count"),
        "context_reader_ref": target.get("context_reader_ref"),
    }


def _safe_torch_load(path: Path):
    import torch  # type: ignore

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class QueryGroupSpool:
    """Append query tensors and compact target metadata into per-bank .pt batches."""

    def __init__(self, root: str | Path, *, batch_size: int, dtype: str) -> None:
        self.root = Path(root)
        self.batch_size = max(1, int(batch_size))
        self.dtype = dtype
        self.buffers: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
        self.counts: collections.Counter[tuple[int, int]] = collections.Counter()
        self.batch_counts: collections.Counter[tuple[int, int]] = collections.Counter()
        self.total = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        *,
        group: tuple[int, int],
        query,
        target: Mapping[str, Any],
        context_checksum: str | None,
        context_policy_label: str,
        model_inference_policy: str,
    ) -> None:
        import torch  # type: ignore

        q = query.detach().cpu().reshape(-1).float()
        if self.dtype == "float16":
            q = q.to(dtype=torch.float16)
        elif self.dtype != "float32":
            raise ValueError(f"unsupported query cache dtype: {self.dtype}")
        self.buffers[group].append(
            {
                "query": q,
                "target": _target_minimal(target),
                "context_checksum": context_checksum,
                "context_policy_label": context_policy_label,
                "model_inference_policy": model_inference_policy,
            }
        )
        self.counts[group] += 1
        self.total += 1
        if len(self.buffers[group]) >= self.batch_size:
            self.flush_group(group)

    def group_dir(self, group: tuple[int, int]) -> Path:
        domain_id, bank_id = group
        return self.root / "groups" / f"domain_{domain_id}" / f"bank_{bank_id:04d}"

    def flush_group(self, group: tuple[int, int]) -> None:
        import torch  # type: ignore

        rows = self.buffers.get(group) or []
        if not rows:
            return
        group_dir = self.group_dir(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        batch_idx = self.batch_counts[group]
        self.batch_counts[group] += 1
        payload = {
            "schema": QUERY_BATCH_SCHEMA,
            "group": {"domain_id": int(group[0]), "bank_id": int(group[1])},
            "queries": torch.stack([row["query"] for row in rows], dim=0),
            "targets": [row["target"] for row in rows],
            "context_checksums": [row["context_checksum"] for row in rows],
            "context_policy_labels": [row["context_policy_label"] for row in rows],
            "model_inference_policies": [row["model_inference_policy"] for row in rows],
        }
        tmp = group_dir / f"batch_{batch_idx:06d}.pt.tmp"
        out = group_dir / f"batch_{batch_idx:06d}.pt"
        torch.save(payload, tmp)
        tmp.replace(out)
        rows.clear()

    def close(self) -> None:
        for group in list(self.buffers):
            self.flush_group(group)

    def write_manifest(self, *, args, generated_at: str, model_digest: str, context_policy_digest: str, stats: Mapping[str, Any]) -> dict[str, Any]:
        groups = [
            {
                "domain_id": int(domain_id),
                "bank_id": int(bank_id),
                "target_count": int(count),
                "batch_count": int(self.batch_counts[(domain_id, bank_id)]),
                "path": str(self.group_dir((domain_id, bank_id)).relative_to(self.root)),
            }
            for (domain_id, bank_id), count in sorted(self.counts.items())
        ]
        manifest = {
            "schema": QUERY_CACHE_SCHEMA,
            "created_at": generated_at,
            "entrypoint": ENTRYPOINT,
            "target_source": args.target_jsonl or args.target_sidecar_glob,
            "selected_bank_subset": args.selected_bank_subset,
            "selected_bank_count": sum(len(v) for v in (args._selected_bank_filter or {}).values()),
            "query_cache_dtype": self.dtype,
            "query_batch_size": self.batch_size,
            "model_digest": model_digest,
            "context_policy_digest": context_policy_digest,
            "target_count": int(self.total),
            "group_count": len(groups),
            "groups": groups,
            "stats": dict(stats),
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest


def iter_query_groups(query_cache_dir: str | Path) -> Iterable[tuple[tuple[int, int], list[Path], dict[str, Any]]]:
    root = Path(query_cache_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != QUERY_CACHE_SCHEMA:
        raise ValueError(f"unsupported query cache schema: {manifest.get('schema')}")
    for group in manifest.get("groups") or []:
        group_key = (int(group["domain_id"]), int(group["bank_id"]))
        group_dir = root / group["path"]
        paths = sorted(group_dir.glob("batch_*.pt"))
        if len(paths) != int(group.get("batch_count") or len(paths)):
            raise RuntimeError(f"query cache batch count mismatch for {group_key}: {len(paths)} files vs manifest {group.get('batch_count')}")
        yield group_key, paths, group


def load_query_group(batch_paths: Sequence[Path], *, device: str):
    import torch  # type: ignore

    queries = []
    targets: list[dict[str, Any]] = []
    checksums: list[str | None] = []
    context_labels: list[str] = []
    policy_labels: list[str] = []
    for path in batch_paths:
        payload = _safe_torch_load(path)
        if payload.get("schema") != QUERY_BATCH_SCHEMA:
            raise ValueError(f"unsupported query batch schema in {path}: {payload.get('schema')}")
        queries.append(payload["queries"].float())
        targets.extend(payload["targets"])
        checksums.extend(payload["context_checksums"])
        context_labels.extend(payload["context_policy_labels"])
        policy_labels.extend(payload["model_inference_policies"])
    if not queries:
        raise RuntimeError("empty query group")
    return torch.cat(queries, dim=0).to(device=device, dtype=torch.float32), targets, checksums, context_labels, policy_labels


def materialize_fast_contexts_from_history(
    reader_mod,
    reader,
    history,
    targets: Sequence[Mapping[str, Any]],
    *,
    checksum_mode: str,
    validate_context_count: bool,
) -> list[tuple[Mapping[str, Any], FastContext]]:
    """Fast context materialization with optional official RawContext checksum.

    The default checksum_mode='none' avoids per-target stable JSON hashing of the
    full prefix. That is intentional for million-target proxy runs where the
    compact output is an execution-derived artifact, not the official audit JSONL.
    """
    if not targets:
        return []
    target_dts = [(target, seq._target_datetime(reader_mod, history, target)) for target in targets]
    valid_events = [
        (ev.event_time, ev.model_visible_dict())
        for ev in history.events
        if ev.fields.get("timestamp_quality_status") == "valid" and reader_mod.T1 <= ev.event_time
    ]
    chronological = not any(valid_events[i][0] > valid_events[i + 1][0] for i in range(len(valid_events) - 1))

    def build(target: Mapping[str, Any], target_dt) -> FastContext:
        events = [ev.model_visible_dict() for ev in history.events if ev.fields.get("timestamp_quality_status") == "valid" and reader_mod.T1 <= ev.event_time < target_dt]
        return _context_from_events(reader_mod, reader, history, target, target_dt, events, checksum_mode=checksum_mode, validate_context_count=validate_context_count)

    if not chronological:
        return [(target, build(target, target_dt)) for target, target_dt in target_dts]

    out_by_id: dict[str, FastContext] = {}
    prefix: list[dict[str, Any]] = []
    event_idx = 0
    for target, target_dt in sorted(target_dts, key=lambda item: (item[1], str(item[0].get("target_id")))):
        while event_idx < len(valid_events) and valid_events[event_idx][0] < target_dt:
            prefix.append(valid_events[event_idx][1])
            event_idx += 1
        out_by_id[str(target["target_id"])] = _context_from_events(
            reader_mod,
            reader,
            history,
            target,
            target_dt,
            list(prefix),
            checksum_mode=checksum_mode,
            validate_context_count=validate_context_count,
        )
    return [(target, out_by_id[str(target["target_id"])]) for target in targets]


def _context_from_events(
    reader_mod,
    reader,
    history,
    target: Mapping[str, Any],
    target_dt,
    events: list[dict[str, Any]],
    *,
    checksum_mode: str,
    validate_context_count: bool,
) -> FastContext:
    if validate_context_count:
        supplied_count = target.get("raw_context_event_count")
        if supplied_count is not None and int(supplied_count) != len(events):
            raise ValueError(
                f"raw_context_event_count mismatch for {target.get('target_id')}: "
                f"target={supplied_count} materialized={len(events)}"
            )
    if checksum_mode == "none":
        return FastContext(events=events, checksum=None)
    if checksum_mode == "contract":
        ctx = seq._build_raw_context(reader_mod, reader, history, target, target_dt, events)
        return FastContext(events=list(ctx.events), checksum=ctx.checksum)
    raise ValueError(f"unsupported context checksum mode: {checksum_mode}")


def validate_query_cache_manifest(args, manifest: Mapping[str, Any], *, model_digest: str, context_policy_digest: str) -> None:
    if manifest.get("schema") != QUERY_CACHE_SCHEMA:
        raise RuntimeError(f"unsupported query cache manifest schema: {manifest.get('schema')}")
    if manifest.get("model_digest") != model_digest:
        raise RuntimeError("query cache model digest does not match current checkpoint; rebuild query cache")
    if manifest.get("context_policy_digest") != context_policy_digest:
        raise RuntimeError("query cache context policy digest does not match current context policy; rebuild query cache")
    if str(manifest.get("selected_bank_subset")) != str(args.selected_bank_subset):
        raise RuntimeError("query cache selected-bank subset path does not match current run; rebuild query cache")


def build_query_cache(args, *, model, reader_mod, history_reader, model_digest: str, context_policy_digest: str, log_f) -> dict[str, Any]:
    import torch  # type: ignore

    generated_at = utc_now()
    started = time.time()
    if args.query_cache_dir.exists() and any(args.query_cache_dir.iterdir()):
        if args.reuse_query_cache and (args.query_cache_dir / "manifest.json").exists():
            manifest = json.loads((args.query_cache_dir / "manifest.json").read_text(encoding="utf-8"))
            validate_query_cache_manifest(args, manifest, model_digest=model_digest, context_policy_digest=context_policy_digest)
            return manifest
        if not args.force_rebuild_query_cache:
            raise RuntimeError(f"query cache dir is non-empty; pass --force-rebuild-query-cache or --reuse-query-cache: {args.query_cache_dir}")
        shutil.rmtree(args.query_cache_dir)
    args.query_cache_dir.mkdir(parents=True, exist_ok=True)
    spool = QueryGroupSpool(args.query_cache_dir, batch_size=args.query_cache_batch_size, dtype=args.query_cache_dtype)

    target_counter = 0
    loaded_histories = 0
    encoder_passes = 0
    short_full_available_targets = 0
    long_latest_200_targets = 0
    zero_context_targets = 0
    batch_counter = 0

    _json_dump_line(log_f, {"progress": "query_cache_start", "query_cache_dir": str(args.query_cache_dir), "at": generated_at})

    for targets in seq.iter_target_batches(args):
        batch_counter += 1
        by_part = seq.grouped_targets(targets)
        _json_dump_line(
            log_f,
            {
                "progress": "query_target_batch_start",
                "batch_index": batch_counter,
                "batch_target_count": len(targets),
                "batch_part_count": len(by_part),
                "targets_cached": target_counter,
                "at": utc_now(),
            },
        )
        if args.log_flush_every == 1 or (args.log_flush_every > 1 and batch_counter % args.log_flush_every == 0):
            log_f.flush()
        for part_file in sorted(by_part):
            found = seq.selected_histories_for_part(
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
                target_contexts = materialize_fast_contexts_from_history(
                    reader_mod,
                    history_reader,
                    history,
                    user_targets,
                    checksum_mode=args.context_checksum_mode,
                    validate_context_count=args.validate_context_count,
                )
                zero_contexts = [(t, c) for t, c in target_contexts if c.event_count == 0]
                short_contexts = [(t, c) for t, c in target_contexts if 0 < c.event_count <= args.max_sequence_length]
                long_contexts = [(t, c) for t, c in target_contexts if c.event_count > args.max_sequence_length]

                if short_contexts:
                    max_ctx = max(short_contexts, key=lambda tc: tc[1].event_count)[1]
                    tensors = seq.events_to_tensors(list(max_ctx.events), max_sequence_length=args.max_sequence_length, device=args.device)
                    positions = {str(t["target_id"]): c.event_count - 1 for t, c in short_contexts}
                    queries = seq.encode_positions_one_pass(model, tensors, positions)
                    encoder_passes += 1
                    for target, ctx in short_contexts:
                        target_counter += 1
                        short_full_available_targets += 1
                        spool.add(
                            group=_target_group(target),
                            query=queries[str(target["target_id"])],
                            target=target,
                            context_checksum=ctx.checksum,
                            context_policy_label="full_available_history_for_p23",
                            model_inference_policy="one_causal_hstu_forward_per_history_extract_position",
                        )

                if zero_contexts:
                    query = seq.zero_context_query(model, device=args.device)
                    for target, ctx in zero_contexts:
                        target_counter += 1
                        zero_context_targets += 1
                        spool.add(
                            group=_target_group(target),
                            query=query,
                            target=target,
                            context_checksum=ctx.checksum,
                            context_policy_label="zero_context_no_history_fallback",
                            model_inference_policy="zero_query_tie_break_by_candidate_id",
                        )

                if long_contexts:
                    long_windows = [list(ctx.events)[-args.max_sequence_length :] for _, ctx in long_contexts]
                    tensors = seq.events_to_batched_tensors(long_windows, max_sequence_length=args.max_sequence_length, device=args.device)
                    positions = {
                        str(target["target_id"]): (idx, len(long_windows[idx]) - 1)
                        for idx, (target, _) in enumerate(long_contexts)
                    }
                    queries = seq.encode_positions_batched(model, tensors, positions)
                    encoder_passes += 1
                    for target, ctx in long_contexts:
                        target_counter += 1
                        long_latest_200_targets += 1
                        spool.add(
                            group=_target_group(target),
                            query=queries[str(target["target_id"])],
                            target=target,
                            context_checksum=ctx.checksum,
                            context_policy_label="latest_200_due_legacy_p23_max_sequence_length",
                            model_inference_policy="latest_200_window_batched_causal_hstu_forward",
                        )

                if args.stdout_progress_every and target_counter and target_counter % args.stdout_progress_every == 0:
                    print(json.dumps({"progress": "queries_cached", "targets": target_counter, "at": utc_now()}), flush=True)

    spool.close()
    if target_counter == 0:
        raise RuntimeError("no targets to score after selected-bank filtering/resume")
    stats = {
        "targets_cached": target_counter,
        "histories_loaded": loaded_histories,
        "target_batches": batch_counter,
        "encoder_passes": encoder_passes,
        "short_full_available_targets": short_full_available_targets,
        "long_latest_200_targets": long_latest_200_targets,
        "zero_context_targets": zero_context_targets,
        "elapsed_s": time.time() - started,
    }
    manifest = spool.write_manifest(
        args=args,
        generated_at=generated_at,
        model_digest=model_digest,
        context_policy_digest=context_policy_digest,
        stats=stats,
    )
    _json_dump_line(log_f, {"progress": "query_cache_done", **stats, "query_cache_manifest": str(args.query_cache_dir / "manifest.json"), "at": utc_now()})
    log_f.flush()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def load_bank_artifact(bank_cache: dict[int, Any], *, args, generator, domain_id: int):
    if domain_id not in bank_cache:
        bank_path = Path(args.bank_root) / "banks" / f"domain_{domain_id}_banks.production.json"
        bank_cache[domain_id] = generator.load_bank_artifact(bank_path)
    return bank_cache[domain_id]


def _raw_cache_enabled(cache: RawCandidateBankCache) -> bool:
    return bool(getattr(cache, "enabled", True))


def _project_item_embeddings(model, ids: Sequence[str], *, chunk_size: int, device: str):
    """Return normalized projected embeddings for ids, preserving input order."""
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    if not ids:
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(ids), max(1, int(chunk_size))):
            cur = [int(x) for x in ids[start : start + max(1, int(chunk_size))]]
            cand = torch.tensor(cur, dtype=torch.long, device=device)
            emb = model.model._embedding_module.get_item_embeddings(cand).float()
            chunks.append(F.normalize(emb, p=2, dim=-1).detach())
    return torch.cat(chunks, dim=0)


def _derive_replacement_ids(
    *,
    args,
    generator,
    bank_artifact,
    target: Mapping[str, Any],
    base_candidate_ids: Sequence[str],
    base_id_set: set[str],
) -> list[str]:
    positive_id = str(target["positive_item_id"])
    if positive_id not in base_id_set:
        if args.candidate_check_mode == "full":
            cand = generator.generate_candidates_for_target(target, bank_artifact)
            _check_candidate_result(args, generator, target, cand)
            return [str(x) for x in getattr(cand, "replacement_item_ids", [])]
        return []

    if args.candidate_check_mode == "none":
        raise RuntimeError(
            f"positive item collides with base bank for target {target['target_id']} but --candidate-check-mode none cannot derive replacements"
        )
    cand = generator.generate_candidates_for_target(target, bank_artifact)
    _check_candidate_result(args, generator, target, cand)
    replacement_ids = [str(x) for x in getattr(cand, "replacement_item_ids", [])]
    if not replacement_ids:
        raise RuntimeError(f"positive collision for target {target['target_id']} produced no replacement ids")
    expected_count = len(base_candidate_ids) - 1 + 1 + len(replacement_ids)
    if hasattr(cand, "candidate_item_ids") and len(cand.candidate_item_ids) != expected_count:
        raise RuntimeError(
            f"candidate count mismatch for collision target {target['target_id']}: "
            f"fast_count={expected_count} generator_count={len(cand.candidate_item_ids)}"
        )
    return replacement_ids


def _check_candidate_result(args, generator, target: Mapping[str, Any], cand) -> None:
    if args.validate_candidate_generation:
        errors = generator.validate_generated_candidates(cand)
        if errors:
            raise RuntimeError(f"candidate generation failed for {target['target_id']}: {errors}")
    expected_digest = target.get("candidate_set_digest")
    actual_digest = getattr(cand, "candidate_set_digest", None)
    if actual_digest is not None and expected_digest is not None and str(actual_digest) != str(expected_digest):
        raise RuntimeError(f"candidate digest mismatch for {target['target_id']}: {actual_digest} != {expected_digest}")


def prepare_bank_major_targets(
    *,
    args,
    model,
    generator,
    bank_artifact,
    targets: Sequence[Mapping[str, Any]],
    checksums: Sequence[str | None],
    context_labels: Sequence[str],
    policy_labels: Sequence[str],
    base_candidate_ids: Sequence[str],
    bank_embeddings,
) -> tuple[list[BankMajorTarget], dict[str, Any]]:
    import torch  # type: ignore

    base_ids = [str(x) for x in base_candidate_ids]
    base_id_set = set(base_ids)
    base_index = {cid: idx for idx, cid in enumerate(base_ids)}
    replacement_by_idx: dict[int, list[str]] = {}
    online_ids: list[str] = []
    online_slots: list[tuple[int, str, int | None]] = []  # (target_idx, kind, repl_idx)
    collision_count = 0

    for idx, target in enumerate(targets):
        pos = str(target["positive_item_id"])
        repl = _derive_replacement_ids(
            args=args,
            generator=generator,
            bank_artifact=bank_artifact,
            target=target,
            base_candidate_ids=base_ids,
            base_id_set=base_id_set,
        )
        replacement_by_idx[idx] = repl
        if pos in base_index:
            collision_count += 1
        else:
            online_slots.append((idx, "positive", None))
            online_ids.append(pos)
        for repl_idx, rid in enumerate(repl):
            online_slots.append((idx, "replacement", repl_idx))
            online_ids.append(rid)

    projected = _project_item_embeddings(model, online_ids, chunk_size=args.extra_embedding_chunk_size, device=args.device)
    projected_by_slot: dict[tuple[int, str, int | None], Any] = {}
    for row_idx, slot in enumerate(online_slots):
        projected_by_slot[slot] = projected[row_idx]

    specs: list[BankMajorTarget] = []
    width = int(bank_embeddings.shape[1])
    empty_repl = torch.empty((0, width), dtype=torch.float32, device=args.device)
    for idx, target in enumerate(targets):
        pos = str(target["positive_item_id"])
        if pos in base_index:
            pos_emb = bank_embeddings[base_index[pos]]
        else:
            pos_emb = projected_by_slot[(idx, "positive", None)]
        repl_ids = replacement_by_idx[idx]
        if repl_ids:
            repl_emb = torch.stack([projected_by_slot[(idx, "replacement", ridx)] for ridx in range(len(repl_ids))], dim=0)
        else:
            repl_emb = empty_repl
        specs.append(
            BankMajorTarget(
                target=target,
                positive_item_id=pos,
                positive_embedding=pos_emb,
                replacement_item_ids=repl_ids,
                replacement_embeddings=repl_emb,
                candidate_set_digest=str(target.get("candidate_set_digest")),
                context_checksum=checksums[idx],
                context_policy_label=context_labels[idx],
                model_inference_policy=policy_labels[idx],
            )
        )
    return specs, {
        "targets": len(targets),
        "positive_collisions": collision_count,
        "online_extra_ids": len(online_ids),
        "replacement_targets": sum(1 for vals in replacement_by_idx.values() if vals),
    }


def run_debug_equivalence(
    *,
    args,
    model,
    generator,
    bank_artifact,
    query,
    target: Mapping[str, Any],
    fast_record: Mapping[str, Any],
    context_checksum: str | None,
    context_policy_label: str,
    model_inference_policy: str,
    generated_at: str,
    model_digest: str,
    context_policy_digest: str,
) -> dict[str, Any]:
    cand = generator.generate_candidates_for_target(target, bank_artifact)
    _check_candidate_result(args, generator, target, cand)
    ranked, timing = score_candidate_set_from_query(
        model,
        query.reshape(1, -1),
        cand,
        chunk_size=args.extra_embedding_chunk_size,
        device=args.device,
        candidate_cache=None,
    )
    rank_stats = pessimistic_rank_from_ranked(ranked, str(target["positive_item_id"]), k=args.compact_top_k)
    old = make_compact_record(
        target=target,
        ranked=ranked,
        rank_stats=rank_stats,
        top_k=args.compact_top_k,
        model_submission_id=args.model_submission_id,
        prediction_run_id=args.prediction_run_id,
        generated_at=generated_at,
        model_digest=model_digest,
        context_policy_digest=context_policy_digest,
        candidate_count=len(cand.candidate_item_ids),
        candidate_set_digest=getattr(cand, "candidate_set_digest", target.get("candidate_set_digest")),
        context_checksum=context_checksum,
        context_policy_label=context_policy_label,
        model_inference_policy=model_inference_policy,
        include_full_score_order_digest=False,
    )
    keys = ["positive_score", "greater_score_count", "equal_score_nonpositive_count", "pessimistic_rank", f"hit_at_{args.compact_top_k}", f"ndcg_at_{args.compact_top_k}", "reciprocal_rank"]
    diffs: list[str] = []
    for key in keys:
        a = old["rank_stats"][key]
        b = fast_record["rank_stats"][key]
        if isinstance(a, float) or isinstance(b, float):
            if not math.isclose(float(a), float(b), rel_tol=args.equivalence_tolerance, abs_tol=args.equivalence_tolerance):
                diffs.append(f"rank_stats.{key}:{a}!={b}")
        elif a != b:
            diffs.append(f"rank_stats.{key}:{a}!={b}")
    old_top = old["top_k"]
    fast_top = fast_record["top_k"]
    if [r["candidate_id"] for r in old_top] != [r["candidate_id"] for r in fast_top]:
        diffs.append("top_k.candidate_id")
    for idx, (old_row, fast_row) in enumerate(zip(old_top, fast_top)):
        if old_row["rank"] != fast_row["rank"]:
            diffs.append(f"top_k[{idx}].rank")
        if not math.isclose(float(old_row["score"]), float(fast_row["score"]), rel_tol=args.equivalence_tolerance, abs_tol=args.equivalence_tolerance):
            diffs.append(f"top_k[{idx}].score:{old_row['score']}!={fast_row['score']}")
    return {
        "target_id": target["target_id"],
        "ok": not diffs,
        "diffs": diffs,
        "sequential_timing": timing,
    }


def score_query_cache(args, *, model, generator, model_digest: str, context_policy_digest: str, log_f) -> dict[str, Any]:
    import torch  # type: ignore

    started = time.time()
    generated_at = utc_now()
    raw_cache = RawCandidateBankCache(
        cache_dir=args.raw_bank_cache_dir,
        model_digest=model_digest,
        max_banks=args.candidate_cache_max_banks,
        device=args.device,
        placement=args.raw_bank_cache_placement,
        timing_sync_cuda=args.timing_sync_cuda,
    )
    if not _raw_cache_enabled(raw_cache):
        raise RuntimeError("raw bank cache did not initialize as enabled")

    bank_artifacts: dict[int, Any] = {}
    aggregator = StreamingMetricAggregator(k=args.compact_top_k)
    target_count = 0
    group_count = 0
    equivalence_reports: list[dict[str, Any]] = []
    group_metrics: list[dict[str, Any]] = []

    Path(args.output_compact).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    if args.equivalence_output:
        Path(args.equivalence_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_compact, "w", encoding="utf-8") as compact_f:
        for group, batch_paths, group_meta in iter_query_groups(args.query_cache_dir):
            group_count += 1
            domain_id, bank_id = group
            group_started = time.time()
            queries, targets, checksums, context_labels, policy_labels = load_query_group(batch_paths, device=args.device)
            bank_artifact = load_bank_artifact(bank_artifacts, args=args, generator=generator, domain_id=domain_id)
            entry, cache_info = raw_cache.get_bank(
                model=model,
                generator_mod=generator,
                bank_artifact=bank_artifact,
                domain_id=domain_id,
                bank_id=bank_id,
            )
            specs, prep_metrics = prepare_bank_major_targets(
                args=args,
                model=model,
                generator=generator,
                bank_artifact=bank_artifact,
                targets=targets,
                checksums=checksums,
                context_labels=context_labels,
                policy_labels=policy_labels,
                base_candidate_ids=entry["candidate_ids"],
                bank_embeddings=entry["embeddings"],
            )
            records, metrics = score_bank_major_compact_records(
                queries=queries,
                bank_embeddings=entry["embeddings"],
                base_candidate_ids=entry["candidate_ids"],
                targets=specs,
                top_k=args.compact_top_k,
                model_submission_id=args.model_submission_id,
                prediction_run_id=args.prediction_run_id,
                generated_at=generated_at,
                model_digest=model_digest,
                context_policy_digest=context_policy_digest,
                query_chunk_size=args.score_query_chunk_size,
                device=args.device,
            )
            for idx, record in enumerate(records):
                compact_f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                aggregator.add_compact_record(record)
                target_count += 1
                if args.debug_equivalence_targets and len(equivalence_reports) < args.debug_equivalence_targets:
                    report = run_debug_equivalence(
                        args=args,
                        model=model,
                        generator=generator,
                        bank_artifact=bank_artifact,
                        query=queries[idx],
                        target=targets[idx],
                        fast_record=record,
                        context_checksum=checksums[idx],
                        context_policy_label=context_labels[idx],
                        model_inference_policy=policy_labels[idx],
                        generated_at=generated_at,
                        model_digest=model_digest,
                        context_policy_digest=context_policy_digest,
                    )
                    equivalence_reports.append(report)
                    if not report["ok"] and args.equivalence_fail_fast:
                        raise RuntimeError(f"equivalence check failed: {report}")
                if args.output_flush_every and target_count % args.output_flush_every == 0:
                    compact_f.flush()
            if args.output_flush_every == 0:
                compact_f.flush()
            group_elapsed = time.time() - group_started
            group_payload = {
                "progress": "bank_group_done",
                "domain_id": domain_id,
                "bank_id": bank_id,
                "group_index": group_count,
                "group_targets": len(targets),
                "total_targets_scored": target_count,
                "raw_cache_event": cache_info.get("event"),
                "prepare": prep_metrics,
                "score": metrics,
                "group_elapsed_s": group_elapsed,
                "targets_per_hour_group": (len(targets) / group_elapsed * 3600.0) if group_elapsed > 0 else None,
                "raw_bank_cache": raw_cache.snapshot(),
                "at": utc_now(),
            }
            group_metrics.append({k: v for k, v in group_payload.items() if k != "raw_bank_cache"})
            _json_dump_line(log_f, group_payload)
            if args.log_flush_every == 1 or (args.log_flush_every > 1 and group_count % args.log_flush_every == 0):
                log_f.flush()
            if args.stdout_progress_every and group_count % args.stdout_progress_every == 0:
                print(json.dumps({"progress": "bank_group_done", "groups": group_count, "targets": target_count, "at": utc_now()}), flush=True)
            del queries, targets, specs, records
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    elapsed_s = time.time() - started
    if target_count == 0:
        raise RuntimeError("no query-cache targets were scored")
    metrics_payload = aggregator.result(
        created_at=utc_now(),
        inputs={
            "target_source": args.target_jsonl or args.target_sidecar_glob,
            "selected_bank_subset": args.selected_bank_subset,
            "raw_bank_cache_dir": args.raw_bank_cache_dir,
            "query_cache_dir": str(args.query_cache_dir),
            "output_compact": args.output_compact,
            "model_submission_id": args.model_submission_id,
            "prediction_run_id": args.prediction_run_id,
            "runner": ENTRYPOINT,
            "bank_major_group_count": group_count,
            "raw_bank_cache": raw_cache.snapshot(),
            "elapsed_s": elapsed_s,
            "targets_per_hour": (target_count / elapsed_s * 3600.0) if elapsed_s > 0 else None,
            "group_metrics": group_metrics,
        },
    )
    Path(args.output_metrics_json).write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    eq_summary = None
    if args.debug_equivalence_targets:
        eq_summary = {
            "schema": "lrm_v001_fast_proxy_equivalence_report_v001",
            "checked": len(equivalence_reports),
            "failed": sum(1 for row in equivalence_reports if not row["ok"]),
            "reports": equivalence_reports,
        }
        if args.equivalence_output:
            Path(args.equivalence_output).write_text(json.dumps(eq_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if eq_summary["failed"]:
            raise RuntimeError(f"equivalence check failures: {eq_summary['failed']} / {eq_summary['checked']}")

    done = {
        "progress": "run_done",
        "targets_scored": target_count,
        "bank_groups_scored": group_count,
        "compact_predictions": args.output_compact,
        "compact_metrics_json": args.output_metrics_json,
        "equivalence": eq_summary,
        "raw_bank_cache": raw_cache.snapshot(),
        "elapsed_s": elapsed_s,
        "targets_per_hour": (target_count / elapsed_s * 3600.0) if elapsed_s > 0 else None,
        "at": utc_now(),
    }
    _json_dump_line(log_f, done)
    log_f.flush()
    return done


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Fast bank-major compact proxy eval for LRM-v001 selected-bank subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-jsonl", help="optional JSONL target sample")
    group.add_argument("--target-sidecar-glob", help="production/eval target sidecar parquet glob")
    ap.add_argument("--target-id-file")
    ap.add_argument("--selected-bank-subset", required=True, help="fixed selected-bank subset manifest; filters targets")
    ap.add_argument("--max-targets", type=int)
    ap.add_argument("--target-batch-size", type=int, default=8192)
    ap.add_argument("--resume", action="store_true", help="accepted for iter_target_batches compatibility; query cache rebuild still owns resume")

    ap.add_argument("--history-prefix-source", required=True, help="canonical row-array root")
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--bank-generator", required=True)
    ap.add_argument("--history-reader", required=True)
    ap.add_argument("--source-root", default="/home/yourslewis/lrm-scaling-all-events")
    ap.add_argument("--gin-config-file", required=True)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--embedding-root", default="/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve")
    ap.add_argument("--context-policy", required=True)

    ap.add_argument("--raw-bank-cache-dir", required=True, help="raw selected-bank cache built by build_raw_candidate_bank_cache.py")
    ap.add_argument("--raw-bank-cache-placement", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--candidate-cache-max-banks", type=int, default=0, help="resident projected bank limit; 0 means all raw-cache banks")

    ap.add_argument("--model-submission-id", required=True)
    ap.add_argument("--prediction-run-id", required=True)
    ap.add_argument("--output-compact", required=True, help="compact_predictions.jsonl output")
    ap.add_argument("--output-metrics-json", required=True, help="compact_metrics.json output")
    ap.add_argument("--output-inference-log", required=True)
    ap.add_argument("--compact-top-k", type=int, default=10)

    ap.add_argument("--query-cache-dir", help="directory for grouped query vectors; default is output dir/query_cache")
    ap.add_argument("--query-cache-batch-size", type=int, default=1024)
    ap.add_argument("--query-cache-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--reuse-query-cache", action="store_true", help="skip query encoding if query-cache manifest exists")
    ap.add_argument("--force-rebuild-query-cache", action="store_true")
    ap.add_argument("--context-checksum-mode", choices=["none", "contract"], default="none", help="contract mode computes official RawContext checksums but is slower")
    ap.add_argument("--validate-context-count", action="store_true")
    ap.add_argument("--candidate-check-mode", choices=["none", "collisions", "full"], default="collisions", help="default avoids 10k candidate generation except positive-bank collisions")
    ap.add_argument("--validate-candidate-generation", action="store_true", help="run generator validation when candidate generation is invoked")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-sequence-length", type=int, default=200)
    ap.add_argument("--history-batch-size", type=int, default=128)
    ap.add_argument("--score-query-chunk-size", type=int, default=8192)
    ap.add_argument("--extra-embedding-chunk-size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--timing-sync-cuda", action="store_true")
    ap.add_argument("--log-flush-every", type=int, default=10, help="flush log every N group/batch events; 0 flushes only at close")
    ap.add_argument("--output-flush-every", type=int, default=0, help="flush compact output every N records; 0 relies on close/group flush")
    ap.add_argument("--stdout-progress-every", type=int, default=1, help="print progress every N query batches/bank groups; 0 disables")

    ap.add_argument("--debug-equivalence-targets", type=int, default=0, help="compare fast records to old sequential scorer for first N targets (max 1000)")
    ap.add_argument("--equivalence-output", help="JSON report for --debug-equivalence-targets")
    ap.add_argument("--equivalence-tolerance", type=float, default=1e-5)
    ap.add_argument("--equivalence-fail-fast", action="store_true")
    args = ap.parse_args(argv)

    if args.compact_top_k <= 0:
        raise SystemExit("--compact-top-k must be positive")
    if args.debug_equivalence_targets < 0 or args.debug_equivalence_targets > 1000:
        raise SystemExit("--debug-equivalence-targets must be between 0 and 1000")
    args.query_cache_dir = Path(args.query_cache_dir) if args.query_cache_dir else Path(args.output_compact).resolve().parent / "query_cache"
    args._selected_bank_filter = seq.load_selected_bank_subset(args.selected_bank_subset)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seq.configure_log_flush(args.log_flush_every)
    import torch  # type: ignore

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1] if ":" in args.device else 0))

    Path(args.output_inference_log).parent.mkdir(parents=True, exist_ok=True)
    generator = seq.load_module("banked_candidate_generator_v001", args.bank_generator)
    reader_mod = seq.load_module("history_prefix_reader_v001", args.history_reader)
    model = seq.build_model(args)
    history_reader = reader_mod.HistoryPrefixReader.open(
        canonical_root=args.history_prefix_source,
        split="eval",
        mode="eval_inference",
    )
    model_digest = seq.sha256_file(args.checkpoint_path)
    context_policy_digest = seq.sha256_file(args.context_policy)

    with ExitStack() as stack:
        log_f = stack.enter_context(open(args.output_inference_log, "a", encoding="utf-8"))
        _json_dump_line(
            log_f,
            {
                "progress": "run_start",
                "entrypoint": ENTRYPOINT,
                "target_source": args.target_jsonl or args.target_sidecar_glob,
                "selected_bank_subset": args.selected_bank_subset,
                "selected_bank_count": sum(len(v) for v in (args._selected_bank_filter or {}).values()),
                "raw_bank_cache_dir": args.raw_bank_cache_dir,
                "query_cache_dir": str(args.query_cache_dir),
                "candidate_check_mode": args.candidate_check_mode,
                "context_checksum_mode": args.context_checksum_mode,
                "compact_top_k": args.compact_top_k,
                "debug_equivalence_targets": args.debug_equivalence_targets,
                "at": utc_now(),
            },
        )
        if args.reuse_query_cache and (args.query_cache_dir / "manifest.json").exists():
            manifest = json.loads((args.query_cache_dir / "manifest.json").read_text(encoding="utf-8"))
            validate_query_cache_manifest(args, manifest, model_digest=model_digest, context_policy_digest=context_policy_digest)
            _json_dump_line(log_f, {"progress": "query_cache_reused", "target_count": manifest.get("target_count"), "group_count": manifest.get("group_count"), "at": utc_now()})
        else:
            build_query_cache(
                args,
                model=model,
                reader_mod=reader_mod,
                history_reader=history_reader,
                model_digest=model_digest,
                context_policy_digest=context_policy_digest,
                log_f=log_f,
            )
        score_query_cache(
            args,
            model=model,
            generator=generator,
            model_digest=model_digest,
            context_policy_digest=context_policy_digest,
            log_f=log_f,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
