#!/usr/bin/env python3
"""Minimal P23 -> lrm_benchmark_v001 smoke inference adapter.

This is a bounded smoke adapter, not a final production entrypoint. It loads the
legacy P23 checkpoint, reads v001 model-facing history prefixes, regenerates the
banked 10k candidate set for each supplied target row, and emits prediction JSONL.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def event_type_id_for_context(ev: dict[str, Any], max_supported: int = 13) -> int:
    """Map v001 event_type_id to the legacy P23 embedding range.

    The checkpoint's event-type embedding has rows 0..13. v001 may expose newer
    ids (e.g. MSN=14). Unsupported ids are mapped to UNK=0 and counted in logs.
    """
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

    # v001 approved static_v3_full_preserve embedding universe; one shard per domain.
    shard_sizes = {
        0: 4_489_403,
        1: 176_044_856,
        2: 47_742_266,
        3: 4_692_056,
        4: 945_621,
    }
    dataset = RecoDataset(
        dataset_name="lrm_benchmark_v001_v3_full_preserve_smoke",
        max_sequence_length=int(args.max_sequence_length),
        positional_sampling_ratio=1.0,
        train_dataset=None,
        eval_dataset=None,
        domain_to_item_id_range={d: (20, shard_sizes[d] - 1) for d in shard_sizes},
        embd_dim=384,
        domain_offset=1_000_000_000,
        # v3_full_preserve stores each domain as a single shard_0.npy; this keeps
        # encoded_id % domain_offset as the direct row offset.
        shard_size=1_000_000_000,
        shard_counts={d: 1 for d in range(5)},
        min_item_id=20,
        max_item_id=4_000_945_620,
        num_ratings=0,
        # Checkpoint compatibility: event-type embedding shape is [14,128], i.e.
        # supported ids are 0..13. v001 id 14 is mapped to 0 in this adapter.
        num_event_types=13,
    )
    embedding_dirs = {d: str(Path(args.embedding_root) / f"domain_{d}") for d in range(5)}
    model = make_model(dataset=dataset, precomputed_embeddings_domain_to_dir=embedding_dirs)
    snapshot = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(snapshot["MODEL_STATE"], strict=True)
    model.to(args.device)
    model.eval()
    return model, snapshot


def context_to_tensors(ctx, *, max_sequence_length: int, device: str):
    import torch  # type: ignore

    events = list(ctx.events)[-max_sequence_length:]
    if not events:
        raise ValueError("P23 adapter cannot score zero-length context targets in this smoke")
    input_ids = [int(ev["encoded_id"]) for ev in events]
    timestamps = [int(ev.get("event_time_unix_s") or 0) for ev in events]
    raw_type_ids = []
    type_ids = []
    mapped_unknown = 0
    for ev in events:
        raw = int(ev.get("event_type_id") or 0)
        mapped = event_type_id_for_context(ev)
        raw_type_ids.append(raw)
        type_ids.append(mapped)
        if raw != mapped:
            mapped_unknown += 1
    actual_len = len(input_ids)
    raw_type_ids_tail = raw_type_ids[-10:]
    type_ids_tail = type_ids[-10:]
    # Legacy HSTU modules are configured with max_sequence_length=200 and their
    # relative-time bias path expects dense [B, 200] payload tensors even when
    # the valid prefix is shorter. Pad on the right; lengths preserves the valid
    # suffix point for get_current_embeddings().
    pad_n = max_sequence_length - actual_len
    if pad_n < 0:
        raise ValueError(f"internal error: context length {actual_len} exceeds max_sequence_length={max_sequence_length}")
    input_ids = input_ids + [0] * pad_n
    timestamps = timestamps + [0] * pad_n
    type_ids = type_ids + [0] * pad_n
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "timestamps": torch.tensor([timestamps], dtype=torch.long, device=device),
        "type_ids": torch.tensor([type_ids], dtype=torch.long, device=device),
        "lengths": torch.tensor([actual_len], dtype=torch.long, device=device),
        "ratings": torch.zeros((1,), dtype=torch.long, device=device),
        "context_event_count_used": actual_len,
        "context_event_count_available": ctx.event_count,
        "mapped_unsupported_event_type_count": mapped_unknown,
        "raw_type_ids_tail": raw_type_ids_tail,
        "type_ids_tail": type_ids_tail,
        "context_checksum": ctx.checksum,
    }


def score_candidates(model, tensors, candidate_ids: list[str], *, chunk_size: int, device: str) -> list[tuple[str, float]]:
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
        query = F.normalize(query.float(), p=2, dim=-1)
        pairs: list[tuple[str, float]] = []
        ids_int = [int(x) for x in candidate_ids]
        for start in range(0, len(ids_int), chunk_size):
            cur_ids = ids_int[start:start + chunk_size]
            cand = torch.tensor(cur_ids, dtype=torch.long, device=device)
            emb = model.model._embedding_module.get_item_embeddings(cand).float()
            emb = F.normalize(emb, p=2, dim=-1)
            scores = torch.mv(emb, query.squeeze(0)).detach().cpu().tolist()
            pairs.extend((str(cid), float(score)) for cid, score in zip(cur_ids, scores))
    # Deterministic total order: descending model score, then candidate id.
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-version", required=True)
    ap.add_argument("--target-sample-jsonl", required=True)
    ap.add_argument("--history-prefix-source", required=True, help="canonical row-array root")
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--bank-generator", required=True)
    ap.add_argument("--history-reader", required=True)
    ap.add_argument("--source-root", default="/home/yourslewis/lrm-scaling-all-events")
    ap.add_argument("--gin-config-file", required=True)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--embedding-root", default="/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve")
    ap.add_argument("--model-submission-id", default="p23_page_s10_p09_m01_o00.v001_smoke")
    ap.add_argument("--prediction-run-id", required=True)
    ap.add_argument("--context-policy", required=True)
    ap.add_argument("--output-predictions", required=True)
    ap.add_argument("--output-inference-log", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-sequence-length", type=int, default=200)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260526)
    args = ap.parse_args()

    if args.benchmark_version != "lrm_benchmark_v001":
        raise SystemExit("unsupported benchmark version")

    import torch  # type: ignore
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1] if ":" in args.device else 0))

    generator = load_module("banked_candidate_generator_v001", args.bank_generator)
    reader_mod = load_module("history_prefix_reader_v001", args.history_reader)
    model, snapshot = build_model(args)
    history_reader = reader_mod.HistoryPrefixReader.open(
        canonical_root=args.history_prefix_source,
        split="eval",
        mode="eval_inference",
    )

    model_digest = sha256_file(args.checkpoint_path)
    context_policy_digest = sha256_file(args.context_policy)
    generated_at = utc_now()
    targets = read_jsonl(args.target_sample_jsonl)
    Path(args.output_predictions).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_inference_log).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_predictions, "w", encoding="utf-8") as pred_f, open(args.output_inference_log, "w", encoding="utf-8") as log_f:
        for target in targets:
            domain_id = int(target["target_canonical_domain_id"])
            bank_path = Path(args.bank_root) / "banks" / f"domain_{domain_id}_banks.production.json"
            bank = generator.load_bank_artifact(bank_path)
            cand_result = generator.generate_candidates_for_target(target, bank)
            cand_errors = generator.validate_generated_candidates(cand_result)
            if cand_errors:
                raise RuntimeError(f"candidate generation failed for {target['target_id']}: {cand_errors}")
            if cand_result.candidate_set_digest != target.get("candidate_set_digest"):
                raise RuntimeError(f"candidate digest mismatch for {target['target_id']}")
            ctx = history_reader.raw_context_for_target(target)
            tensors = context_to_tensors(ctx, max_sequence_length=args.max_sequence_length, device=args.device)
            ranked = score_candidates(model, tensors, cand_result.candidate_item_ids, chunk_size=args.chunk_size, device=args.device)
            record = {
                "schema_version": "lrm_prediction_record_v001",
                "benchmark_version": "lrm_benchmark_v001",
                "model_submission_id": args.model_submission_id,
                "prediction_run_id": args.prediction_run_id,
                "target_id": target["target_id"],
                "candidate_protocol_label": target["candidate_protocol_label"],
                "candidate_set_id": target["candidate_set_id"],
                "predictions": [
                    {"candidate_id": cid, "rank": rank, "score": score}
                    for rank, (cid, score) in enumerate(ranked, start=1)
                ],
                "inference_metadata": {
                    "generated_at": generated_at,
                    "entrypoint_name": "p23_v001_infer.py:base_hstu_query_smoke",
                    "model_artifact_digest": model_digest,
                    "context_policy_digest": context_policy_digest,
                    "seed": args.seed,
                    "context_policy_mode": "declared_transforms_over_full_available_prefix",
                    "notes": "Smoke-only model-backed scoring. Query uses legacy base HSTU encode path; unsupported v001 event_type_id values above checkpoint range are mapped to UNK=0.",
                },
            }
            pred_f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            log = {
                "target_id": target["target_id"],
                "candidate_count": len(cand_result.candidate_item_ids),
                "candidate_set_digest": cand_result.candidate_set_digest,
                "context_event_count_available": tensors["context_event_count_available"],
                "context_event_count_used": tensors["context_event_count_used"],
                "context_checksum": tensors["context_checksum"],
                "mapped_unsupported_event_type_count": tensors["mapped_unsupported_event_type_count"],
                "raw_type_ids_tail": tensors["raw_type_ids_tail"],
                "type_ids_tail": tensors["type_ids_tail"],
                "positive_item_rank": next(i for i, (cid, _) in enumerate(ranked, 1) if cid == target["positive_item_id"]),
                "positive_item_id": target["positive_item_id"],
                "target_domain": target.get("target_domain"),
                "target_event_type": target.get("target_event_type"),
            }
            log_f.write(json.dumps(log, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
