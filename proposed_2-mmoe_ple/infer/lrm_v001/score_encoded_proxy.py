#!/usr/bin/env python3
"""Testing-owned model-free scorer for encoded LRM-v001 proxy artifacts.

Consumes ML-owned query/doc caches and emits compact predictions + metrics.
This command should not load gin/checkpoint/model code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compact_metrics import StreamingMetricAggregator  # noqa: E402
from fast_proxy_bank_scorer import BankMajorTarget, score_bank_major_compact_records  # noqa: E402
import fast_proxy_eval_runner as fast  # noqa: E402
import sequential_submission_infer as seq  # noqa: E402

DOC_CACHE_SCHEMA = "lrm_v001_projected_doc_cache_v001"
DOC_BANK_SCHEMA = "lrm_v001_projected_doc_bank_v001"
ENTRYPOINT = "proposed_2-mmoe_ple/infer/lrm_v001/score_encoded_proxy.py"


def _safe_torch_load(path: Path):
    return fast._safe_torch_load(path)


def _load_doc_manifest(doc_cache_dir: Path) -> dict[str, Any]:
    manifest = json.loads((doc_cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != DOC_CACHE_SCHEMA:
        raise ValueError(f"unsupported doc cache schema: {manifest.get('schema')}")
    return manifest


def _load_doc_bank(doc_cache_dir: Path, manifest: Mapping[str, Any], *, domain_id: int, bank_id: int, device: str):
    import torch  # type: ignore

    by_key = {(int(row["domain_id"]), int(row["bank_id"])): row for row in manifest.get("banks") or []}
    meta = by_key.get((int(domain_id), int(bank_id)))
    if meta is None:
        raise KeyError(f"doc cache missing domain={domain_id} bank={bank_id}")
    payload = _safe_torch_load(doc_cache_dir / meta["path"])
    if payload.get("schema") != DOC_BANK_SCHEMA:
        raise ValueError(f"unsupported doc bank schema in {meta['path']}: {payload.get('schema')}")
    return {
        "candidate_ids": [str(x) for x in payload["candidate_ids"]],
        "embeddings": payload["embeddings"].float().to(device=device),
        "extra_ids": [str(x) for x in payload.get("extra_ids", [])],
        "extra_embeddings": payload.get("extra_embeddings", torch.empty((0, int(payload["embeddings"].shape[1])), dtype=torch.float32)).float().to(device=device),
    }


def _prepare_encoded_targets(
    *,
    args,
    generator,
    bank_artifact,
    targets: Sequence[Mapping[str, Any]],
    checksums: Sequence[str | None],
    context_labels: Sequence[str],
    policy_labels: Sequence[str],
    doc_bank: Mapping[str, Any],
) -> tuple[list[BankMajorTarget], dict[str, Any]]:
    import torch  # type: ignore

    base_ids = [str(x) for x in doc_bank["candidate_ids"]]
    base_index = {cid: idx for idx, cid in enumerate(base_ids)}
    base_id_set = set(base_ids)
    extra_ids = [str(x) for x in doc_bank.get("extra_ids", [])]
    extra_index = {cid: idx for idx, cid in enumerate(extra_ids)}
    base_embeddings = doc_bank["embeddings"]
    extra_embeddings = doc_bank["extra_embeddings"]

    width = int(base_embeddings.shape[1])
    empty_repl = torch.empty((0, width), dtype=torch.float32, device=args.device)
    specs: list[BankMajorTarget] = []
    collision_count = 0
    extra_lookup_count = 0
    replacement_targets = 0

    for idx, target in enumerate(targets):
        pos = str(target["positive_item_id"])
        replacement_ids = fast._derive_replacement_ids(
            args=args,
            generator=generator,
            bank_artifact=bank_artifact,
            target=target,
            base_candidate_ids=base_ids,
            base_id_set=base_id_set,
        )
        if pos in base_index:
            positive_embedding = base_embeddings[base_index[pos]]
            collision_count += 1
        else:
            if pos not in extra_index:
                raise KeyError(f"positive item {pos} for target {target['target_id']} is not present in doc cache extras")
            positive_embedding = extra_embeddings[extra_index[pos]]
            extra_lookup_count += 1
        repl_tensors = []
        for rid in replacement_ids:
            rid = str(rid)
            if rid in base_index:
                repl_tensors.append(base_embeddings[base_index[rid]])
            elif rid in extra_index:
                repl_tensors.append(extra_embeddings[extra_index[rid]])
                extra_lookup_count += 1
            else:
                raise KeyError(f"replacement item {rid} for target {target['target_id']} is not present in doc cache")
        if repl_tensors:
            replacement_embeddings = torch.stack(repl_tensors, dim=0)
            replacement_targets += 1
        else:
            replacement_embeddings = empty_repl
        specs.append(
            BankMajorTarget(
                target=target,
                positive_item_id=pos,
                positive_embedding=positive_embedding,
                replacement_item_ids=[str(x) for x in replacement_ids],
                replacement_embeddings=replacement_embeddings,
                candidate_set_digest=str(target.get("candidate_set_digest")),
                context_checksum=checksums[idx],
                context_policy_label=context_labels[idx],
                model_inference_policy=policy_labels[idx],
            )
        )
    return specs, {"targets": len(targets), "positive_collisions": collision_count, "extra_lookup_count": extra_lookup_count, "replacement_targets": replacement_targets}


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Score encoded LRM-v001 query/doc caches and emit compact metrics without loading model internals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--query-cache-dir", required=True)
    ap.add_argument("--doc-cache-dir", required=True)
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--bank-generator", required=True)
    ap.add_argument("--selected-bank-subset", required=True)
    ap.add_argument("--model-submission-id", required=True)
    ap.add_argument("--prediction-run-id", required=True)
    ap.add_argument("--output-compact", required=True)
    ap.add_argument("--output-metrics-json", required=True)
    ap.add_argument("--output-inference-log", required=True)
    ap.add_argument("--compact-top-k", type=int, default=10)
    ap.add_argument("--score-query-chunk-size", type=int, default=4096)
    ap.add_argument("--candidate-check-mode", choices=("none", "collisions", "full"), default="collisions")
    ap.add_argument("--validate-candidate-generation", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log-flush-every", type=int, default=100)
    ap.add_argument("--output-flush-every", type=int, default=0)
    ap.add_argument("--stdout-progress-every", type=int, default=10000)
    args = ap.parse_args(argv)
    args.query_cache_dir = Path(args.query_cache_dir)
    args.doc_cache_dir = Path(args.doc_cache_dir)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch  # type: ignore

    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1] if ":" in args.device else 0))

    generator = seq.load_module("banked_candidate_generator_v001", args.bank_generator)
    doc_manifest = _load_doc_manifest(args.doc_cache_dir)
    query_manifest = json.loads((args.query_cache_dir / "manifest.json").read_text(encoding="utf-8"))
    model_digest = doc_manifest.get("model_digest") or query_manifest.get("model_digest")
    context_policy_digest = query_manifest.get("context_policy_digest")
    generated_at = fast.utc_now()
    bank_artifacts: dict[int, Any] = {}
    aggregator = StreamingMetricAggregator(k=args.compact_top_k)
    target_count = 0
    group_count = 0
    group_metrics: list[dict[str, Any]] = []
    started = time.time()

    Path(args.output_compact).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_inference_log).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_inference_log, "a", encoding="utf-8") as log_f, open(args.output_compact, "w", encoding="utf-8") as compact_f:
        fast._json_dump_line(log_f, {"progress": "encoded_score_start", "entrypoint": ENTRYPOINT, "query_cache_dir": str(args.query_cache_dir), "doc_cache_dir": str(args.doc_cache_dir), "at": generated_at})
        for group, batch_paths, _group_meta in fast.iter_query_groups(args.query_cache_dir):
            group_count += 1
            domain_id, bank_id = group
            group_started = time.time()
            queries, targets, checksums, context_labels, policy_labels = fast.load_query_group(batch_paths, device=args.device)
            doc_bank = _load_doc_bank(args.doc_cache_dir, doc_manifest, domain_id=domain_id, bank_id=bank_id, device=args.device)
            bank_artifact = fast.load_bank_artifact(bank_artifacts, args=args, generator=generator, domain_id=domain_id)
            specs, prep_metrics = _prepare_encoded_targets(
                args=args,
                generator=generator,
                bank_artifact=bank_artifact,
                targets=targets,
                checksums=checksums,
                context_labels=context_labels,
                policy_labels=policy_labels,
                doc_bank=doc_bank,
            )
            records, metrics = score_bank_major_compact_records(
                queries=queries,
                bank_embeddings=doc_bank["embeddings"],
                base_candidate_ids=doc_bank["candidate_ids"],
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
            for record in records:
                compact_f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                aggregator.add_compact_record(record)
                target_count += 1
                if args.output_flush_every and target_count % args.output_flush_every == 0:
                    compact_f.flush()
            if args.output_flush_every == 0:
                compact_f.flush()
            group_elapsed = time.time() - group_started
            payload = {"progress": "encoded_bank_group_done", "domain_id": domain_id, "bank_id": bank_id, "group_index": group_count, "group_targets": len(targets), "total_targets_scored": target_count, "prepare": prep_metrics, "score": metrics, "group_elapsed_s": group_elapsed, "at": fast.utc_now()}
            group_metrics.append(payload)
            fast._json_dump_line(log_f, payload)
            if args.log_flush_every == 1 or (args.log_flush_every > 1 and group_count % args.log_flush_every == 0):
                log_f.flush()
            if args.stdout_progress_every and group_count % args.stdout_progress_every == 0:
                print(json.dumps({"progress": "encoded_bank_group_done", "groups": group_count, "targets": target_count, "at": fast.utc_now()}), flush=True)
            del queries, targets, specs, records, doc_bank
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        elapsed_s = time.time() - started
        if target_count == 0:
            raise RuntimeError("no encoded targets were scored")
        metrics_payload = aggregator.result(
            created_at=fast.utc_now(),
            inputs={
                "query_cache_dir": str(args.query_cache_dir),
                "doc_cache_dir": str(args.doc_cache_dir),
                "selected_bank_subset": args.selected_bank_subset,
                "output_compact": args.output_compact,
                "model_submission_id": args.model_submission_id,
                "prediction_run_id": args.prediction_run_id,
                "runner": ENTRYPOINT,
                "bank_major_group_count": group_count,
                "elapsed_s": elapsed_s,
                "targets_per_hour": (target_count / elapsed_s * 3600.0) if elapsed_s > 0 else None,
                "group_metrics": group_metrics,
            },
        )
        Path(args.output_metrics_json).write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        done = {"progress": "encoded_score_done", "targets_scored": target_count, "bank_groups_scored": group_count, "compact_predictions": args.output_compact, "compact_metrics_json": args.output_metrics_json, "elapsed_s": elapsed_s, "targets_per_hour": (target_count / elapsed_s * 3600.0) if elapsed_s > 0 else None, "at": fast.utc_now()}
        fast._json_dump_line(log_f, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
