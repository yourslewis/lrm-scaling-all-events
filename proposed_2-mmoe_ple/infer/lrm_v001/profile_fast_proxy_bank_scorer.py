#!/usr/bin/env python3
"""Smoke profiler for the LRM-v001 500-bank proxy bank-major scorer.

This is deliberately synthetic and bounded: it profiles the scoring shape used by
selected-bank proxy evals without touching production sidecars or launching a
full run. Use it as a quick regression/sanity gate before trying a real proxy
sample on the GPU host.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_embedding_cache import score_candidate_set_from_query  # noqa: E402
from fast_proxy_bank_scorer import BankMajorTarget, score_bank_major_compact_records  # noqa: E402


class FakeEmbeddingModule:
    def __init__(self, embeddings_by_id: dict[str, torch.Tensor]) -> None:
        self.embeddings_by_id = {str(k): v.detach().cpu().float() for k, v in embeddings_by_id.items()}

    def get_item_embeddings(self, cand: torch.Tensor) -> torch.Tensor:
        rows = [self.embeddings_by_id[str(int(x))] for x in cand.detach().cpu().tolist()]
        return torch.stack(rows, dim=0).to(device=cand.device, dtype=torch.float32)


def _fake_model(embeddings_by_id: dict[str, torch.Tensor]):
    return SimpleNamespace(model=SimpleNamespace(_embedding_module=FakeEmbeddingModule(embeddings_by_id)))


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def gpu_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except Exception as exc:  # nvidia-smi is absent on local Mac/CPU-only hosts.
        return {"available": False, "reason": str(exc)}
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx, util, mem_used, mem_total = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(idx),
                "utilization_gpu_pct": float(util),
                "memory_used_mib": float(mem_used),
                "memory_total_mib": float(mem_total),
            }
        )
    return {"available": True, "gpus": rows}


def _target(target_id: str, positive: str, *, domain_id: int, bank_id: int) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "user_id": f"user-{int(target_id.split('-')[-1]) % 100}",
        "target_domain": "Ads",
        "target_event_type": "AdClick",
        "target_canonical_domain_id": domain_id,
        "negative_bank_id": bank_id,
        "positive_item_id": positive,
        "candidate_protocol_label": "banked_domain_negatives_10k_b1000_v001",
        "candidate_set_id": f"cs-{target_id}",
        "candidate_set_digest": f"sha256:{target_id}",
        "headline_slices": ["all_domain", "all_ads"],
        "diagnostic_buckets": {"context_length": "synthetic"},
    }


def build_case(*, targets: int, bank_size: int, dim: int, seed: int, device: str):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.perf_counter()
    base_ids = [str(1000000 + i) for i in range(bank_size)]
    bank_cpu = F.normalize(torch.randn((bank_size, dim), generator=gen), p=2, dim=-1)
    setup_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    queries_cpu = F.normalize(torch.randn((targets, dim), generator=gen), p=2, dim=-1)
    embeddings_by_id = {cid: bank_cpu[i].clone() for i, cid in enumerate(base_ids)}
    specs: list[BankMajorTarget] = []
    sequential_cands: list[SimpleNamespace] = []
    for idx in range(targets):
        collides = (idx % 17) == 0
        if collides:
            positive_id = base_ids[(idx * 13) % bank_size]
            positive_emb = embeddings_by_id[positive_id]
            replacement_id = str(3000000 + idx)
            replacement_emb = F.normalize(torch.randn((dim,), generator=gen), p=2, dim=-1)
            replacement_ids = [replacement_id]
            replacement_embs = replacement_emb.reshape(1, -1)
            embeddings_by_id[replacement_id] = replacement_emb
            candidate_item_ids = [cid for cid in base_ids if cid != positive_id] + [positive_id, replacement_id]
        else:
            positive_id = str(2000000 + idx)
            positive_emb = F.normalize(torch.randn((dim,), generator=gen), p=2, dim=-1)
            replacement_ids = []
            replacement_embs = torch.empty((0, dim), dtype=torch.float32)
            embeddings_by_id[positive_id] = positive_emb
            candidate_item_ids = base_ids + [positive_id]
        target = _target(f"synthetic-{idx}", positive_id, domain_id=idx % 5, bank_id=idx % 100)
        specs.append(
            BankMajorTarget(
                target=target,
                positive_item_id=positive_id,
                positive_embedding=positive_emb.to(device=device),
                replacement_item_ids=replacement_ids,
                replacement_embeddings=replacement_embs.to(device=device),
                candidate_set_digest=target["candidate_set_digest"],
                model_inference_policy="synthetic_bank_major_profile",
            )
        )
        sequential_cands.append(
            SimpleNamespace(
                target_id=target["target_id"],
                candidate_item_ids=candidate_item_ids,
                positive_item_id=positive_id,
                target_canonical_domain_id=idx % 5,
                negative_bank_id=idx % 100,
                replacement_item_ids=replacement_ids,
            )
        )
    query_extraction_s = time.perf_counter() - t1
    queries = queries_cpu.to(device=device)
    bank = bank_cpu.to(device=device)
    return setup_s, query_extraction_s, queries, bank, base_ids, specs, sequential_cands, _fake_model(embeddings_by_id)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"requested {args.device}, but torch.cuda.is_available() is false")
    before_gpu = gpu_snapshot()
    setup_s, query_extraction_s, queries, bank, base_ids, specs, sequential_cands, model = build_case(
        targets=args.targets,
        bank_size=args.bank_size,
        dim=args.dim,
        seed=args.seed,
        device=args.device,
    )
    _sync(args.device)
    score_started = time.perf_counter()
    records, score_metrics = score_bank_major_compact_records(
        queries=queries,
        bank_embeddings=bank,
        base_candidate_ids=base_ids,
        targets=specs,
        top_k=args.top_k,
        model_submission_id="synthetic-profile",
        prediction_run_id="synthetic-bank-major",
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:synthetic-model",
        context_policy_digest="sha256:synthetic-context-policy",
        query_chunk_size=args.query_chunk_size,
        device=args.device,
    )
    _sync(args.device)
    score_wall_s = time.perf_counter() - score_started
    after_gpu = gpu_snapshot()

    sequential_s = None
    sequential_targets = 0
    if args.compare_sequential_targets > 0:
        limit = min(args.compare_sequential_targets, args.targets)
        _sync(args.device)
        t0 = time.perf_counter()
        for idx in range(limit):
            # Old target-major path: online project+score every candidate, then full sort.
            score_candidate_set_from_query(
                model,
                queries[idx].reshape(1, -1),
                sequential_cands[idx],
                chunk_size=args.sequential_chunk_size,
                device=args.device,
                candidate_cache=None,
            )
        _sync(args.device)
        sequential_s = time.perf_counter() - t0
        sequential_targets = limit

    total_s = setup_s + query_extraction_s + score_wall_s
    return {
        "schema": "lrm_v001_fast_proxy_bank_scorer_profile_v001",
        "device": args.device,
        "targets": args.targets,
        "bank_size": args.bank_size,
        "dim": args.dim,
        "top_k": args.top_k,
        "query_chunk_size": args.query_chunk_size,
        "record_count": len(records),
        "gpu_before": before_gpu,
        "gpu_after": after_gpu,
        "timing_s": {
            "setup_bank_synthetic_s": setup_s,
            "query_extraction_s": query_extraction_s,
            "bank_major_wall_s": score_wall_s,
            "bank_major_reported_total_s": score_metrics["total_s"],
            "bank_major_matmul_s": score_metrics["matmul_s"],
            "bank_major_derive_s": score_metrics["derive_s"],
            "end_to_end_s": total_s,
            "old_sequential_full_sort_s": sequential_s,
        },
        "throughput": {
            "targets_per_s_end_to_end": args.targets / total_s if total_s > 0 else math.inf,
            "targets_per_s_bank_major_wall": args.targets / score_wall_s if score_wall_s > 0 else math.inf,
            "old_sequential_targets": sequential_targets,
            "old_sequential_targets_per_s": (sequential_targets / sequential_s) if sequential_s and sequential_s > 0 else None,
        },
        "score_metrics": score_metrics,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=int, default=100, help="synthetic targets sharing one bank; use 100 then 1000 for smoke gates")
    ap.add_argument("--bank-size", type=int, default=1000, help="base candidates in the reusable selected bank; real v001 banks are ~10000")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--query-chunk-size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--compare-sequential-targets", type=int, default=0, help="also time old target-major scorer on this many targets; keep small")
    ap.add_argument("--sequential-chunk-size", type=int, default=4096)
    ap.add_argument("--output-json")
    args = ap.parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
