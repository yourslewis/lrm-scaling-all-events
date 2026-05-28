#!/usr/bin/env python3
"""Bank-major compact scoring helpers for LRM-v001 selected-bank proxy evals.

The sequential runner scores one target at a time and fully sorts/digests all
candidates. For the 500-bank proxy that is the wrong shape: many targets share
one negative bank, so the fast path should batch queries per bank and compute a
single `queries @ bank_embeddings.T` score matrix, then derive only the exact
positive rank and topK rows needed by compact metrics.

This module is intentionally execution-side only. It does not alter the official
v001 prediction JSONL contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

from compact_metrics import make_compact_record_from_parts

# The legacy target-major scorer uses torch.mv over per-target candidate chunks.
# The fast path uses batched GEMM and sometimes separate positive/replacement
# dot products. Those mathematically equivalent kernels can differ by a few
# float32 ulps, which is enough to break exact-equality pessimistic tie counts
# for highly duplicated/tied scores. Treat near-identical scores as ties so the
# fast path stays conservative and matches legacy metric intent.
DEFAULT_TIE_EPSILON = 1e-7


@dataclass(frozen=True)
class BankMajorTarget:
    """Per-target inputs needed once the query and reusable bank matrix exist."""

    target: Mapping[str, Any]
    positive_item_id: str
    positive_embedding: Any
    replacement_item_ids: Sequence[str] = ()
    replacement_embeddings: Any | None = None
    candidate_set_digest: str | None = None
    context_checksum: str | None = None
    context_policy_label: str | None = None
    model_inference_policy: str | None = "bank_major_compact_proxy"


def _as_1d_float_tensor(value, *, device: str):
    import torch  # type: ignore

    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32).flatten()
    return torch.as_tensor(value, device=device, dtype=torch.float32).flatten()


def _as_2d_float_tensor(value, *, device: str, width: int):
    import torch  # type: ignore

    if value is None:
        return torch.empty((0, width), device=device, dtype=torch.float32)
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.float32)
    else:
        out = torch.as_tensor(value, device=device, dtype=torch.float32)
    if out.numel() == 0:
        return torch.empty((0, width), device=device, dtype=torch.float32)
    return out.reshape((-1, width))


def _score_token(score: float) -> float:
    """Normalize scalar tensor/list scores to Python float for JSON output."""
    return float(score)


def _topk_from_scores(
    *,
    base_candidate_ids: Sequence[str],
    base_scores,
    positive_item_id: str,
    positive_score: float,
    replacement_item_ids: Sequence[str],
    replacement_scores: Sequence[float],
    k: int,
    base_positive_index: int | None,
) -> list[dict[str, Any]]:
    """Exact topK with score desc / candidate_id asc tie handling.

    We use torch.topk only to find a score threshold, then include all candidates
    at/above the threshold before Python sorting. This preserves deterministic
    lexicographic tie-breaks without sorting all 10k candidates in normal cases.
    If a degenerate query ties all candidates, correctness wins and the tie set
    can be all candidates.
    """
    import torch  # type: ignore

    k = max(0, int(k))
    if k == 0:
        return []

    candidate_total = len(base_candidate_ids) - (1 if base_positive_index is not None else 0) + 1 + len(replacement_item_ids)
    want = min(k, candidate_total)
    if want <= 0:
        return []

    # If the positive item is present in the reusable base bank, it is not a
    # separate non-positive candidate for this target. Exclude that base row
    # before computing the kth threshold; otherwise a very high positive-bank
    # collision can occupy two topK threshold slots and make us return <K rows.
    if base_positive_index is not None:
        idx = int(base_positive_index)
        valid_base_scores = torch.cat((base_scores[:idx], base_scores[idx + 1 :]))
    else:
        valid_base_scores = base_scores
    components = [valid_base_scores]
    extra_scores = [positive_score, *[_score_token(x) for x in replacement_scores]]
    if extra_scores:
        components.append(torch.as_tensor(extra_scores, device=base_scores.device, dtype=base_scores.dtype))
    all_scores_for_threshold = torch.cat(components)
    kth_score = float(torch.topk(all_scores_for_threshold, k=want).values[-1].detach().cpu().item())

    rows: list[tuple[str, float]] = []
    mask = base_scores >= kth_score
    if bool(mask.any().detach().cpu().item()):
        idxs = mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for idx in idxs:
            if base_positive_index is not None and int(idx) == int(base_positive_index):
                continue
            rows.append((str(base_candidate_ids[int(idx)]), float(base_scores[int(idx)].detach().cpu().item())))
    if positive_score >= kth_score:
        rows.append((str(positive_item_id), float(positive_score)))
    for cid, score in zip(replacement_item_ids, replacement_scores):
        fscore = float(score)
        if fscore >= kth_score:
            rows.append((str(cid), fscore))

    rows.sort(key=lambda kv: (-kv[1], kv[0]))
    return [
        {"candidate_id": cid, "rank": rank, "score": score}
        for rank, (cid, score) in enumerate(rows[:want], start=1)
    ]


def _rank_stats_from_scores(
    *,
    base_scores,
    positive_item_id: str,
    positive_score: float,
    replacement_scores: Sequence[float],
    k: int,
    base_positive_index: int | None,
    tie_epsilon: float = DEFAULT_TIE_EPSILON,
) -> dict[str, Any]:
    """Exact pessimistic positive rank from score tensors without full sort."""
    import torch  # type: ignore

    eps = float(tie_epsilon)
    greater = int((base_scores > (positive_score + eps)).sum().detach().cpu().item())
    equal_nonpositive = int((torch.abs(base_scores - positive_score) <= eps).sum().detach().cpu().item())
    if base_positive_index is not None:
        base_pos_score = float(base_scores[int(base_positive_index)].detach().cpu().item())
        if base_pos_score > positive_score + eps:
            greater -= 1
        if abs(base_pos_score - positive_score) <= eps:
            equal_nonpositive -= 1

    if replacement_scores:
        repl = torch.as_tensor(list(replacement_scores), device=base_scores.device, dtype=base_scores.dtype)
        greater += int((repl > (positive_score + eps)).sum().detach().cpu().item())
        equal_nonpositive += int((torch.abs(repl - positive_score) <= eps).sum().detach().cpu().item())

    # Numerical tie tolerance plus positive-bank collision removal can very rarely
    # over-subtract the reusable positive row by one fp32 ulp. Counts below zero
    # are not meaningful; the best possible positive rank is 1.
    greater = max(0, int(greater))
    equal_nonpositive = max(0, int(equal_nonpositive))
    rank = max(1, 1 + greater + equal_nonpositive)
    return {
        "positive_score": float(positive_score),
        "greater_score_count": int(greater),
        "equal_score_nonpositive_count": int(equal_nonpositive),
        "pessimistic_rank": int(rank),
        f"hit_at_{int(k)}": int(rank <= int(k)),
        f"ndcg_at_{int(k)}": (1.0 / math.log2(rank + 1)) if rank <= int(k) else 0.0,
        "reciprocal_rank": 1.0 / rank,
    }


def score_bank_major_compact_records(
    *,
    queries,
    bank_embeddings,
    base_candidate_ids: Sequence[str],
    targets: Sequence[BankMajorTarget],
    top_k: int,
    model_submission_id: str,
    prediction_run_id: str,
    generated_at: str,
    model_digest: str,
    context_policy_digest: str,
    query_chunk_size: int = 4096,
    device: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score many targets sharing one bank and emit compact records.

    Correctness properties:
    - positive rank uses the same pessimistic tie formula as the official eval;
    - topK uses score desc, candidate_id asc tie-break;
    - positive-bank collisions are excluded from reusable negatives and replaced
      with target-specific replacement ids, matching `score_candidate_set_from_query`.
    """
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    started = time.perf_counter()
    if device is None:
        if torch.is_tensor(bank_embeddings):
            device = str(bank_embeddings.device)
        elif torch.is_tensor(queries):
            device = str(queries.device)
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bank = bank_embeddings.to(device=device, dtype=torch.float32) if torch.is_tensor(bank_embeddings) else torch.as_tensor(bank_embeddings, device=device, dtype=torch.float32)
    # Match the legacy online path exactly: get_item_embeddings(...) is normalized
    # immediately before dot-product scoring, even if a projected-bank cache
    # already stores near-unit vectors. Re-normalizing the reusable bank preserves
    # exact tie behavior for pessimistic rank/topK equivalence.
    bank = F.normalize(bank, p=2, dim=-1)
    query_tensor = queries.to(device=device, dtype=torch.float32) if torch.is_tensor(queries) else torch.as_tensor(queries, device=device, dtype=torch.float32)
    if query_tensor.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"expected queries and bank_embeddings to be 2D, got {tuple(query_tensor.shape)} and {tuple(bank.shape)}")
    if query_tensor.shape[0] != len(targets):
        raise ValueError(f"query count {query_tensor.shape[0]} != target count {len(targets)}")
    if query_tensor.shape[1] != bank.shape[1]:
        raise ValueError(f"query dim {query_tensor.shape[1]} != bank dim {bank.shape[1]}")
    if len(base_candidate_ids) != int(bank.shape[0]):
        raise ValueError(f"base_candidate_ids count {len(base_candidate_ids)} != bank rows {bank.shape[0]}")

    base_candidate_ids = [str(x) for x in base_candidate_ids]
    base_index = {cid: idx for idx, cid in enumerate(base_candidate_ids)}
    records: list[dict[str, Any]] = []
    matmul_s = 0.0
    derive_s = 0.0
    chunks = 0
    with torch.inference_mode():
        for start in range(0, len(targets), max(1, int(query_chunk_size))):
            end = min(len(targets), start + max(1, int(query_chunk_size)))
            chunks += 1
            t0 = time.perf_counter()
            scores = torch.matmul(query_tensor[start:end], bank.T)
            matmul_s += time.perf_counter() - t0
            t1 = time.perf_counter()
            for local_idx, spec in enumerate(targets[start:end]):
                row_idx = start + local_idx
                query = query_tensor[row_idx]
                row_scores = scores[local_idx]
                positive_id = str(spec.positive_item_id)
                positive_emb = F.normalize(_as_1d_float_tensor(spec.positive_embedding, device=device), p=2, dim=-1)
                positive_score = float(torch.dot(query, positive_emb).detach().cpu().item())

                repl_ids = [str(x) for x in (spec.replacement_item_ids or [])]
                repl_emb = _as_2d_float_tensor(spec.replacement_embeddings, device=device, width=int(bank.shape[1]))
                if len(repl_ids) != int(repl_emb.shape[0]):
                    raise ValueError(
                        f"replacement id/embedding count mismatch for target {spec.target.get('target_id')}: "
                        f"{len(repl_ids)} ids vs {int(repl_emb.shape[0])} embeddings"
                    )
                repl_scores_tensor = torch.mv(F.normalize(repl_emb, p=2, dim=-1), query) if len(repl_ids) else torch.empty((0,), device=device)
                repl_scores = [float(x) for x in repl_scores_tensor.detach().cpu().tolist()]
                base_positive_index = base_index.get(positive_id)

                rank_stats = _rank_stats_from_scores(
                    base_scores=row_scores,
                    positive_item_id=positive_id,
                    positive_score=positive_score,
                    replacement_scores=repl_scores,
                    k=top_k,
                    base_positive_index=base_positive_index,
                )
                topk_rows = _topk_from_scores(
                    base_candidate_ids=base_candidate_ids,
                    base_scores=row_scores,
                    positive_item_id=positive_id,
                    positive_score=positive_score,
                    replacement_item_ids=repl_ids,
                    replacement_scores=repl_scores,
                    k=top_k,
                    base_positive_index=base_positive_index,
                )
                candidate_count = len(base_candidate_ids) - (1 if base_positive_index is not None else 0) + 1 + len(repl_ids)
                records.append(
                    make_compact_record_from_parts(
                        target=spec.target,
                        rank_stats=rank_stats,
                        top_k_records_value=topk_rows,
                        top_k=top_k,
                        model_submission_id=model_submission_id,
                        prediction_run_id=prediction_run_id,
                        generated_at=generated_at,
                        model_digest=model_digest,
                        context_policy_digest=context_policy_digest,
                        candidate_count=candidate_count,
                        candidate_set_digest=spec.candidate_set_digest or str(spec.target.get("candidate_set_digest")),
                        context_checksum=spec.context_checksum,
                        context_policy_label=spec.context_policy_label,
                        model_inference_policy=spec.model_inference_policy,
                        full_score_order_digest_value=None,
                    )
                )
            derive_s += time.perf_counter() - t1

    metrics = {
        "mode": "bank_major_compact",
        "targets": len(targets),
        "bank_candidates": len(base_candidate_ids),
        "query_dim": int(query_tensor.shape[1]),
        "chunks": chunks,
        "matmul_s": matmul_s,
        "derive_s": derive_s,
        "total_s": time.perf_counter() - started,
        "full_score_order_digest": "omitted",
    }
    return records, metrics
