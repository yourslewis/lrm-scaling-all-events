#!/usr/bin/env python3
"""Compact/streaming metric helpers for LRM-v001 inference outputs.

This module is intentionally execution-side only. It does not replace the
official v001 prediction JSONL contract; it mirrors the official evaluator's
rank-derived metric formulas when a runner has already scored every candidate
for a target in memory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

BENCHMARK_VERSION = "lrm_benchmark_v001"
SPLIT_PROTOCOL = "primary_user_disjoint_same_period"
COMPACT_RECORD_SCHEMA_VERSION = "lrm_compact_prediction_record_v001.derived"
COMPACT_RESULT_ARTIFACT = "lrm_benchmark_v001_compact_streaming_evaluation_result"
IMPLEMENTATION_VERSION = "compact_streaming_evaluator_v001.0.1"
DEFAULT_K = 10
HEADLINE_SLICES = ("cold_ads", "warm_ads", "all_ads", "all_domain")
DIAGNOSTIC_BUCKET_FAMILIES = (
    "context_length",
    "ads_history",
    "target_time",
    "last_event_recency",
    "last_ads_recency",
)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def score_token(score: float) -> str:
    """Return a stable token for digesting Python/JSON finite float scores."""
    return float(score).hex()


def full_score_order_digest(ranked: Sequence[tuple[str, float]]) -> str:
    """Digest the full sorted candidate-score order without storing it in JSON.

    The digest commits to rank order, candidate id, and the exact binary64 score
    observed by Python after model scoring. It is an audit/reproducibility aid,
    not an official benchmark artifact.
    """
    h = hashlib.sha256()
    for rank, (candidate_id, score) in enumerate(ranked, start=1):
        h.update(str(rank).encode("utf-8"))
        h.update(b"\t")
        h.update(str(candidate_id).encode("utf-8"))
        h.update(b"\t")
        h.update(score_token(float(score)).encode("ascii"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def topk_records(ranked: Sequence[tuple[str, float]], *, k: int) -> list[dict[str, Any]]:
    return [
        {"candidate_id": str(candidate_id), "rank": rank, "score": float(score)}
        for rank, (candidate_id, score) in enumerate(ranked[: max(int(k), 0)], start=1)
    ]


def pessimistic_rank_from_ranked(
    ranked: Sequence[tuple[str, float]],
    positive_item_id: str,
    *,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Mirror official evaluator pessimistic tie rank from in-memory scores."""
    positive = str(positive_item_id)
    positive_scores = [float(score) for cid, score in ranked if str(cid) == positive]
    if len(positive_scores) != 1:
        raise ValueError(f"positive item {positive!r} must appear exactly once; found {len(positive_scores)}")
    positive_score = positive_scores[0]
    greater = sum(1 for _, score in ranked if float(score) > positive_score)
    equal_nonpositive = sum(
        1 for cid, score in ranked if str(cid) != positive and float(score) == positive_score
    )
    rank = 1 + greater + equal_nonpositive
    return {
        "positive_score": positive_score,
        "greater_score_count": greater,
        "equal_score_nonpositive_count": equal_nonpositive,
        "pessimistic_rank": rank,
        f"hit_at_{k}": int(rank <= k),
        f"ndcg_at_{k}": (1.0 / math.log2(rank + 1)) if rank <= k else 0.0,
        "reciprocal_rank": 1.0 / rank,
    }


def is_ads_target(target_or_record: Mapping[str, Any]) -> bool:
    return str(target_or_record.get("target_domain")) == "Ads"


def average(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


@dataclass
class UserMetricState:
    count: int = 0
    hit_sum: float = 0.0
    ndcg_sum: float = 0.0
    rr_sum: float = 0.0

    def add(self, *, hit: float, ndcg: float, rr: float) -> None:
        self.count += 1
        self.hit_sum += float(hit)
        self.ndcg_sum += float(ndcg)
        self.rr_sum += float(rr)


@dataclass
class SliceMetricState:
    slice_id: str
    slice_kind: str
    metric_family: str
    k: int
    target_count: int = 0
    hit_sum: float = 0.0
    ndcg_sum: float = 0.0
    rr_sum: float = 0.0
    domain_counts: Counter[str] = field(default_factory=Counter)
    users: dict[str, UserMetricState] = field(default_factory=lambda: defaultdict(UserMetricState))

    def add(self, *, user_id: str, target_domain: str, hit: float, ndcg: float, rr: float) -> None:
        self.target_count += 1
        self.hit_sum += float(hit)
        self.ndcg_sum += float(ndcg)
        self.rr_sum += float(rr)
        self.domain_counts[str(target_domain)] += 1
        self.users[str(user_id)].add(hit=hit, ndcg=ndcg, rr=rr)

    @property
    def user_count(self) -> int:
        return len(self.users)

    def target_per_user_stats(self) -> dict[str, Any]:
        counts = [state.count for state in self.users.values()]
        if not counts:
            return {
                "targets_per_user_min": None,
                "targets_per_user_median": None,
                "targets_per_user_max": None,
            }
        return {
            "targets_per_user_min": min(counts),
            "targets_per_user_median": statistics.median(counts),
            "targets_per_user_max": max(counts),
        }

    def support_flags(self) -> dict[str, bool]:
        if self.slice_kind == "headline":
            low_support = self.target_count < 1000 or self.user_count < 100
        else:
            low_support = self.target_count < 300 or self.user_count < 50
        return {
            "low_support": low_support,
            "invalid_support": self.target_count < 100 or self.user_count < 20,
        }

    def micro(self, key: str) -> float | None:
        if self.target_count == 0:
            return None
        if key == "hit":
            return self.hit_sum / self.target_count
        if key == "ndcg":
            return self.ndcg_sum / self.target_count
        if key == "rr":
            return self.rr_sum / self.target_count
        raise KeyError(key)

    def macro_user(self, key: str) -> float | None:
        values: list[float] = []
        for state in self.users.values():
            if state.count <= 0:
                continue
            if key == "hit":
                values.append(state.hit_sum / state.count)
            elif key == "ndcg":
                values.append(state.ndcg_sum / state.count)
            elif key == "rr":
                values.append(state.rr_sum / state.count)
            else:
                raise KeyError(key)
        return average(values)

    def metric_record(
        self,
        *,
        candidate_protocol_label: str,
        target_manifest_checksum: str | None,
        metric_name: str,
        metric_family: str,
        cutoff: int | None,
        value: float | None,
        headline_metric: bool,
        diagnostic_metric: bool,
    ) -> dict[str, Any]:
        return {
            "benchmark_id": BENCHMARK_VERSION,
            "split_protocol": SPLIT_PROTOCOL,
            "candidate_protocol_label": candidate_protocol_label,
            "manifest_checksum": target_manifest_checksum,
            "metric_impl_version": IMPLEMENTATION_VERSION,
            "slice_id": self.slice_id,
            "slice_kind": self.slice_kind,
            "metric_name": metric_name,
            "metric_family": metric_family,
            "k": cutoff,
            "value": value,
            "target_count": self.target_count,
            "user_count": self.user_count,
            "target_domain_distribution": dict(sorted(self.domain_counts.items())),
            "headline": headline_metric,
            "diagnostic": diagnostic_metric,
            "not_used_for_primary_model_ranking": diagnostic_metric,
            **self.support_flags(),
            **self.target_per_user_stats(),
        }

    def records(self, *, candidate_protocol_label: str, target_manifest_checksum: str | None) -> list[dict[str, Any]]:
        out = [
            self.metric_record(
                candidate_protocol_label=candidate_protocol_label,
                target_manifest_checksum=target_manifest_checksum,
                metric_name=f"micro_{self.metric_family}@{self.k}",
                metric_family=self.metric_family,
                cutoff=self.k,
                value=self.micro("hit"),
                headline_metric=self.slice_kind == "headline",
                diagnostic_metric=self.slice_kind != "headline",
            ),
            self.metric_record(
                candidate_protocol_label=candidate_protocol_label,
                target_manifest_checksum=target_manifest_checksum,
                metric_name=f"macro_user_{self.metric_family}@{self.k}",
                metric_family=self.metric_family,
                cutoff=self.k,
                value=self.macro_user("hit"),
                headline_metric=self.slice_kind == "headline",
                diagnostic_metric=self.slice_kind != "headline",
            ),
        ]
        for metric_name, family, cutoff, key in [
            (f"micro_NDCG@{self.k}", "NDCG", self.k, "ndcg"),
            (f"macro_user_NDCG@{self.k}", "NDCG", self.k, "ndcg"),
            ("micro_MRR", "MRR", None, "rr"),
            ("macro_user_MRR", "MRR", None, "rr"),
        ]:
            value = self.micro(key) if metric_name.startswith("micro_") else self.macro_user(key)
            out.append(
                self.metric_record(
                    candidate_protocol_label=candidate_protocol_label,
                    target_manifest_checksum=target_manifest_checksum,
                    metric_name=metric_name,
                    metric_family=family,
                    cutoff=cutoff,
                    value=value,
                    headline_metric=False,
                    diagnostic_metric=True,
                )
            )
        return out


class StreamingMetricAggregator:
    """Exact online aggregate for all official rank-derived v001 metrics.

    The aggregator stores per-slice/user sums, not full candidate lists. That is
    enough for exact micro and macro-by-user HR/AHR/OHR, NDCG@K, and MRR because
    all formulas depend only on the target's pessimistic positive rank.
    """

    def __init__(
        self,
        *,
        k: int = DEFAULT_K,
        candidate_protocol_label: str = "banked_domain_negatives_10k_b1000_v001",
        target_manifest_checksum: str | None = None,
    ) -> None:
        self.k = int(k)
        self.candidate_protocol_label = candidate_protocol_label
        self.target_manifest_checksum = target_manifest_checksum
        self.target_count = 0
        self._states: dict[tuple[str, str, str], SliceMetricState] = {}
        # The official evaluator emits all headline slices even when a bounded
        # sample has zero rows for that slice. Pre-create them for byte-level
        # comparable metric structure.
        for slice_id in HEADLINE_SLICES:
            self._state(slice_id, "headline", "OHR" if slice_id == "all_domain" else "AHR")

    def _state(self, slice_id: str, slice_kind: str, metric_family: str) -> SliceMetricState:
        key = (slice_id, slice_kind, metric_family)
        if key not in self._states:
            self._states[key] = SliceMetricState(
                slice_id=slice_id,
                slice_kind=slice_kind,
                metric_family=metric_family,
                k=self.k,
            )
        return self._states[key]

    def add_target(self, target: Mapping[str, Any], rank_stats: Mapping[str, Any]) -> None:
        hit = float(rank_stats[f"hit_at_{self.k}"])
        ndcg = float(rank_stats[f"ndcg_at_{self.k}"])
        rr = float(rank_stats["reciprocal_rank"])
        user_id = str(target["user_id"])
        target_domain = str(target.get("target_domain"))
        headline_slices = list(target.get("headline_slices") or [])
        diagnostic_buckets = dict(target.get("diagnostic_buckets") or {})

        self.target_count += 1
        for slice_id in HEADLINE_SLICES:
            if slice_id not in headline_slices:
                continue
            family = "OHR" if slice_id == "all_domain" else "AHR"
            self._state(slice_id, "headline", family).add(
                user_id=user_id,
                target_domain=target_domain,
                hit=hit,
                ndcg=ndcg,
                rr=rr,
            )

        for family_name in DIAGNOSTIC_BUCKET_FAMILIES:
            bucket_id = diagnostic_buckets.get(family_name)
            if not bucket_id:
                continue
            self._state(f"{family_name}:{bucket_id}:all_domain", "diagnostic", "OHR").add(
                user_id=user_id,
                target_domain=target_domain,
                hit=hit,
                ndcg=ndcg,
                rr=rr,
            )
            ads_state = self._state(f"{family_name}:{bucket_id}:ads", "diagnostic", "AHR")
            if is_ads_target(target):
                ads_state.add(
                    user_id=user_id,
                    target_domain=target_domain,
                    hit=hit,
                    ndcg=ndcg,
                    rr=rr,
                )

    def add_compact_record(self, record: Mapping[str, Any]) -> None:
        target = dict(record.get("target") or {})
        # Keep target_id/candidate fields available for future auditing but only
        # rank-derived fields are needed for exact metrics.
        target.setdefault("target_id", record.get("target_id"))
        rank_stats = record.get("rank_stats") or {}
        self.add_target(target, rank_stats)

    def metric_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        # Match the official evaluator's headline order first.
        for slice_id in HEADLINE_SLICES:
            family = "OHR" if slice_id == "all_domain" else "AHR"
            state = self._states.get((slice_id, "headline", family))
            if state is not None:
                records.extend(
                    state.records(
                        candidate_protocol_label=self.candidate_protocol_label,
                        target_manifest_checksum=self.target_manifest_checksum,
                    )
                )
        # Then stable diagnostic order.
        for key in sorted(k for k in self._states if k[1] == "diagnostic"):
            records.extend(
                self._states[key].records(
                    candidate_protocol_label=self.candidate_protocol_label,
                    target_manifest_checksum=self.target_manifest_checksum,
                )
            )
        return records

    def result(self, *, created_at: str, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        metrics = self.metric_records()
        return {
            "artifact": COMPACT_RESULT_ARTIFACT,
            "created_at": created_at,
            "run_status": "passed",
            "benchmark_version": BENCHMARK_VERSION,
            "evaluator_impl_version": IMPLEMENTATION_VERSION,
            "cutoff_k": self.k,
            "tie_handling": "pessimistic_tie_rank",
            "tie_rank_formula": "1 + count(score > positive_score) + count(score == positive_score and candidate != positive)",
            "macro_definition": "macro_by_user",
            "output_contract": "derived_compact_execution_output_not_official_prediction_jsonl",
            "inputs": dict(inputs or {}),
            "metrics": metrics,
            "headline_metrics": [
                row for row in metrics if row["slice_kind"] == "headline" and row["metric_family"] in {"AHR", "OHR"}
            ],
            "diagnostic_metrics": [row for row in metrics if row["diagnostic"]],
            "per_target_count": self.target_count,
        }


def make_compact_record_from_parts(
    *,
    target: Mapping[str, Any],
    rank_stats: Mapping[str, Any],
    top_k_records_value: Sequence[Mapping[str, Any]],
    top_k: int,
    model_submission_id: str,
    prediction_run_id: str,
    generated_at: str,
    model_digest: str,
    context_policy_digest: str,
    candidate_count: int,
    candidate_set_digest: str,
    context_checksum: str | None = None,
    context_policy_label: str | None = None,
    model_inference_policy: str | None = None,
    full_score_order_digest_value: str | None = None,
) -> dict[str, Any]:
    """Build a compact record from precomputed rank/topK pieces.

    Fast proxy evaluators can compute exact positive rank and topK from a
    bank-major score matrix without fully sorting/digesting all 10k candidates.
    This helper keeps the compact schema construction shared with the legacy
    per-target scorer while making the expensive full-order audit digest
    optional.
    """
    digests = {
        "model_artifact_digest": model_digest,
        "context_policy_digest": context_policy_digest,
        "context_checksum": context_checksum,
    }
    if full_score_order_digest_value is not None:
        digests["full_score_order_digest"] = full_score_order_digest_value
    else:
        digests["full_score_order_digest"] = None
        digests["full_score_order_digest_omitted"] = True

    return {
        "schema_version": COMPACT_RECORD_SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "model_submission_id": model_submission_id,
        "prediction_run_id": prediction_run_id,
        "target_id": target["target_id"],
        "candidate_protocol_label": target["candidate_protocol_label"],
        "candidate_set_id": target["candidate_set_id"],
        "candidate_set_digest": candidate_set_digest,
        "candidate_count": int(candidate_count),
        "positive_item_id": target["positive_item_id"],
        "rank_stats": dict(rank_stats),
        "top_k": [dict(row) for row in top_k_records_value],
        "top_k_k": int(top_k),
        "digests": digests,
        "target": {
            "target_id": target["target_id"],
            "user_id": target["user_id"],
            "target_domain": target.get("target_domain"),
            "target_event_type": target.get("target_event_type"),
            "headline_slices": list(target.get("headline_slices") or []),
            "diagnostic_buckets": dict(target.get("diagnostic_buckets") or {}),
        },
        "inference_metadata": {
            "generated_at": generated_at,
            "context_policy_label": context_policy_label,
            "model_inference_policy": model_inference_policy,
            "compact_output_note": "Derived execution/evaluation output. Official v001 submission/audit JSONL remains lrm_prediction_record_v001.",
        },
    }


def make_compact_record(
    *,
    target: Mapping[str, Any],
    ranked: Sequence[tuple[str, float]],
    rank_stats: Mapping[str, Any],
    top_k: int,
    model_submission_id: str,
    prediction_run_id: str,
    generated_at: str,
    model_digest: str,
    context_policy_digest: str,
    candidate_count: int,
    candidate_set_digest: str,
    context_checksum: str | None = None,
    context_policy_label: str | None = None,
    model_inference_policy: str | None = None,
    include_full_score_order_digest: bool = True,
) -> dict[str, Any]:
    digest = full_score_order_digest(ranked) if include_full_score_order_digest else None
    return make_compact_record_from_parts(
        target=target,
        rank_stats=rank_stats,
        top_k_records_value=topk_records(ranked, k=top_k),
        top_k=top_k,
        model_submission_id=model_submission_id,
        prediction_run_id=prediction_run_id,
        generated_at=generated_at,
        model_digest=model_digest,
        context_policy_digest=context_policy_digest,
        candidate_count=candidate_count,
        candidate_set_digest=candidate_set_digest,
        context_checksum=context_checksum,
        context_policy_label=context_policy_label,
        model_inference_policy=model_inference_policy,
        full_score_order_digest_value=digest,
    )
