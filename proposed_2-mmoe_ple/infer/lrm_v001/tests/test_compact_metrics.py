from __future__ import annotations

import json
import math
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from compact_metrics import (  # noqa: E402
    StreamingMetricAggregator,
    make_compact_record,
    pessimistic_rank_from_ranked,
)


def _target(target_id: str, user: str, domain: str, positive: str, slices: list[str]) -> dict:
    return {
        "target_id": target_id,
        "user_id": user,
        "target_domain": domain,
        "target_event_type": "SearchClick" if domain == "Ads" else "PageAction",
        "positive_item_id": positive,
        "candidate_protocol_label": "banked_domain_negatives_10k_b1000_v001",
        "candidate_set_id": f"cs_{target_id}",
        "headline_slices": slices,
        "diagnostic_buckets": {
            "context_length": "short",
            "ads_history": "has_ads" if domain == "Ads" else "no_ads",
        },
    }


def _full_prediction(target: dict, ranked: list[tuple[str, float]]) -> dict:
    return {
        "target_id": target["target_id"],
        "predictions": [
            {"candidate_id": candidate_id, "rank": rank, "score": score}
            for rank, (candidate_id, score) in enumerate(ranked, start=1)
        ],
    }


def _rank_from_full_prediction(prediction: dict, positive_item_id: str, k: int) -> dict:
    scores = {row["candidate_id"]: float(row["score"]) for row in prediction["predictions"]}
    positive_score = scores[positive_item_id]
    greater = sum(1 for score in scores.values() if score > positive_score)
    equal_nonpositive = sum(
        1 for candidate_id, score in scores.items() if candidate_id != positive_item_id and score == positive_score
    )
    rank = 1 + greater + equal_nonpositive
    return {
        "rank": rank,
        "hit": float(rank <= k),
        "ndcg": (1.0 / math.log2(rank + 1)) if rank <= k else 0.0,
        "rr": 1.0 / rank,
    }


def _reference_full_jsonl_metrics(targets: list[dict], predictions: list[dict], *, k: int) -> dict[tuple[str, str], float]:
    predictions_by_target = {row["target_id"]: row for row in predictions}
    ranks = {
        target["target_id"]: _rank_from_full_prediction(predictions_by_target[target["target_id"]], target["positive_item_id"], k)
        for target in targets
    }

    # Minimal independent full-JSONL evaluator reference for headline metrics.
    out: dict[tuple[str, str], float] = {}
    for slice_id in ("cold_ads", "warm_ads", "all_ads", "all_domain"):
        rows = [target for target in targets if slice_id in target["headline_slices"]]
        if not rows:
            continue
        family = "OHR" if slice_id == "all_domain" else "AHR"
        out[(slice_id, f"micro_{family}@{k}")] = sum(ranks[row["target_id"]]["hit"] for row in rows) / len(rows)
        out[(slice_id, f"micro_NDCG@{k}")] = sum(ranks[row["target_id"]]["ndcg"] for row in rows) / len(rows)
        out[(slice_id, "micro_MRR")] = sum(ranks[row["target_id"]]["rr"] for row in rows) / len(rows)

        by_user: dict[str, list[dict]] = {}
        for row in rows:
            by_user.setdefault(row["user_id"], []).append(ranks[row["target_id"]])
        out[(slice_id, f"macro_user_{family}@{k}")] = sum(
            sum(rank["hit"] for rank in user_rows) / len(user_rows) for user_rows in by_user.values()
        ) / len(by_user)
    return out


def _metric_map(result: dict) -> dict[tuple[str, str], float]:
    return {
        (row["slice_id"], row["metric_name"]): row["value"]
        for row in result["metrics"]
        if row["slice_kind"] == "headline"
    }


def test_positive_rank_topk_and_tie_policy_are_exact() -> None:
    ranked = [("a", 1.0), ("p", 0.5), ("b", 0.5), ("c", 0.2)]

    stats = pessimistic_rank_from_ranked(ranked, "p", k=2)

    assert stats["positive_score"] == 0.5
    assert stats["greater_score_count"] == 1
    assert stats["equal_score_nonpositive_count"] == 1
    assert stats["pessimistic_rank"] == 3
    assert stats["hit_at_2"] == 0
    assert stats["ndcg_at_2"] == 0.0

    target = _target("t", "u", "Ads", "p", ["all_domain", "all_ads", "cold_ads"])
    compact = make_compact_record(
        target=target,
        ranked=ranked,
        rank_stats=stats,
        top_k=2,
        model_submission_id="m",
        prediction_run_id="r",
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:model",
        context_policy_digest="sha256:ctx",
        candidate_count=len(ranked),
        candidate_set_digest="sha256:candidates",
    )

    assert compact["rank_stats"]["pessimistic_rank"] == 3
    assert [row["candidate_id"] for row in compact["top_k"]] == ["a", "p"]
    assert compact["digests"]["full_score_order_digest"].startswith("sha256:")
    assert "predictions" not in compact


def test_streaming_metrics_match_full_jsonl_reference_on_same_sample() -> None:
    k = 2
    cases = [
        (_target("t1", "u1", "Ads", "p1", ["all_domain", "all_ads", "cold_ads"]), [("n1", 1.0), ("p1", 0.9), ("n2", 0.1)]),
        (_target("t2", "u1", "Page", "p2", ["all_domain"]), [("n3", 0.5), ("p2", 0.5), ("n4", 0.5)]),
        (_target("t3", "u2", "Ads", "p3", ["all_domain", "all_ads", "warm_ads"]), [("p3", 0.8), ("n5", 0.7), ("n6", 0.1)]),
    ]
    targets = [target for target, _ in cases]
    full_predictions = [_full_prediction(target, ranked) for target, ranked in cases]

    aggregator = StreamingMetricAggregator(k=k)
    for target, ranked in cases:
        compact = make_compact_record(
            target=target,
            ranked=ranked,
            rank_stats=pessimistic_rank_from_ranked(ranked, target["positive_item_id"], k=k),
            top_k=k,
            model_submission_id="m",
            prediction_run_id="r",
            generated_at="2026-05-27T00:00:00Z",
            model_digest="sha256:model",
            context_policy_digest="sha256:ctx",
            candidate_count=len(ranked),
            candidate_set_digest="sha256:candidates",
        )
        aggregator.add_compact_record(compact)

    compact_metrics = _metric_map(aggregator.result(created_at="2026-05-27T00:00:00Z"))
    reference_metrics = _reference_full_jsonl_metrics(targets, full_predictions, k=k)

    for key, expected in reference_metrics.items():
        assert key in compact_metrics
        assert compact_metrics[key] == expected


def test_compact_output_reduces_storage_after_scoring_all_10001_candidates() -> None:
    target = _target("t_big", "u", "Ads", "1000009999", ["all_domain", "all_ads", "warm_ads"])
    ranked = [(str(1000000000 + idx), 1.0 / (idx + 1)) for idx in range(10001)]
    stats = pessimistic_rank_from_ranked(ranked, target["positive_item_id"], k=10)
    full = {
        "schema_version": "lrm_prediction_record_v001",
        "benchmark_version": "lrm_benchmark_v001",
        "target_id": target["target_id"],
        "candidate_protocol_label": target["candidate_protocol_label"],
        "candidate_set_id": target["candidate_set_id"],
        "predictions": [
            {"candidate_id": candidate_id, "rank": rank, "score": score}
            for rank, (candidate_id, score) in enumerate(ranked, start=1)
        ],
    }
    compact = make_compact_record(
        target=target,
        ranked=ranked,
        rank_stats=stats,
        top_k=10,
        model_submission_id="m",
        prediction_run_id="r",
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:model",
        context_policy_digest="sha256:ctx",
        candidate_count=len(ranked),
        candidate_set_digest="sha256:candidates",
    )

    full_bytes = len(json.dumps(full, separators=(",", ":")).encode("utf-8"))
    compact_bytes = len(json.dumps(compact, separators=(",", ":")).encode("utf-8"))

    assert compact["candidate_count"] == 10001
    assert len(compact["top_k"]) == 10
    assert compact_bytes / full_bytes < 0.02
