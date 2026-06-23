#!/usr/bin/env python3
"""Streaming compact rank evaluator for LRM-v001 execution-side runs.

This module computes the same target-level rank quantities needed for HR/AHR,
NDCG, and MRR without requiring full 10,001-candidate prediction JSONL output to
be persisted. It can also read full prediction JSONL and emit equivalent compact
rank records for smoke/equivalence validation.

All artifacts produced here are non-submission execution/evaluation artifacts;
the official v001 prediction schema and manifests are left unchanged.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BENCHMARK_VERSION = "lrm_benchmark_v001"
COMPACT_RANK_SCHEMA_VERSION = "lrm_compact_rank_record_v001"
COMPACT_EVALUATION_SCHEMA_VERSION = "lrm_compact_evaluation_result_v001"
HEADLINE_SLICES = ("all_domain", "all_ads", "cold_ads", "warm_ads")
DIAGNOSTIC_BUCKET_FAMILIES = (
    "context_length",
    "ads_history",
    "target_time",
    "last_event_recency",
    "last_ads_recency",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def load_targets_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def load_targets_from_manifest(path: str | Path) -> list[dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if "targets" not in obj:
        raise ValueError(f"target manifest {path} does not contain embedded targets; use --target-jsonl for sidecar scopes")
    return list(obj["targets"])


def is_ads_target(row: Mapping[str, Any]) -> bool:
    domain = str(row.get("target_domain") or "").lower()
    if domain == "ads":
        return True
    try:
        return int(row.get("target_canonical_domain_id")) == 4
    except Exception:  # noqa: BLE001
        return False


def average(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def user_macro(target_rows: Sequence[Mapping[str, Any]], rank_by_target: Mapping[str, Mapping[str, Any]], value_key: str) -> float | None:
    by_user: dict[str, list[float]] = collections.defaultdict(list)
    for row in target_rows:
        target_id = str(row["target_id"])
        if target_id in rank_by_target:
            by_user[str(row.get("user_id"))].append(float(rank_by_target[target_id][value_key]))
    return average([average(v) for v in by_user.values() if v])


def support_flags(target_rows: Sequence[Mapping[str, Any]], *, headline: bool) -> dict[str, bool]:
    target_count = len(target_rows)
    user_count = len({str(r.get("user_id")) for r in target_rows})
    if headline:
        return {"low_support": target_count < 1000 or user_count < 100, "invalid_support": target_count < 100 or user_count < 20}
    return {"low_support": target_count < 300 or user_count < 50, "invalid_support": target_count < 100 or user_count < 20}


def target_distribution(target_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter(str(r.get("target_domain") or r.get("target_canonical_domain_id")) for r in target_rows)
    return dict(sorted(c.items()))


def targets_per_user_stats(target_rows: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    c: collections.Counter[str] = collections.Counter(str(r.get("user_id")) for r in target_rows)
    counts = sorted(c.values())
    if not counts:
        return {"targets_per_user_min": None, "targets_per_user_median": None, "targets_per_user_max": None}
    return {
        "targets_per_user_min": counts[0],
        "targets_per_user_median": counts[len(counts) // 2],
        "targets_per_user_max": counts[-1],
    }


def metric_record(
    *,
    candidate_protocol_label: str,
    slice_id: str,
    slice_kind: str,
    metric_family: str,
    metric_name: str,
    value: float | None,
    target_rows: Sequence[Mapping[str, Any]],
    diagnostic: bool,
    headline: bool,
    k: int,
) -> dict[str, Any]:
    out = {
        "candidate_protocol_label": candidate_protocol_label,
        "slice_id": slice_id,
        "slice_kind": slice_kind,
        "metric_family": metric_family,
        "metric_name": metric_name,
        "cutoff_k": k if "@" in metric_name else None,
        "value": value,
        "target_count": len(target_rows),
        "user_count": len({str(r.get("user_id")) for r in target_rows}),
        "target_domain_distribution": target_distribution(target_rows),
        "headline": headline,
        "diagnostic": diagnostic,
        "not_used_for_primary_model_ranking": diagnostic,
    }
    out.update(targets_per_user_stats(target_rows))
    out.update(support_flags(target_rows, headline=headline))
    return out


def compute_metric_records(
    *,
    candidate_protocol_label: str,
    slice_id: str,
    slice_kind: str,
    family: str,
    rows: Sequence[Mapping[str, Any]],
    rank_by_target: Mapping[str, Mapping[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    hit_key = f"hit_at_{k}"
    ndcg_key = f"ndcg_at_{k}"
    hits = [float(rank_by_target[str(r["target_id"])][hit_key]) for r in rows if str(r["target_id"]) in rank_by_target]
    ndcgs = [float(rank_by_target[str(r["target_id"])][ndcg_key]) for r in rows if str(r["target_id"]) in rank_by_target]
    rrs = [float(rank_by_target[str(r["target_id"])] ["reciprocal_rank"]) for r in rows if str(r["target_id"]) in rank_by_target]
    headline = slice_kind == "headline" and family in {"AHR", "OHR"}
    diagnostic = not headline
    prefix = "AHR" if family == "AHR" else "OHR"
    records = [
        metric_record(
            candidate_protocol_label=candidate_protocol_label,
            slice_id=slice_id,
            slice_kind=slice_kind,
            metric_family=family,
            metric_name=f"micro_{prefix}@{k}",
            value=average(hits),
            target_rows=rows,
            diagnostic=diagnostic,
            headline=headline,
            k=k,
        ),
        metric_record(
            candidate_protocol_label=candidate_protocol_label,
            slice_id=slice_id,
            slice_kind=slice_kind,
            metric_family=family,
            metric_name=f"macro_user_{prefix}@{k}",
            value=user_macro(rows, rank_by_target, hit_key),
            target_rows=rows,
            diagnostic=diagnostic,
            headline=headline,
            k=k,
        ),
    ]
    for base_name, value, macro_key in (
        (f"micro_NDCG@{k}", average(ndcgs), None),
        (f"macro_user_NDCG@{k}", user_macro(rows, rank_by_target, ndcg_key), ndcg_key),
        ("micro_MRR", average(rrs), None),
        ("macro_user_MRR", user_macro(rows, rank_by_target, "reciprocal_rank"), "reciprocal_rank"),
    ):
        records.append(
            metric_record(
                candidate_protocol_label=candidate_protocol_label,
                slice_id=slice_id,
                slice_kind=slice_kind,
                metric_family="diagnostic_ranking",
                metric_name=base_name,
                value=value,
                target_rows=rows,
                diagnostic=True,
                headline=False,
                k=k,
            )
        )
    return records


def pessimistic_rank_from_scores(scores: Mapping[str, float], positive_item_id: str, k: int) -> dict[str, Any]:
    positive_score = float(scores[str(positive_item_id)])
    greater = sum(1 for score in scores.values() if float(score) > positive_score)
    equal_nonpositive = sum(1 for cid, score in scores.items() if str(cid) != str(positive_item_id) and float(score) == positive_score)
    rank = 1 + greater + equal_nonpositive
    return rank_fields(rank=rank, positive_score=positive_score, greater_score_count=greater, equal_score_nonpositive_count=equal_nonpositive, k=k)


def rank_fields(*, rank: int, positive_score: float | None, greater_score_count: int | None, equal_score_nonpositive_count: int | None, k: int) -> dict[str, Any]:
    return {
        "positive_rank": int(rank),
        "pessimistic_rank": int(rank),
        "positive_score": positive_score,
        "greater_score_count": greater_score_count,
        "equal_score_nonpositive_count": equal_score_nonpositive_count,
        f"hit_at_{k}": int(rank <= k),
        f"ndcg_at_{k}": (1.0 / math.log2(rank + 1)) if rank <= k else 0.0,
        "reciprocal_rank": 1.0 / rank,
    }


def compact_rank_record_from_ranked(
    *,
    target: Mapping[str, Any],
    ranked_pairs: Sequence[tuple[str, float]],
    model_submission_id: str,
    prediction_run_id: str,
    candidate_set_digest: str | None,
    k: int = 10,
    top_k: int = 0,
) -> dict[str, Any]:
    positive = str(target["positive_item_id"])
    scores = {str(cid): float(score) for cid, score in ranked_pairs}
    rank = pessimistic_rank_from_scores(scores, positive, k)
    top: list[dict[str, Any]] | None = None
    if top_k > 0:
        top = [
            {"candidate_id": str(cid), "rank": idx, "score": float(score)}
            for idx, (cid, score) in enumerate(ranked_pairs[:top_k], start=1)
        ]
    return {
        "schema_version": COMPACT_RANK_SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "model_submission_id": model_submission_id,
        "prediction_run_id": prediction_run_id,
        "target_id": target["target_id"],
        "candidate_protocol_label": target.get("candidate_protocol_label"),
        "candidate_set_id": target.get("candidate_set_id"),
        "candidate_set_digest": candidate_set_digest or target.get("candidate_set_digest"),
        "candidate_count": len(ranked_pairs),
        "rank_semantics": "pessimistic_tie_rank",
        "created_at": utc_now(),
        **rank,
        "top_k": top,
        "official_contract_change": False,
    }


def compact_rank_record_from_prediction(
    *,
    target: Mapping[str, Any],
    prediction: Mapping[str, Any],
    k: int = 10,
    top_k: int = 0,
) -> dict[str, Any]:
    ranked_predictions = sorted(prediction["predictions"], key=lambda p: int(p["rank"]))
    ranked_pairs = [(str(p["candidate_id"]), float(p["score"])) for p in ranked_predictions]
    return compact_rank_record_from_ranked(
        target=target,
        ranked_pairs=ranked_pairs,
        model_submission_id=str(prediction.get("model_submission_id")),
        prediction_run_id=str(prediction.get("prediction_run_id")),
        candidate_set_digest=target.get("candidate_set_digest"),
        k=k,
        top_k=top_k,
    )


def compact_records_from_full_predictions(target_rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *, k: int, top_k: int = 0) -> list[dict[str, Any]]:
    targets = {str(t["target_id"]): t for t in target_rows}
    out: list[dict[str, Any]] = []
    for pred in predictions:
        target_id = str(pred["target_id"])
        if target_id not in targets:
            raise ValueError(f"prediction target_id not present in target rows: {target_id}")
        out.append(compact_rank_record_from_prediction(target=targets[target_id], prediction=pred, k=k, top_k=top_k))
    return out


def evaluate_compact_ranks(
    target_rows: Sequence[Mapping[str, Any]],
    compact_records: Sequence[Mapping[str, Any]],
    *,
    candidate_protocol_label: str | None = None,
    k: int = 10,
) -> dict[str, Any]:
    target_by_id = {str(t["target_id"]): t for t in target_rows}
    rank_by_target: dict[str, Mapping[str, Any]] = {}
    missing_targets: list[str] = []
    for rec in compact_records:
        target_id = str(rec["target_id"])
        if target_id not in target_by_id:
            missing_targets.append(target_id)
            continue
        rank_by_target[target_id] = rec
    omitted = sorted(set(target_by_id) - set(rank_by_target))
    protocol = candidate_protocol_label or next((str(r.get("candidate_protocol_label")) for r in compact_records if r.get("candidate_protocol_label")), "unknown")
    metric_records: list[dict[str, Any]] = []
    for slice_id in HEADLINE_SLICES:
        rows = [row for row in target_rows if slice_id in row.get("headline_slices", []) and str(row["target_id"]) in rank_by_target]
        family = "OHR" if slice_id == "all_domain" else "AHR"
        metric_records.extend(compute_metric_records(candidate_protocol_label=protocol, slice_id=slice_id, slice_kind="headline", family=family, rows=rows, rank_by_target=rank_by_target, k=k))
    for family_name in DIAGNOSTIC_BUCKET_FAMILIES:
        bucket_ids = sorted({row.get("diagnostic_buckets", {}).get(family_name) for row in target_rows if row.get("diagnostic_buckets", {}).get(family_name)})
        for bucket_id in bucket_ids:
            all_rows = [row for row in target_rows if row.get("diagnostic_buckets", {}).get(family_name) == bucket_id and str(row["target_id"]) in rank_by_target]
            ads_rows = [row for row in all_rows if is_ads_target(row)]
            metric_records.extend(compute_metric_records(candidate_protocol_label=protocol, slice_id=f"{family_name}:{bucket_id}:all_domain", slice_kind="diagnostic", family="OHR", rows=all_rows, rank_by_target=rank_by_target, k=k))
            metric_records.extend(compute_metric_records(candidate_protocol_label=protocol, slice_id=f"{family_name}:{bucket_id}:ads", slice_kind="diagnostic", family="AHR", rows=ads_rows, rank_by_target=rank_by_target, k=k))
    return {
        "schema_version": COMPACT_EVALUATION_SCHEMA_VERSION,
        "created_at": utc_now(),
        "run_status": "passed" if not missing_targets else "failed",
        "benchmark_version": BENCHMARK_VERSION,
        "cutoff_k": k,
        "tie_handling": "pessimistic_tie_rank",
        "compact_rank_count": len(compact_records),
        "per_target_count": len(rank_by_target),
        "missing_prediction_target_ids": missing_targets[:20],
        "omitted_target_count": len(omitted),
        "metrics": metric_records,
        "headline_metrics": [r for r in metric_records if r["slice_kind"] == "headline" and r["metric_family"] in {"AHR", "OHR"}],
        "diagnostic_metrics": [r for r in metric_records if r["diagnostic"]],
        "official_contract_change": False,
    }


def metric_signature(result: Mapping[str, Any]) -> dict[tuple[str, str, str], float | None]:
    sig: dict[tuple[str, str, str], float | None] = {}
    for row in result.get("metrics", []):
        sig[(str(row.get("slice_id")), str(row.get("metric_family")), str(row.get("metric_name")))] = row.get("value")
    return sig


def assert_metric_equivalence(left: Mapping[str, Any], right: Mapping[str, Any], *, atol: float = 0.0) -> None:
    l_sig = metric_signature(left)
    r_sig = metric_signature(right)
    if set(l_sig) != set(r_sig):
        raise AssertionError(f"metric key mismatch: left-only={sorted(set(l_sig)-set(r_sig))[:10]} right-only={sorted(set(r_sig)-set(l_sig))[:10]}")
    for key in sorted(l_sig):
        lv = l_sig[key]
        rv = r_sig[key]
        if lv is None or rv is None:
            if lv != rv:
                raise AssertionError(f"metric {key} differs: {lv!r} != {rv!r}")
            continue
        if abs(float(lv) - float(rv)) > atol:
            raise AssertionError(f"metric {key} differs: {lv!r} != {rv!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate LRM-v001 compact positive-rank artifacts")
    tgt = ap.add_mutually_exclusive_group(required=True)
    tgt.add_argument("--target-jsonl")
    tgt.add_argument("--target-manifest")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--compact-ranks-jsonl")
    src.add_argument("--predictions-jsonl")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-compact-ranks-jsonl")
    ap.add_argument("--candidate-protocol-label")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    targets = load_targets_from_jsonl(args.target_jsonl) if args.target_jsonl else load_targets_from_manifest(args.target_manifest)
    if args.predictions_jsonl:
        compact = compact_records_from_full_predictions(targets, read_jsonl(args.predictions_jsonl), k=args.k, top_k=args.top_k)
        if args.output_compact_ranks_jsonl:
            write_jsonl(args.output_compact_ranks_jsonl, compact)
    else:
        compact = read_jsonl(args.compact_ranks_jsonl)
    result = evaluate_compact_ranks(targets, compact, candidate_protocol_label=args.candidate_protocol_label, k=args.k)
    result["inputs"] = {
        "target_jsonl": args.target_jsonl,
        "target_manifest": args.target_manifest,
        "compact_ranks_jsonl": args.compact_ranks_jsonl,
        "predictions_jsonl": args.predictions_jsonl,
        "target_sha256": sha256_file(args.target_jsonl or args.target_manifest),
        "compact_or_prediction_sha256": sha256_file(args.compact_ranks_jsonl or args.predictions_jsonl),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_status": result["run_status"], "per_target_count": result["per_target_count"], "output_json": args.output_json}, sort_keys=True))
    return 0 if result["run_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
