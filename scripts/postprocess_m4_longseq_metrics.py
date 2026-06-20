#!/usr/bin/env python3
"""Postprocess compact_predictions.jsonl into long-sequence headline metrics.

Filters compact prediction records to:
  target.diagnostic_buckets.context_length == "ctx_len_gt_1000"
then recomputes the headline slices cold_ads, warm_ads, all_ads, all_domain using
record.target.headline_slices membership.

Ads slices report AHR@10 (hit_at_10); all_domain reports OHR@10 (hit_at_10).
Both micro/target-weighted and macro-user averages are emitted, with support.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

SLICES = ("cold_ads", "warm_ads", "all_ads", "all_domain")
ADS_SLICES = {"cold_ads", "warm_ads", "all_ads"}
DEFAULT_CONTEXT_BUCKET = "ctx_len_gt_1000"


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SliceAgg:
    def __init__(self) -> None:
        self.target_count = 0
        self.hit_sum = 0.0
        self.users: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # hits, targets
        self.domain_counts: Counter[str] = Counter()

    def add(self, user_id: str, domain: str, hit: float) -> None:
        self.target_count += 1
        self.hit_sum += hit
        self.users[user_id][0] += hit
        self.users[user_id][1] += 1.0
        if domain:
            self.domain_counts[domain] += 1

    @property
    def user_count(self) -> int:
        return len(self.users)

    def micro(self) -> float | None:
        if self.target_count == 0:
            return None
        return self.hit_sum / self.target_count

    def macro_user(self) -> float | None:
        if not self.users:
            return None
        return sum(h / n for h, n in self.users.values() if n) / len(self.users)

    def targets_per_user_summary(self) -> tuple[int | None, float | None, int | None]:
        counts = [int(v[1]) for v in self.users.values()]
        if not counts:
            return None, None, None
        return min(counts), float(median(counts)), max(counts)


def metric_record(
    *,
    slice_id: str,
    metric_name: str,
    value: float | None,
    agg: SliceAgg,
    k: int,
    context_bucket: str,
) -> dict[str, Any]:
    tmin, tmed, tmax = agg.targets_per_user_summary()
    return {
        "slice_id": slice_id,
        "slice_kind": "headline",
        "diagnostic_filter": {"target.diagnostic_buckets.context_length": context_bucket},
        "metric_name": metric_name,
        "metric_family": "AHR" if slice_id in ADS_SLICES else "OHR",
        "k": k,
        "value": value,
        "target_count": agg.target_count,
        "user_count": agg.user_count,
        "target_domain_distribution": dict(sorted(agg.domain_counts.items())),
        "targets_per_user_min": tmin,
        "targets_per_user_median": tmed,
        "targets_per_user_max": tmax,
        "headline": True,
        "low_support": agg.target_count < 100,
        "invalid_support": agg.target_count == 0,
        "metric_impl_version": "m4_seq_len_longseq_postprocess_v001",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", required=True, help="input compact_predictions.jsonl")
    ap.add_argument("--output", required=True, help="output compact_metrics_longseq_gt1000.json")
    ap.add_argument("--context-bucket", default=DEFAULT_CONTEXT_BUCKET)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    aggs = {s: SliceAgg() for s in SLICES}
    records_read = 0
    records_kept = 0
    compact_path = Path(args.compact)

    with compact_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records_read += 1
            rec = json.loads(line)
            target = rec.get("target") or {}
            buckets = target.get("diagnostic_buckets") or {}
            if buckets.get("context_length") != args.context_bucket:
                continue
            records_kept += 1
            headline_slices = set(target.get("headline_slices") or [])
            user_id = str(target.get("user_id") or "__missing_user__")
            domain = str(target.get("target_domain") or "")
            rank_stats = rec.get("rank_stats") or {}
            hit = float(rank_stats.get("hit_at_10") or 0.0)
            for slice_id in SLICES:
                if slice_id in headline_slices:
                    aggs[slice_id].add(user_id, domain, hit)

    metrics: list[dict[str, Any]] = []
    for slice_id in SLICES:
        family = "AHR" if slice_id in ADS_SLICES else "OHR"
        agg = aggs[slice_id]
        metrics.append(metric_record(
            slice_id=slice_id,
            metric_name=f"micro_{family}@{args.k}",
            value=agg.micro(),
            agg=agg,
            k=args.k,
            context_bucket=args.context_bucket,
        ))
        metrics.append(metric_record(
            slice_id=slice_id,
            metric_name=f"macro_user_{family}@{args.k}",
            value=agg.macro_user(),
            agg=agg,
            k=args.k,
            context_bucket=args.context_bucket,
        ))

    out = {
        "artifact": "m4_seq_len_long_sequence_compact_metrics",
        "created_at": utc(),
        "source_compact_predictions": str(compact_path),
        "context_length_filter": args.context_bucket,
        "cutoff_k": args.k,
        "records_read": records_read,
        "records_kept": records_kept,
        "headline_slices": list(SLICES),
        "headline_metrics": metrics,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path} records_read={records_read} records_kept={records_kept}")


if __name__ == "__main__":
    main()
