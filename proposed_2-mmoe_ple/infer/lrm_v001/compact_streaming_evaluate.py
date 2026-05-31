#!/usr/bin/env python3
"""Stream aggregate metrics from LRM-v001 compact prediction records.

The input records are derived execution artifacts produced by
`sequential_submission_infer.py --output-mode compact|both`. They already contain
exact positive-rank statistics computed immediately after scoring all candidates
for each target. This script recomputes aggregate metrics without loading or
writing the full 10,001-candidate prediction arrays.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compact_metrics import StreamingMetricAggregator  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}: line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"{path}: line {line_no}: record is not a JSON object")
            yield record


def evaluate_compact_jsonl(
    compact_jsonl: Path,
    *,
    k: int,
    candidate_protocol_label: str,
    target_manifest_checksum: str | None = None,
) -> dict[str, Any]:
    aggregator = StreamingMetricAggregator(
        k=k,
        candidate_protocol_label=candidate_protocol_label,
        target_manifest_checksum=target_manifest_checksum,
    )
    for record in iter_jsonl(compact_jsonl):
        aggregator.add_compact_record(record)
    return aggregator.result(
        created_at=utc_now(),
        inputs={
            "compact_jsonl": str(compact_jsonl),
            "target_manifest_checksum": target_manifest_checksum,
        },
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream exact aggregate metrics from compact LRM-v001 records")
    parser.add_argument("--compact-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-protocol-label", default="banked_domain_negatives_10k_b1000_v001")
    parser.add_argument("--target-manifest-checksum")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.k <= 0:
        raise SystemExit("--k must be positive")
    result = evaluate_compact_jsonl(
        args.compact_jsonl,
        k=args.k,
        candidate_protocol_label=args.candidate_protocol_label,
        target_manifest_checksum=args.target_manifest_checksum,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"compact streaming evaluator passed: wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
