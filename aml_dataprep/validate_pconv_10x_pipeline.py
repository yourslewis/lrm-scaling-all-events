#!/usr/bin/env python3
"""Local validation for generated pconv 10x pipeline YAML."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def static_validate(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors = []
    required = [
        "discover_raw_shards:",
        "merge_relay_raw:",
        "merge_vocab_spill:",
        "merge_vocab_reduced:",
        "vocab_prefix_sum:",
        "vocab_finalize:",
        "merge_seqview:",
        "check_parquet_ready:",
        "aggregate_seqview_manifest:",
        "encode_embeddings:",
        "train:",
        "evaluate:",
    ]
    for needle in required:
        if needle not in text:
            errors.append(f"missing job {needle}")
    relay = len(re.findall(r"^  relay_shard_\d{4}:", text, re.MULTILINE))
    parquet = len(re.findall(r"^  parquet_shard_\d{4}:", text, re.MULTILINE))
    if relay == 0 or relay != parquet:
        errors.append(f"relay/parquet shard mismatch: relay={relay} parquet={parquet}")
    if "gpu_instance_count > 1 is gated" not in text:
        errors.append("missing gpu multi-node guard")
    if "{gpu_guard()}" in text:
        errors.append("unexpanded gpu guard placeholder remains")
    if "parent.jobs.aggregate_seqview_manifest.outputs.metadata" not in text:
        errors.append("train/eval dependency on aggregate metadata missing")
    if "parent.jobs.train.outputs.model" not in text:
        errors.append("evaluate dependency on train output missing")
    return errors


def aml_load_validate(path: Path) -> str:
    try:
        from azure.ai.ml import load_job
    except Exception as e:
        return f"skipped azure.ai.ml.load_job: {e}"
    load_job(str(path))
    return "azure.ai.ml.load_job: ok"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("pipeline")
    args = p.parse_args()
    path = Path(args.pipeline)
    errors = static_validate(path)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2
    print("static validation: ok")
    print(aml_load_validate(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
