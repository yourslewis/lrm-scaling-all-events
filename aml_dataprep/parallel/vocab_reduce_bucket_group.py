#!/usr/bin/env python3
"""Reduce one hash bucket across all vocab domains.

This keeps the number of AML reducer jobs equal to ``num_buckets`` while still
hash-partitioning every domain. For example, ``num_buckets=5`` produces five
balanced reducers; reducer ``bucket=2`` processes ``domain_0/bucket_0002``
through ``domain_4/bucket_0002`` in the same AML job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.vocab_common_v3 import NUM_DOMAINS
from aml_dataprep.parallel.partition import write_ready_manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spill_root", required=True)
    p.add_argument("--bucket", required=True, type=int)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_domains", type=int, default=NUM_DOMAINS)
    args = p.parse_args()

    script = Path(__file__).resolve().parent / "vocab_reduce_bucket.py"
    completed_domains = []
    for domain in range(args.num_domains):
        subprocess.check_call([
            sys.executable,
            str(script),
            "--spill_root",
            args.spill_root,
            "--domain",
            str(domain),
            "--bucket",
            str(args.bucket),
            "--output_dir",
            args.output_dir,
        ])
        completed_domains.append(domain)

    ready_path = write_ready_manifest(
        Path(args.output_dir) / "_ready",
        f"vocab_reduce_bucket_group_b{args.bucket:04d}.json",
        {
            "stage": "vocab_reduce_bucket_group",
            "bucket": args.bucket,
            "domains": completed_domains,
            "num_domains": args.num_domains,
        },
    )
    print(
        f"reduced bucket={args.bucket:04d} across domains={completed_domains}; ready={ready_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
