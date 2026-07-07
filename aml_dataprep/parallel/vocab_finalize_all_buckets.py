#!/usr/bin/env python3
"""Finalize all reduced vocab buckets after prefix-sum fan-in."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.vocab_common_v3 import NUM_DOMAINS
from aml_dataprep.parallel.partition import write_ready_manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reduced_root", required=True)
    p.add_argument("--vocab_root", required=True)
    p.add_argument("--offsets_root", required=True)
    p.add_argument("--num_buckets", type=int, default=4096)
    args = p.parse_args()

    os.makedirs(args.vocab_root, exist_ok=True)
    for name in ("vocab_offsets.json", "vocab_meta.json"):
        shutil.copy2(Path(args.offsets_root) / name, Path(args.vocab_root) / name)
    for entry in Path(args.offsets_root).glob("domain_*_*"):
        if not entry.is_dir():
            continue
        dst = Path(args.vocab_root) / entry.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(entry, dst)

    script = Path(__file__).resolve().parent / "vocab_finalize_bucket.py"
    finalized = 0
    for domain in range(NUM_DOMAINS):
        for bucket in range(args.num_buckets):
            subprocess.check_call([
                sys.executable,
                str(script),
                "--reduced_root",
                args.reduced_root,
                "--vocab_root",
                args.vocab_root,
                "--domain",
                str(domain),
                "--bucket",
                str(bucket),
            ])
            finalized += 1
    ready = write_ready_manifest(Path(args.vocab_root) / "_ready", "vocab_finalize_ready.json", {
        "stage": "vocab_finalize",
        "num_domains": NUM_DOMAINS,
        "num_buckets": args.num_buckets,
        "finalized_buckets": finalized,
    })
    print(f"finalized {finalized} vocab buckets; ready={ready}", flush=True)


if __name__ == "__main__":
    main()
