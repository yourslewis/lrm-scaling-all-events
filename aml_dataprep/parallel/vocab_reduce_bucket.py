#!/usr/bin/env python3
"""Deduplicate and sort one vocab spill domain/bucket."""

# Workflow notes:
# Reduce one domain/bucket by reading all spill parts, deduplicating normalized
# texts, and writing a sorted unique text list plus a count sidecar.
# Performance tricks:
# - Bucket-level reduction bounds memory by domain/hash bucket.
# - Sorting after dedupe makes downstream id assignment stable across retries.

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aml_dataprep.parallel.partition import write_ready_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spill_root", required=True)
    p.add_argument("--domain", required=True, type=int)
    p.add_argument("--bucket", required=True, type=int)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    part_dir = os.path.join(args.spill_root, f"domain_{args.domain}", f"bucket_{args.bucket:04d}")
    uniq = set()
    for path in sorted(glob.glob(os.path.join(part_dir, "part_*.txt"))):
        with open(path, encoding="utf-8", errors="surrogatepass") as f:
            for line in f:
                uniq.add(line.rstrip("\n"))
    out_dir = os.path.join(args.output_dir, f"domain_{args.domain}", f"bucket_{args.bucket:04d}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "texts.txt"), "w", encoding="utf-8", errors="surrogatepass") as f:
        for text in sorted(uniq):
            f.write(text + "\n")
    count = {"domain": args.domain, "bucket": args.bucket, "count": len(uniq)}
    with open(os.path.join(out_dir, "count.json"), "w", encoding="utf-8") as f:
        json.dump(count, f, sort_keys=True)
    write_ready_manifest(Path(args.output_dir) / "_ready", f"vocab_reduce_d{args.domain}_b{args.bucket:04d}.json", count)
    print(f"reduced domain={args.domain} bucket={args.bucket:04d} count={len(uniq)}", flush=True)


if __name__ == "__main__":
    main()
