#!/usr/bin/env python3
"""Compute deterministic vocab bucket offsets and metadata manifests."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.vocab_common_v3 import MIN_ITEM_ID, NORMALIZER_VERSION, NUM_DOMAINS


def _count(reduced_root, domain, bucket):
    path = os.path.join(reduced_root, f"domain_{domain}", f"bucket_{bucket:04d}", "count.json")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return int(json.load(f).get("count", 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reduced_root", required=True)
    p.add_argument("--output_dir", required=True, help="Final vocab root.")
    p.add_argument("--num_buckets", type=int, default=4096)
    p.add_argument("--data_version", default="unknown")
    p.add_argument("--layout_version", default="layout_v1")
    p.add_argument("--git_sha", default=os.environ.get("GIT_COMMIT", "unknown"))
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    offsets = {"num_buckets": args.num_buckets, "min_item_id": MIN_ITEM_ID, "domains": {}}
    meta_domains = {}
    total = 0
    for domain in range(NUM_DOMAINS):
        next_id = MIN_ITEM_ID
        buckets = {}
        for bucket in range(args.num_buckets):
            count = _count(args.reduced_root, domain, bucket)
            buckets[str(bucket)] = {"start_id": next_id, "count": count}
            next_id += count
        n_unique = next_id - MIN_ITEM_ID
        total += n_unique
        offsets["domains"][str(domain)] = {"buckets": buckets}
        manifest = {"num_buckets": args.num_buckets, "min_item_id": MIN_ITEM_ID, "buckets": buckets}
        for kind in ("text2id", "id2text"):
            mdir = os.path.join(args.output_dir, f"domain_{domain}_{kind}")
            os.makedirs(mdir, exist_ok=True)
            with open(os.path.join(mdir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
        meta_domains[str(domain)] = {
            "num_unique_texts": n_unique,
            "max_item_id": next_id - 1,
            "shard_size": n_unique + MIN_ITEM_ID,
            "num_buckets": args.num_buckets,
        }
    offsets.update({"data_version": args.data_version, "layout_version": args.layout_version})
    with open(os.path.join(args.output_dir, "vocab_offsets.json"), "w", encoding="utf-8") as f:
        json.dump(offsets, f, indent=2, sort_keys=True)
    meta = {
        "min_item_id": MIN_ITEM_ID,
        "url_normalized": True,
        "normalizer_version": NORMALIZER_VERSION,
        "builder": "parallel_step1_v3_hashbucket",
        "num_buckets": args.num_buckets,
        "data_version": args.data_version,
        "layout_version": args.layout_version,
        "git_sha": args.git_sha,
        "domains": meta_domains,
        "total_unique_texts": total,
    }
    with open(os.path.join(args.output_dir, "vocab_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"wrote offsets and vocab_meta for {total} texts", flush=True)


if __name__ == "__main__":
    main()
