#!/usr/bin/env python3
"""Convert one raw TSV shard to one seqview parquet part."""

# Workflow notes:
# Convert exactly one raw TSV shard into one seqview parquet part after vocab ids
# are finalized. This is designed for AML fan-out: many instances can run the
# same script with different split/shard_index values.
# Performance tricks:
# - Use a bounded LRU cache of vocab buckets to avoid loading the full text2id map.
# - Keep the output partition one-to-one with input shards for simple retries.
# - Prefer orjson when available for hot JSON event parsing.

import argparse
import json
import os
import pickle
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.vocab_common_v3 import EVENT_TO_DOMAIN, bucket_of, extract_text_normalized

AD_EVENTS = {"SearchClick", "NativeClick"}
ALL_EVENTS = set(EVENT_TO_DOMAIN.keys())
DOMAIN_OFFSET = 1_000_000_000

try:
    import orjson as _json

    def loads(s):
        return _json.loads(s)
except Exception:
    import json as _json

    def loads(s):
        return _json.loads(s)


class ShardedText2Id:
    def __init__(self, vocab_dir, cache_buckets=64):
        self.vocab_dir = vocab_dir
        self.cache_buckets = cache_buckets
        self.manifests = {}
        self.cache = OrderedDict()
        for domain in range(5):
            mpath = os.path.join(vocab_dir, f"domain_{domain}_text2id", "manifest.json")
            with open(mpath, encoding="utf-8") as f:
                self.manifests[domain] = json.load(f)

    def get(self, domain, text):
        nb = int(self.manifests[domain]["num_buckets"])
        bucket = bucket_of(text, nb)
        key = (domain, bucket)
        if key not in self.cache:
            if len(self.cache) >= self.cache_buckets:
                self.cache.popitem(last=False)
            path = os.path.join(self.vocab_dir, f"domain_{domain}_text2id", f"bucket_{bucket:04d}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.cache[key] = pickle.load(f)
            else:
                self.cache[key] = {}
        else:
            self.cache.move_to_end(key)
        return self.cache[key].get(text)


def parse_time(s):
    try:
        return int(datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp())
    except Exception:
        return 0


def _manifest_row(manifest, split, shard_index):
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == split and int(row["shard_index"]) == shard_index:
                return row
    raise ValueError(f"no manifest row for {split} shard {shard_index}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_manifest", required=True)
    p.add_argument("--raw_root", required=True)
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--shard_index", required=True, type=int)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["ads_only", "all_events"], default="all_events")
    p.add_argument("--bucket_cache_size", type=int, default=64)
    args = p.parse_args()

    import pandas as pd

    row = _manifest_row(args.raw_manifest, args.split, args.shard_index)
    raw_path = os.path.join(args.raw_root, row["dest_relpath"])
    out_split = "train" if args.split == "train" else "eval"
    out_dir = os.path.join(args.output_dir, out_split)
    stats_dir = os.path.join(args.output_dir, "_stats")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    allowed = ALL_EVENTS if args.mode == "all_events" else AD_EVENTS
    lookup = ShardedText2Id(args.vocab_dir, args.bucket_cache_size)
    rows = []
    miss_count = 0
    max_seq_len = 0
    with open(raw_path, encoding="utf-8", errors="surrogatepass") as f:
        try:
            next(f)
        except StopIteration:
            pass
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            user_id = line[:tab]
            try:
                events = loads(line[tab + 1:].rstrip("\n"))
            except Exception:
                continue
            encoded_ids, types, timestamps = [], [], []
            for ev in events:
                etype = ev.get("Type", "")
                if etype not in allowed or etype not in EVENT_TO_DOMAIN:
                    continue
                domain = EVENT_TO_DOMAIN[etype]
                item_id = lookup.get(domain, extract_text_normalized(ev))
                if item_id is None:
                    miss_count += 1
                    continue
                encoded_ids.append(domain * DOMAIN_OFFSET + item_id)
                types.append(etype)
                timestamps.append(parse_time(ev.get("time", "")))
            if len(encoded_ids) >= 2:
                max_seq_len = max(max_seq_len, len(encoded_ids))
                rows.append({"user_id": user_id, "encoded_ids": encoded_ids, "types": types, "timestamps_unix": timestamps})
    part_name = f"part_{args.split}_{args.shard_index:04d}.parquet"
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, part_name), index=False)
    stats = {
        "split": args.split,
        "shard_index": args.shard_index,
        "parquet_relpath": f"{out_split}/{part_name}",
        "row_count": len(rows),
        "miss_count": miss_count,
        "max_seq_len": max_seq_len,
    }
    with open(os.path.join(stats_dir, f"part_{args.split}_{args.shard_index:04d}.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, sort_keys=True)
    print(f"wrote {part_name}: rows={len(rows)} misses={miss_count}", flush=True)


if __name__ == "__main__":
    main()
