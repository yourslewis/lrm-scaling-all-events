#!/usr/bin/env python3
"""Spill normalized texts from one raw shard into domain/bucket part files."""

# Workflow notes:
# Parse one raw TSV shard and spill normalized event text into domain/bucket text
# files. This is the first vocab fan-out stage and has no dependency on other
# shards.
# Performance tricks:
# - Use stable hashing so all workers agree on bucket placement.
# - Buffer open file handles with an LRU to avoid OS fd exhaustion on 512 buckets.
# - Prefer orjson when present for the JSON event hot path.

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_prep.vocab_common_v3 import EVENT_TO_DOMAIN, bucket_of, extract_text_normalized

try:
    import orjson as _json

    def loads(s):
        return _json.loads(s)
except Exception:
    import json as _json

    def loads(s):
        return _json.loads(s)


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
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--shard_index", required=True, type=int)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_buckets", type=int, default=4096)
    p.add_argument("--max_open_handles", type=int, default=128)
    args = p.parse_args()

    row = _manifest_row(args.raw_manifest, args.split, args.shard_index)
    raw_path = os.path.join(args.raw_root, row["dest_relpath"])
    handles = OrderedDict()

    def hfor(domain, bucket):
        key = (domain, bucket)
        h = handles.get(key)
        if h is not None:
            handles.move_to_end(key)
            return h
        if len(handles) >= args.max_open_handles:
            _, old = handles.popitem(last=False)
            old.close()
        bdir = os.path.join(args.output_dir, f"domain_{domain}", f"bucket_{bucket:04d}")
        os.makedirs(bdir, exist_ok=True)
        path = os.path.join(bdir, f"part_{args.split}_{args.shard_index:04d}.txt")
        h = open(path, "w", encoding="utf-8", errors="surrogatepass")
        handles[key] = h
        return h

    n_events = 0
    with open(raw_path, encoding="utf-8", errors="surrogatepass") as f:
        try:
            next(f)
        except StopIteration:
            pass
        for line in f:
            tab = line.find("\t")
            if tab < 0:
                continue
            try:
                events = loads(line[tab + 1:].rstrip("\n"))
            except Exception:
                continue
            for ev in events:
                domain = EVENT_TO_DOMAIN.get(ev.get("Type", ""))
                if domain is None:
                    continue
                text = extract_text_normalized(ev).replace("\n", " ")
                bucket = bucket_of(text, args.num_buckets)
                hfor(domain, bucket).write(text + "\n")
                n_events += 1
    for h in handles.values():
        h.close()
    print(f"spilled {n_events} events from {raw_path}", flush=True)


if __name__ == "__main__":
    main()
