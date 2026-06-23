#!/usr/bin/env python3
"""Step 3 v3: parquet conversion using hash-bucketed vocab shards.

This is the memory-bounded companion to step1_collect_vocab_v3.py. It never
loads all domain text2id pickles at once. For each normalized text it hashes to
the same bucket as step1_v3 and lazily loads that bucket's small mapping.
"""
import argparse
import glob
import hashlib
import json
import logging
import os
import pickle
from collections import OrderedDict
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EVENT_TO_DOMAIN = {
    "SearchClick": 0, "NativeClick": 0,
    "EdgePageTitle": 1, "MSN": 1, "ChromePageTitle": 1, "UET": 1, "UETShoppingView": 1,
    "OrganicSearchQuery": 2, "EdgeSearchQuery": 2,
    "UETShoppingCart": 3, "AbandonCart": 3, "EdgeShoppingCart": 3, "EdgeShoppingPurchase": 3,
    "OutlookSenderDomain": 4,
}
AD_EVENTS = {"SearchClick", "NativeClick"}
ALL_EVENTS = set(EVENT_TO_DOMAIN.keys())
DOMAIN_OFFSET = 1_000_000_000

try:
    import orjson as _json
    def loads(s): return _json.loads(s)
except Exception:
    import json as _json
    def loads(s): return _json.loads(s)


def normalize_url_to_domain(text):
    text = text.strip()
    if not text:
        return ""
    if "://" in text or text.startswith("www."):
        if not text.startswith("http"):
            text = "https://" + text
        try:
            parsed = urlparse(text)
            domain = parsed.netloc or parsed.path.split("/")[0]
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            pass
    if "." in text and " " not in text and "/" not in text:
        if text.startswith("www."):
            text = text[4:]
        return text
    return text


def extract_text_normalized(event):
    texts = event.get("Texts", ["", ""])
    t0 = str(texts[0]).strip() if len(texts) > 0 and texts[0] else ""
    t1 = str(texts[1]).strip() if len(texts) > 1 and texts[1] else ""
    if t1:
        t1 = normalize_url_to_domain(t1)
    if t0 and t1:
        return f"{t0} {t1}"
    if t0:
        return t0
    if t1:
        return t1
    return event.get("Type", "UNK")


def bucket_of(text, num_buckets):
    h = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(h, "big") % num_buckets


class ShardedText2Id:
    def __init__(self, vocab_dir, cache_buckets=64):
        self.vocab_dir = vocab_dir
        self.cache_buckets = cache_buckets
        self.manifests = {}
        self.cache = OrderedDict()
        self.monolithic = {}
        for d in range(5):
            mpath = os.path.join(vocab_dir, f"domain_{d}_text2id", "manifest.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    self.manifests[d] = json.load(f)
            else:
                self.manifests[d] = None

    def _load_bucket(self, domain, bucket):
        key = (domain, bucket)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if len(self.cache) >= self.cache_buckets:
            self.cache.popitem(last=False)
        path = os.path.join(self.vocab_dir, f"domain_{domain}_text2id", f"bucket_{bucket:04d}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                mapping = pickle.load(f)
        else:
            mapping = {}
        self.cache[key] = mapping
        return mapping

    def get(self, domain, text):
        manifest = self.manifests.get(domain)
        if manifest is not None:
            nb = int(manifest["num_buckets"])
            b = bucket_of(text, nb)
            return self._load_bucket(domain, b).get(text)
        # Backward-compatible fallback for old v2 vocab dirs.
        if domain not in self.monolithic:
            with open(os.path.join(self.vocab_dir, f"domain_{domain}_text2id.pkl"), "rb") as f:
                self.monolithic[domain] = pickle.load(f)
        return self.monolithic[domain].get(text)


def parse_time(s):
    try:
        return int(datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp())
    except Exception:
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["ads_only", "all_events"], default="all_events")
    p.add_argument("--bucket_cache_size", type=int, default=64)
    args = p.parse_args()

    allowed = ALL_EVENTS if args.mode == "all_events" else AD_EVENTS
    lookup = ShardedText2Id(args.vocab_dir, cache_buckets=args.bucket_cache_size)
    with open(os.path.join(args.vocab_dir, "vocab_meta.json")) as f:
        meta = json.load(f)

    train_dir = os.path.join(args.output_dir, "train")
    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    miss_count = 0

    def process_file(fpath):
        nonlocal miss_count
        rows = []
        with open(fpath, encoding="utf-8", errors="surrogatepass") as f:
            try:
                next(f)
            except StopIteration:
                return rows
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
                    text = extract_text_normalized(ev)
                    item_id = lookup.get(domain, text)
                    if item_id is None:
                        miss_count += 1
                        continue
                    encoded_ids.append(domain * DOMAIN_OFFSET + item_id)
                    types.append(etype)
                    timestamps.append(parse_time(ev.get("time", "")))
                if len(encoded_ids) >= 2:
                    rows.append({
                        "user_id": user_id,
                        "encoded_ids": encoded_ids,
                        "types": types,
                        "timestamps_unix": timestamps,
                    })
        return rows

    train_files = sorted(glob.glob(os.path.join(args.train_dir, "train_chunk_*.tsv")))
    logging.info(f"Processing {len(train_files)} train chunks with sharded vocab")
    total_train = 0
    for i, fpath in enumerate(train_files):
        rows = process_file(fpath)
        df = pd.DataFrame(rows)
        out = os.path.join(train_dir, f"part_{i:04d}.parquet")
        df.to_parquet(out, index=False)
        total_train += len(rows)
        logging.info(f"[{i+1}/{len(train_files)}] {os.path.basename(fpath)}: {len(rows)} users, total={total_train}, misses={miss_count}")
        del rows, df

    eval_rows = []
    if args.val_dir:
        val_files = sorted(glob.glob(os.path.join(args.val_dir, "val_chunk_*.tsv")))
        logging.info(f"Processing {len(val_files)} val chunks")
        for fpath in val_files:
            eval_rows.extend(process_file(fpath))

    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_parquet(os.path.join(eval_dir, "part_0000.parquet"), index=False)
    logging.info(f"Eval: {len(eval_rows)} users")

    dataset_meta = {
        "mode": args.mode,
        "num_train_users": total_train,
        "num_eval_users": len(eval_rows),
        "domain_offset": DOMAIN_OFFSET,
        "miss_count": miss_count,
        "vocab_dir": args.vocab_dir,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": 384,
        "vocab_format": meta.get("builder", "unknown"),
        "domains": {
            str(d): {
                "shard_size": meta["domains"][str(d)]["shard_size"],
                "num_items": meta["domains"][str(d)]["num_unique_texts"],
            }
            for d in range(5)
        },
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(dataset_meta, f, indent=2)
    logging.info(f"Done! {total_train} train + {len(eval_rows)} eval, {miss_count} misses")


if __name__ == "__main__":
    main()
