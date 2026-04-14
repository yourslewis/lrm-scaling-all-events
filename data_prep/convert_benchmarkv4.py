"""
Convert benchmark v4 TSV data into the format expected by HSTU training pipeline.

Produces:
  1. Parquet files (train/ and eval/) with columns:
     [user_id, encoded_ids, types, timestamps_unix]
  2. Random embedding shards (.npy) per domain

The encoded_id scheme: domain_id * DOMAIN_OFFSET + item_id
  - Domain 0 (Ads): SearchClick, NativeClick
  - Domain 1 (Browsing): EdgePageTitle, MSN, ChromePageTitle, UET, UETShoppingView
  - Domain 2 (Search): OrganicSearchQuery, EdgeSearchQuery
  - Domain 3 (Purchase): UETShoppingCart, AbandonCart, EdgeShoppingCart, EdgeShoppingPurchase
  - Domain 4 (Others): OutlookSenderDomain

For ads_only mode, only domain 0 events are kept.

Usage:
    python convert_benchmarkv4.py \\
        --input /path/to/train_chunk_00.tsv \\
        --output_dir /path/to/processed/ads_only \\
        --mode ads_only \\
        --eval_users 500 \\
        --min_ad_events_eval 5 \\
        --seed 42
"""

import argparse
import json
import hashlib
import os
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ── Event type string → integer (matches codebase EVENT_TYPE_DICT) ──
EVENT_TYPE_DICT = {
    "UNK": 0,
    "NativeClick": 1,
    "SearchClick": 2,
    "EdgePageTitle": 3,
    "EdgeSearchQuery": 4,
    "OrganicSearchQuery": 5,
    "UET": 6,
    "OutlookSenderDomain": 7,
    "UETShoppingCart": 8,
    "UETShoppingView": 9,
    "AbandonCart": 10,
    "EdgeShoppingCart": 11,
    "EdgeShoppingPurchase": 12,
    "ChromePageTitle": 13,
    "MSN": 14,
}

# ── Event groups ──
AD_EVENTS = {"SearchClick", "NativeClick"}
BROWSING_EVENTS = {"EdgePageTitle", "MSN", "ChromePageTitle", "UET", "UETShoppingView"}
SEARCH_EVENTS = {"OrganicSearchQuery", "EdgeSearchQuery"}
PURCHASE_EVENTS = {"UETShoppingCart", "AbandonCart", "EdgeShoppingCart", "EdgeShoppingPurchase"}
OTHER_EVENTS = {"OutlookSenderDomain"}
ALL_EVENTS = AD_EVENTS | BROWSING_EVENTS | SEARCH_EVENTS | PURCHASE_EVENTS | OTHER_EVENTS

EVENT_TO_DOMAIN = {}
for e in AD_EVENTS:       EVENT_TO_DOMAIN[e] = 0
for e in BROWSING_EVENTS: EVENT_TO_DOMAIN[e] = 1
for e in SEARCH_EVENTS:   EVENT_TO_DOMAIN[e] = 2
for e in PURCHASE_EVENTS: EVENT_TO_DOMAIN[e] = 3
for e in OTHER_EVENTS:    EVENT_TO_DOMAIN[e] = 4

DOMAIN_OFFSET = 1_000_000_000
HASH_SPACE = 500_000          # max item ID within each domain (keeps embedding shards small)
MIN_ITEM_ID = 20              # 0 = padding, 1-19 reserved
EMBD_DIM = 64                 # raw embedding dimension before projection


# ── Helpers ──

def text_to_item_id(text: str, max_id: int = HASH_SPACE, min_id: int = MIN_ITEM_ID) -> int:
    """Deterministic hash of text content → integer in [min_id, max_id)."""
    h = int(hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest(), 16)
    return min_id + (h % (max_id - min_id))


def encode_event(event: dict) -> Tuple[Optional[int], str, int]:
    """Return (encoded_id, type_str, unix_timestamp) or (None, ...) for unknown types."""
    etype = event.get("Type", "UNK")
    if etype not in EVENT_TO_DOMAIN:
        return None, etype, 0

    domain = EVENT_TO_DOMAIN[etype]
    texts = event.get("Texts", ["", ""])
    text_content = " ".join(str(t) for t in texts if t).strip()
    if not text_content:
        text_content = etype

    item_id = text_to_item_id(text_content)
    encoded_id = domain * DOMAIN_OFFSET + item_id

    time_str = event.get("time", "")
    try:
        ts = int(datetime.strptime(time_str, "%Y-%m-%d %H:%M").timestamp())
    except Exception:
        ts = 0

    return encoded_id, etype, ts


def mode_events(mode: str) -> Set[str]:
    if mode == "ads_only":
        return AD_EVENTS
    elif mode == "all_events":
        return ALL_EVENTS
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Load / Split / Convert ──

def load_users(tsv_path: str) -> List[Dict]:
    users = []
    errors = 0
    with open(tsv_path, "r") as f:
        _header = f.readline()
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            try:
                events = json.loads(parts[1])
                users.append({"uid": parts[0], "events": events})
            except json.JSONDecodeError:
                errors += 1
    logging.info(f"Loaded {len(users)} users ({errors} parse errors)")
    return users


def split_users(
    users: List[Dict],
    eval_count: int,
    min_ad_events: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    eligible, rest = [], []
    for u in users:
        n_ads = sum(1 for e in u["events"] if e.get("Type") in AD_EVENTS)
        (eligible if n_ads >= min_ad_events else rest).append(u)
    logging.info(f"Eval-eligible users (>={min_ad_events} ad events): {len(eligible)}")

    idx = rng.permutation(len(eligible))
    n = min(eval_count, len(eligible))
    eval_users = [eligible[i] for i in idx[:n]]
    train_users = [eligible[i] for i in idx[n:]] + rest
    logging.info(f"Split → {len(train_users)} train, {len(eval_users)} eval")
    return train_users, eval_users


def users_to_parquet(
    users: List[Dict],
    output_dir: str,
    allowed_types: Set[str],
    num_files: int = 1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    skipped = 0
    for u in users:
        encoded_ids, types, timestamps = [], [], []
        for ev in u["events"]:
            if ev.get("Type") not in allowed_types:
                continue
            eid, etype, ts = encode_event(ev)
            if eid is None:
                continue
            encoded_ids.append(eid)
            types.append(etype)
            timestamps.append(ts)
        if len(encoded_ids) < 2:
            skipped += 1
            continue
        rows.append({
            "user_id": u["uid"],
            "encoded_ids": encoded_ids,
            "types": types,
            "timestamps_unix": timestamps,
        })
    logging.info(f"Converted {len(rows)} users ({skipped} skipped < 2 events)")

    per_file = max(1, len(rows) // num_files)
    for i in range(num_files):
        start = i * per_file
        end = start + per_file if i < num_files - 1 else len(rows)
        df = pd.DataFrame(rows[start:end])
        path = os.path.join(output_dir, f"part_{i:04d}.parquet")
        df.to_parquet(path, index=False)
        logging.info(f"Wrote {path} ({len(df)} rows)")


def generate_embedding_shards(
    output_dir: str,
    domains: List[int],
    seed: int = 42,
    embd_dim: int = EMBD_DIM,
    num_items: int = HASH_SPACE,
) -> None:
    """Generate random normalized embedding shards per domain."""
    rng = np.random.default_rng(seed)
    for d in domains:
        d_dir = os.path.join(output_dir, f"domain_{d}")
        os.makedirs(d_dir, exist_ok=True)
        emb = rng.standard_normal((num_items, embd_dim)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-6)
        emb = (emb / norms).astype(np.float16)
        path = os.path.join(d_dir, "shard_0.npy")
        np.save(path, emb)
        logging.info(f"Wrote {path} shape={emb.shape}")


def print_stats(users: List[Dict], label: str, allowed: Set[str]) -> None:
    tc = Counter()
    total, lens = 0, []
    for u in users:
        n = sum(1 for e in u["events"] if e.get("Type") in allowed)
        lens.append(n)
        for e in u["events"]:
            t = e.get("Type")
            if t in allowed:
                tc[t] += 1
                total += 1
    logging.info(f"\n{'='*40}\n{label} | users={len(users)} events={total}")
    if lens:
        logging.info(f"  seq_len  min={min(lens)} max={max(lens)} "
                      f"mean={np.mean(lens):.1f} median={np.median(lens):.0f}")
    for t, c in tc.most_common():
        logging.info(f"  {t:30s} {c:>8d} ({c/max(total,1)*100:5.1f}%)")


# ── CLI ──

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["ads_only", "all_events"], default="ads_only")
    p.add_argument("--eval_users", type=int, default=500)
    p.add_argument("--min_ad_events_eval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_train_files", type=int, default=4)
    p.add_argument("--num_eval_files", type=int, default=1)
    args = p.parse_args()

    allowed = mode_events(args.mode)
    users = load_users(args.input)
    train_u, eval_u = split_users(users, args.eval_users, args.min_ad_events_eval, args.seed)

    print_stats(train_u, "TRAIN", allowed)
    print_stats(eval_u,  "EVAL",  allowed)

    users_to_parquet(train_u, os.path.join(args.output_dir, "train"), allowed, args.num_train_files)
    users_to_parquet(eval_u,  os.path.join(args.output_dir, "eval"),  allowed, args.num_eval_files)

    # Determine which domains appear
    if args.mode == "ads_only":
        domains = [0]
    else:
        domains = list(range(5))
    generate_embedding_shards(os.path.join(args.output_dir, "embeddings"), domains, args.seed)

    # Save metadata for gin config
    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "num_train_users": len(train_u),
        "num_eval_users": len(eval_u),
        "hash_space": HASH_SPACE,
        "min_item_id": MIN_ITEM_ID,
        "embd_dim": EMBD_DIM,
        "domain_offset": DOMAIN_OFFSET,
        "domains": domains,
    }
    import json as j
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        j.dump(meta, f, indent=2)
    logging.info(f"Metadata → {args.output_dir}/metadata.json")
    logging.info("Done!")


if __name__ == "__main__":
    main()
