"""Chunked converter: process TSV files one chunk at a time to avoid OOM."""
import argparse, json, hashlib, os, logging, glob
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import Set, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EVENT_TYPE_DICT = {
    "UNK":0,"NativeClick":1,"SearchClick":2,"EdgePageTitle":3,"EdgeSearchQuery":4,
    "OrganicSearchQuery":5,"UET":6,"OutlookSenderDomain":7,"UETShoppingCart":8,
    "UETShoppingView":9,"AbandonCart":10,"EdgeShoppingCart":11,"EdgeShoppingPurchase":12,
    "ChromePageTitle":13,"MSN":14,
}
AD_EVENTS = {"SearchClick","NativeClick"}
BROWSING_EVENTS = {"EdgePageTitle","MSN","ChromePageTitle","UET","UETShoppingView"}
SEARCH_EVENTS = {"OrganicSearchQuery","EdgeSearchQuery"}
PURCHASE_EVENTS = {"UETShoppingCart","AbandonCart","EdgeShoppingCart","EdgeShoppingPurchase"}
OTHER_EVENTS = {"OutlookSenderDomain"}
ALL_EVENTS = AD_EVENTS|BROWSING_EVENTS|SEARCH_EVENTS|PURCHASE_EVENTS|OTHER_EVENTS

EVENT_TO_DOMAIN = {}
for e in AD_EVENTS:       EVENT_TO_DOMAIN[e]=0
for e in BROWSING_EVENTS: EVENT_TO_DOMAIN[e]=1
for e in SEARCH_EVENTS:   EVENT_TO_DOMAIN[e]=2
for e in PURCHASE_EVENTS: EVENT_TO_DOMAIN[e]=3
for e in OTHER_EVENTS:    EVENT_TO_DOMAIN[e]=4

DOMAIN_OFFSET = 1_000_000_000
HASH_SPACE = 500_000
MIN_ITEM_ID = 20
EMBD_DIM = 64

def text_to_item_id(text, max_id=HASH_SPACE, min_id=MIN_ITEM_ID):
    h = int(hashlib.md5(text.encode("utf-8",errors="replace")).hexdigest(),16)
    return min_id + (h % (max_id - min_id))

def encode_event(event):
    etype = event.get("Type","UNK")
    if etype not in EVENT_TO_DOMAIN: return None, etype, 0
    domain = EVENT_TO_DOMAIN[etype]
    texts = event.get("Texts",["",""])
    text_content = " ".join(str(t) for t in texts if t).strip()
    if not text_content: text_content = etype
    item_id = text_to_item_id(text_content)
    encoded_id = domain * DOMAIN_OFFSET + item_id
    time_str = event.get("time","")
    try: ts = int(datetime.strptime(time_str,"%Y-%m-%d %H:%M").timestamp())
    except: ts = 0
    return encoded_id, etype, ts

def process_chunk_file(tsv_path, allowed, min_events=2):
    """Process one TSV chunk file, yield rows."""
    rows = []
    errors = 0
    with open(tsv_path) as f:
        _header = f.readline()
        for line in f:
            parts = line.strip().split("\t",1)
            if len(parts)<2: continue
            try: events = json.loads(parts[1])
            except json.JSONDecodeError:
                errors += 1
                continue
            encoded_ids, types, timestamps = [],[],[]
            for ev in events:
                if ev.get("Type") not in allowed: continue
                eid, etype, ts = encode_event(ev)
                if eid is None: continue
                encoded_ids.append(eid)
                types.append(etype)
                timestamps.append(ts)
            if len(encoded_ids) >= min_events:
                rows.append({
                    "user_id": parts[0],
                    "encoded_ids": encoded_ids,
                    "types": types,
                    "timestamps_unix": timestamps,
                    "n_ads": sum(1 for t in types if t in AD_EVENTS),
                })
    logging.info(f"Processed {tsv_path}: {len(rows)} users ({errors} errors)")
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help="Dir with train_chunk_*.tsv")
    p.add_argument("--val_dir", default=None, help="Dir with val_chunk_*.tsv (optional)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["ads_only","all_events"], default="all_events")
    p.add_argument("--eval_users", type=int, default=5000)
    p.add_argument("--min_ad_events_eval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    allowed = ALL_EVENTS if args.mode == "all_events" else AD_EVENTS
    rng = np.random.default_rng(args.seed)

    # Process train chunks
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "train_chunk_*.tsv")))
    logging.info(f"Found {len(train_files)} train chunks")

    train_dir = os.path.join(args.output_dir, "train")
    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    total_train = 0
    eval_rows = []
    need_eval = args.eval_users if args.val_dir is None else 0

    for i, tf in enumerate(train_files):
        rows = process_chunk_file(tf, allowed)

        if need_eval > 0 and len(eval_rows) < args.eval_users:
            # Sample eval users from this chunk
            eligible = [r for r in rows if r["n_ads"] >= args.min_ad_events_eval]
            n_take = min(len(eligible), args.eval_users - len(eval_rows))
            if n_take > 0:
                idx = rng.permutation(len(eligible))[:n_take]
                eval_set = {id(eligible[j]) for j in idx}
                eval_rows.extend(eligible[j] for j in idx)
                rows = [r for r in rows if id(r) not in eval_set]

        # Remove n_ads helper
        for r in rows: del r["n_ads"]
        
        df = pd.DataFrame(rows)
        path = os.path.join(train_dir, f"part_{i:04d}.parquet")
        df.to_parquet(path, index=False)
        total_train += len(rows)
        logging.info(f"Wrote {path} ({len(df)} rows), total_train={total_train}")
        del rows, df

    # Process val chunks if provided
    if args.val_dir:
        val_files = sorted(glob.glob(os.path.join(args.val_dir, "val_chunk_*.tsv")))
        logging.info(f"Found {len(val_files)} val chunks")
        for vf in val_files:
            eval_rows.extend(process_chunk_file(vf, allowed))

    # Write eval
    for r in eval_rows:
        if "n_ads" in r: del r["n_ads"]
    df_eval = pd.DataFrame(eval_rows)
    eval_path = os.path.join(eval_dir, "part_0000.parquet")
    df_eval.to_parquet(eval_path, index=False)
    logging.info(f"Wrote {eval_path} ({len(df_eval)} eval rows)")

    # Generate embeddings
    domains = list(range(5)) if args.mode == "all_events" else [0]
    emb_dir = os.path.join(args.output_dir, "embeddings")
    for d in domains:
        d_dir = os.path.join(emb_dir, f"domain_{d}")
        os.makedirs(d_dir, exist_ok=True)
        emb = rng.standard_normal((HASH_SPACE, EMBD_DIM)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-6)
        emb = (emb / norms).astype(np.float16)
        np.save(os.path.join(d_dir, "shard_0.npy"), emb)
        logging.info(f"Wrote domain_{d}/shard_0.npy shape={emb.shape}")

    meta = {"mode":args.mode,"seed":args.seed,"num_train_users":total_train,
            "num_eval_users":len(eval_rows),"hash_space":HASH_SPACE,
            "min_item_id":MIN_ITEM_ID,"embd_dim":EMBD_DIM,
            "domain_offset":DOMAIN_OFFSET,"domains":domains}
    with open(os.path.join(args.output_dir,"metadata.json"),"w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"Done! {total_train} train + {len(eval_rows)} eval users")

if __name__ == "__main__":
    main()
