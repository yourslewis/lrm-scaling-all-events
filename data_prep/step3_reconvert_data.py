"""
Step 3: Re-convert benchmark v4 data using the new text→ID mapping (no hash collisions).
Produces new parquet files with stable sequential IDs matching the embedding shards.
"""
import argparse, json, os, glob, logging, pickle
import numpy as np
import pandas as pd
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EVENT_TO_DOMAIN = {
    'SearchClick': 0, 'NativeClick': 0,
    'EdgePageTitle': 1, 'MSN': 1, 'ChromePageTitle': 1, 'UET': 1, 'UETShoppingView': 1,
    'OrganicSearchQuery': 2, 'EdgeSearchQuery': 2,
    'UETShoppingCart': 3, 'AbandonCart': 3, 'EdgeShoppingCart': 3, 'EdgeShoppingPurchase': 3,
    'OutlookSenderDomain': 4,
}
AD_EVENTS = {"SearchClick", "NativeClick"}
ALL_EVENTS = set(EVENT_TO_DOMAIN.keys())

DOMAIN_OFFSET = 1_000_000_000

EVENT_TYPE_DICT = {
    "UNK":0,"NativeClick":1,"SearchClick":2,"EdgePageTitle":3,"EdgeSearchQuery":4,
    "OrganicSearchQuery":5,"UET":6,"OutlookSenderDomain":7,"UETShoppingCart":8,
    "UETShoppingView":9,"AbandonCart":10,"EdgeShoppingCart":11,"EdgeShoppingPurchase":12,
    "ChromePageTitle":13,"MSN":14,
}

from datetime import datetime

def extract_text(event):
    texts = event.get("Texts", ["", ""])
    text = " ".join(str(x) for x in texts if x).strip()
    return text if text else event.get("Type", "UNK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_dir", required=True, help="Dir with domain_*_text2id.pkl")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["ads_only", "all_events"], default="all_events")
    args = p.parse_args()

    allowed = ALL_EVENTS if args.mode == "all_events" else AD_EVENTS

    # Load text→ID mappings
    text2id = {}
    for d in range(5):
        pkl_path = os.path.join(args.vocab_dir, f"domain_{d}_text2id.pkl")
        with open(pkl_path, "rb") as f:
            text2id[d] = pickle.load(f)
        logging.info(f"Loaded domain {d}: {len(text2id[d])} texts")

    # Load metadata for shard sizes
    with open(os.path.join(args.vocab_dir, "vocab_meta.json")) as f:
        meta = json.load(f)

    train_dir = os.path.join(args.output_dir, "train")
    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Process train chunks
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "train_chunk_*.tsv")))
    logging.info(f"Processing {len(train_files)} train chunks")

    total_train = 0
    for i, fpath in enumerate(train_files):
        rows = []
        with open(fpath) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) < 2:
                    continue
                try:
                    events = json.loads(parts[1])
                except json.JSONDecodeError:
                    continue

                encoded_ids, types, timestamps = [], [], []
                for ev in events:
                    etype = ev.get("Type", "")
                    if etype not in allowed or etype not in EVENT_TO_DOMAIN:
                        continue
                    domain = EVENT_TO_DOMAIN[etype]
                    text = extract_text(ev)
                    if text not in text2id[domain]:
                        continue  # shouldn't happen
                    item_id = text2id[domain][text]
                    encoded_id = domain * DOMAIN_OFFSET + item_id
                    
                    time_str = ev.get("time", "")
                    try:
                        ts = int(datetime.strptime(time_str, "%Y-%m-%d %H:%M").timestamp())
                    except:
                        ts = 0

                    encoded_ids.append(encoded_id)
                    types.append(etype)
                    timestamps.append(ts)

                if len(encoded_ids) >= 2:
                    rows.append({
                        "user_id": parts[0],
                        "encoded_ids": encoded_ids,
                        "types": types,
                        "timestamps_unix": timestamps,
                    })

        df = pd.DataFrame(rows)
        path = os.path.join(train_dir, f"part_{i:04d}.parquet")
        df.to_parquet(path, index=False)
        total_train += len(rows)
        logging.info(f"[{i+1}/{len(train_files)}] {os.path.basename(fpath)}: "
                     f"{len(rows)} users, total={total_train}")
        del rows, df

    # Process val chunks
    eval_rows = []
    if args.val_dir:
        val_files = sorted(glob.glob(os.path.join(args.val_dir, "val_chunk_*.tsv")))
        logging.info(f"Processing {len(val_files)} val chunks")
        for fpath in val_files:
            with open(fpath) as f:
                next(f)
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) < 2:
                        continue
                    try:
                        events = json.loads(parts[1])
                    except json.JSONDecodeError:
                        continue

                    encoded_ids, types, timestamps = [], [], []
                    for ev in events:
                        etype = ev.get("Type", "")
                        if etype not in allowed or etype not in EVENT_TO_DOMAIN:
                            continue
                        domain = EVENT_TO_DOMAIN[etype]
                        text = extract_text(ev)
                        if text not in text2id[domain]:
                            continue
                        item_id = text2id[domain][text]
                        encoded_id = domain * DOMAIN_OFFSET + item_id

                        time_str = ev.get("time", "")
                        try:
                            ts = int(datetime.strptime(time_str, "%Y-%m-%d %H:%M").timestamp())
                        except:
                            ts = 0

                        encoded_ids.append(encoded_id)
                        types.append(etype)
                        timestamps.append(ts)

                    if len(encoded_ids) >= 2:
                        eval_rows.append({
                            "user_id": parts[0],
                            "encoded_ids": encoded_ids,
                            "types": types,
                            "timestamps_unix": timestamps,
                        })

    df_eval = pd.DataFrame(eval_rows)
    eval_path = os.path.join(eval_dir, "part_0000.parquet")
    df_eval.to_parquet(eval_path, index=False)
    logging.info(f"Eval: {len(eval_rows)} users")

    # Save metadata
    dataset_meta = {
        "mode": args.mode,
        "num_train_users": total_train,
        "num_eval_users": len(eval_rows),
        "domain_offset": DOMAIN_OFFSET,
        "domains": {
            str(d): {
                "shard_size": meta["domains"][str(d)]["shard_size"],
                "num_items": meta["domains"][str(d)]["num_unique_texts"],
            }
            for d in range(5)
        }
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(dataset_meta, f, indent=2)
    logging.info(f"Done! {total_train} train + {len(eval_rows)} eval users")


if __name__ == "__main__":
    main()
