"""
Step 1: Scan all TSV chunks, collect unique texts per domain, build text→ID mapping.
Outputs a JSON mapping file per domain + summary stats.
"""
import argparse, json, glob, os, logging, pickle
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EVENT_TO_DOMAIN = {
    'SearchClick': 0, 'NativeClick': 0,
    'EdgePageTitle': 1, 'MSN': 1, 'ChromePageTitle': 1, 'UET': 1, 'UETShoppingView': 1,
    'OrganicSearchQuery': 2, 'EdgeSearchQuery': 2,
    'UETShoppingCart': 3, 'AbandonCart': 3, 'EdgeShoppingCart': 3, 'EdgeShoppingPurchase': 3,
    'OutlookSenderDomain': 4,
}

MIN_ITEM_ID = 20  # 0=padding, 1-19 reserved

def extract_text(event):
    texts = event.get("Texts", ["", ""])
    text = " ".join(str(x) for x in texts if x).strip()
    return text if text else event.get("Type", "UNK")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", default=None)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # domain_id -> {text: sequential_id}
    domain_texts = defaultdict(dict)
    # domain_id -> next available ID
    domain_next_id = defaultdict(lambda: MIN_ITEM_ID)

    all_files = sorted(glob.glob(os.path.join(args.train_dir, "train_chunk_*.tsv")))
    if args.val_dir:
        all_files += sorted(glob.glob(os.path.join(args.val_dir, "val_chunk_*.tsv")))

    logging.info(f"Scanning {len(all_files)} files...")

    for fi, fpath in enumerate(all_files):
        n_events = 0
        n_new = 0
        with open(fpath) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) < 2:
                    continue
                try:
                    events = json.loads(parts[1])
                except json.JSONDecodeError:
                    continue
                for ev in events:
                    etype = ev.get("Type", "")
                    if etype not in EVENT_TO_DOMAIN:
                        continue
                    domain = EVENT_TO_DOMAIN[etype]
                    text = extract_text(ev)
                    n_events += 1
                    if text not in domain_texts[domain]:
                        item_id = domain_next_id[domain]
                        domain_texts[domain][text] = item_id
                        domain_next_id[domain] = item_id + 1
                        n_new += 1

        logging.info(f"[{fi+1}/{len(all_files)}] {os.path.basename(fpath)}: "
                     f"{n_events} events, {n_new} new texts")

    # Summary
    logging.info("=" * 60)
    logging.info("DOMAIN SUMMARY:")
    total_texts = 0
    for d in sorted(domain_texts.keys()):
        n = len(domain_texts[d])
        max_id = domain_next_id[d] - 1
        total_texts += n
        logging.info(f"  Domain {d}: {n:>10,} unique texts, ID range [{MIN_ITEM_ID}, {max_id}]")
    logging.info(f"  TOTAL: {total_texts:>10,} unique texts")

    # Save mappings
    for d in sorted(domain_texts.keys()):
        # Save as pickle (faster for large dicts)
        pkl_path = os.path.join(args.output_dir, f"domain_{d}_text2id.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(domain_texts[d], f)
        logging.info(f"Saved {pkl_path}")

        # Also save the reverse mapping (id -> text) for embedding inference
        id2text = {v: k for k, v in domain_texts[d].items()}
        pkl_path = os.path.join(args.output_dir, f"domain_{d}_id2text.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(id2text, f)
        logging.info(f"Saved {pkl_path}")

    # Save metadata
    meta = {
        "min_item_id": MIN_ITEM_ID,
        "domains": {
            d: {
                "num_unique_texts": len(domain_texts[d]),
                "max_item_id": domain_next_id[d] - 1,
                "shard_size": len(domain_texts[d]) + MIN_ITEM_ID,  # total size needed
            }
            for d in sorted(domain_texts.keys())
        },
        "total_unique_texts": total_texts,
    }
    meta_path = os.path.join(args.output_dir, "vocab_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"Saved {meta_path}")
    logging.info("Done!")


if __name__ == "__main__":
    main()
