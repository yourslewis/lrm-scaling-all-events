"""
Step 1 (v2): Scan all TSV chunks, collect unique texts per domain.
Normalizes Texts[1] URLs to domain before concatenation.
"""
import argparse, json, glob, os, logging, pickle, re
from collections import defaultdict
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

EVENT_TO_DOMAIN = {
    'SearchClick': 0, 'NativeClick': 0,
    'EdgePageTitle': 1, 'MSN': 1, 'ChromePageTitle': 1, 'UET': 1, 'UETShoppingView': 1,
    'OrganicSearchQuery': 2, 'EdgeSearchQuery': 2,
    'UETShoppingCart': 3, 'AbandonCart': 3, 'EdgeShoppingCart': 3, 'EdgeShoppingPurchase': 3,
    'OutlookSenderDomain': 4,
}

MIN_ITEM_ID = 20

def normalize_url_to_domain(text):
    """Extract domain from URL-like strings, stripping www. prefix."""
    text = text.strip()
    if not text:
        return ""
    # If it looks like a URL, parse it
    if "://" in text or text.startswith("www."):
        if not text.startswith("http"):
            text = "https://" + text
        try:
            parsed = urlparse(text)
            domain = parsed.netloc or parsed.path.split("/")[0]
            # Strip www.
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except:
            pass
    # If it's already a domain-like string (e.g., "amazon.com")
    if "." in text and " " not in text and "/" not in text:
        if text.startswith("www."):
            text = text[4:]
        return text
    # Otherwise return as-is (not a URL)
    return text

def extract_text_normalized(event):
    """Extract text from event, normalizing Texts[1] URL to domain."""
    texts = event.get("Texts", ["", ""])
    t0 = str(texts[0]).strip() if len(texts) > 0 and texts[0] else ""
    t1 = str(texts[1]).strip() if len(texts) > 1 and texts[1] else ""
    
    # Normalize t1 (URL/domain)
    if t1:
        t1 = normalize_url_to_domain(t1)
    
    # Concatenate
    if t0 and t1:
        return f"{t0} {t1}"
    elif t0:
        return t0
    elif t1:
        return t1
    else:
        return event.get("Type", "UNK")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", default=None)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    domain_texts = defaultdict(dict)
    domain_next_id = defaultdict(lambda: MIN_ITEM_ID)

    all_files = sorted(glob.glob(os.path.join(args.train_dir, "train_chunk_*.tsv")))
    if args.val_dir:
        all_files += sorted(glob.glob(os.path.join(args.val_dir, "val_chunk_*.tsv")))

    logging.info(f"Scanning {len(all_files)} files...")

    for fi, fpath in enumerate(all_files):
        n_events = 0
        n_new = 0
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
                for ev in events:
                    etype = ev.get("Type", "")
                    if etype not in EVENT_TO_DOMAIN:
                        continue
                    domain = EVENT_TO_DOMAIN[etype]
                    text = extract_text_normalized(ev)
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
    logging.info("DOMAIN SUMMARY (with URL normalization):")
    total_texts = 0
    for d in sorted(domain_texts.keys()):
        n = len(domain_texts[d])
        max_id = domain_next_id[d] - 1
        total_texts += n
        logging.info(f"  Domain {d}: {n:>10,} unique texts, ID range [{MIN_ITEM_ID}, {max_id}]")
    logging.info(f"  TOTAL: {total_texts:>10,} unique texts")

    # Show some examples per domain
    for d in sorted(domain_texts.keys()):
        samples = list(domain_texts[d].keys())[:3]
        logging.info(f"  Domain {d} samples: {samples}")

    # Save mappings
    for d in sorted(domain_texts.keys()):
        pkl_path = os.path.join(args.output_dir, f"domain_{d}_text2id.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(domain_texts[d], f)

        id2text = {v: k for k, v in domain_texts[d].items()}
        pkl_path = os.path.join(args.output_dir, f"domain_{d}_id2text.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(id2text, f)

        logging.info(f"Saved domain_{d} mappings")

    meta = {
        "min_item_id": MIN_ITEM_ID,
        "url_normalized": True,
        "domains": {
            d: {
                "num_unique_texts": len(domain_texts[d]),
                "max_item_id": domain_next_id[d] - 1,
                "shard_size": len(domain_texts[d]) + MIN_ITEM_ID,
            }
            for d in sorted(domain_texts.keys())
        },
        "total_unique_texts": total_texts,
    }
    with open(os.path.join(args.output_dir, "vocab_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"Saved vocab_meta.json")
    logging.info("Done!")


if __name__ == "__main__":
    main()
