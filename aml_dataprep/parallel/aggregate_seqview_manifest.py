#!/usr/bin/env python3
"""Aggregate per-shard seqview stats into metadata.json."""
import argparse
import glob
import json
import os

DOMAIN_OFFSET = 1_000_000_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqview_dir", required=True)
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--output_dir", required=True, help="Writable output directory for metadata.json")
    p.add_argument("--mode", choices=["ads_only", "all_events"], default="all_events")
    p.add_argument("--embedding_model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--embedding_dim", type=int, default=384)
    args = p.parse_args()

    stats = []
    for path in sorted(glob.glob(os.path.join(args.seqview_dir, "_stats", "part_*.json"))):
        with open(path, encoding="utf-8") as f:
            stats.append(json.load(f))
    with open(os.path.join(args.vocab_dir, "vocab_meta.json"), encoding="utf-8") as f:
        vocab_meta = json.load(f)
    num_train = sum(s["row_count"] for s in stats if s["split"] == "train")
    num_eval = sum(s["row_count"] for s in stats if s["split"] == "val")
    metadata = {
        "mode": args.mode,
        "num_train_users": num_train,
        "num_eval_users": num_eval,
        "domain_offset": DOMAIN_OFFSET,
        "miss_count": sum(s["miss_count"] for s in stats),
        "max_sequence_length": max((s["max_seq_len"] for s in stats), default=0),
        "parts": stats,
        "vocab_dir": args.vocab_dir,
        "embedding_model": args.embedding_model,
        "embedding_dim": args.embedding_dim,
        "vocab_format": vocab_meta.get("builder", "unknown"),
        "domains": {
            str(d): {
                "shard_size": vocab_meta["domains"][str(d)]["shard_size"],
                "num_items": vocab_meta["domains"][str(d)]["num_unique_texts"],
            }
            for d in range(5)
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"wrote {out_path} train={num_train} eval={num_eval}", flush=True)


if __name__ == "__main__":
    main()
