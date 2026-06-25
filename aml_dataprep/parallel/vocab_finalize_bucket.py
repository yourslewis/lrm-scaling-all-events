#!/usr/bin/env python3
"""Write one finalized vocab bucket mapping from reduced texts and offsets."""
import argparse
import json
import os
import pickle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reduced_root", required=True)
    p.add_argument("--vocab_root", required=True)
    p.add_argument("--domain", required=True, type=int)
    p.add_argument("--bucket", required=True, type=int)
    args = p.parse_args()

    offsets_path = os.path.join(args.vocab_root, "vocab_offsets.json")
    with open(offsets_path, encoding="utf-8") as f:
        offsets = json.load(f)
    info = offsets["domains"][str(args.domain)]["buckets"][str(args.bucket)]
    start_id = int(info["start_id"])
    count = int(info["count"])
    texts_path = os.path.join(args.reduced_root, f"domain_{args.domain}", f"bucket_{args.bucket:04d}", "texts.txt")
    text2id = {}
    if os.path.exists(texts_path):
        with open(texts_path, encoding="utf-8", errors="surrogatepass") as f:
            for i, line in enumerate(f):
                text2id[line.rstrip("\n")] = start_id + i
    if len(text2id) != count:
        raise ValueError(f"expected {count} texts, got {len(text2id)}")
    id2text = {item_id: text for text, item_id in text2id.items()}
    tdir = os.path.join(args.vocab_root, f"domain_{args.domain}_text2id")
    idir = os.path.join(args.vocab_root, f"domain_{args.domain}_id2text")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(idir, exist_ok=True)
    with open(os.path.join(tdir, f"bucket_{args.bucket:04d}.pkl"), "wb") as f:
        pickle.dump(text2id, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(idir, f"bucket_{args.bucket:04d}.pkl"), "wb") as f:
        pickle.dump(id2text, f, protocol=pickle.HIGHEST_PROTOCOL)
    part_dir = os.path.join(args.vocab_root, "manifest_parts", f"domain_{args.domain}")
    os.makedirs(part_dir, exist_ok=True)
    with open(os.path.join(part_dir, f"bucket_{args.bucket:04d}.json"), "w", encoding="utf-8") as f:
        json.dump({"domain": args.domain, "bucket": args.bucket, **info}, f, sort_keys=True)
    print(f"finalized domain={args.domain} bucket={args.bucket:04d} count={count}", flush=True)


if __name__ == "__main__":
    main()
