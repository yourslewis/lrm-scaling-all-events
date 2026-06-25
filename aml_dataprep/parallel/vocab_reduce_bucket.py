#!/usr/bin/env python3
"""Deduplicate and sort one vocab spill domain/bucket."""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spill_root", required=True)
    p.add_argument("--domain", required=True, type=int)
    p.add_argument("--bucket", required=True, type=int)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    part_dir = os.path.join(args.spill_root, f"domain_{args.domain}", f"bucket_{args.bucket:04d}")
    uniq = set()
    for path in sorted(glob.glob(os.path.join(part_dir, "part_*.txt"))):
        with open(path, encoding="utf-8", errors="surrogatepass") as f:
            for line in f:
                uniq.add(line.rstrip("\n"))
    out_dir = os.path.join(args.output_dir, f"domain_{args.domain}", f"bucket_{args.bucket:04d}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "texts.txt"), "w", encoding="utf-8", errors="surrogatepass") as f:
        for text in sorted(uniq):
            f.write(text + "\n")
    count = {"domain": args.domain, "bucket": args.bucket, "count": len(uniq)}
    with open(os.path.join(out_dir, "count.json"), "w", encoding="utf-8") as f:
        json.dump(count, f, sort_keys=True)
    print(f"reduced domain={args.domain} bucket={args.bucket:04d} count={len(uniq)}", flush=True)


if __name__ == "__main__":
    main()
