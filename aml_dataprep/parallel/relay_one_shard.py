#!/usr/bin/env python3
"""Copy one manifest row into the deterministic raw layout."""
import argparse
import json
import os
import shutil


def _open_source(uri):
    if uri.startswith("azureml://"):
        from azureml.fsspec import AzureMachineLearningFileSystem

        fs = AzureMachineLearningFileSystem(uri.rsplit("/", 1)[0])
        return fs.open(uri, "rb")
    return open(uri, "rb")


def _select_row(manifest, split, shard_index):
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == split and int(row["shard_index"]) == shard_index:
                return row
    raise ValueError(f"no manifest row for {split} shard {shard_index}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--shard_index", required=True, type=int)
    p.add_argument("--output_dir", required=True, help="Raw output root.")
    p.add_argument("--chunk_bytes", type=int, default=1 << 20)
    args = p.parse_args()

    row = _select_row(args.manifest, args.split, args.shard_index)
    dst = os.path.join(args.output_dir, row["dest_relpath"])
    tmp = dst + ".tmp"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with _open_source(row["source_uri"]) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, length=args.chunk_bytes)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)
    marker = dst + ".done"
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"source_uri": row["source_uri"], "dest_relpath": row["dest_relpath"]}, f)
    print(f"relayed {row['source_uri']} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
