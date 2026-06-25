#!/usr/bin/env python3
"""Discover raw train/val TSV shards and write a source manifest."""
import argparse
import json
import os
from urllib.parse import urlparse


def _fs_for(root):
    if root.startswith("azureml://"):
        from azureml.fsspec import AzureMachineLearningFileSystem

        return AzureMachineLearningFileSystem(root), root.rstrip("/")
    return None, root.rstrip("/")


def _list_tsv(root, split):
    fs, base = _fs_for(root)
    rel_dir = f"{base}/{split}"
    if fs is None:
        names = [os.path.join(rel_dir, n) for n in os.listdir(rel_dir)] if os.path.isdir(rel_dir) else []
    else:
        names = fs.ls(rel_dir)
    return sorted(n for n in names if n.endswith(".tsv"))


def _size(root, path):
    try:
        if root.startswith("azureml://"):
            from azureml.fsspec import AzureMachineLearningFileSystem

            return AzureMachineLearningFileSystem(root).info(path).get("size")
        return os.path.getsize(path)
    except Exception:
        return None


def _uri_join(root, path):
    if root.startswith("azureml://"):
        return path if path.startswith("azureml://") else root.rstrip("/") + "/" + path.lstrip("/")
    return os.path.abspath(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_root", required=True, help="Root containing train/ and val/ TSV shards.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_version", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "raw_source_manifest.jsonl")
    rows = []
    for split in ("train", "val"):
        for shard_index, path in enumerate(_list_tsv(args.source_root, split)):
            name = os.path.basename(urlparse(path).path)
            rows.append({
                "split": split,
                "shard_index": shard_index,
                "source_uri": _uri_join(args.source_root, path),
                "dest_relpath": f"{split}/{name}",
                "size_bytes": _size(args.source_root, path),
                "etag": None,
                "data_version": args.data_version,
            })
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with open(os.path.join(args.output_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write(f"{len(rows)}\n")
    print(f"wrote {len(rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
