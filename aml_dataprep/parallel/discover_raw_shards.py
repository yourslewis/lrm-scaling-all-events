#!/usr/bin/env python3
"""Discover raw train/val TSV shards and write a source manifest."""

# Workflow notes:
# This is the discovery/root-of-truth step for the generated fan-out workflow:
# enumerate train/val TSV shards once and emit a deterministic JSONL manifest
# that downstream relay, vocab, and parquet workers address by split+shard_index.
# Performance tricks:
# - Store only URIs and sizes in the manifest; data bytes are streamed later.
# - Sort shard names so retries and generated AML jobs remain deterministic.

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


def _azureml_paths_prefix(root):
    marker = "/paths/"
    if marker not in root:
        return root.rstrip("/"), ""
    prefix, path = root.split(marker, 1)
    return prefix.rstrip("/") + "/paths", path.strip("/")


def _uri_join(root, path):
    """Return a canonical source URI for fsspec-openable manifest rows.

    azureml.fsspec.fs.ls() may return either a fully-qualified azureml:// URI or
    a datastore-relative path like local/User/.../train/foo.tsv. The relay step
    needs fully-qualified /datastores/.../paths/... URIs, and blindly appending
    the listed path to source_root can duplicate the datastore path and yield
    StreamError(NotFound).
    """
    if not root.startswith("azureml://"):
        return os.path.abspath(path)
    if path.startswith("azureml://"):
        return path

    prefix, root_path = _azureml_paths_prefix(root.rstrip("/"))
    listed = path.lstrip("/")
    if root_path and listed.startswith(root_path.rstrip("/") + "/"):
        return prefix + "/" + listed
    return root.rstrip("/") + "/" + listed


def _size(root, path):
    try:
        if root.startswith("azureml://"):
            from azureml.fsspec import AzureMachineLearningFileSystem

            uri = _uri_join(root, path)
            return AzureMachineLearningFileSystem(uri.rsplit("/", 1)[0]).info(uri).get("size")
        return os.path.getsize(path)
    except Exception:
        return None


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
            source_uri = _uri_join(args.source_root, path)
            name = os.path.basename(urlparse(source_uri).path)
            rows.append({
                "split": split,
                "shard_index": shard_index,
                "source_uri": source_uri,
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
