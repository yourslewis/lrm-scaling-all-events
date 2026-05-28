#!/usr/bin/env python3
"""Pre-cook raw frozen item embeddings for selected v001 negative banks.

The output is model-independent: it stores candidate ids plus raw frozen
embedding vectors for selected (domain_id, bank_id) pairs. Model-specific
projection is still applied by the inference runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


def load_module(name: str, path: str | Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def stable_sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_bank_artifact(bank_root: Path, domain_id: int) -> dict[str, Any]:
    path = bank_root / "banks" / f"domain_{domain_id}_banks.production.json"
    if not path.exists():
        # Some callers may pass the banks directory directly.
        path = bank_root / f"domain_{domain_id}_banks.production.json"
    if not path.exists():
        raise FileNotFoundError(f"missing bank artifact for domain {domain_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--bank-generator", required=True)
    ap.add_argument("--embedding-root", required=True)
    ap.add_argument("--selected-bank-subset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--domain-offset", type=int, default=1_000_000_000)
    ap.add_argument("--dtype", choices=["float16"], default="float16")
    args = ap.parse_args()

    bank_root = Path(args.bank_root)
    embedding_root = Path(args.embedding_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_manifest = json.loads(Path(args.selected_bank_subset).read_text(encoding="utf-8"))
    domains = {int(k): [int(x) for x in v] for k, v in selected_manifest["domains"].items()}
    generator = load_module("banked_candidate_generator_v001", args.bank_generator)

    domain_meta: dict[str, Any] = {}
    total_banks = 0
    total_bytes = 0
    started = time.time()
    for domain_id, bank_ids in sorted(domains.items()):
        t0 = time.time()
        artifact = load_bank_artifact(bank_root, domain_id)
        raw_path = embedding_root / f"domain_{domain_id}" / "shard_0.npy"
        raw = np.load(raw_path, mmap_mode="r")
        if raw.ndim != 2:
            raise RuntimeError(f"expected 2D raw embedding shard, got {raw.shape}: {raw_path}")
        raw_dim = int(raw.shape[1])
        count = None
        domain_dir = output_dir / f"domain_{domain_id}"
        domain_dir.mkdir(parents=True, exist_ok=True)
        bank_ids_arr = np.asarray(bank_ids, dtype=np.int32)
        np.save(domain_dir / "bank_ids.npy", bank_ids_arr, allow_pickle=False)

        candidate_ids_path = domain_dir / "candidate_ids.int64.npy"
        embeddings_path = domain_dir / f"raw_embeddings.{args.dtype}.npy"

        candidate_ids_mm = None
        embeddings_mm = None
        for bank_pos, bank_id in enumerate(bank_ids):
            base_ids = [int(x) for x in generator.materialize_bank(artifact, bank_id, expected_domain_id=domain_id)]
            if count is None:
                count = len(base_ids)
                candidate_ids_mm = np.lib.format.open_memmap(candidate_ids_path, mode="w+", dtype=np.int64, shape=(len(bank_ids), count))
                embeddings_mm = np.lib.format.open_memmap(embeddings_path, mode="w+", dtype=np.float16, shape=(len(bank_ids), count, raw_dim))
            if len(base_ids) != count:
                raise RuntimeError(f"domain={domain_id} bank={bank_id} count {len(base_ids)} != expected {count}")
            local_ids = np.asarray([x % args.domain_offset for x in base_ids], dtype=np.int64)
            if int(local_ids.min()) < 0 or int(local_ids.max()) >= raw.shape[0]:
                raise RuntimeError(f"domain={domain_id} bank={bank_id} local id outside raw shape {raw.shape}")
            assert candidate_ids_mm is not None and embeddings_mm is not None
            candidate_ids_mm[bank_pos, :] = np.asarray(base_ids, dtype=np.int64)
            embeddings_mm[bank_pos, :, :] = raw[local_ids].astype(np.float16, copy=False)
            if (bank_pos + 1) % 10 == 0 or bank_pos + 1 == len(bank_ids):
                print(json.dumps({"domain_id": domain_id, "banks_done": bank_pos + 1, "banks_total": len(bank_ids), "elapsed_s": round(time.time() - t0, 1)}), flush=True)
        if candidate_ids_mm is not None:
            candidate_ids_mm.flush()
        if embeddings_mm is not None:
            embeddings_mm.flush()
        bytes_written = embeddings_path.stat().st_size + candidate_ids_path.stat().st_size + (domain_dir / "bank_ids.npy").stat().st_size
        total_bytes += bytes_written
        total_banks += len(bank_ids)
        domain_meta[str(domain_id)] = {
            "bank_ids": bank_ids,
            "bank_count": len(bank_ids),
            "candidate_count": int(count or 0),
            "raw_dim": raw_dim,
            "raw_embedding_path": str(raw_path),
            "raw_embedding_shape": list(raw.shape),
            "bank_ids_file": f"domain_{domain_id}/bank_ids.npy",
            "candidate_ids_file": f"domain_{domain_id}/candidate_ids.int64.npy",
            "raw_embeddings_file": f"domain_{domain_id}/raw_embeddings.{args.dtype}.npy",
            "shape": [len(bank_ids), int(count or 0), raw_dim],
            "bytes": bytes_written,
        }

    manifest = {
        "schema": "lrm_v001_raw_candidate_bank_cache_v001",
        "created_at_unix_s": int(time.time()),
        "bank_root": str(bank_root),
        "bank_generator": str(Path(args.bank_generator)),
        "bank_generator_digest": sha256_file(args.bank_generator),
        "embedding_root": str(embedding_root),
        "selected_bank_subset": str(Path(args.selected_bank_subset)),
        "selected_bank_subset_digest": stable_sha256_json(selected_manifest),
        "dtype": args.dtype,
        "domain_offset": args.domain_offset,
        "total_banks": total_banks,
        "total_bytes": total_bytes,
        "domains": domain_meta,
        "elapsed_s": round(time.time() - started, 3),
    }
    manifest["digest"] = stable_sha256_json(manifest)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest_digest": manifest["digest"], "total_banks": total_banks, "total_gib": total_bytes / 1024**3}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
