#!/usr/bin/env python3
"""Multi-GPU semantic encoder for step1_v3 sharded id2text vocab.

Unlike encode_embeddings_multigpu.py, this does not load a giant
`domain_<d>_id2text.pkl`. It reads `domain_<d>_id2text/bucket_XXXX.pkl` shards
emitted by step1_collect_vocab_v3.py and encodes one bucket at a time.
"""

# Workflow notes:
# 1. Iterate the sharded id2text vocab produced by step1_collect_vocab_v3.py,
# 2. split bucket work across GPUs, 3. encode each bucket, 4. stitch bucket
#    outputs into the domain_<d>/shard_0.npy arrays consumed by training.
# Performance tricks:
# - Read one bucket at a time instead of materializing the full vocab in RAM.
# - Use one process per GPU so sentence-transformers keeps each device saturated.
# - Persist float16 arrays to reduce upload/download time and GPU memory pressure.

import argparse
import json
import logging
import os
import pickle
import multiprocessing as mp

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s")

MODEL_NAME_DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_DIM_DEFAULT = 384


def _bucket_count(manifest, bucket_id):
    """Return manifest-estimated rows for a bucket, with compatibility fallbacks."""
    buckets = manifest.get("buckets") or {}
    counts = manifest.get("bucket_counts") or manifest.get("counts") or []
    info = buckets.get(str(bucket_id), buckets.get(bucket_id))
    if isinstance(info, dict):
        value = info.get("count") or info.get("num_items") or info.get("rows")
        if value is not None:
            return int(value)
    elif isinstance(info, (int, float)):
        return int(info)
    if bucket_id < len(counts):
        return int(counts[bucket_id] or 0)
    return 0


def _build_balanced_gpu_plan(domains, vocab_dir, n_gpus):
    """Greedily assign largest (domain, bucket) work items to least-loaded GPU."""
    work = []
    for domain in domains:
        manifest_path = os.path.join(vocab_dir, f"domain_{domain}_id2text", "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        nb = int(manifest["num_buckets"])
        for b in range(nb):
            work.append((domain, b, _bucket_count(manifest, b)))

    plan = {gpu: [] for gpu in range(n_gpus)}
    gpu_load = {gpu: 0 for gpu in range(n_gpus)}
    gpu_bucket_counts = {gpu: 0 for gpu in range(n_gpus)}
    for domain, bucket_id, est_count in sorted(work, key=lambda x: (-x[2], x[0], x[1])):
        gpu_id = min(range(n_gpus), key=lambda g: (gpu_load[g], gpu_bucket_counts[g], g))
        plan[gpu_id].append((domain, bucket_id, est_count))
        gpu_load[gpu_id] += est_count
        gpu_bucket_counts[gpu_id] += 1

    total_est = sum(gpu_load.values())
    logging.info(
        "balanced assignment: total_buckets=%d total_est_ids=%d n_gpus=%d",
        len(work), total_est, n_gpus)
    for gpu_id in range(n_gpus):
        preview = ",".join(f"d{d}:b{b}:n{n}" for d, b, n in plan[gpu_id][:12])
        logging.info(
            "balanced assignment planned gpu=%d buckets=%d est_ids=%d preview=%s%s",
            gpu_id, len(plan[gpu_id]), gpu_load[gpu_id], preview,
            "..." if len(plan[gpu_id]) > 12 else "")
    return plan


def _gpu_worker(args_tuple):
    """One long-lived worker per GPU; avoids multiple processes sharing a GPU."""
    (gpu_id, assigned_buckets, vocab_dir, out_dir, model_name, batch_size, emb_dim) = args_tuple
    import torch
    from sentence_transformers import SentenceTransformer

    torch.cuda.set_device(gpu_id)
    model = SentenceTransformer(model_name, device=f"cuda:{gpu_id}")
    total = 0
    total_buckets = 0

    logging.info("starting gpu=%d assigned_buckets=%d est_ids=%d", gpu_id,
                 len(assigned_buckets), sum(int(x[2]) for x in assigned_buckets))
    for domain, b, est_count in assigned_buckets:
        part_dir = os.path.join(out_dir, f"domain_{domain}", "_parts")
        os.makedirs(part_dir, exist_ok=True)
        pkl_path = os.path.join(vocab_dir, f"domain_{domain}_id2text", f"bucket_{b:04d}.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            id2text = pickle.load(f)
        if not id2text:
            continue
        ids = sorted(id2text.keys())
        rows = np.zeros((len(ids), emb_dim), dtype=np.float16)
        for s in range(0, len(ids), batch_size):
            e = min(s + batch_size, len(ids))
            texts = [id2text[i] for i in ids[s:e]]
            with torch.no_grad():
                emb = model.encode(texts, batch_size=batch_size,
                                   show_progress_bar=False, normalize_embeddings=True)
            rows[s:e] = emb.astype(np.float16)
            if e == len(ids) or (e // batch_size) % 200 == 0:
                logging.info(f"domain {domain} bucket {b:04d} gpu{gpu_id} {e}/{len(ids)} est={est_count}")
        np.save(os.path.join(part_dir, f"bucket_{b:04d}.npy"), rows)
        np.save(os.path.join(part_dir, f"bucket_{b:04d}.ids.npy"), np.asarray(ids, dtype=np.int64))
        total += len(ids)
        total_buckets += 1
        del id2text, ids, rows
    return (gpu_id, total_buckets, total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--domains", default="0,1,2,3,4")
    p.add_argument("--model_name", default=MODEL_NAME_DEFAULT)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--emb_dim", type=int, default=EMB_DIM_DEFAULT)
    p.add_argument("--num_gpus", type=int, default=None)
    args = p.parse_args()

    import torch
    n_gpus = args.num_gpus or torch.cuda.device_count()
    if n_gpus < 1:
        raise SystemExit("No CUDA GPUs visible")
    logging.info(f"Using {n_gpus} GPUs")

    with open(os.path.join(args.vocab_dir, "vocab_meta.json")) as f:
        meta = json.load(f)
    emb_dim = args.emb_dim
    domains = [int(d) for d in args.domains.split(",") if d != ""]

    # Validate manifests up front.
    for d in domains:
        manifest_path = os.path.join(args.vocab_dir, f"domain_{d}_id2text", "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing sharded id2text manifest: {manifest_path}")

    plan = _build_balanced_gpu_plan(domains, args.vocab_dir, n_gpus)
    jobs = [(gpu, plan[gpu], args.vocab_dir, args.output_dir,
             args.model_name, args.batch_size, emb_dim)
            for gpu in range(n_gpus)]
    logging.info(f"Planned {len(jobs)} GPU workers")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_gpus) as pool:
        for res in pool.imap_unordered(_gpu_worker, jobs):
            logging.info(f"finished gpu={res[0]} buckets={res[1]} ids={res[2]}")

    # Merge partial bucket arrays into the single shard_0.npy training expects.
    for d in domains:
        dm = meta["domains"][str(d)]
        shard_size = int(dm["shard_size"])
        d_dir = os.path.join(args.output_dir, f"domain_{d}")
        part_dir = os.path.join(d_dir, "_parts")
        shard = np.zeros((shard_size, emb_dim), dtype=np.float16)
        filled = 0
        for pf in sorted(os.listdir(part_dir)):
            if not (pf.startswith("bucket_") and pf.endswith(".npy") and not pf.endswith(".ids.npy")):
                continue
            rows = np.load(os.path.join(part_dir, pf))
            ids = np.load(os.path.join(part_dir, pf[:-4] + ".ids.npy"))
            if len(ids):
                shard[ids] = rows
                filled += len(ids)
        out_path = os.path.join(d_dir, "shard_0.npy")
        np.save(out_path, shard)
        logging.info(f"domain {d}: merged {filled} ids -> {out_path} shape={shard.shape}")
        with open(os.path.join(d_dir, "meta_v3_bucketed.json"), "w") as f:
            json.dump({"domain": d, "shard_size": shard_size,
                       "emb_dim": emb_dim, "model_name": args.model_name,
                       "num_encoded": filled}, f, indent=2)

    with open(os.path.join(args.output_dir, "embedding_meta.json"), "w") as f:
        json.dump({"model_name": args.model_name, "emb_dim": emb_dim,
                   "dtype": "float16",
                   "domains": {str(d): meta["domains"][str(d)] for d in domains}},
                  f, indent=2)
    logging.info("Merge complete. Done.")


if __name__ == "__main__":
    main()
