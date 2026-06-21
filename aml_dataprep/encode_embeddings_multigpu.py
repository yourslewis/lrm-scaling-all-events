#!/usr/bin/env python3
"""
Multi-GPU semantic embedding encoder for AML (8x A100 on one Singularity node).

Wraps the deterministic single-GPU encoder logic from data_prep/step2_v2.py but
parallelizes across all visible GPUs by assigning non-overlapping ID ranges, then
merges per-(GPU, domain) partial shards into the single `shard_0.npy` per domain
that the training pipeline (reco_dataset.py) expects:

    <output_dir>/domain_<d>/shard_0.npy   shape=(shard_size, emb_dim) float16

Determinism: embedding(text) is a pure function of text + model; vocab ids come
from step1, so the produced shard is reproducible regardless of how ranges are
split across GPUs. We assign each id its own row by item_id, so there is never a
write conflict between GPU workers (disjoint id ranges -> disjoint rows), and we
allocate the full (shard_size, dim) array once per domain and fill it.

Usage (run once per node; spawns one process per GPU internally):
    python encode_embeddings_multigpu.py \
        --vocab_dir   /path/to/vocab \
        --output_dir  /path/to/semantic_embeddings_v3_full_preserve \
        --domains     0,1,2,3,4 \
        --batch_size  1024

By default uses every GPU reported by torch.cuda.device_count().
"""
import argparse
import json
import logging
import math
import os
import pickle
import time
import multiprocessing as mp

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s")

MODEL_NAME_DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _encode_range(args_tuple):
    """Encode a disjoint slice of ids for one domain on one GPU.

    Writes a partial .npy holding only the rows for [id_start, id_end) plus a
    companion .npz index of which absolute ids were written, so the merge step
    can place them at the right rows. Pure function of (text, model) => stable.
    """
    (gpu_id, domain, id_start, id_end, vocab_dir, out_dir,
     model_name, batch_size, emb_dim) = args_tuple

    import torch
    from sentence_transformers import SentenceTransformer

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    with open(os.path.join(vocab_dir, f"domain_{domain}_id2text.pkl"), "rb") as f:
        id2text = pickle.load(f)

    ids = [i for i in sorted(id2text.keys()) if id_start <= i < id_end]
    part_dir = os.path.join(out_dir, f"domain_{domain}", "_parts")
    os.makedirs(part_dir, exist_ok=True)
    part_path = os.path.join(part_dir, f"part_{id_start}_{id_end}.npy")
    idx_path = os.path.join(part_dir, f"part_{id_start}_{id_end}.ids.npy")

    if not ids:
        np.save(part_path, np.zeros((0, emb_dim), dtype=np.float16))
        np.save(idx_path, np.zeros((0,), dtype=np.int64))
        return (domain, id_start, id_end, 0)

    model = SentenceTransformer(model_name, device=device)
    rows = np.zeros((len(ids), emb_dim), dtype=np.float16)
    total_batches = math.ceil(len(ids) / batch_size)
    for b in range(total_batches):
        s = b * batch_size
        e = min(s + batch_size, len(ids))
        texts = [id2text[i] for i in ids[s:e]]
        with torch.no_grad():
            emb = model.encode(texts, batch_size=batch_size,
                               show_progress_bar=False, normalize_embeddings=True)
        rows[s:e] = emb.astype(np.float16)
        if (b + 1) % 200 == 0 or b == total_batches - 1:
            logging.info(f"domain {domain} gpu{gpu_id} {e}/{len(ids)} "
                         f"({100.0*e/len(ids):.1f}%)")

    np.save(part_path, rows)
    np.save(idx_path, np.asarray(ids, dtype=np.int64))
    return (domain, id_start, id_end, len(ids))


def _ranges_for_domain(num_ids_lo, num_ids_hi, n_splits):
    """Split [lo, hi) into n_splits contiguous ranges."""
    span = num_ids_hi - num_ids_lo
    if span <= 0:
        return [(num_ids_lo, num_ids_hi)]
    step = math.ceil(span / n_splits)
    out = []
    a = num_ids_lo
    while a < num_ids_hi:
        out.append((a, min(a + step, num_ids_hi)))
        a += step
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--domains", default="0,1,2,3,4",
                   help="Comma-separated domain ids to encode")
    p.add_argument("--model_name", default=MODEL_NAME_DEFAULT)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_gpus", type=int, default=None,
                   help="Default: all visible GPUs")
    args = p.parse_args()

    import torch
    n_gpus = args.num_gpus or torch.cuda.device_count()
    if n_gpus < 1:
        raise SystemExit("No CUDA GPUs visible")
    logging.info(f"Using {n_gpus} GPUs")

    with open(os.path.join(args.vocab_dir, "vocab_meta.json")) as f:
        meta = json.load(f)
    min_item_id = int(meta["min_item_id"])
    # emb_dim is fixed for this model; confirm once on cpu-free path
    emb_dim = 384

    domains = [int(d) for d in args.domains.split(",") if d != ""]

    # Build a global task list. We give the largest domains more GPU splits.
    # Total parallel slots == n_gpus; we round-robin tasks onto GPUs.
    tasks = []
    for d in domains:
        dm = meta["domains"][str(d)]
        shard_size = int(dm["shard_size"])
        max_id = int(dm["max_item_id"])
        lo, hi = min_item_id, max_id + 1
        n_ids = max(0, hi - lo)
        # proportion GPUs by id count, but at least 1 split, at most n_gpus
        # (simple: split every domain into n_gpus contiguous ranges; small
        # domains will have some empty ranges which return instantly)
        n_splits = n_gpus
        for (a, b) in _ranges_for_domain(lo, hi, n_splits):
            tasks.append((d, a, b, shard_size))
    logging.info(f"Planned {len(tasks)} encode tasks across {n_gpus} GPUs")

    # Assign each task a GPU round-robin and run with a pool of size n_gpus.
    job_args = []
    for i, (d, a, b, shard_size) in enumerate(tasks):
        gpu_id = i % n_gpus
        job_args.append((gpu_id, d, a, b, args.vocab_dir, args.output_dir,
                         args.model_name, args.batch_size, emb_dim))

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_gpus) as pool:
        for res in pool.imap_unordered(_encode_range, job_args):
            logging.info(f"finished slice domain={res[0]} "
                         f"[{res[1]},{res[2]}) n={res[3]}")
    logging.info(f"All encode slices done in {time.time()-t0:.0f}s")

    # Merge per-domain partials into the single shard_0.npy training expects.
    for d in domains:
        dm = meta["domains"][str(d)]
        shard_size = int(dm["shard_size"])
        d_dir = os.path.join(args.output_dir, f"domain_{d}")
        part_dir = os.path.join(d_dir, "_parts")
        shard = np.zeros((shard_size, emb_dim), dtype=np.float16)
        parts = sorted(f for f in os.listdir(part_dir)
                       if f.endswith(".npy") and not f.endswith(".ids.npy"))
        filled = 0
        for pf in parts:
            rows = np.load(os.path.join(part_dir, pf))
            ids = np.load(os.path.join(part_dir, pf[:-4] + ".ids.npy"))
            if len(ids):
                shard[ids] = rows
                filled += len(ids)
        out_path = os.path.join(d_dir, "shard_0.npy")
        np.save(out_path, shard)
        logging.info(f"domain {d}: merged {filled} ids -> {out_path} "
                     f"shape={shard.shape}")
        # write meta compatible with original step2 output
        with open(os.path.join(d_dir, "meta_v3_incremental.json"), "w") as f:
            json.dump({"domain": d, "shard_size": shard_size,
                       "emb_dim": emb_dim, "model_name": args.model_name,
                       "num_encoded": filled}, f, indent=2)

    # top-level embedding meta (mirrors original embedding_meta files)
    with open(os.path.join(args.output_dir, "embedding_meta.json"), "w") as f:
        json.dump({"model_name": args.model_name, "emb_dim": emb_dim,
                   "dtype": "float16",
                   "domains": {str(d): meta["domains"][str(d)] for d in domains}},
                  f, indent=2)
    logging.info("Merge complete. Done.")


if __name__ == "__main__":
    main()
