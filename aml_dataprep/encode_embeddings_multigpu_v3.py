#!/usr/bin/env python3
"""Multi-GPU semantic encoder for step1_v3 sharded id2text vocab.

Unlike encode_embeddings_multigpu.py, this does not load a giant
`domain_<d>_id2text.pkl`. It reads `domain_<d>_id2text/bucket_XXXX.pkl` shards
emitted by step1_collect_vocab_v3.py and encodes one bucket at a time.
"""
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


def _gpu_worker(args_tuple):
    """One long-lived worker per GPU; avoids multiple processes sharing a GPU."""
    (gpu_id, n_gpus, domains, vocab_dir, out_dir, model_name, batch_size, emb_dim) = args_tuple
    import torch
    from sentence_transformers import SentenceTransformer

    torch.cuda.set_device(gpu_id)
    model = SentenceTransformer(model_name, device=f"cuda:{gpu_id}")
    total = 0
    total_buckets = 0

    for domain in domains:
        manifest_path = os.path.join(vocab_dir, f"domain_{domain}_id2text", "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        nb = int(manifest["num_buckets"])
        bucket_ids = [b for b in range(nb) if b % n_gpus == gpu_id]
        part_dir = os.path.join(out_dir, f"domain_{domain}", "_parts")
        os.makedirs(part_dir, exist_ok=True)
        for b in bucket_ids:
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
                    logging.info(f"domain {domain} bucket {b:04d} gpu{gpu_id} {e}/{len(ids)}")
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
    p.add_argument("--num_gpus", type=int, default=None)
    args = p.parse_args()

    import torch
    n_gpus = args.num_gpus or torch.cuda.device_count()
    if n_gpus < 1:
        raise SystemExit("No CUDA GPUs visible")
    logging.info(f"Using {n_gpus} GPUs")

    with open(os.path.join(args.vocab_dir, "vocab_meta.json")) as f:
        meta = json.load(f)
    emb_dim = EMB_DIM_DEFAULT
    domains = [int(d) for d in args.domains.split(",") if d != ""]

    # Validate manifests up front.
    for d in domains:
        manifest_path = os.path.join(args.vocab_dir, f"domain_{d}_id2text", "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing sharded id2text manifest: {manifest_path}")

    jobs = [(gpu, n_gpus, domains, args.vocab_dir, args.output_dir,
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
