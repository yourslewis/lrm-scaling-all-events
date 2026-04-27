"""
Step 2 v2: Encode texts with balanced GPU splitting.
Supports --id_start and --id_end to process a slice of a domain.
Multiple instances can write to same shard (non-overlapping ranges).
"""
import argparse, json, os, logging, pickle, math
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--domain", type=int, required=True, help="Single domain to process")
    p.add_argument("--id_start", type=int, default=None, help="Start ID (inclusive)")
    p.add_argument("--id_end", type=int, default=None, help="End ID (exclusive)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.vocab_dir, "vocab_meta.json")) as f:
        meta = json.load(f)
    min_item_id = meta["min_item_id"]
    domain_meta = meta["domains"][str(args.domain)]
    shard_size = domain_meta["shard_size"]
    num_texts = domain_meta["num_unique_texts"]

    logging.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=args.device)
    emb_dim = model.get_sentence_embedding_dimension()
    logging.info(f"Embedding dim: {emb_dim}")

    # Load id->text mapping
    id2text_path = os.path.join(args.vocab_dir, f"domain_{args.domain}_id2text.pkl")
    with open(id2text_path, "rb") as f:
        id2text = pickle.load(f)

    # Filter to requested range
    all_ids = sorted(id2text.keys())
    if args.id_start is not None:
        all_ids = [i for i in all_ids if i >= args.id_start]
    if args.id_end is not None:
        all_ids = [i for i in all_ids if i < args.id_end]

    logging.info(f"Domain {args.domain}: processing {len(all_ids)} IDs "
                 f"(range [{all_ids[0] if all_ids else 'N/A'}, {all_ids[-1] if all_ids else 'N/A'}]), "
                 f"shard_size={shard_size}")

    # Check if shard already exists (for merging partial writes)
    d_dir = os.path.join(args.output_dir, f"domain_{args.domain}")
    os.makedirs(d_dir, exist_ok=True)
    shard_path = os.path.join(d_dir, "shard_0.npy")

    if os.path.exists(shard_path):
        logging.info(f"Loading existing shard to merge")
        shard = np.load(shard_path, mmap_mode='r+')
        # mmap_mode='r+' allows writing
        shard = np.array(shard)  # copy to memory for writing
    else:
        shard = np.zeros((shard_size, emb_dim), dtype=np.float16)

    # Process in batches
    total_batches = math.ceil(len(all_ids) / args.batch_size)
    for batch_idx in range(total_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(all_ids))
        batch_ids = all_ids[start:end]
        batch_texts = [id2text[i] for i in batch_ids]

        with torch.no_grad():
            embeddings = model.encode(
                batch_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        for i, item_id in enumerate(batch_ids):
            shard[item_id] = embeddings[i].astype(np.float16)

        if (batch_idx + 1) % 200 == 0 or batch_idx == total_batches - 1:
            pct = end / len(all_ids) * 100
            logging.info(f"  Domain {args.domain}: {end}/{len(all_ids)} texts ({pct:.1f}%)")

    np.save(shard_path, shard)
    logging.info(f"Saved {shard_path} shape={shard.shape}")

    # Save partial metadata
    part_meta = {
        "domain": args.domain,
        "id_start": args.id_start,
        "id_end": args.id_end,
        "num_encoded": len(all_ids),
        "shard_size": shard_size,
        "emb_dim": emb_dim,
        "model_name": args.model_name,
    }
    part_path = os.path.join(d_dir, f"meta_{'full' if args.id_start is None else f'{args.id_start}_{args.id_end}'}.json")
    with open(part_path, "w") as f:
        json.dump(part_meta, f, indent=2)

    logging.info("Done!")


if __name__ == "__main__":
    main()
