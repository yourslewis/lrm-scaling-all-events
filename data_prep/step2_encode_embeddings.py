"""
Step 2: Encode all unique texts per domain using all-MiniLM-L6-v2.
Reads id2text mappings from Step 1, produces .npy embedding shards.
"""
import argparse, json, os, logging, pickle, math
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_dir", required=True, help="Dir with domain_*_id2text.pkl from step1")
    p.add_argument("--output_dir", required=True, help="Dir to write embedding shards")
    p.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--domains", default="0,1,2,3,4", help="Comma-separated domain IDs to process")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    with open(os.path.join(args.vocab_dir, "vocab_meta.json")) as f:
        meta = json.load(f)
    min_item_id = meta["min_item_id"]

    logging.info(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=args.device)
    emb_dim = model.get_sentence_embedding_dimension()
    logging.info(f"Embedding dim: {emb_dim}")

    domains_to_process = [int(d) for d in args.domains.split(",")]

    for domain_id in domains_to_process:
        domain_meta = meta["domains"][str(domain_id)]
        shard_size = domain_meta["shard_size"]
        num_texts = domain_meta["num_unique_texts"]

        logging.info(f"=== Domain {domain_id}: {num_texts} texts, shard_size={shard_size} ===")

        # Load id->text mapping
        id2text_path = os.path.join(args.vocab_dir, f"domain_{domain_id}_id2text.pkl")
        with open(id2text_path, "rb") as f:
            id2text = pickle.load(f)

        # Pre-allocate shard array (fp16 to save memory)
        shard = np.zeros((shard_size, emb_dim), dtype=np.float16)

        # Sort IDs for sequential processing
        sorted_ids = sorted(id2text.keys())

        # Process in batches
        total_batches = math.ceil(len(sorted_ids) / args.batch_size)
        for batch_idx in range(total_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(sorted_ids))
            batch_ids = sorted_ids[start:end]
            batch_texts = [id2text[i] for i in batch_ids]

            # Encode
            with torch.no_grad():
                embeddings = model.encode(
                    batch_texts,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

            # Store in shard
            for i, item_id in enumerate(batch_ids):
                shard[item_id] = embeddings[i].astype(np.float16)

            if (batch_idx + 1) % 100 == 0 or batch_idx == total_batches - 1:
                logging.info(f"  Domain {domain_id}: {end}/{len(sorted_ids)} texts encoded "
                             f"({end/len(sorted_ids)*100:.1f}%)")

        # Save shard
        d_dir = os.path.join(args.output_dir, f"domain_{domain_id}")
        os.makedirs(d_dir, exist_ok=True)
        shard_path = os.path.join(d_dir, "shard_0.npy")
        np.save(shard_path, shard)
        logging.info(f"  Saved {shard_path} shape={shard.shape}")

        # Free memory
        del shard, id2text
        import gc
        gc.collect()

    # Save embedding metadata
    emb_meta = {
        "model_name": args.model_name,
        "emb_dim": emb_dim,
        "dtype": "float16",
        "domains": {
            str(d): meta["domains"][str(d)] for d in domains_to_process
        }
    }
    with open(os.path.join(args.output_dir, "embedding_meta.json"), "w") as f:
        json.dump(emb_meta, f, indent=2)
    logging.info("Done!")


if __name__ == "__main__":
    main()
