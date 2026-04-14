"""
Generate local test data for the HSTU retrieval pipeline.

Generates:
1. Parquet files with user interaction sequences (train + eval splits)
2. .npy embedding shard files for 4 domains

Usage:
    python scripts/generate_local_data.py \
        --output_dir /tmp/hstu_local/data \
        --embd_output_dir /tmp/hstu_local/embds
"""

import argparse
import os
import numpy as np
import pandas as pd

DOMAIN_OFFSET = 1_000_000_000
NUM_DOMAINS = 4
ITEMS_PER_DOMAIN = 1000
EMBD_DIM = 64
SEED = 42

# Domain 0 item IDs start at 20 (0 = padding, 1 = mask, 2-19 reserved)
DOMAIN_MIN_ITEM_ID = {0: 20, 1: 0, 2: 0, 3: 0}
DOMAIN_MAX_ITEM_ID = {0: ITEMS_PER_DOMAIN - 1, 1: ITEMS_PER_DOMAIN - 1, 2: ITEMS_PER_DOMAIN - 1, 3: ITEMS_PER_DOMAIN - 1}

# Domain sampling weights for sequence generation
DOMAIN_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # domains 0,1,2,3


def encode_id(domain_id: int, item_id: int) -> int:
    return domain_id * DOMAIN_OFFSET + item_id


def generate_sequence(rng: np.random.Generator, min_len: int = 10, max_len: int = 200) -> tuple:
    seq_len = rng.integers(min_len, max_len + 1)
    domains = rng.choice(NUM_DOMAINS, size=seq_len, p=DOMAIN_WEIGHTS)
    # Force last event to be domain 0 (ad event) for meaningful eval
    domains[-1] = 0

    encoded_ids = []
    for d in domains:
        min_id = DOMAIN_MIN_ITEM_ID[d]
        max_id = DOMAIN_MAX_ITEM_ID[d]
        item_id = rng.integers(min_id, max_id + 1)
        encoded_ids.append(encode_id(int(d), int(item_id)))

    # Generate monotonically increasing timestamps
    base_ts = 1700000000  # ~Nov 2023
    timestamps = sorted(rng.integers(base_ts, base_ts + 86400 * 30, size=seq_len).tolist())

    return encoded_ids, timestamps


def generate_parquet_files(output_dir: str, num_users: int, num_files: int, rng: np.random.Generator) -> None:
    os.makedirs(output_dir, exist_ok=True)
    users_per_file = num_users // num_files

    for file_idx in range(num_files):
        rows = []
        start_user = file_idx * users_per_file
        end_user = start_user + users_per_file
        for user_id in range(start_user, end_user):
            encoded_ids, timestamps = generate_sequence(rng)
            rows.append({
                "user_id": user_id,
                "encoded_ids": encoded_ids,
                "timestamps_unix": timestamps,
            })

        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, f"part_{file_idx:04d}.parquet")
        df.to_parquet(path, index=False)
        print(f"  Wrote {path} ({len(df)} rows)")


def generate_embedding_shards(embd_output_dir: str, rng: np.random.Generator) -> None:
    for domain_id in range(NUM_DOMAINS):
        domain_dir = os.path.join(embd_output_dir, f"domain_{domain_id}")
        os.makedirs(domain_dir, exist_ok=True)

        embeddings = rng.standard_normal((ITEMS_PER_DOMAIN, EMBD_DIM)).astype(np.float16)
        # Normalize rows to unit length for more realistic embeddings
        norms = np.linalg.norm(embeddings.astype(np.float32), axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        embeddings = (embeddings.astype(np.float32) / norms).astype(np.float16)

        shard_path = os.path.join(domain_dir, "shard_0.npy")
        np.save(shard_path, embeddings)
        print(f"  Wrote {shard_path} (shape={embeddings.shape}, dtype={embeddings.dtype})")


def main():
    parser = argparse.ArgumentParser(description="Generate local test data for HSTU retrieval")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for parquet files")
    parser.add_argument("--embd_output_dir", type=str, required=True, help="Directory for embedding shards")
    parser.add_argument("--num_train_users", type=int, default=200_000, help="Number of training users")
    parser.add_argument("--num_eval_users", type=int, default=20_000, help="Number of eval users")
    parser.add_argument("--num_train_files", type=int, default=10, help="Number of train parquet files")
    parser.add_argument("--num_eval_files", type=int, default=2, help="Number of eval parquet files")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Generating training data ({args.num_train_users} users, {args.num_train_files} files)...")
    generate_parquet_files(
        os.path.join(args.output_dir, "train"),
        args.num_train_users, args.num_train_files, rng
    )

    print(f"Generating eval data ({args.num_eval_users} users, {args.num_eval_files} files)...")
    generate_parquet_files(
        os.path.join(args.output_dir, "eval"),
        args.num_eval_users, args.num_eval_files, rng
    )

    print(f"Generating embedding shards ({NUM_DOMAINS} domains, {ITEMS_PER_DOMAIN} items each, {EMBD_DIM}d)...")
    generate_embedding_shards(args.embd_output_dir, rng)

    print("Done!")


if __name__ == "__main__":
    main()
