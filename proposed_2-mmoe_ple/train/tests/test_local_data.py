"""
Tests for validating generated local data format.

Run with:
    cd large-rec-model-main/src/hstu_retrieval
    pytest tests/test_local_data.py -v \
        --data_dir=/tmp/hstu_local/data \
        --embd_dir=/tmp/hstu_local/embds
"""

import os
import numpy as np
import pandas as pd
import pytest

DOMAIN_OFFSET = 1_000_000_000
NUM_DOMAINS = 4
ITEMS_PER_DOMAIN = 1000
EMBD_DIM = 64


@pytest.fixture
def data_dir(request):
    return request.config.getoption("--data_dir")


@pytest.fixture
def embd_dir(request):
    return request.config.getoption("--embd_dir")


class TestEmbeddingShards:
    """Test embedding shard files."""

    def test_all_domain_shards_exist(self, embd_dir):
        for domain_id in range(NUM_DOMAINS):
            shard_path = os.path.join(embd_dir, f"domain_{domain_id}", "shard_0.npy")
            assert os.path.exists(shard_path), f"Missing shard: {shard_path}"

    def test_shard_shapes(self, embd_dir):
        for domain_id in range(NUM_DOMAINS):
            shard_path = os.path.join(embd_dir, f"domain_{domain_id}", "shard_0.npy")
            arr = np.load(shard_path)
            assert arr.shape == (ITEMS_PER_DOMAIN, EMBD_DIM), \
                f"Domain {domain_id} shard shape {arr.shape} != expected ({ITEMS_PER_DOMAIN}, {EMBD_DIM})"

    def test_shard_dtype(self, embd_dir):
        for domain_id in range(NUM_DOMAINS):
            shard_path = os.path.join(embd_dir, f"domain_{domain_id}", "shard_0.npy")
            arr = np.load(shard_path)
            assert arr.dtype == np.float16, \
                f"Domain {domain_id} shard dtype {arr.dtype} != expected float16"

    def test_shard_no_nans(self, embd_dir):
        for domain_id in range(NUM_DOMAINS):
            shard_path = os.path.join(embd_dir, f"domain_{domain_id}", "shard_0.npy")
            arr = np.load(shard_path)
            assert not np.any(np.isnan(arr)), f"Domain {domain_id} shard contains NaN values"


class TestParquetFiles:
    """Test parquet data files."""

    def test_train_files_exist(self, data_dir):
        train_dir = os.path.join(data_dir, "train")
        assert os.path.isdir(train_dir), f"Train directory missing: {train_dir}"
        parquet_files = [f for f in os.listdir(train_dir) if f.endswith(".parquet")]
        assert len(parquet_files) > 0, "No train parquet files found"

    def test_eval_files_exist(self, data_dir):
        eval_dir = os.path.join(data_dir, "eval")
        assert os.path.isdir(eval_dir), f"Eval directory missing: {eval_dir}"
        parquet_files = [f for f in os.listdir(eval_dir) if f.endswith(".parquet")]
        assert len(parquet_files) > 0, "No eval parquet files found"

    def test_train_schema(self, data_dir):
        train_dir = os.path.join(data_dir, "train")
        parquet_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".parquet")])
        df = pd.read_parquet(os.path.join(train_dir, parquet_files[0]))
        required_cols = {"user_id", "encoded_ids", "timestamps_unix"}
        assert required_cols.issubset(set(df.columns)), \
            f"Missing columns. Found: {df.columns.tolist()}, need: {required_cols}"

    def test_encoded_ids_valid(self, data_dir):
        """Check that encoded_ids decode to valid domain/item pairs."""
        train_dir = os.path.join(data_dir, "train")
        parquet_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".parquet")])
        df = pd.read_parquet(os.path.join(train_dir, parquet_files[0]))

        for idx, row in df.head(100).iterrows():
            encoded_ids = row["encoded_ids"]
            for eid in encoded_ids:
                domain_id = eid // DOMAIN_OFFSET
                item_id = eid % DOMAIN_OFFSET
                assert 0 <= domain_id < NUM_DOMAINS, \
                    f"Invalid domain_id {domain_id} from encoded_id {eid}"
                assert 0 <= item_id < ITEMS_PER_DOMAIN, \
                    f"Invalid item_id {item_id} from encoded_id {eid} (domain {domain_id})"

    def test_timestamps_monotonic(self, data_dir):
        """Check timestamps are monotonically non-decreasing."""
        train_dir = os.path.join(data_dir, "train")
        parquet_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".parquet")])
        df = pd.read_parquet(os.path.join(train_dir, parquet_files[0]))

        for idx, row in df.head(100).iterrows():
            ts = row["timestamps_unix"]
            for i in range(1, len(ts)):
                assert ts[i] >= ts[i - 1], \
                    f"Timestamps not monotonic at user {row['user_id']}: {ts[i-1]} > {ts[i]}"

    def test_enough_training_rows(self, data_dir):
        """Ensure enough training data for at least 1000 batches at batch_size=64."""
        train_dir = os.path.join(data_dir, "train")
        total_rows = 0
        for f in os.listdir(train_dir):
            if f.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(train_dir, f), columns=["user_id"])
                total_rows += len(df)
        min_required = 64 * 1000  # batch_size * min_batches
        assert total_rows >= min_required, \
            f"Only {total_rows} train rows, need at least {min_required} for 1000 batches"

    def test_last_event_is_domain_0(self, data_dir):
        """Check that the last event in each sequence is from domain 0 (ad event)."""
        train_dir = os.path.join(data_dir, "train")
        parquet_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".parquet")])
        df = pd.read_parquet(os.path.join(train_dir, parquet_files[0]))

        for idx, row in df.head(100).iterrows():
            last_id = row["encoded_ids"][-1]
            domain_id = last_id // DOMAIN_OFFSET
            assert domain_id == 0, \
                f"Last event domain should be 0, got {domain_id} for user {row['user_id']}"
