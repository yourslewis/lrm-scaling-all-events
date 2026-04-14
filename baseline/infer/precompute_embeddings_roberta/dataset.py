from torch.utils.data import Dataset
import logging
import pandas as pd
import pyarrow.dataset as ds
import math
import torch

class DomainDataset(Dataset):
    def __init__(
            self, 
            filesystem, 
            parquet_dir: str, 
            rank: int = 0, 
            world_size: int = 1,
            shard_size: int = 25_000_000  # 25 million rows per shard, adjust as needed
            ):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.rank = rank
        self.world_size = world_size
        self.shard_size = shard_size

        # Load single Parquet file
        files = self.fs.glob(f"{parquet_dir}/*.parquet")
        if not files:
            raise ValueError(f"No Parquet files found in {parquet_dir}")
        assert len(files) == 1, "Expected exactly one Parquet file"
        path = files[0]

        logging.info(f"[DomainDataset] Rank {rank}: Scanning index range from {path}")

        # Read only index column to get full index range
        with self.fs.open(path) as f:
            df_index = pd.read_parquet(f, columns=["index"])

        min_index = df_index["index"].min()
        max_index = df_index["index"].max()
        total_ids = max_index + 1  # because shards start from 0

        # Total number of shards based on absolute index range
        num_shards = math.ceil(total_ids / shard_size)

        # Shards per rank
        shards_per_rank = math.ceil(num_shards / world_size)
        shard_start = rank * shards_per_rank
        shard_end = min((rank + 1) * shards_per_rank, num_shards)

        # Global-aligned shard ranges
        self.shard_ranges = [
            (i * shard_size, min((i + 1) * shard_size, max_index + 1))
            for i in range(shard_start, shard_end)
            if (i + 1) * shard_size > min_index  # skip shards before actual data
        ]

        logging.info(
            f"num_shards: {num_shards}, shards_per_rank: {shards_per_rank}, "
            f"[DomainDataset] Rank {rank}: Assigned shards {shard_start} to {shard_end - 1}, "
            f"index ranges: {self.shard_ranges}"
        )

        # Efficiently read only matching rows from Parquet file
        dataset = ds.dataset(path, format="parquet", filesystem=self.fs)

        all_shards = []
        for start, end in self.shard_ranges:
            table = dataset.to_table(
                columns=["index", "title", "url"],
                filter=(ds.field("index") >= start) & (ds.field("index") < end),
            )
            logging.info(f"[DomainDataset] Rank {rank}: Loaded {table.num_rows} rows for shard {start}–{end - 1}")
            all_shards.append(table)

        if not all_shards:
            logging.info(f"[DomainDataset] Rank {rank}: No assigned shards. This rank will do no work.")
            self.df = pd.DataFrame(columns=["index", "title", "url"])
            return

        # Convert to Pandas
        self.df = pd.concat([t.to_pandas() for t in all_shards]).reset_index(drop=True)

        logging.info(
            f"[DomainDataset] Rank {rank}: Loaded {len(self.df)} rows for index range {min_index}–{max_index}"
        )

        logging.info(
            f"[DomainDataset] Rank {rank}: Assigned shards {shard_start} to {shard_end - 1}, "
            f"covering index ranges: {self.shard_ranges}. Loaded {len(self.df)} rows."
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "prompt": f"Title: {row['title']} </s> URL: {row['url']}",
            "index": row["index"],
        }


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompts = [item["prompt"] for item in batch]
        indices = [int(item["index"]) for item in batch]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "inputs": encoded,  # for model
            "index": torch.tensor(indices, dtype=torch.long)  # for downstream use
        }


def get_domain_dataset(
    mode: str = "job",
    path: str = "",
    rank: int = 0,
    world_size: int = 1,
    shard_size: int = 25_000_000,  # 25 million rows per shard, adjust as needed
):
    """
    Get the ADS dataset for precomputing embeddings.

    Args:
        mode (str): The mode of the dataset.
        dataset_name (str): The name of the dataset.
        path (str): The path to the dataset.
        rank (int, optional): The rank of the process. Defaults to 0.
        world_size (int, optional): The total number of processes. Defaults to 1.

    Returns:
        Dataset: The ADS dataset.
    """
    if mode=='local':
        # access data during interactive local development
        from azureml.fsspec import AzureMachineLearningFileSystem
        subscription = 'f920ee3b-6bdc-48c6-a487-9e0397b69322'
        resource_group = 'msan-aml'
        workspace = 'msan-retrieval-ranking-aml'
        datastore_name = 'bingads_algo_prod_networkprotection_c08'
        uri = f'azureml://subscriptions/{subscription}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/{datastore_name}'
        fs = AzureMachineLearningFileSystem(uri)

        if domain == "ads":
            prefix = 'shares/bingads.hm/local/NativeAds/Relevance/Data/sequential/hstu/v2/ads_vocab_07012025'
    else:
        # access data in jobs from mounted local file system
        import fsspec
        fs = fsspec.filesystem('file')
        prefix = path
    return DomainDataset(fs, prefix, rank, world_size, shard_size)
    