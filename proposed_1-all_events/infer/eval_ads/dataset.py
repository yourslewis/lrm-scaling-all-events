from torch.utils.data import Dataset
import logging
import numpy as np
import pandas as pd
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from inference.util.normalize import normalize_title, normalize_url

PINSAGE_MAX_NTOKENS = 32   # PinSage model was trained with max number of tokens to be 32, with format as "title [SEP] url"

class DomainDataset(Dataset):
    def __init__(
            self,
            filesystem,
            parquet_dir: str,
            domain: str = "ads",
            rank: int = 0,
            world_size: int = 1,
            ):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.domain = domain
        self.rank = rank
        self.world_size = world_size
 
        # Load single Parquet file
        files = sorted(self.fs.glob(f"{parquet_dir}/*.parquet"))
        if not files:
            raise ValueError(f"No Parquet files found in {parquet_dir}")
        my_files = files[self.rank::self.world_size]
        logging.info(f"rank: {rank}, assigned files: {my_files}")
        
        if not my_files:      
            logging.info(f"Rank {rank}: No assigned shards. This rank will do no work.")
            self.df = pd.DataFrame(columns=["InferAdId", "DocTitle", "DocUrl"])
            return
 
        dfs = []
        for path in my_files:
            with self.fs.open(path) as f:
                df = pd.read_parquet(f, columns=["InferAdId", "DocTitle", "DocUrl"])
                df["DocTitle"] = np.array(df["DocTitle"].tolist())[:, 0]
                df["DocUrl"] = np.array(df["DocUrl"].tolist())[:, 0]
                dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        # print(f"get item for {idx}")
        row = self.df.iloc[idx]
        prompt = f"{normalize_title(row['DocTitle'])} [SEP] {normalize_url(row['DocUrl'])}"
        # print(f"prompt: {prompt}")
        return {
            "prompt": prompt,
            "InferAdId": row["InferAdId"],
        }


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompts = [item["prompt"] for item in batch]
        indices = [int(item["InferAdId"]) for item in batch]

        enc = self.tokenizer(                   # enc["input_ids"], enc["attention_mask"]: [B, T]
            prompts,
            padding="max_length",
            max_length=PINSAGE_MAX_NTOKENS,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        return {
            "inputs": enc["input_ids"],  # for model
            "InferAdId": torch.tensor(indices, dtype=torch.long)  # for downstream use
        }


def get_domain_dataset(
    domain: str = "ads",
    mode: str = "job",
    path: str = "",
    rank: int = 0,
    world_size: int = 1,
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

    # access data in jobs from mounted local file system
    import fsspec
    fs = fsspec.filesystem('file')
    prefix = path
    return DomainDataset(fs, prefix, domain, rank, world_size)
    