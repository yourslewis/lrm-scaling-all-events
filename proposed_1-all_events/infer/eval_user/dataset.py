from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import List, Tuple, Union
import os
import sys
import gin

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
            train_max_nevents: int = None,            # the maximum number of events in a sequence seen during training, it should be equal to the model max length by default
            model_max_nevents: int = 200,             # the maximum number of events in a sequence a model can support
            ):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.domain = domain
        self.rank = rank
        self.world_size = world_size
        self.train_max_len = train_max_nevents
        self.model_max_len = model_max_nevents

        # Load single Parquet file
        files = self.fs.glob(f"{parquet_dir}/*.parquet")
        if not files:
            raise ValueError(f"No Parquet files found in {parquet_dir}")
        assert len(files) == 1, "Expected exactly one Parquet file"
        path = files[0]

        # Read only index column to get full index range
        with self.fs.open(path) as f:
            self.df = pd.read_parquet(f, columns=["InferUserId", "timestamps_unix", "titles", "urls", "domains", "len"])

    def __len__(self) -> int:
        return len(self.df)


    def _truncate_or_pad_seq(self, y: List[int], target_len: int, pad_value: Union[str, int]) -> Tuple[List[int], int]:
        y_len = len(y)
        if y_len < target_len:
            y = y + [pad_value] * (target_len - y_len)
        else:
            y = y[-target_len:]
        assert len(y) == target_len
        return y, min(y_len, target_len)


    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        titles, urls, timestamps, length, user_id = list(row["titles"]), list(row["urls"]), list(row["timestamps_unix"]), row["len"], row["InferUserId"]
        
        # Truncate eval data first if train_max_len is not None
        if self.train_max_len:
            titles, urls, timestamps = titles[-self.train_max_len:], urls[-self.train_max_len:], timestamps[-self.train_max_len:]
            length = min(length, self.train_max_len)

        timestamps, _ = self._truncate_or_pad_seq(timestamps, self.model_max_len, pad_value=0)
        prompts = [
                    f"{normalize_title(title)} [SEP] {normalize_url(url)}"
                    for title, url in zip(titles, urls)
                 ]
        prompts, _ = self._truncate_or_pad_seq(prompts, self.model_max_len, pad_value="")

        return {
            "prompts": prompts,
            "InferUserId": user_id,
            "len": length if length<=self.model_max_len else self.model_max_len,
            "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        }


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def tokenize(self, prompts) -> torch.Tensor:
        """
        prompts: List[List[str]] of shape [B][L]
        returns: tensor shaped [B, L, T]
        """
        B = len(prompts)
        L = len(prompts[0]) if B > 0 else 0

        # Flatten to [B*L]
        flat = []
        for seq in prompts:
            flat.extend([s for s in seq])

        enc = self.tokenizer(       # enc["input_ids"], enc["attention_mask"]: [B*L, T]
            flat,
            padding="max_length",
            max_length=PINSAGE_MAX_NTOKENS,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        T = enc["input_ids"].size(1)
        input_ids = enc["input_ids"].view(B, L, T)
        return input_ids  

    def __call__(self, batch):
        indices = [int(item["InferUserId"]) for item in batch]
        lengths = [item["len"] for item in batch]
        timestamps = [item["timestamps"] for item in batch]

        prompts = [item["prompts"] for item in batch]    # [B, L]
        encoded = self.tokenize(prompts)
        return {
            "inputs": encoded,  # for model
            "InferUserId": torch.tensor(indices, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long), 
            "timestamps": torch.stack(timestamps) 
        }


@gin.configurable
def get_domain_dataset(
    domain: str = "ads",
    mode: str = "job",
    path: str = "",
    rank: int = 0,
    world_size: int = 1,
    train_max_nevents: int = None,
    model_max_nevents: int = 200,
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
            prefix = ''
    else:
        # access data in jobs from mounted local file system
        import fsspec
        fs = fsspec.filesystem('file')
        prefix = path
    return DomainDataset(fs, prefix, domain, rank, world_size, train_max_nevents=train_max_nevents, model_max_nevents=model_max_nevents)
    