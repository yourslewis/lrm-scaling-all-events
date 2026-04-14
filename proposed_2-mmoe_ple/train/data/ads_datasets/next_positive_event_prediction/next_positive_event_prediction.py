import pandas as pd
import torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
import random
from torch.utils.data import get_worker_info
import gin
from ..special_tokens import MASK_TOKEN 

MASK_TOKEN = 1

@gin.configurable
class TrainIterableDataset(IterableDataset):
    def __init__(
            self, 
            filesystem, 
            parquet_dir: str, 
            max_sequence_length: int, 
            rank: int = 0, 
            world_size: int = 1,
            mask_prob = 0,
            mask_token = MASK_TOKEN
        ):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.max_sequence_length = max_sequence_length
        self.rank = rank
        self.world_size = world_size
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        
        # Collect file paths
        self.file_paths = []
        offset = 0
        for path in tqdm(self.fs.glob(f'{parquet_dir}/*.parquet'), desc="Loading Parquet Metadata"):
            with self.fs.open(path) as f:
                num_rows = pd.read_parquet(f, columns=[]).shape[0]
                self.file_paths.append(path)
                offset += num_rows
        self.total_len = offset
        logging.info(f"Total rows in train dataset for conditional next event prediction experiment: {self.total_len}")

    def _truncate_or_pad_seq(self, y: List[int], target_len: int) -> Tuple[List[int], int]:
        y_len = len(y)
        if y_len < target_len:
            y = y + [0] * (target_len - y_len)
        else:
            y = y[-target_len:]
        assert len(y) == target_len
        return y, min(y_len, target_len)
    
    def _build_next_positive_event_labels(self, ad_ids, action_ids, mask_token) -> Tuple[List[int], List[int], List[int]]:
        assert len(ad_ids) == len(action_ids)
        labels = [0] * len(ad_ids)
        masked_ad_ids = ad_ids.copy()

        # Precompute all future-positive ad_ids for each time step (from right to left)
        future_positive_ids = set()
        future_positives_by_pos = [set() for _ in range(len(ad_ids))]

        for i in reversed(range(len(ad_ids))):
            future_positives_by_pos[i] = future_positive_ids.copy()
            if action_ids[i] > 0:
                future_positive_ids.add(ad_ids[i])

        # Assign next-positive labels and apply masking
        next_pos = 0
        positive_indices = [i for i, a in enumerate(action_ids) if a > 0]

        for i in range(len(ad_ids)):
            while next_pos < len(positive_indices) and i >= positive_indices[next_pos]:
                next_pos += 1
            if next_pos < len(positive_indices):
                labels[i] = ad_ids[positive_indices[next_pos]]
            else:
                labels[i] = 0

            # Mask if this ad will be clicked later
            if ad_ids[i] in future_positives_by_pos[i]:
                masked_ad_ids[i] = mask_token

        return masked_ad_ids, action_ids, labels

    def _process_row(self, row) -> Dict[str, torch.Tensor]:
        user_id, ad_ids, action_ids, timestamps = row.user_id, list(row.ad_ids), list(row.action_ids), list(row.timestamps)
        # get masked_ad_ids, action_ids, and labels
        masked_ad_ids, action_ids, label  = self._build_next_positive_event_labels(ad_ids, action_ids, mask_token=self.mask_token)
        # Decide whether to apply masking
        if random.random() < self.mask_prob:
            input = masked_ad_ids
        else:
            input = ad_ids  
        timestamps, label, ratings = timestamps, label, action_ids

        input, length = self._truncate_or_pad_seq(input, self.max_sequence_length)
        timestamps, _ = self._truncate_or_pad_seq(timestamps, self.max_sequence_length)
        label, _ = self._truncate_or_pad_seq(label, self.max_sequence_length)
        ratings, _ = self._truncate_or_pad_seq(ratings, self.max_sequence_length)

        return {
            "user_id": user_id,
            "length": length,
            "input": torch.tensor(input, dtype=torch.int64),
            "ratings": torch.tensor(ratings, dtype=torch.int64),
            "label": torch.tensor(label, dtype=torch.int64),
            "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        }

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Each sample is assigned to exactly one (rank, worker) pair
        # Global worker id across all DDP processes
        global_worker_id = self.rank * num_workers + worker_id
        global_num_workers = self.world_size * num_workers

        sample_idx = 0
        for path in self.file_paths:
            with self.fs.open(path) as f:
                df = pd.read_parquet(f, columns=['user_id', 'ad_ids', 'action_ids', 'timestamps'])
            for row in df.itertuples(index=False):
                if sample_idx % global_num_workers == global_worker_id:
                    yield self._process_row(row)
                sample_idx += 1
                
        if global_worker_id == 0:
            logging.info(f"Total rows yielded for train dataset: {sample_idx}")


class EvalIterableDataset(IterableDataset):
    def __init__(self, filesystem, parquet_dir: str, max_sequence_length: int, rank: int = 0, world_size: int = 1):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.max_sequence_length = max_sequence_length
        self.rank = rank
        self.world_size = world_size

        # Collect file paths
        self.file_paths = []
        offset = 0
        for path in tqdm(self.fs.glob(f'{parquet_dir}/*.parquet'), desc="Loading Parquet Metadata"):
            with self.fs.open(path) as f:
                num_rows = pd.read_parquet(f, columns=[]).shape[0]
                self.file_paths.append(path)
                offset += num_rows
        self.total_len = offset
        logging.info(f"Total rows in eval dataset for conditional next event prediction experiment: {self.total_len}")

    def _truncate_or_pad_seq(self, y: List[int], target_len: int) -> Tuple[List[int], int]:
        y_len = len(y)
        if y_len < target_len:
            y = y + [0] * (target_len - y_len)
        else:
            y = y[-target_len:]
        assert len(y) == target_len
        return y, min(y_len, target_len)

    def _process_row(self, row) -> Dict[str, torch.Tensor]:
        user_id, ad_ids, action_ids, timestamps = row.user_id, list(row.ad_ids), list(row.action_ids), list(row.timestamps)
        input, ratings, timestamps, label = ad_ids[:-1], action_ids[:-1], timestamps[:-1], ad_ids[-1]
        assert label>0, "The last ad_id in the sequence must be a positive event for evaluation."

        input, length = self._truncate_or_pad_seq(input, self.max_sequence_length)
        ratings, _ = self._truncate_or_pad_seq(ratings, self.max_sequence_length)
        timestamps, _ = self._truncate_or_pad_seq(timestamps, self.max_sequence_length)

        return {
            "user_id": user_id,
            "length": length,
            "input": torch.tensor(input, dtype=torch.int64),
            "ratings": torch.tensor(ratings, dtype=torch.int64),
            "label": torch.tensor(label, dtype=torch.int64),
            "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        }

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Each sample is assigned to exactly one (rank, worker) pair
        # Global worker id across all DDP processes
        global_worker_id = self.rank * num_workers + worker_id
        global_num_workers = self.world_size * num_workers

        sample_idx = 0
        for path in self.file_paths:
            with self.fs.open(path) as f:
                df = pd.read_parquet(f, columns=['user_id', 'ad_ids', 'action_ids', 'timestamps'])
            for row in df.itertuples(index=False):
                if sample_idx % global_num_workers == global_worker_id:
                    yield self._process_row(row)
                sample_idx += 1

        if global_worker_id == 0:
            logging.info(f"Total rows yielded for eval dataset: {sample_idx}")