import pandas as pd
import torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
from torch.utils.data import get_worker_info


class TrainIterableDataset(IterableDataset):
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
        logging.info(f"Total rows in train dataset for conditional next event prediction experiment: {self.total_len}")

    def _truncate_or_pad_seq(self, y: List[int], target_len: int) -> Tuple[List[int], int]:
        y_len = len(y)
        if y_len < target_len:
            y = y + [0] * (target_len - y_len)
        else:
            y = y[-target_len:]
        assert len(y) == target_len
        return y, min(y_len, target_len)
    
    def _build_conditional_next_event_labels(self, ad_ids, action_ids):
        labels = []
        for i in range(len(ad_ids)):
            if i + 1 < len(ad_ids) and action_ids[i + 1] > 0:
                labels.append(ad_ids[i + 1])
            else:
                labels.append(0)  
        return labels

    def _process_row(self, row) -> Dict[str, torch.Tensor]:
        user_id, ad_ids, action_ids, timestamps = row.user_id, list(row.ad_ids), list(row.action_ids), list(row.timestamps)
        label = self._build_conditional_next_event_labels(ad_ids, action_ids)
        input, timestamps, label, ratings = ad_ids, timestamps, label, action_ids

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
