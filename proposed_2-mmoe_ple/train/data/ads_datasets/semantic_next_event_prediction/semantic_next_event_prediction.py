import hashlib
import math
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from torch.utils.data import get_worker_info



def _stable_row_seed(user_id, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{user_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def _window_and_sample_history(
    user_id,
    encoded_ids: List[int],
    types: List[str],
    timestamps: List[int],
    original_sequence_length: Optional[int],
    history_keep_rate: float,
    history_sample_seed: int,
) -> Tuple[List[int], List[str], List[int]]:
    if not (0.0 < history_keep_rate <= 1.0):
        raise ValueError(f"history_keep_rate must be in (0, 1], got {history_keep_rate}")
    if not encoded_ids:
        return encoded_ids, types, timestamps

    # Keep the prediction target fixed. Window/drop only the history before it.
    target_id = encoded_ids[-1]
    target_type = types[-1] if types else "UNK"
    target_ts = timestamps[-1] if timestamps else 0

    history_ids = encoded_ids[:-1]
    history_types = types[:-1] if types else []
    history_ts = timestamps[:-1] if timestamps else []

    if original_sequence_length is not None and original_sequence_length > 0:
        history_ids = history_ids[-original_sequence_length:]
        history_types = history_types[-original_sequence_length:] if history_types else []
        history_ts = history_ts[-original_sequence_length:] if history_ts else []

    if history_keep_rate < 1.0 and history_ids:
        keep_n = int(math.ceil(len(history_ids) * history_keep_rate))
        keep_n = max(1, min(len(history_ids), keep_n))
        generator = torch.Generator()
        generator.manual_seed(_stable_row_seed(user_id, history_sample_seed))
        keep_idx = torch.randperm(len(history_ids), generator=generator)[:keep_n].sort().values.tolist()
        history_ids = [history_ids[i] for i in keep_idx]
        history_types = [history_types[i] for i in keep_idx] if history_types else []
        history_ts = [history_ts[i] for i in keep_idx] if history_ts else []

    return (
        history_ids + [target_id],
        (history_types + [target_type]) if types else [],
        (history_ts + [target_ts]) if timestamps else [],
    )

EVENT_TYPE_DICT = {
    "UNK": 0,
    "NativeClick": 1,
    "SearchClick": 2,
    "EdgePageTitle": 3,
    "EdgeSearchQuery": 4,
    "OrganicSearchQuery": 5,
    "UET": 6,
    "OutlookSenderDomain": 7,
    "UETShoppingCart": 8,
    "UETShoppingView": 9,
    "AbandonCart": 10,
    "EdgeShoppingCart": 11,
    "EdgeShoppingPurchase": 12,
}


class TrainIterableDataset(IterableDataset):
    def __init__(
        self,
        filesystem,
        parquet_dir: str,
        max_sequence_length: int,
        rank: int = 0,
        world_size: int = 1,
        original_sequence_length: Optional[int] = None,
        history_keep_rate: float = 1.0,
        history_sample_seed: int = 42,
    ):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.max_sequence_length = max_sequence_length+1     # apply shifting and lengths-1 inside model
        self.rank = rank
        self.world_size = world_size
        self.original_sequence_length = original_sequence_length
        self.history_keep_rate = history_keep_rate
        self.history_sample_seed = history_sample_seed

        # Collect file paths
        self.file_paths = []
        offset = 0
        for path in tqdm(self.fs.glob(f'{parquet_dir}/*.parquet'), desc="Loading Parquet Metadata"):
            with self.fs.open(path) as f:
                num_rows = pd.read_parquet(f, columns=[]).shape[0]
                self.file_paths.append(path)
                offset += num_rows
        self.total_len = offset
        logging.info(f"Total rows in train dataset: {self.total_len}")

    def _truncate_or_pad_seq(self, y: List[int], target_len: int) -> Tuple[List[int], int]:
        y_len = len(y)
        if y_len < target_len:
            y = y + [0] * (target_len - y_len)
        else:
            y = y[-target_len:]
        assert len(y) == target_len
        return y, min(y_len, target_len)

    def _process_row(self, row) -> Dict[str, torch.Tensor]:
        user_id, encoded_ids, timestamps = row.user_id, list(row.encoded_ids), list(row.timestamps_unix)
        types = list(row.types) if hasattr(row, 'types') else []
        input_ids, types, timestamps = _window_and_sample_history(
            user_id,
            encoded_ids,
            types,
            timestamps,
            self.original_sequence_length,
            self.history_keep_rate,
            self.history_sample_seed,
        )

        input_ids, length = self._truncate_or_pad_seq(input_ids, self.max_sequence_length)
        type_ids = [EVENT_TYPE_DICT.get(t, 0) for t in types]
        type_ids, _ = self._truncate_or_pad_seq(type_ids, self.max_sequence_length)
        timestamps, _ = self._truncate_or_pad_seq(timestamps, self.max_sequence_length)

        return {
            "user_id": user_id,
            "length": length,
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "type_ids": torch.tensor(type_ids, dtype=torch.int64),
            "ratings": -1, # Placeholder for ratings, not used
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
                df = pd.read_parquet(f, columns=['user_id', 'encoded_ids', 'types', 'timestamps_unix'])
            for row in df.itertuples(index=False):
                if sample_idx % global_num_workers == global_worker_id:
                    yield self._process_row(row)
                sample_idx += 1
                
        if global_worker_id == 0:
            logging.info(f"Total rows yielded for train dataset: {sample_idx}")


class EvalIterableDataset(IterableDataset):
    def __init__(
        self,
        filesystem,
        parquet_dir: str,
        max_sequence_length: int,
        rank: int = 0,
        world_size: int = 1,
        original_sequence_length: Optional[int] = None,
        history_keep_rate: float = 1.0,
        history_sample_seed: int = 42,
    ):
        self.fs = filesystem
        self.parquet_dir = parquet_dir
        self.max_sequence_length = max_sequence_length+1     # apply shifting and lengths-1 inside model
        self.rank = rank
        self.world_size = world_size
        self.original_sequence_length = original_sequence_length
        self.history_keep_rate = history_keep_rate
        self.history_sample_seed = history_sample_seed

        # Collect file paths
        self.file_paths = []
        offset = 0
        for path in tqdm(self.fs.glob(f'{parquet_dir}/*.parquet'), desc="Loading Parquet Metadata"):
            with self.fs.open(path) as f:
                num_rows = pd.read_parquet(f, columns=[]).shape[0]
                self.file_paths.append(path)
                offset += num_rows
        self.total_len = offset
        logging.info(f"Total rows in eval dataset: {self.total_len}")

    def _truncate_or_pad_seq(self, y: List[int], target_len: int) -> Tuple[List[int], int]:
        y_len = len(y)
        if y_len < target_len:
            y = y + [0] * (target_len - y_len)
        else:
            y = y[-target_len:]
        assert len(y) == target_len
        return y, min(y_len, target_len)

    def _process_row(self, row) -> Dict[str, torch.Tensor]:
        user_id, encoded_ids, timestamps = row.user_id, list(row.encoded_ids), list(row.timestamps_unix)
        types = list(row.types) if hasattr(row, 'types') else []
        input_ids, types, timestamps = _window_and_sample_history(
            user_id,
            encoded_ids,
            types,
            timestamps,
            self.original_sequence_length,
            self.history_keep_rate,
            self.history_sample_seed,
        )

        input_ids, length = self._truncate_or_pad_seq(input_ids, self.max_sequence_length)
        type_ids = [EVENT_TYPE_DICT.get(t, 0) for t in types]
        type_ids, _ = self._truncate_or_pad_seq(type_ids, self.max_sequence_length)
        timestamps, _ = self._truncate_or_pad_seq(timestamps, self.max_sequence_length)

        return {
            "user_id": user_id,
            "length": length,
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "type_ids": torch.tensor(type_ids, dtype=torch.int64),
            "ratings": -1,  # Placeholder for ratings, not used
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
                df = pd.read_parquet(f, columns=['user_id', 'encoded_ids', 'types', 'timestamps_unix'])
            for row in df.itertuples(index=False):
                if sample_idx % global_num_workers == global_worker_id:
                    yield self._process_row(row)
                sample_idx += 1

        if global_worker_id == 0:
            logging.info(f"Total rows yielded for eval dataset: {sample_idx}")