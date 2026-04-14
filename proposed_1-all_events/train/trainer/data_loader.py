# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import os
from typing import Optional, Tuple
from data.ads_datasets.buffered_shuffle import BufferedShuffleDataset

import gin
import numpy as np
import random
import torch
import logging


def _worker_init_fn(worker_id: int) -> None:
    """Initialize each DataLoader worker with a unique seed and zombie prevention.

    1. Sets PR_SET_PDEATHSIG so workers are killed when the parent dies,
       preventing zombie processes that can exhaust system RAM.
    2. Seeds each worker uniquely to avoid duplicate samples across workers
       and DDP ranks.
    """
    # Prevent zombie workers: auto-SIGTERM when parent dies (Linux only)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, 15)  # SIGTERM on parent death
    except Exception:
        pass  # Non-Linux platforms

    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed  # set by PyTorch per-worker
    rank = int(os.environ.get("RANK", 0))
    combined = base_seed + worker_id + rank * 1000
    np.random.seed(combined % (2**32))
    random.seed(combined)
    torch.manual_seed(combined)
    logging.debug(
        "Worker %d (rank %d) seeded with %d", worker_id, rank, combined
    )


@gin.configurable
def create_data_loader(
    dataset: torch.utils.data.IterableDataset,  # Updated to accept IterableDataset
    batch_size: int,
    world_size: int,
    rank: int,
    shuffle: bool = True,  
    prefetch_factor: int = 128,
    num_workers: Optional[int] = 4,
    drop_last: bool = True,
    random_seed: Optional[int] = 42,
    collate_fn: Optional[callable] = None,
) -> torch.utils.data.DataLoader:  
    if shuffle:
        logging.info("Enable Buffered Shuffling with seed: %s", random_seed)
        dataset = BufferedShuffleDataset(dataset, seed=random_seed)

    # Create the DataLoader
    if rank == 0:
        logging.info(f'Creating DataLoader with batch_size: {batch_size}, num_workers: {num_workers}, prefetch_factor: {prefetch_factor}, drop_last: {drop_last}')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = False,
        num_workers=num_workers,  # number of workers per process
        drop_last=drop_last,  # Drop the last incomplete batch if specified
        # prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if num_workers and num_workers > 0 else None,
    )

    return data_loader
