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
from dataclasses import dataclass
from typing import List

import pandas as pd

import torch

from data.dataset import DatasetV2, MultiFileDatasetV2
from data.item_features import ItemFeatures
from data.preprocessor import get_common_preprocessors
from data.ads_datasets.conditional_next_event_prediction import conditional_next_event_prediction
from data.ads_datasets.next_event_prediction import next_event_prediction
from data.ads_datasets.next_positive_event_prediction import next_positive_event_prediction
from data.ads_datasets.semantic_next_event_prediction import semantic_next_event_prediction
from typing import List, Optional, Dict, Tuple
import logging
import gin
import fsspec
try:
    from azureml.fsspec import AzureMachineLearningFileSystem
except ImportError:
    AzureMachineLearningFileSystem = None

SUBSCRIPTION = 'f920ee3b-6bdc-48c6-a487-9e0397b69322'
RESOURCE_GROUP = 'msan-aml'
WORKSPACE = 'msan-retrieval-ranking-aml'
DATASTORE_NAME = 'bingads_algo_prod_networkprotection_c08'

@dataclass
class RecoDataset:
    dataset_name: str
    max_sequence_length: int
    positional_sampling_ratio: float
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset
    domain_to_item_id_range: Dict[int, Tuple[int, int]]
    embd_dim: Optional[int] = None
    domain_offset: Optional[int] = None
    shard_size: Optional[int] = None
    shard_counts: Dict[int, int] = None     # {0: 35, 1: 32, 2: 25} for sequential v2
    min_item_id: int = 0
    max_item_id: int = 0
    num_ratings: int = 0
    num_event_types: int = 0


@gin.configurable
def get_reco_dataset(
    mode: str,
    dataset_name: str,
    experiment_name: str,
    path: str,
    max_sequence_length: int,
    chronological: bool,
    positional_sampling_ratio: float = 1.0,
    rank: int = 0,
    world_size: int = 1,
) -> RecoDataset:
    """
    Example path for corp: 
    path = 'shares/bingads.hm/local/NativeAds/Relevance/Data/sequential/hstu/training_data_05012025/'
    Example path for pme:
    path = 'azureml://datastores/bingads_algo_prod_networkprotection_c08/paths/shares/bingads.hm/local/NativeAds/Relevance/Data/sequential/hstu/v2/training_data_09042025'
    """
    prefix = path
    if mode == 'local':
        # access data during interactive local development
        uri = f'azureml://subscriptions/{SUBSCRIPTION}/resourcegroups/{RESOURCE_GROUP}/workspaces/{WORKSPACE}/datastores/{DATASTORE_NAME}'
        fs = AzureMachineLearningFileSystem(uri)
    else:
        # access data in jobs from mounted local file system
        fs = fsspec.filesystem('file')
    if dataset_name == "training_data_05012025":
        if experiment_name == "next_event_prediction":
            train_dataset = next_event_prediction.TrainIterableDataset(fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size) 
            eval_dataset = next_event_prediction.EvalIterableDataset(fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
            num_ratings = 0
        elif experiment_name == "conditional_next_event_prediction":
            train_dataset = conditional_next_event_prediction.TrainIterableDataset(fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size)
            eval_dataset = conditional_next_event_prediction.EvalIterableDataset(fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
            num_ratings = 3    # impression, click, conversion
        elif experiment_name == "next_positive_event_prediction":
            train_dataset = next_positive_event_prediction.TrainIterableDataset(fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size)
            eval_dataset = next_positive_event_prediction.EvalIterableDataset(fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
            num_ratings = 3    # impression, click, conversion
        else:
            raise ValueError(f"Unknown experiment {experiment_name} for dataset {dataset_name}")

        # early exit for ads domain dataset
        return RecoDataset(
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            positional_sampling_ratio=positional_sampling_ratio,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_item_id = 16_386_645,
            min_item_id= 20,  # 0 is reserved for padding
            num_ratings=num_ratings,
        )
    elif dataset_name == "training_data_07082025":
        if experiment_name == "semantic_next_event_prediction":
            train_dataset = semantic_next_event_prediction.TrainIterableDataset(fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size) 
            eval_dataset = semantic_next_event_prediction.EvalIterableDataset(fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
        elif experiment_name == "semantic_next_event_prediction_finetune_ads":
            train_dataset = semantic_next_event_prediction.TrainIterableDataset(fs, os.path.join(prefix, "filetune_ads"), max_sequence_length, rank, world_size) 
            eval_dataset = semantic_next_event_prediction.EvalIterableDataset(fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)            
        return RecoDataset(
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            domain_to_item_id_range={0: (20, 853_521_661), 1: (0, 790_114_583), 2: (0, 621_877_842)},  
            embd_dim=64,  # robertta 768, pinsage 64
            domain_offset=1_000_000_000,
            shard_size=25_000_000,
            positional_sampling_ratio=positional_sampling_ratio,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    elif dataset_name == "training_data_11032025":
        if experiment_name == "semantic_next_event_prediction":
            train_dataset = semantic_next_event_prediction.TrainIterableDataset(fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size) 
            eval_dataset = semantic_next_event_prediction.EvalIterableDataset(fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
        return RecoDataset(
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            domain_to_item_id_range={0: (20, 42_262_200), 1: (0, 301_422_400), 2: (0, 40_592_094), 3: (0, 199_999_999)},  
            embd_dim=64,  # robertta 768, pinsage 64
            domain_offset=1_000_000_000,
            shard_size=25_000_000,
            shard_counts={0: 2, 1: 13, 2: 2, 3: 8},  
            positional_sampling_ratio=positional_sampling_ratio,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    elif dataset_name == "astrov6":
        if experiment_name == "semantic_next_event_prediction":
            train_dataset = semantic_next_event_prediction.TrainIterableDataset(fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size)
            eval_dataset = semantic_next_event_prediction.EvalIterableDataset(fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
        else:
            raise ValueError(f"Unknown experiment {experiment_name} for dataset {dataset_name}")
        return RecoDataset(
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            domain_to_item_id_range={0: (20, 42_262_200), 1: (0, 301_422_400), 2: (0, 40_592_094)},
            embd_dim=64,
            domain_offset=1_000_000_000,
            shard_size=25_000_000,
            shard_counts={0: 2, 1: 13, 2: 2},
            num_event_types=len(semantic_next_event_prediction.EVENT_TYPE_DICT),
            positional_sampling_ratio=positional_sampling_ratio,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    elif dataset_name == "local_data":
        if experiment_name == "semantic_next_event_prediction":
            local_fs = fsspec.filesystem('file')
            train_dataset = semantic_next_event_prediction.TrainIterableDataset(local_fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size)
            eval_dataset = semantic_next_event_prediction.EvalIterableDataset(local_fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
        else:
            raise ValueError(f"Unknown experiment {experiment_name} for dataset {dataset_name}")
        return RecoDataset(
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            domain_to_item_id_range={0: (20, 999), 1: (0, 999), 2: (0, 999), 3: (0, 999)},
            embd_dim=64,
            domain_offset=1_000_000_000,
            shard_size=1000,
            shard_counts={0: 1, 1: 1, 2: 1, 3: 1},
            positional_sampling_ratio=positional_sampling_ratio,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    elif dataset_name == "benchmarkv4_ads_only":
        if experiment_name == "semantic_next_event_prediction":
            local_fs = fsspec.filesystem('file')
            train_dataset = semantic_next_event_prediction.TrainIterableDataset(local_fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size)
            eval_dataset = semantic_next_event_prediction.EvalIterableDataset(local_fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
        else:
            raise ValueError(f"Unknown experiment {experiment_name} for dataset {dataset_name}")
        return RecoDataset(
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            domain_to_item_id_range={0: (20, 500_000)},
            embd_dim=64,
            domain_offset=1_000_000_000,
            shard_size=500_000,
            shard_counts={0: 1},
            num_event_types=len(semantic_next_event_prediction.EVENT_TYPE_DICT),
            positional_sampling_ratio=positional_sampling_ratio,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    elif dataset_name == "benchmarkv4_all_events":
        if experiment_name == "semantic_next_event_prediction":
            local_fs = fsspec.filesystem('file')
            train_dataset = semantic_next_event_prediction.TrainIterableDataset(local_fs, os.path.join(prefix, "train"), max_sequence_length, rank, world_size)
            eval_dataset = semantic_next_event_prediction.EvalIterableDataset(local_fs, os.path.join(prefix, "eval"), max_sequence_length, rank, world_size)
        else:
            raise ValueError(f"Unknown experiment {experiment_name} for dataset {dataset_name}")
        return RecoDataset(
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            domain_to_item_id_range={d: (20, 500_000) for d in range(5)},
            embd_dim=64,
            domain_offset=1_000_000_000,
            shard_size=500_000,
            shard_counts={d: 1 for d in range(5)},
            num_event_types=len(semantic_next_event_prediction.EVENT_TYPE_DICT),
            positional_sampling_ratio=positional_sampling_ratio,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    elif dataset_name == "ml-1m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
            sample_ratio=1.0,  # do not sample
        )
    elif dataset_name == "ml-20m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif dataset_name == "ml-3b":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = MultiFileDatasetV2(
            file_prefix="tmp/ml-3b/16x32",
            num_files=16,
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = MultiFileDatasetV2(
            file_prefix="tmp/ml-3b/16x32",
            num_files=16,
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif dataset_name == "amzn-books":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    # TODO: revisit this logic for item features
    if dataset_name == "ml-1m" or dataset_name == "ml-20m":
        items = pd.read_csv(dp.processed_item_csv(), delimiter=",")
        max_jagged_dimension = 16
        expected_max_item_id = dp.expected_max_item_id()
        assert expected_max_item_id is not None
        item_features: ItemFeatures = ItemFeatures(
            max_ind_range=[63, 16383, 511],
            num_items=expected_max_item_id + 1,
            max_jagged_dimension=max_jagged_dimension,
            lengths=[
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
            ],
            values=[
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
            ],
        )
        all_item_ids = []
        for df_index, row in items.iterrows():
            # print(f"index {df_index}: {row}")
            movie_id = int(row["movie_id"])
            genres = row["genres"].split("|")
            titles = row["cleaned_title"].split(" ")
            # print(f"{index}: genres{genres}, title{titles}")
            genres_vector = [hash(x) % item_features.max_ind_range[0] for x in genres]
            titles_vector = [hash(x) % item_features.max_ind_range[1] for x in titles]
            years_vector = [hash(row["year"]) % item_features.max_ind_range[2]]
            item_features.lengths[0][movie_id] = min(
                len(genres_vector), max_jagged_dimension
            )
            item_features.lengths[1][movie_id] = min(
                len(titles_vector), max_jagged_dimension
            )
            item_features.lengths[2][movie_id] = min(
                len(years_vector), max_jagged_dimension
            )
            for f, f_values in enumerate([genres_vector, titles_vector, years_vector]):
                for j in range(min(len(f_values), max_jagged_dimension)):
                    item_features.values[f][movie_id][j] = f_values[j]
            all_item_ids.append(movie_id)
        max_item_id = dp.expected_max_item_id()
        for x in all_item_ids:
            assert x > 0, "x in all_item_ids should be positive"
    else:
        # expected_max_item_id and item_features are not set for Amazon datasets.
        item_features = None
        max_item_id = dp.expected_num_unique_items()
        all_item_ids = [x + 1 for x in range(max_item_id)]  # pyre-ignore [6]

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=dp.expected_num_unique_items(),  # pyre-ignore [6]
        max_item_id=max_item_id,  # pyre-ignore [6]
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )