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

import torch

from rails.indexing.candidate_index import TopKModule
from rails.indexing.mips_top_k import (
    MIPSBruteForceTopK,
    MIPSBruteForceShardedTopK,
)
from rails.indexing.mol_top_k import MoLBruteForceTopK

from typing import Tuple, Callable, List


def get_top_k_module(
    top_k_method: str,
    item_embeddings: torch.Tensor = None,
    item_ids: torch.Tensor = None,
    domain_id: int = 0,
    embedding_lookup_fn: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor]] = None,
    assigned_shards: List[int] = [],
    device: torch.device = torch.device("cpu"),
) -> TopKModule:
    if top_k_method == "MIPSBruteForceTopK":
        top_k_module = MIPSBruteForceTopK(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    elif top_k_method == "MoLBruteForceTopK":
        top_k_module = MoLBruteForceTopK(  # pyre-ignore [20]
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    elif top_k_method == "MIPSBruteForceShardedTopK":
        if embedding_lookup_fn is None or not assigned_shards:
            raise ValueError("embedding_lookup_fn and assigned_shards must be provided for MIPSBruteForceShardedTopK")
        top_k_module = MIPSBruteForceShardedTopK(
            embedding_lookup_fn=embedding_lookup_fn,
            domain_id=domain_id,
            assigned_shards=assigned_shards,
            device=device,
        )
    else:
        raise ValueError(f"Invalid top-k method {top_k_method}")
    return top_k_module
