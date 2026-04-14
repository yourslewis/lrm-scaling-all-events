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

from typing import Tuple, Callable, List, Optional

import torch

from rails.indexing.candidate_index import TopKModule

import logging


class MIPSTopKModule(TopKModule):
    def __init__(
        self,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        """
        Args:
            item_embeddings: (1, X, D)
            item_ids: (1, X,)
        """
        super().__init__()

        self._item_embeddings: torch.Tensor = item_embeddings
        self._item_ids: torch.Tensor = item_ids


class MIPSBruteForceTopK(MIPSTopKModule):       # maximum inner product search (MIPS)
    def __init__(
        self,
        item_embeddings: torch.Tensor,          # (1, X, D)
        item_ids: torch.Tensor,                 # (1, X,)
    ) -> None:
        super().__init__(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
        del self._item_embeddings
        self._item_embeddings_t: torch.Tensor = item_embeddings.permute(            # (D, X)
            2, 1, 0
        ).squeeze(2)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, ...). Implementation-specific.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.

        Returns:
            Tuple of (top_k_scores x float, top_k_ids x int), both of shape (B, K,)
        """
        # (B, X,)
        all_logits = torch.mm(query_embeddings, self._item_embeddings_t)            # (B, D) @ (D, X) -> (B, X)
        top_k_logits, top_k_indices = torch.topk(
            all_logits,
            dim=1,
            k=k,
            sorted=sorted,
            largest=True,
        )  # (B, k,)
        return top_k_logits, self._item_ids.squeeze(0)[top_k_indices]



class MIPSBruteForceShardedTopK(TopKModule):
    def __init__(
        self,
        embedding_lookup_fn: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor]],
        domain_id: int,
        assigned_shards: List[int],
        device: torch.device = torch.device("cuda"),
        float_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.float_dtype = float_dtype
        self.prefetched_shards = {}

        # hack: if we have more than 2 shards, we only use the first 2 for simplicity
        if len(assigned_shards) >2:
            logging.warning("More than 2 shards assigned, only using the first 2 for simplicity.")
            assigned_shards = assigned_shards[:2]  

        # Prefetch all assigned shards
        rank  = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logging.info(f"[Rank {rank}] Assigned shards: {assigned_shards}")
        for shard_idx in assigned_shards:
            logging.info(f"[Rank {rank}] Prefetching shard {shard_idx} for domain {domain_id}")
            item_ids, item_embs = embedding_lookup_fn(domain_id, shard_idx)

            item_ids = item_ids.unsqueeze(0)
            item_embs = item_embs.unsqueeze(0)

            if float_dtype is not None:
                item_embs = item_embs.to(dtype=float_dtype)

            self.prefetched_shards[shard_idx] = (item_ids, item_embs)


    def forward(self, query_embeddings: torch.Tensor, k: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_scores_all = []
        topk_ids_all = []

        device = self.device
        query_embeddings = query_embeddings.to(device=device, dtype=torch.bfloat16)

        for i, (shard_idx, (item_ids, item_embs)) in enumerate(self.prefetched_shards.items()):
            # item_embs: (1, Shard_size, D) → item_embs_t: (D, Shard_size)
            item_embs_t = item_embs.permute(2, 1, 0).squeeze(2).to(device=device, dtype=torch.bfloat16)

            # Matrix multiply in bfloat16: [B, D] @ [D, X] → [B, X]
            all_logits = torch.mm(query_embeddings, item_embs_t)

            scores, ids = torch.topk(
                all_logits,
                dim=1,
                k=k,
                sorted=True,
                largest=True,
            )

            topk_scores_all.append(scores)
            topk_ids_all.append(ids)

            # Explicit cleanup
            del item_embs_t
            del all_logits
            torch.cuda.empty_cache()

        # Merge results across shards
        all_scores = torch.cat(topk_scores_all, dim=1)  # [B, total_candidates]
        all_ids = torch.cat(topk_ids_all, dim=1)        # [B, total_candidates]

        # Final top-k over all shards
        top_k_scores, top_k_idx = torch.topk(all_scores, k=k, dim=1)
        top_k_ids = all_ids.gather(dim=1, index=top_k_idx)

        return top_k_scores, top_k_ids

