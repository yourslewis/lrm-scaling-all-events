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

import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Union, Tuple

import torch
import torch.distributed as dist

from indexing.candidate_index import (
    TopKModule,
)
from modeling.sequential.features import (
    SequentialFeatures,
)
from rails.similarities.module import SimilarityModule
from torch.utils.tensorboard import SummaryWriter
from trainer.util import SequentialRetrieval
from indexing.utils import get_top_k_module

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@dataclass
class EvalState:
    item_id_range: Tuple[int,int]
    top_k_module: TopKModule

@dataclass
class EvalStateV2:
    item_ids: torch.Tensor
    top_k_module: TopKModule

def get_eval_state(
    model: SequentialRetrieval,                                            
    item_id_range: Tuple[int, int],
    top_k_method: str,    
    domain_id: int = 0,  # default to be 0
    assigned_shards: List[int] = [],
    device: torch.device = torch.device("cpu"),
    float_dtype: Optional[torch.dtype] = None,  # default to be None
) -> EvalState:
    if top_k_method == "MIPSBruteForceShardedTopK":
        embedding_lookup_fn = model.model._embedding_module.get_shard_embeddings
        top_k_module = get_top_k_module(
            top_k_method=top_k_method,
            domain_id=domain_id,
            embedding_lookup_fn=embedding_lookup_fn,
            assigned_shards=assigned_shards,
            device=device
        )
    else:
        # Exhaustively eval all items (incl. seen ids).
        start_id, end_id = item_id_range
        all_item_ids = list(range(start_id, end_id+1))
        eval_negatives_ids = torch.as_tensor(all_item_ids).to(device).unsqueeze(0)  # [1, X]
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        eval_negative_embeddings = model.negatives_sampler.normalize_embeddings(          # l2 norm
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            model.model.get_item_embeddings(eval_negatives_ids)
        )
        if float_dtype is not None:
            eval_negative_embeddings = eval_negative_embeddings.to(float_dtype)
        top_k_module = get_top_k_module(top_k_method, eval_negative_embeddings, eval_negatives_ids)

    return EvalState(
        item_id_range=item_id_range,
        top_k_module=top_k_module,
    )

def get_eval_state_v2(
    model: SequentialRetrieval,
    top_k_method: str,
) -> EvalStateV2:
    device = next(model.parameters()).device
    negatives_sampler = model.negatives_sampler['eval']
    sampled_negatives_ids, sampled_negatives_embeddings = negatives_sampler(
        positive_ids=torch.tensor([0], device=device),        # sample 10k negatives from ads domain, use fake id=0
        num_to_sample=10000,
    )
    top_k_module = get_top_k_module(top_k_method, sampled_negatives_embeddings, sampled_negatives_ids)
    return EvalStateV2(
        item_ids=sampled_negatives_ids.squeeze(0),
        top_k_module=top_k_module,
    )


@torch.inference_mode  # pyre-ignore [56]
def eval_metrics_v2_from_tensors(
    eval_state: EvalStateV2,
    model: SimilarityModule,                        # type is wrong
    input_ids: torch.Tensor,   
    raw_input_embeddings: torch.Tensor,  
    ratings: torch.Tensor,                         
    label_ids: torch.Tensor, 
    raw_label_embeddings: torch.Tensor, 
    timestamps: torch.Tensor,                      
    lengths: torch.Tensor,
    type_ids: torch.Tensor = None,
    filter_invalid_ids: bool = False,                # default to be true
    user_max_batch_size: Optional[int] = None,      # default to be None
    dtype: Optional[torch.dtype] = None,            # default to be None
) -> Dict[str, Union[float, torch.Tensor]]: 
    """
    Args:
        eval_negatives_ids: Optional[Tensor]. If not present, defaults to eval over
            the entire corpus (`num_items`) excluding all the items that users have
            seen in the past (historical_ids, target_ids). This is consistent with
            papers like SASRec and TDM but may not be fair in practice as retrieval
            modules don't have access to read state during the initial fetch stage.
        filter_invalid_ids: bool. If true, filters seen ids by default.
    Returns:
        keyed metric -> list of values for each example.
    """
    B = label_ids.size(0)
    device = label_ids.device

    negatives_sampler, model = model.negatives_sampler['eval'], model.model

    past_embeddings = model._embedding_module(raw_input_embeddings)
    label_embeddings = negatives_sampler.normalize_embeddings(model._embedding_module(raw_label_embeddings))  # need to normalize item emb
    shared_input_embeddings = model.encode(                                # [B, D]
        past_lengths=lengths,
        past_ids=input_ids,
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        past_embeddings=past_embeddings,
        past_payloads={"timestamps": timestamps, 'ratings': ratings, 'type_ids': type_ids},                                  # past_ratings, (past_timestamps + 1)
    )
    if dtype is not None:                                                  # default to be None
        shared_input_embeddings = shared_input_embeddings.to(dtype)

    MAX_K = 2500
    #print(f"item_ids shape: {eval_state.item_ids.shape}")
    num_items = eval_state.item_ids.size(0)
    k = min(MAX_K, num_items)
    query_embeddings=shared_input_embeddings
    eval_top_k_ids, eval_top_k_scores, _ = get_top_k_outputs(
                                            query_embeddings=query_embeddings,
                                            top_k_module=eval_state.top_k_module,
                                            k=k,
                                            invalid_ids=None,
                                            return_embeddings=False,
                                        )
    assert eval_top_k_ids.size(1) == k
    #print(f"topk ids shape: {eval_top_k_ids.shape}")

    pos_scores = (query_embeddings * label_embeddings).sum(-1, keepdim=True)  #  [B, D] * [B, D] = [B, 1]
    all_ids = torch.cat([label_ids.unsqueeze(1), eval_top_k_ids], dim=1)  # [B, K+1]
    all_scores = torch.cat([pos_scores, eval_top_k_scores], dim=1)        # [B, K+1]
    sorted_scores, sorted_indices = all_scores.sort(dim=1, descending=True)
    sorted_ids = torch.gather(all_ids, dim=1, index=sorted_indices)

    _, eval_rank_indices = torch.max(
        sorted_ids == label_ids.unsqueeze(1),
        dim=1,
    )
    eval_ranks = eval_rank_indices + 1     # shape [B]
    # print(f"pos score: {pos_scores}")
    # print(f"neg score: {eval_top_k_scores}")
    # print(f"sorted score: {sorted_scores}")
    # print(f"eval ranks: {eval_ranks}")

    output = {
        "ndcg_1": torch.where(
            eval_ranks <= 1,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg_10": torch.where(
            eval_ranks <= 10,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg_50": torch.where(
            eval_ranks <= 50,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg_100": torch.where(
            eval_ranks <= 100,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg_200": torch.where(
            eval_ranks <= 200,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "hr_1": (eval_ranks <= 1),
        "hr_10": (eval_ranks <= 10),
        "hr_50": (eval_ranks <= 50),
        "hr_100": (eval_ranks <= 100),
        "hr_200": (eval_ranks <= 200),
        "hr_500": (eval_ranks <= 500),
        "hr_1000": (eval_ranks <= 1000),
        "mrr": torch.div(1.0, eval_ranks),
    }
    return output  # pyre-ignore [7]


def get_top_k_outputs(
    query_embeddings: torch.Tensor,
    k: int,
    top_k_module: TopKModule,
    invalid_ids: Optional[torch.Tensor],
    return_embeddings: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Gets top-k outputs specified by `policy_fn', while filtering out
    invalid ids per row as specified by `invalid_ids'.

    Args:
        k: int. top k to return.
        policy_fn: lambda that takes in item-side embeddings (B, X, D,) and user-side
            embeddings (B * r, ...), and returns predictions (unnormalized logits)
            of shape (B * r, X,).
        invalid_ids: (B * r, N_0) x int64. The list of ids (if > 0) to filter from
            results if present. Expect N_0 to be a small constant.
        return_embeddings: bool if we should additionally return embeddings for the
            top k results.

    Returns:
        A tuple of (top_k_ids, top_k_prs, top_k_embeddings) of shape (B * r, k, ...).
    """
    B: int = query_embeddings.size(0)
    max_num_invalid_ids = 0
    if invalid_ids is not None:
        max_num_invalid_ids = invalid_ids.size(1)

    k_prime = k + max_num_invalid_ids
    top_k_prime_scores, top_k_prime_ids = top_k_module(
        query_embeddings=query_embeddings, k=k_prime
    )
    # Masks out invalid items rowwise.
    if invalid_ids is not None:
        id_is_valid = ~(
            (top_k_prime_ids.unsqueeze(2) == invalid_ids.unsqueeze(1)).max(2)[0]
        )  # [B, K + N_0]
        id_is_valid = torch.logical_and(
            id_is_valid, torch.cumsum(id_is_valid.int(), dim=1) <= k
        )
        # [[1, 0, 1, 0], [0, 1, 1, 1]], k=2 -> [[0, 2], [1, 2]]
        top_k_rowwise_offsets = torch.nonzero(id_is_valid, as_tuple=True)[1].view(
            -1, k
        )
        top_k_scores = torch.gather(
            top_k_prime_scores, dim=1, index=top_k_rowwise_offsets
        )
        top_k_ids = torch.gather(
            top_k_prime_ids, dim=1, index=top_k_rowwise_offsets
        )
    else:
        top_k_scores = top_k_prime_scores
        top_k_ids = top_k_prime_ids

    if return_embeddings:
        raise ValueError("return_embeddings not supported yet.")
    else:
        top_k_embeddings = None
    return top_k_ids, top_k_scores, top_k_embeddings



def _avg(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    _sum_and_numel = torch.tensor(
        [x.sum(), x.numel()], dtype=torch.float32, device=x.device
    )
    if world_size > 1:
        #logging.info(f"[Rank {rank}] Tensor shape: {_sum_and_numel.shape}, dtype: {_sum_and_numel.dtype}, device: {_sum_and_numel.device}")
        dist.all_reduce(_sum_and_numel, op=dist.ReduceOp.SUM)
    return _sum_and_numel[0] / _sum_and_numel[1]


def add_to_summary_writer(
    writer: Optional[SummaryWriter],
    batch_id: int,
    metrics: Dict[str, torch.Tensor],
    prefix: str,
) -> None:
    for key, value in metrics.items():
        if writer is not None:
            writer.add_scalar(f"{prefix}/{key}", value, batch_id)


@torch.inference_mode  # pyre-ignore [56]
def eval_metrics_v3_from_tensors(
    eval_state: EvalState,
    model: torch.nn.Module,                       
    input_ids: torch.Tensor,   
    raw_input_embeddings: torch.Tensor,  
    ratings: torch.Tensor,                                                   
    timestamps: torch.Tensor,                      
    lengths: torch.Tensor,     
    user_ids,
    type_ids: torch.Tensor = None,
    leak_next_type_ids: bool = False,
) -> Dict[str, Union[float, torch.Tensor]]: 
    
    new_input_ids = input_ids[:, :-1]                            # [B, N-1]
    label_ids     = input_ids[:, 1:]                             # [B, N-1]
    new_raw_input_embeddings = raw_input_embeddings[:, :-1, :]   # [B, N-1, D]
    raw_label_embeddings     = raw_input_embeddings[:, 1:, :]    # [B, N-1, D]
    # Ratings are intentionally not sliced here because they are batch-shaped
    # metadata, not sequence-shaped payloads like input_ids/timestamps/type_ids.
    assert ratings.dim() == 1 and ratings.size(0) == input_ids.size(0), (
        f"Expected batch-shaped ratings [B], got {tuple(ratings.shape)}. "
        "If ratings becomes sequence-shaped, slice it alongside the other sequence payloads."
    )
    new_ratings = ratings
    new_timestamps = timestamps[:, :-1]                          # [B, N-1]
    if type_ids is not None:
        new_type_ids = type_ids[:, 1:] if leak_next_type_ids else type_ids[:, :-1]
    else:
        new_type_ids = None
    new_lengths = lengths - 1                                    # [B]

    # next_type_ids are the next-event type labels used by proposed7 conditioning.
    next_type_ids = type_ids[:, 1:] if type_ids is not None else None  # [B, N-1]

    logits, loss, metrics = model(
        input_ids=new_input_ids,
        raw_input_embeddings=new_raw_input_embeddings,
        input_lengths=new_lengths,
        label_ids=label_ids,
        raw_label_embeddings=raw_label_embeddings,
        ratings=new_ratings,
        type_ids=new_type_ids,
        next_type_ids=next_type_ids,
        timestamps=new_timestamps,
        user_ids=user_ids,
    )
    metrics["log_pplx"] = loss.detach()

    # eval recall metrics
    new_input_ids = input_ids[:, :-1]                                                                                              # [B, N-1]
    label_ids = input_ids[torch.arange(input_ids.size(0)), lengths - 1]                                                            # [B]
    new_raw_input_embeddings = raw_input_embeddings[:, :-1, :]                                                                     # [B, N-1, D]
    raw_label_embeddings = raw_input_embeddings[torch.arange(input_ids.size(0)), lengths - 1, :]                                   # [B, D]
    new_ratings = ratings                                                               # ignore ratings for now
    new_timestamps = timestamps[:, :-1]                                                                                            # [B, N-1]
    new_lengths = lengths - 1
    recall_metrics = eval_metrics_v2_from_tensors(
        eval_state,
        model,  
        new_input_ids,
        new_raw_input_embeddings,
        new_ratings,
        label_ids,
        raw_label_embeddings,
        new_timestamps,
        new_lengths,
        new_type_ids,
    )
    metrics.update(recall_metrics)

    return metrics  # pyre-ignore [7]
