import abc
import torch
from typing import List, Optional, Tuple, Dict
from modeling.sequential.embedding_modules import EmbeddingModule
import torch.distributed as dist
from modeling.sequential.utils import (
    pack_and_all_gather,
)
import gc
import logging
from registry import register

class NegativesSampler(torch.nn.Module):
    def __init__(self, l2_norm: bool, l2_norm_eps: float) -> None:
        super().__init__()

        self._l2_norm: bool = l2_norm
        self._l2_norm_eps: float = l2_norm_eps

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_l2_norm(x)

    def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._l2_norm:
            x = x / torch.clamp(
                torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
                min=self._l2_norm_eps,
            )
        return x

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass

    @abc.abstractmethod
    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        pass


@register("sampler", "local")
class LocalNegativesSampler(NegativesSampler):
    def __init__(
        self,
        num_items: int,
        item_emb: torch.nn.Embedding,
        all_item_ids: List[int],
        l2_norm: bool,
        l2_norm_eps: float,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._num_items: int = len(all_item_ids)
        self._item_emb: torch.nn.Embedding = item_emb
        self.register_buffer("_all_item_ids", torch.tensor(all_item_ids))

    def debug_str(self) -> str:
        sampling_debug_str = (
            f"local{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )
        return sampling_debug_str

    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        # TODO: ensure negative samples are distinct from positive samples
        
        output_shape = positive_ids.size() + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0,
            high=self._num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        return sampled_ids, self.normalize_embeddings(self._item_emb(sampled_ids))


@register("sampler", "InBatch")
class InBatchNegativesSampler(NegativesSampler):
    """
    In-batch negatives sampler for contrastive training (e.g., InfoNCE).

    Workflow:
      1. Call `process_batch` once per batch to cache the valid candidate pool
         (ids, embeddings, and optionally deduplicated).
      2. Call `forward` to sample negatives for a given set of positives.
         - Negatives are drawn from the cached pool.
         - Support seq-level masking (exclude items from the same sequence).
         - Also support item-level masking (exclude the same item id).
    """
    def __init__(
        self,
        l2_norm: bool,
        l2_norm_eps: float,
        dedup_embeddings: bool,
        cross_rank: bool = False,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)

        self._dedup_embeddings: bool = dedup_embeddings
        self.cross_rank: bool = cross_rank

    def debug_str(self) -> str:
        sampling_debug_str = (
            f"in-batch{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )
        if self._dedup_embeddings:
            sampling_debug_str += "-dedup"
        return sampling_debug_str

    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        """
        Args:
           ids: (N') or (B, N) x int64
           presences: (N') or (B, N) x bool
           embeddings: (N', D) or (B, N, D) x float
        """
        assert ids.size() == presences.size()
        assert ids.size() == embeddings.size()[:-1]
        if self._dedup_embeddings:
            valid_ids = ids[presences]
            valid_seq_ids = torch.arange(ids.size(0), device=ids.device).unsqueeze(1).expand_as(ids)[presences]

            unique_ids, unique_ids_inverse_indices = torch.unique(
                input=valid_ids, sorted=False, return_inverse=True
            )
            device = unique_ids.device
            unique_embedding_offsets = torch.empty(
                (unique_ids.numel(),),
                dtype=torch.int64,
                device=device,
            )
            unique_embedding_offsets[unique_ids_inverse_indices] = torch.arange(
                valid_ids.numel(), dtype=torch.int64, device=device
            )

            unique_embeddings = embeddings[presences][unique_embedding_offsets, :]
            unique_seq_ids = valid_seq_ids[unique_embedding_offsets]

            self._cached_embeddings = self._maybe_l2_norm(  # pyre-ignore [16]
                unique_embeddings
            )
            self._cached_ids = unique_ids  # pyre-ignore [16]
            self._cached_seq_ids = unique_seq_ids  # pyre-ignore [16]
        else:
            self._cached_embeddings = self._maybe_l2_norm(embeddings[presences])
            self._cached_ids = ids[presences]
            self._cached_seq_ids = (
                torch.arange(ids.size(0), device=ids.device).unsqueeze(1).expand_as(ids)[presences]
            )
        
        if self.cross_rank:
            rank = dist.get_rank()
            self.batch_size = ids.size(0)
            self._cached_seq_ids += rank * self.batch_size
            self._cached_ids, self._cached_seq_ids, self._cached_embeddings = pack_and_all_gather(
                self._cached_ids, self._cached_seq_ids, self._cached_embeddings
            )

    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings  # pyre-ignore [7]

    def forward(
        self,
        positive_ids: torch.Tensor,                # shape: [N]
        num_to_sample: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings,).
        """
        positive_seq_ids: torch.Tensor = kwargs["jagged_seq_ids"]
        if self.cross_rank:
            rank = dist.get_rank()
            positive_seq_ids += rank * self.batch_size

        # mask shape: [num_pos, num_candidates], it's a combination of seq-level mask and item mask
        mask = (positive_seq_ids.unsqueeze(-1) != self._cached_seq_ids.unsqueeze(0)) & \
            (positive_ids.unsqueeze(-1) != self._cached_ids.unsqueeze(0))

        probs = mask.float()
        row_sums = probs.sum(dim=-1, keepdim=True).clamp_min(1)
        probs = probs / row_sums

        sampled_offsets = torch.multinomial(probs, num_samples=num_to_sample, replacement=True)

        return (
            self._cached_ids[sampled_offsets],
            self._cached_embeddings[sampled_offsets],
        )


@register("sampler", "RotateInDomainGlobalNegativesSampler")
class RotateInDomainGlobalNegativesSampler(NegativesSampler):
    def __init__(self, item_emb: EmbeddingModule, domain_offset: int, shard_size: int, shard_counts: Dict[int, int], l2_norm: bool, l2_norm_eps: float) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)
        self._item_emb: EmbeddingModule = item_emb
        self.domain_offset: int = domain_offset
        self.shard_size: int = shard_size
        self.shard_counts: Dict[int, int] = shard_counts
        self.pools: Dict[int, Tuple[int, Tuple[torch.Tensor, torch.Tensor]]] = {}   # domain_id -> (current_shard_idx, (index_tensor, embedding_tensor))
        # Historical get_eval_state_v2 behavior: it calls this sampler with a
        # fake positive id from domain 0 and no event-type labels, so keep domain
        # 0 sampling as the same 0/3 mixture when domain 3 exists. Other encoded
        # domains route to themselves so eval loss/perplexity can handle all 5
        # all_events_v2 domains without changing top-k eval candidate sampling.
        if 3 in shard_counts:
            self.domain_pools_map: Dict[int, List[Tuple[int, float]]] = {
                0: [(0, 0.5), (3, 0.5)],
                1: [(1, 1.0)],
                2: [(2, 1.0)],
                3: [(3, 1.0)],
                4: [(4, 1.0)],
            }
        else:
            self.domain_pools_map: Dict[int, List[Tuple[int, float]]] = {
                0: [(0, 1.0)],
                1: [(1, 1.0)],
                2: [(2, 1.0)],
                3: [(3, 1.0)],
                4: [(4, 1.0)],
            }

        # Train-time event-type-aware routing for current all_events_v2 domains.
        # This is used only when SampledSoftmaxLoss passes supervision_type_ids.
        self.event_type_to_domain: Dict[int, int] = {
            1: 0,   # NativeClick -> Ads
            2: 0,   # SearchClick -> Ads
            3: 1,   # EdgePageTitle -> Browsing/Web
            4: 2,   # EdgeSearchQuery -> SearchQuery
            5: 2,   # OrganicSearchQuery -> SearchQuery
            6: 1,   # UET -> Browsing/Web
            7: 4,   # OutlookSenderDomain -> OutlookSender
            8: 3,   # UETShoppingCart -> PurchaseCart
            9: 1,   # UETShoppingView -> Browsing/Web
            10: 3,  # AbandonCart -> PurchaseCart
            11: 3,  # EdgeShoppingCart -> PurchaseCart
            12: 3,  # EdgeShoppingPurchase -> PurchaseCart
            13: 1,  # ChromePageTitle -> Browsing/Web
            14: 1,  # MSN -> Browsing/Web
        }
        self.train_domain_pools_map: Dict[int, List[Tuple[int, float]]] = {
            domain_id: [(domain_id, 1.0)] for domain_id in shard_counts.keys()
        }

    def debug_str(self) -> str:
        sampling_debug_str = (
            f"Rotate{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )
        return sampling_debug_str


    def rotate(self):
        for domain_id in self.shard_counts.keys():
            # Get current pool entry
            current_idx, old_entry = self.pools.get(domain_id, (-1, (None, None)))
    
            # Explicitly free old shard
            if old_entry is not None:
                old_item_ids, old_raw_embeddings = old_entry
                del old_item_ids
                del old_raw_embeddings
                gc.collect()  # force free of CPU memory
    
            # Move to next shard
            next_idx = (current_idx + 1) % self.shard_counts[domain_id]
    
            # Get raw shard data (CPU tensors)
            item_ids, raw_embeddings = self._item_emb.get_raw_shard_embeddings(domain_id, next_idx)
            assert raw_embeddings.device.type == "cpu", (
                f"raw_embeddings must be on CPU, got {raw_embeddings.device}"
            )
    
            # Update pool
            self.pools[domain_id] = (next_idx, (item_ids, raw_embeddings))
            logging.info(f"[Rotate] domain={domain_id}, shard={next_idx}, pool size={len(item_ids)}")


    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass


    def forward(
        self,
        positive_ids: torch.Tensor,  # shape: [N]
        num_to_sample: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = positive_ids.device
        N, K = positive_ids.size(0), num_to_sample
        supervision_type_ids = kwargs.get("supervision_type_ids")
        if supervision_type_ids is not None:
            if supervision_type_ids.size() != positive_ids.size():
                raise ValueError(
                    f"supervision_type_ids shape {tuple(supervision_type_ids.shape)} "
                    f"must match positive_ids shape {tuple(positive_ids.shape)}"
                )
            mapped = torch.full_like(positive_ids, -1)
            for event_type_id, domain_id in self.event_type_to_domain.items():
                mapped = torch.where(
                    supervision_type_ids == event_type_id,
                    torch.full_like(mapped, domain_id),
                    mapped,
                )
            unknown = mapped < 0
            if torch.any(unknown):
                bad = torch.unique(supervision_type_ids[unknown]).detach().cpu().tolist()
                raise ValueError(f"Unmapped supervision event type ids for negative sampling: {bad}")
            domain_ids = mapped
            pools_map = self.train_domain_pools_map
        else:
            # Eval/default path: preserve historical positive-id-domain routing.
            domain_ids = positive_ids // self.domain_offset  # [N]
            pools_map = self.domain_pools_map

        sampled_ids = torch.zeros(N, K, dtype=positive_ids.dtype, device=device)
        sampled_negative_embeddings = torch.zeros(N, K, self._item_emb.output_dim, dtype=torch.float32, device=device)

        for domain_id in torch.unique(domain_ids).tolist():
            if domain_id not in pools_map:
                raise ValueError(
                    f"No negative pool mapping for domain {domain_id}; "
                    f"available mappings={sorted(pools_map.keys())}; "
                    f"supervision_type_ids provided={supervision_type_ids is not None}"
                )
            pool_weight_pairs = pools_map[domain_id]
            if not pool_weight_pairs:
                raise ValueError(f"Empty negative pool mapping for domain {domain_id}")
            for pool_id, _ in pool_weight_pairs:
                if pool_id not in self.pools:
                    raise RuntimeError(
                        f"Negative pool {pool_id} has not been initialized. "
                        "Call rotate() before using RotateInDomainGlobalNegativesSampler."
                    )

            weights = torch.tensor([w for _, w in pool_weight_pairs], dtype=torch.float)
            probs = weights / weights.sum()
            # counts[k] = index of which pool each negative comes from, will group indices later
            counts = torch.multinomial(probs, num_samples=num_to_sample, replacement=True)

            # sample separately from each pool
            encoded_neg_ids_list, neg_embs_list = [], []
            for i, (pool_id, _) in enumerate(pool_weight_pairs):
                k_i = (counts == i).sum().item()
                if k_i == 0:
                    continue

                _, (item_ids, raw_embeddings) = self.pools[pool_id]
                sampled_offsets = torch.randint(
                    low=0,
                    high=item_ids.numel(),
                    size=(k_i,),
                    dtype=torch.long,
                )
                neg_ids = item_ids[sampled_offsets].to(dtype=positive_ids.dtype)
                raw_offsets = torch.clamp(neg_ids % self.shard_size, max=raw_embeddings.shape[0] - 1).long()
                encoded_neg_ids = pool_id * self.domain_offset + neg_ids
                neg_embs = self.normalize_embeddings(
                    self._item_emb(raw_embeddings[raw_offsets].to(dtype=torch.float32, device=device))
                )  # [K, D]

                encoded_neg_ids_list.append(encoded_neg_ids.to(device))
                neg_embs_list.append(neg_embs)

            encoded_neg_ids = torch.cat(encoded_neg_ids_list, dim=0)
            neg_embs = torch.cat(neg_embs_list, dim=0)

            indices = (domain_ids == domain_id).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue

            sampled_ids[indices] = encoded_neg_ids.unsqueeze(0).expand(len(indices), -1).to(device)
            sampled_negative_embeddings[indices] = neg_embs.unsqueeze(0).expand(len(indices), -1, -1)

        return sampled_ids, sampled_negative_embeddings
    

@register("sampler", "Hybrid")
class HybridNegativesSampler(NegativesSampler):
    """
    Hybrid negatives sampler that combines in-batch and rotate-based global negative sampling.
    
    This sampler blends two sampling strategies:
      1. In-batch sampling: Draws negatives from the current batch's item pool (fast, good diversity within batch).
      2. Rotate sampling: Draws negatives from pre-loaded global item shards (broader coverage).
    
    The mixing ratio is controlled by `in_batch_ratio`, which determines what proportion of negatives
    come from in-batch vs. rotate sampling.
    
    Workflow:
      1. Call `process_batch` to update both underlying samplers with current batch data.
      2. Call `forward` to sample negatives, which splits `num_to_sample` between the two strategies
         and concatenates the results.
    """
    def __init__(
        self,
        l2_norm: bool,
        l2_norm_eps: float,
        in_batch_sampler: InBatchNegativesSampler,
        rotate_sampler: RotateInDomainGlobalNegativesSampler,
        in_batch_ratio: float = 0.5,
    ) -> None:
        super().__init__(
            l2_norm=l2_norm,
            l2_norm_eps=l2_norm_eps,
        )
        self._in_batch_sampler = in_batch_sampler
        self._rotate_sampler = rotate_sampler
        self._in_batch_ratio = in_batch_ratio

    def debug_str(self) -> str:
        return (
            f"hybrid({self._in_batch_sampler.debug_str()}|"
            f"{self._rotate_sampler.debug_str()})"
        )
    
    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        self._in_batch_sampler.process_batch(ids, presences, embeddings)
        self._rotate_sampler.process_batch(ids, presences, embeddings)

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_to_sample <= 0:
            raise ValueError("num_to_sample must be > 0")

        in_batch_k = int(num_to_sample * self._in_batch_ratio)
        rotate_k = num_to_sample - in_batch_k 

        sampled_ids_chunks: List[torch.Tensor] = []
        sampled_emb_chunks: List[torch.Tensor] = []

        in_batch_ids, in_batch_embs = self._in_batch_sampler(positive_ids, in_batch_k, **kwargs)
        sampled_ids_chunks.append(in_batch_ids)
        sampled_emb_chunks.append(in_batch_embs)

        rotate_ids, rotate_embs = self._rotate_sampler(positive_ids, rotate_k, **kwargs)
        sampled_ids_chunks.append(rotate_ids)
        sampled_emb_chunks.append(rotate_embs)

        sampled_ids = torch.cat(sampled_ids_chunks, dim=1)
        sampled_negative_embeddings = torch.cat(sampled_emb_chunks, dim=1)
        return sampled_ids, sampled_negative_embeddings