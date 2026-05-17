from typing import Dict, Tuple

import gc
import logging

import torch

from modeling.sequential.embedding_modules import EmbeddingModule
from modeling.sequential.nagatives_sampler import NegativesSampler
from registry import register


@register("sampler", "MixedHardGlobalNegativesSampler")
class MixedHardGlobalNegativesSampler(NegativesSampler):
    """Event-type-aware global sampler with a controlled hard-negative mix.

    This sampler keeps the random/global-negative logic separate from the
    historical RotateInDomainGlobalNegativesSampler. It is intended for training
    only; eval should continue to use the existing rotate sampler to preserve
    comparable metrics.

    Per positive target, negatives are sampled from the target event type's
    physical embedding domain. A configurable fraction is mined from a random
    candidate pool by query-item similarity, using a rank window rather than the
    absolute hardest candidates to reduce false-negative risk.
    """

    EVENT_TYPE_TO_DOMAIN: Dict[int, int] = {
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

    def __init__(
        self,
        item_emb: EmbeddingModule,
        domain_offset: int,
        shard_size: int,
        shard_counts: Dict[int, int],
        l2_norm: bool,
        l2_norm_eps: float,
        hard_fraction: float = 0.25,
        hard_candidate_pool_size: int = 1024,
        hard_rank_start: int = 32,
        hard_rank_end: int = 512,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)
        if not 0.0 <= hard_fraction <= 1.0:
            raise ValueError(f"hard_fraction must be in [0, 1], got {hard_fraction}")
        if hard_candidate_pool_size <= 0:
            raise ValueError("hard_candidate_pool_size must be > 0")
        if hard_rank_start < 0 or hard_rank_end <= 0:
            raise ValueError("hard rank bounds must be non-negative with rank_end > 0")

        self._item_emb: EmbeddingModule = item_emb
        self.domain_offset: int = domain_offset
        self.shard_size: int = shard_size
        self.shard_counts: Dict[int, int] = shard_counts
        self.hard_fraction: float = hard_fraction
        self.hard_candidate_pool_size: int = hard_candidate_pool_size
        self.hard_rank_start: int = hard_rank_start
        self.hard_rank_end: int = hard_rank_end
        self.pools: Dict[int, Tuple[int, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def debug_str(self) -> str:
        return (
            f"mixed-hard-global-hf{self.hard_fraction:g}"
            f"-c{self.hard_candidate_pool_size}"
            f"-r{self.hard_rank_start}-{self.hard_rank_end}"
            f"{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )

    def rotate(self) -> None:
        for domain_id in self.shard_counts.keys():
            current_idx, old_entry = self.pools.get(domain_id, (-1, (None, None)))
            if old_entry is not None:
                old_item_ids, old_raw_embeddings = old_entry
                del old_item_ids
                del old_raw_embeddings
                gc.collect()

            next_idx = (current_idx + 1) % self.shard_counts[domain_id]
            item_ids, raw_embeddings = self._item_emb.get_raw_shard_embeddings(domain_id, next_idx)
            assert raw_embeddings.device.type == "cpu", (
                f"raw_embeddings must be on CPU, got {raw_embeddings.device}"
            )
            self.pools[domain_id] = (next_idx, (item_ids, raw_embeddings))
            logging.info(
                f"[MixedHardGlobalNegativesSampler.rotate] domain={domain_id}, "
                f"shard={next_idx}, pool size={len(item_ids)}"
            )

    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        pass

    def _domains_from_event_types(
        self,
        positive_ids: torch.Tensor,
        supervision_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if supervision_type_ids.size() != positive_ids.size():
            raise ValueError(
                f"supervision_type_ids shape {tuple(supervision_type_ids.shape)} "
                f"must match positive_ids shape {tuple(positive_ids.shape)}"
            )
        mapped = torch.full_like(positive_ids, -1)
        for event_type_id, domain_id in self.EVENT_TYPE_TO_DOMAIN.items():
            mapped = torch.where(
                supervision_type_ids == event_type_id,
                torch.full_like(mapped, domain_id),
                mapped,
            )
        # Padding/UNK can appear in dense tensors with zero supervision weight.
        mapped = torch.where(
            supervision_type_ids == 0,
            positive_ids // self.domain_offset,
            mapped,
        )
        unknown = mapped < 0
        if torch.any(unknown):
            bad = torch.unique(supervision_type_ids[unknown]).detach().cpu().tolist()
            raise ValueError(f"Unmapped supervision event type ids for hard negative sampling: {bad}")
        return mapped

    def _sample_from_pool(
        self,
        pool_id: int,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pool_id not in self.pools:
            raise RuntimeError(
                f"Negative pool {pool_id} has not been initialized. "
                "Call rotate() before using MixedHardGlobalNegativesSampler."
            )
        _, (item_ids, raw_embeddings) = self.pools[pool_id]
        sampled_offsets = torch.randint(
            low=0,
            high=item_ids.numel(),
            size=(num_samples,),
            dtype=torch.long,
        )
        neg_ids = item_ids[sampled_offsets].to(dtype=dtype)
        raw_offsets = torch.clamp(neg_ids % self.shard_size, max=raw_embeddings.shape[0] - 1).long()
        encoded_neg_ids = (pool_id * self.domain_offset + neg_ids).to(device)
        neg_embs = self.normalize_embeddings(
            self._item_emb(raw_embeddings[raw_offsets].to(dtype=torch.float32, device=device))
        )
        return encoded_neg_ids, neg_embs

    def _sample_uniform(
        self,
        pool_id: int,
        num_rows: int,
        num_to_sample: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_to_sample == 0:
            return (
                torch.empty(num_rows, 0, dtype=dtype, device=device),
                torch.empty(num_rows, 0, self._item_emb.output_dim, dtype=torch.float32, device=device),
            )
        ids, embs = self._sample_from_pool(pool_id, num_rows * num_to_sample, device, dtype)
        return ids.view(num_rows, num_to_sample), embs.view(num_rows, num_to_sample, -1)

    def _sample_hard(
        self,
        pool_id: int,
        query_embeddings: torch.Tensor,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_rows = query_embeddings.size(0)
        device = query_embeddings.device
        dtype = positive_ids.dtype
        if num_to_sample == 0 or num_rows == 0:
            return (
                torch.empty(num_rows, 0, dtype=dtype, device=device),
                torch.empty(num_rows, 0, self._item_emb.output_dim, dtype=torch.float32, device=device),
            )

        candidate_ids, candidate_embs = self._sample_from_pool(
            pool_id,
            self.hard_candidate_pool_size,
            device,
            dtype,
        )
        scores = torch.matmul(query_embeddings.float(), candidate_embs.float().t())
        scores = torch.where(
            candidate_ids.unsqueeze(0) == positive_ids.unsqueeze(1),
            torch.full_like(scores, -torch.inf),
            scores,
        )
        sorted_idx = torch.argsort(scores, dim=1, descending=True)
        rank_start = min(self.hard_rank_start, max(0, sorted_idx.size(1) - 1))
        rank_end = min(self.hard_rank_end, sorted_idx.size(1))
        if rank_end <= rank_start:
            rank_start = 0
            rank_end = sorted_idx.size(1)
        window = sorted_idx[:, rank_start:rank_end]
        choices = torch.randint(
            low=0,
            high=window.size(1),
            size=(num_rows, num_to_sample),
            dtype=torch.long,
            device=device,
        )
        selected = torch.gather(window, dim=1, index=choices)
        return candidate_ids[selected], candidate_embs[selected]

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_to_sample <= 0:
            raise ValueError("num_to_sample must be > 0")
        query_embeddings = kwargs.get("query_embeddings")
        if query_embeddings is None:
            raise ValueError("MixedHardGlobalNegativesSampler requires query_embeddings")
        supervision_type_ids = kwargs.get("supervision_type_ids")
        if supervision_type_ids is None:
            # Fallback keeps the class usable in smoke tests, but train configs
            # should pass supervision_type_ids for event-type-aware routing.
            domain_ids = positive_ids // self.domain_offset
        else:
            domain_ids = self._domains_from_event_types(positive_ids, supervision_type_ids)

        device = positive_ids.device
        N, K = positive_ids.size(0), num_to_sample
        hard_k = int(round(K * self.hard_fraction))
        hard_k = max(0, min(K, hard_k))
        uniform_k = K - hard_k

        sampled_ids_chunks = []
        sampled_emb_chunks = []
        for domain_id in torch.unique(domain_ids).tolist():
            if domain_id not in self.shard_counts:
                raise ValueError(
                    f"No hard-negative pool for domain {domain_id}; "
                    f"available domains={sorted(self.shard_counts.keys())}"
                )
            indices = (domain_ids == domain_id).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue

            uniform_ids, uniform_embs = self._sample_uniform(
                domain_id,
                indices.numel(),
                uniform_k,
                device,
                positive_ids.dtype,
            )
            hard_ids, hard_embs = self._sample_hard(
                domain_id,
                query_embeddings[indices],
                positive_ids[indices],
                hard_k,
            )
            sampled_ids_chunks.append((indices, torch.cat([uniform_ids, hard_ids], dim=1)))
            sampled_emb_chunks.append((indices, torch.cat([uniform_embs, hard_embs], dim=1)))

        sampled_ids = torch.zeros(N, K, dtype=positive_ids.dtype, device=device)
        sampled_negative_embeddings = torch.zeros(N, K, self._item_emb.output_dim, dtype=torch.float32, device=device)
        for indices, ids in sampled_ids_chunks:
            sampled_ids[indices] = ids
        for indices, embs in sampled_emb_chunks:
            sampled_negative_embeddings[indices] = embs
        return sampled_ids, sampled_negative_embeddings
