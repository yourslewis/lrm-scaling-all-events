from typing import Dict, Optional, Tuple

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
        filter_batch_positives: bool = False,
        filter_resample_attempts: int = 3,
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
        self.filter_batch_positives: bool = filter_batch_positives
        self.filter_resample_attempts: int = filter_resample_attempts
        self.pools: Dict[int, Tuple[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._batch_positive_ids: Optional[torch.Tensor] = None

    def debug_str(self) -> str:
        return (
            f"mixed-hard-global-hf{self.hard_fraction:g}"
            f"-c{self.hard_candidate_pool_size}"
            f"-r{self.hard_rank_start}-{self.hard_rank_end}"
            f"{'-filter-batch-pos' if self.filter_batch_positives else ''}"
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
        # Optional denoising filter: cache all valid supervision ids in each
        # sequence row so sampled negatives can exclude other positives from
        # the same user/session window, not just the exact current target.
        # This is intentionally lightweight and batch-local; eval remains
        # unchanged through the separate rotate sampler.
        if not self.filter_batch_positives:
            self._batch_positive_ids = None
            return
        self._batch_positive_ids = torch.where(
            presences & (ids != 0),
            ids,
            torch.zeros_like(ids),
        ).detach()

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
        row_positive_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_to_sample == 0:
            return (
                torch.empty(num_rows, 0, dtype=dtype, device=device),
                torch.empty(num_rows, 0, self._item_emb.output_dim, dtype=torch.float32, device=device),
            )
        ids, embs = self._sample_from_pool(pool_id, num_rows * num_to_sample, device, dtype)
        ids = ids.view(num_rows, num_to_sample)
        embs = embs.view(num_rows, num_to_sample, -1)
        if row_positive_ids is not None:
            ids, embs = self._resample_conflicts(pool_id, ids, embs, row_positive_ids, device, dtype)
        return ids, embs

    def _sample_hard(
        self,
        pool_id: int,
        query_embeddings: torch.Tensor,
        positive_ids: torch.Tensor,
        num_to_sample: int,
        row_positive_ids: Optional[torch.Tensor] = None,
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
        if row_positive_ids is not None:
            row_positive_ids = row_positive_ids.to(device=device, dtype=dtype)
            conflict = torch.zeros_like(scores, dtype=torch.bool)
            for col in range(row_positive_ids.size(1)):
                pos = row_positive_ids[:, col]
                valid = pos != 0
                if torch.any(valid):
                    conflict |= valid.unsqueeze(1) & (candidate_ids.unsqueeze(0) == pos.unsqueeze(1))
            scores = torch.where(conflict, torch.full_like(scores, -torch.inf), scores)
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

    def _resample_conflicts(
        self,
        pool_id: int,
        ids: torch.Tensor,
        embs: torch.Tensor,
        row_positive_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row_positive_ids = row_positive_ids.to(device=device, dtype=dtype)
        for _ in range(max(0, self.filter_resample_attempts)):
            conflict = torch.zeros(ids.shape, dtype=torch.bool, device=device)
            for col in range(row_positive_ids.size(1)):
                pos = row_positive_ids[:, col]
                valid = pos != 0
                if torch.any(valid):
                    conflict |= valid.unsqueeze(1) & (ids == pos.unsqueeze(1))
            if not torch.any(conflict):
                break
            n = int(conflict.sum().item())
            new_ids, new_embs = self._sample_from_pool(pool_id, n, device, dtype)
            ids = ids.clone()
            embs = embs.clone()
            ids[conflict] = new_ids
            embs[conflict] = new_embs
        return ids, embs

    def _row_positive_ids_for(
        self,
        jagged_seq_ids: Optional[torch.Tensor],
        indices: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.filter_batch_positives or self._batch_positive_ids is None or jagged_seq_ids is None:
            return None
        seq_ids = jagged_seq_ids[indices].to(device=self._batch_positive_ids.device).long()
        return self._batch_positive_ids[seq_ids].to(device=device)

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
        jagged_seq_ids = kwargs.get("jagged_seq_ids")
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

            row_positive_ids = self._row_positive_ids_for(jagged_seq_ids, indices, device)
            uniform_ids, uniform_embs = self._sample_uniform(
                domain_id,
                indices.numel(),
                uniform_k,
                device,
                positive_ids.dtype,
                row_positive_ids=row_positive_ids,
            )
            hard_ids, hard_embs = self._sample_hard(
                domain_id,
                query_embeddings[indices],
                positive_ids[indices],
                hard_k,
                row_positive_ids=row_positive_ids,
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


@register("sampler", "HybridDomainInBatchHardGlobalNegativesSampler")
class HybridDomainInBatchHardGlobalNegativesSampler(MixedHardGlobalNegativesSampler):
    """Hybrid sampler: domain-aware in-batch + global hard + global random.

    For each positive, splits K negatives into:
      1. domain-aware in-batch random negatives;
      2. event-type/domain-aware global hard negatives;
      3. event-type/domain-aware global random negatives.

    If the in-batch same-domain pool cannot provide enough unique candidates for
    a row, the missing slots are filled from the global hard-negative pool.
    """

    def __init__(
        self,
        item_emb: EmbeddingModule,
        domain_offset: int,
        shard_size: int,
        shard_counts: Dict[int, int],
        l2_norm: bool,
        l2_norm_eps: float,
        in_batch_fraction: float = 0.50,
        global_hard_fraction: float = 0.30,
        hard_candidate_pool_size: int = 1024,
        hard_rank_start: int = 32,
        hard_rank_end: int = 512,
    ) -> None:
        if in_batch_fraction < 0 or global_hard_fraction < 0 or in_batch_fraction + global_hard_fraction > 1:
            raise ValueError(
                "in_batch_fraction and global_hard_fraction must be non-negative "
                "and sum to <= 1"
            )
        super().__init__(
            item_emb=item_emb,
            domain_offset=domain_offset,
            shard_size=shard_size,
            shard_counts=shard_counts,
            l2_norm=l2_norm,
            l2_norm_eps=l2_norm_eps,
            hard_fraction=global_hard_fraction,
            hard_candidate_pool_size=hard_candidate_pool_size,
            hard_rank_start=hard_rank_start,
            hard_rank_end=hard_rank_end,
            filter_batch_positives=True,
        )
        self.in_batch_fraction = in_batch_fraction
        self.global_hard_fraction = global_hard_fraction
        self._cached_ids: Optional[torch.Tensor] = None
        self._cached_embeddings: Optional[torch.Tensor] = None
        self._cached_seq_ids: Optional[torch.Tensor] = None
        self._cached_domain_ids: Optional[torch.Tensor] = None

    def debug_str(self) -> str:
        return (
            f"hybrid-domain-inbatch{self.in_batch_fraction:g}"
            f"-hard{self.global_hard_fraction:g}"
            f"-globalrand{max(0.0, 1.0 - self.in_batch_fraction - self.global_hard_fraction):g}"
            f"-c{self.hard_candidate_pool_size}"
            f"-r{self.hard_rank_start}-{self.hard_rank_end}"
            f"{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )

    def process_batch(
        self,
        ids: torch.Tensor,
        presences: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> None:
        super().process_batch(ids, presences, embeddings)
        valid = presences & (ids != 0)
        self._cached_ids = ids[valid]
        self._cached_embeddings = self.normalize_embeddings(embeddings[valid])
        self._cached_seq_ids = torch.arange(ids.size(0), device=ids.device).unsqueeze(1).expand_as(ids)[valid]
        self._cached_domain_ids = self._cached_ids // self.domain_offset

    def _sample_in_batch(
        self,
        domain_id: int,
        seq_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        num_to_sample: int,
        row_positive_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = positive_ids.device
        dtype = positive_ids.dtype
        num_rows = positive_ids.size(0)
        ids = torch.zeros(num_rows, num_to_sample, dtype=dtype, device=device)
        embs = torch.zeros(num_rows, num_to_sample, self._item_emb.output_dim, dtype=torch.float32, device=device)
        valid_out = torch.zeros(num_rows, num_to_sample, dtype=torch.bool, device=device)
        if num_to_sample == 0 or self._cached_ids is None or self._cached_ids.numel() == 0:
            return ids, embs, valid_out

        cached_ids = self._cached_ids.to(device=device, dtype=dtype)
        cached_embs = self._cached_embeddings.to(device=device)
        cached_seq_ids = self._cached_seq_ids.to(device=device)
        cached_domains = self._cached_domain_ids.to(device=device)
        row_positive_ids = row_positive_ids.to(device=device, dtype=dtype) if row_positive_ids is not None else None

        for row in range(num_rows):
            mask = (cached_domains == domain_id) & (cached_seq_ids != seq_ids[row]) & (cached_ids != positive_ids[row])
            if row_positive_ids is not None:
                row_pos = row_positive_ids[row]
                for col in range(row_pos.numel()):
                    pos = row_pos[col]
                    if pos != 0:
                        mask &= cached_ids != pos
            offsets = torch.nonzero(mask, as_tuple=True)[0]
            if offsets.numel() == 0:
                continue
            take = min(num_to_sample, int(offsets.numel()))
            perm = torch.randperm(offsets.numel(), device=device)[:take]
            selected = offsets[perm]
            ids[row, :take] = cached_ids[selected]
            embs[row, :take] = cached_embs[selected]
            valid_out[row, :take] = True
        return ids, embs, valid_out

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
            raise ValueError("HybridDomainInBatchHardGlobalNegativesSampler requires query_embeddings")
        supervision_type_ids = kwargs.get("supervision_type_ids")
        if supervision_type_ids is None:
            domain_ids = positive_ids // self.domain_offset
        else:
            domain_ids = self._domains_from_event_types(positive_ids, supervision_type_ids)

        device = positive_ids.device
        N, K = positive_ids.size(0), num_to_sample
        in_batch_k = int(round(K * self.in_batch_fraction))
        hard_k = int(round(K * self.global_hard_fraction))
        in_batch_k = max(0, min(K, in_batch_k))
        hard_k = max(0, min(K - in_batch_k, hard_k))
        random_k = K - in_batch_k - hard_k
        jagged_seq_ids = kwargs.get("jagged_seq_ids")
        if jagged_seq_ids is None:
            raise ValueError("HybridDomainInBatchHardGlobalNegativesSampler requires jagged_seq_ids")

        sampled_ids = torch.zeros(N, K, dtype=positive_ids.dtype, device=device)
        sampled_negative_embeddings = torch.zeros(N, K, self._item_emb.output_dim, dtype=torch.float32, device=device)

        for domain_id in torch.unique(domain_ids).tolist():
            if domain_id not in self.shard_counts:
                raise ValueError(
                    f"No negative pool for domain {domain_id}; "
                    f"available domains={sorted(self.shard_counts.keys())}"
                )
            indices = (domain_ids == domain_id).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue

            row_positive_ids = self._row_positive_ids_for(jagged_seq_ids, indices, device)
            seq_ids = jagged_seq_ids[indices]
            in_ids, in_embs, in_valid = self._sample_in_batch(
                domain_id=domain_id,
                seq_ids=seq_ids,
                positive_ids=positive_ids[indices],
                num_to_sample=in_batch_k,
                row_positive_ids=row_positive_ids,
            )
            # Draw enough hard negatives to fill the normal hard quota plus any
            # unfilled in-batch slots. Unused fallback hard samples are ignored.
            hard_ids_all, hard_embs_all = self._sample_hard(
                domain_id,
                query_embeddings[indices],
                positive_ids[indices],
                hard_k + in_batch_k,
                row_positive_ids=row_positive_ids,
            )
            rand_ids, rand_embs = self._sample_uniform(
                domain_id,
                indices.numel(),
                random_k,
                device,
                positive_ids.dtype,
                row_positive_ids=row_positive_ids,
            )

            hard_fallback_ids = hard_ids_all[:, :in_batch_k]
            hard_fallback_embs = hard_embs_all[:, :in_batch_k]
            hard_ids = hard_ids_all[:, in_batch_k:in_batch_k + hard_k]
            hard_embs = hard_embs_all[:, in_batch_k:in_batch_k + hard_k]
            mixed_in_ids = torch.where(in_valid, in_ids, hard_fallback_ids)
            mixed_in_embs = torch.where(in_valid.unsqueeze(2), in_embs, hard_fallback_embs)
            row_ids = torch.cat([mixed_in_ids, hard_ids, rand_ids], dim=1)
            row_embs = torch.cat([mixed_in_embs, hard_embs, rand_embs], dim=1)
            sampled_ids[indices] = row_ids
            sampled_negative_embeddings[indices] = row_embs

        return sampled_ids, sampled_negative_embeddings
