"""P11/P12/P13/P14 residual target-event conditioning modules.

P11/P12 introduced residual output conditioning by target event type/group.
P13/P14 stabilize that path with small learnable gates and post-condition L2
normalization to avoid query-norm/score-scale collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from proposed12_group_residual.event_groups import build_event_type_to_group_tensor


class EventTypeResidualConditioner(torch.nn.Module):
    """Residual output conditioning by target event type or event group.

    output = seq_embeddings + residual_scale * adapter([seq_embeddings, condition_emb])

    The final adapter layer is zero initialized, so the initial model is exactly
    the unconditioned base model. This makes event-conditioned specialization a
    residual correction rather than a replacement head from random init.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        num_event_types: int,
        hidden_dim: int,
        granularity: str = "event",
        residual_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if granularity not in ("event", "group"):
            raise ValueError(f"Unsupported event residual granularity: {granularity}")
        self.granularity = granularity
        self.residual_scale = residual_scale
        self.num_event_types = num_event_types
        num_conditions = (num_event_types + 1) if granularity == "event" else 6  # pad + five groups
        self.condition_emb = nn.Embedding(num_conditions, condition_dim, padding_idx=0)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        last = self.adapter[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        nn.init.xavier_normal_(self.condition_emb.weight)
        with torch.no_grad():
            self.condition_emb.weight[0].zero_()

        self.register_buffer(
            "event_type_to_group",
            build_event_type_to_group_tensor(num_event_types),
            persistent=False,
        )
        self._last_metrics = {}

    def _condition_ids(self, next_type_ids: torch.Tensor) -> torch.Tensor:
        ids = next_type_ids.long().clamp(min=0, max=self.num_event_types)
        if self.granularity == "group":
            ids = self.event_type_to_group[ids]
        return ids

    def forward(self, seq_embeddings: torch.Tensor, next_type_ids: torch.Tensor) -> torch.Tensor:
        cond_ids = self._condition_ids(next_type_ids)
        cond_emb = self.condition_emb(cond_ids)
        residual = self.adapter(torch.cat([seq_embeddings, cond_emb], dim=-1))
        out = seq_embeddings + self.residual_scale * residual
        with torch.no_grad():
            base_norm = seq_embeddings.norm(dim=-1).mean()
            residual_norm = residual.norm(dim=-1).mean()
            self._last_metrics = {
                "residual_norm": residual_norm.detach(),
                "base_norm": base_norm.detach(),
                "residual_over_base": (residual_norm / (base_norm + 1e-8)).detach(),
            }
        return out


class StabilizedEventTypeResidualConditioner(EventTypeResidualConditioner):
    """P13/P14 stabilized residual conditioner.

    q = L2Norm(base + alpha_condition * residual)

    alpha_condition is a small learnable gate bounded by max_scale. The adapter
    remains zero-initialized, so the initial output is normalized base.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        num_event_types: int,
        hidden_dim: int,
        granularity: str = "event",
        residual_scale: float = 1.0,
        dropout: float = 0.1,
        max_scale: float = 0.05,
        l2_normalize: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            condition_dim=condition_dim,
            num_event_types=num_event_types,
            hidden_dim=hidden_dim,
            granularity=granularity,
            residual_scale=residual_scale,
            dropout=dropout,
        )
        num_conditions = (num_event_types + 1) if granularity == "event" else 6
        self.max_scale = max_scale
        self.l2_normalize = l2_normalize
        # sigmoid(-4) ~= 0.018, so with max_scale=0.05 the initial alpha is ~0.0009.
        self.raw_alpha = nn.Parameter(torch.full((num_conditions,), -4.0))

    def forward(self, seq_embeddings: torch.Tensor, next_type_ids: torch.Tensor) -> torch.Tensor:
        cond_ids = self._condition_ids(next_type_ids)
        cond_emb = self.condition_emb(cond_ids)
        residual = self.adapter(torch.cat([seq_embeddings, cond_emb], dim=-1))
        alpha = self.max_scale * torch.sigmoid(self.raw_alpha[cond_ids]).unsqueeze(-1)
        out = seq_embeddings + self.residual_scale * alpha * residual
        if self.l2_normalize:
            out = F.normalize(out, p=2, dim=-1, eps=1e-6)
        with torch.no_grad():
            base_norm = seq_embeddings.norm(dim=-1).mean()
            residual_norm = residual.norm(dim=-1).mean()
            out_norm = out.norm(dim=-1).mean()
            alpha_values = alpha.squeeze(-1)
            self._last_metrics = {
                "residual_norm": residual_norm.detach(),
                "base_norm": base_norm.detach(),
                "query_norm": out_norm.detach(),
                "residual_over_base": (residual_norm / (base_norm + 1e-8)).detach(),
                "alpha_mean": alpha_values.mean().detach(),
                "alpha_max": alpha_values.max().detach(),
            }
        return out


class StabilizedEventGroupExpertResidualConditioner(torch.nn.Module):
    """P17: separate bounded residual adapter per target event group.

    This is hierarchical: domain/task MMoE produces the base embedding first;
    then the target event group selects one residual expert adapter. Unlike P14,
    the group specialization is a separate adapter bank, not only a condition
    embedding into one shared adapter.
    """
    def __init__(self, input_dim:int, condition_dim:int, num_event_types:int, hidden_dim:int, granularity:str='group_expert', residual_scale:float=1.0, dropout:float=0.1, max_scale:float=0.05, l2_normalize:bool=True):
        super().__init__()
        self.input_dim = input_dim; self.residual_scale = residual_scale; self.max_scale = max_scale; self.l2_normalize = l2_normalize
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))
            for _ in range(6)
        ])
        for adapter in self.adapters:
            last = adapter[-1]; nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.raw_alpha = nn.Parameter(torch.full((6,), -4.0))
        self.register_buffer('event_type_to_group', build_event_type_to_group_tensor(num_event_types), persistent=False)
        self._last_metrics = {}
    def _group_ids(self, next_type_ids: torch.Tensor) -> torch.Tensor:
        safe = next_type_ids.long().clamp_min(0).clamp_max(self.event_type_to_group.numel() - 1)
        return self.event_type_to_group[safe]
    def forward(self, seq_embeddings: torch.Tensor, next_type_ids: torch.Tensor) -> torch.Tensor:
        group_ids = self._group_ids(next_type_ids.to(seq_embeddings.device))
        residuals = torch.stack([adapter(seq_embeddings) for adapter in self.adapters], dim=2)  # [B,N,6,D]
        gather_idx = group_ids.view(*group_ids.shape,1,1).expand(-1,-1,1,seq_embeddings.size(-1))
        residual = residuals.gather(2, gather_idx).squeeze(2)
        alpha_all = self.max_scale * torch.sigmoid(self.raw_alpha)
        alpha = alpha_all[group_ids].unsqueeze(-1)
        out = seq_embeddings + self.residual_scale * alpha * residual
        if self.l2_normalize:
            out = F.normalize(out, p=2, dim=-1)
        with torch.no_grad():
            residual_norm = residual.norm(dim=-1).mean(); base_norm = seq_embeddings.norm(dim=-1).mean(); query_norm = out.norm(dim=-1).mean()
            self._last_metrics = {
                'residual_norm': residual_norm.detach(),
                'base_norm': base_norm.detach(),
                'query_norm': query_norm.detach(),
                'residual_over_base': (residual_norm / (base_norm + 1e-8)).detach(),
                'alpha_mean': alpha.mean().detach(),
                'alpha_max': alpha.max().detach(),
            }
        return out
