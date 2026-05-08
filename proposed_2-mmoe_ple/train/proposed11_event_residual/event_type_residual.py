"""P11/P12 residual target-event conditioning module.

The module lives under proposed11 because P11 is the base mechanism:
use target event type at scoring time without replacing the shared trunk score.
P12 reuses the same residual conditioner with group granularity, using the
mapping in proposed12_group_residual.event_groups.
"""

import torch
import torch.nn as nn

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

    def _condition_ids(self, next_type_ids: torch.Tensor) -> torch.Tensor:
        ids = next_type_ids.long().clamp(min=0, max=self.num_event_types)
        if self.granularity == "group":
            ids = self.event_type_to_group[ids]
        return ids

    def forward(self, seq_embeddings: torch.Tensor, next_type_ids: torch.Tensor) -> torch.Tensor:
        cond_ids = self._condition_ids(next_type_ids)
        cond_emb = self.condition_emb(cond_ids)
        residual = self.adapter(torch.cat([seq_embeddings, cond_emb], dim=-1))
        return seq_embeddings + self.residual_scale * residual
