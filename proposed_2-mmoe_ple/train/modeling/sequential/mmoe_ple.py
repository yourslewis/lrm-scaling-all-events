"""
MMoE and PLE modules for multi-task learning on top of HSTU encoder.

MMoE: Multi-gate Mixture-of-Experts (Ma et al., 2018)
PLE: Progressive Layered Extraction (Tang et al., 2020)

These modules take the HSTU seq_embeddings and produce task-specific
output embeddings, one per domain/task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class Expert(nn.Module):
    """A single expert network (2-layer MLP with ReLU)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerExpert(nn.Module):
    """Causal mini-transformer expert on top of shared sequence embeddings."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.proj_in = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.proj_out = nn.Identity()

        layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # True = masked
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        h = self.proj_in(x)
        n = h.size(1)
        mask = self._causal_mask(n, h.device)
        h = self.encoder(h, mask=mask)
        return self.proj_out(h)


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts.

    All tasks share the same set of experts, but each task has its own
    gating network that learns to select/weight experts differently.

    Input:  (B, N, D) — HSTU sequence embeddings
    Output: Dict[task_id, (B, N, D)] — task-specific embeddings
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        expert_hidden_dim: int,
        output_dim: int,
        task_ids: List[int],
        dropout: float = 0.1,
        expert_type: str = "mlp",  # mlp | transformer
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn_multiplier: int = 4,
        gate_hidden_dim: int = 0,
        load_balance_alpha: float = 0.0,
        router_z_beta: float = 0.0,
    ):
        super().__init__()
        self.task_ids = task_ids
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha
        self.router_z_beta = router_z_beta
        self.expert_type = expert_type

        def build_expert() -> nn.Module:
            if expert_type == "transformer":
                return TransformerExpert(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_layers=transformer_layers,
                    num_heads=transformer_heads,
                    ffn_multiplier=transformer_ffn_multiplier,
                    dropout=dropout,
                )
            return Expert(input_dim, expert_hidden_dim, output_dim, dropout)

        # Shared experts
        self.experts = nn.ModuleList([build_expert() for _ in range(num_experts)])

        # Per-task gating networks
        if gate_hidden_dim and gate_hidden_dim > 0:
            self.gates = nn.ModuleDict(
                {
                    str(tid): nn.Sequential(
                        nn.Linear(input_dim, gate_hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(gate_hidden_dim, num_experts),
                    )
                    for tid in task_ids
                }
            )
        else:
            self.gates = nn.ModuleDict({str(tid): nn.Linear(input_dim, num_experts) for tid in task_ids})

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        # x: (B, N, D)
        # Expert outputs: list of (B, N, D_out)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        # expert_outputs: (B, N, num_experts, D_out)

        task_outputs = {}
        self._last_gate_entropy = {}  # for diagnostics
        gate_weights_per_task = {}
        gate_logits_per_task = {}
        for tid in self.task_ids:
            gate_logits = self.gates[str(tid)](x)  # (B, N, num_experts)
            gate_weights = F.softmax(gate_logits, dim=-1)  # (B, N, num_experts)
            gate_logits_per_task[tid] = gate_logits
            gate_weights_per_task[tid] = gate_weights
            # Gate entropy: -sum(p * log(p)), averaged over batch and sequence
            gate_entropy = -(gate_weights * (gate_weights + 1e-8).log()).sum(dim=-1).mean()
            self._last_gate_entropy[tid] = gate_entropy.detach()
            # Weighted sum of experts
            task_out = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
            task_outputs[tid] = task_out

        # Load balancing loss (Switch Transformer)
        if self.load_balance_alpha > 0:
            all_gate_weights = torch.stack([gate_weights_per_task[tid] for tid in self.task_ids])
            f = all_gate_weights.mean(dim=[0, 1, 2])  # mean fraction per expert
            P = all_gate_weights.mean(dim=[0, 1, 2])  # (same for soft routing)
            self._load_balance_loss = self.num_experts * (f * P).sum()
        else:
            self._load_balance_loss = torch.tensor(0.0, device=x.device)

        # Router z-loss (ST-MoE)
        if self.router_z_beta > 0:
            all_logits = torch.stack([gate_logits_per_task[tid] for tid in self.task_ids])
            self._router_z_loss = torch.logsumexp(all_logits, dim=-1).square().mean()
        else:
            self._router_z_loss = torch.tensor(0.0, device=x.device)

        return task_outputs


class HSTUMMoE(nn.Module):
    """
    Option C: Independent HSTU experts with a shared HSTU for gating.

    Architecture:
        - 1 shared (lighter) HSTU encodes the input → gate embeddings
        - N independent (full) HSTU encoders each process raw input → expert embeddings
        - Per-task gating networks combine expert outputs

    Unlike MMoE (which puts MLP/transformer heads on top of a single shared HSTU),
    here each expert is a full independent HSTU encoder that sees the raw input.

    This module does NOT call the HSTU forward itself — it is given the
    expert HSTU models at construction time and calls them during forward.

    Input:  raw inputs (past_lengths, past_ids, past_embeddings, past_payloads)
    Output: Dict[task_id, (B, N, D)] — task-specific embeddings
    """

    def __init__(
        self,
        gate_dim: int,
        num_experts: int,
        output_dim: int,
        task_ids: List[int],
        dropout: float = 0.1,
        gate_hidden_dim: int = 0,
        load_balance_alpha: float = 0.0,
        router_z_beta: float = 0.0,
    ):
        super().__init__()
        self.task_ids = task_ids
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha
        self.router_z_beta = router_z_beta
        self.output_dim = output_dim
        # Expert HSTU models are set externally via set_expert_models()
        self.expert_models = None

        # Per-task gating networks (operate on gate HSTU output)
        if gate_hidden_dim and gate_hidden_dim > 0:
            self.gates = nn.ModuleDict(
                {
                    str(tid): nn.Sequential(
                        nn.Linear(gate_dim, gate_hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(gate_hidden_dim, num_experts),
                    )
                    for tid in task_ids
                }
            )
        else:
            self.gates = nn.ModuleDict(
                {str(tid): nn.Linear(gate_dim, num_experts) for tid in task_ids}
            )

    def set_expert_models(self, expert_models: nn.ModuleList):
        """Set the expert HSTU encoder models (called from SequentialRetrieval.__init__)."""
        self.expert_models = expert_models

    def forward(
        self,
        gate_embeddings: torch.Tensor,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: dict,
    ) -> Dict[int, torch.Tensor]:
        """
        Args:
            gate_embeddings: (B, N, D_gate) — output of the shared/gate HSTU
            past_lengths, past_ids, past_embeddings, past_payloads: raw inputs
                for each expert HSTU to encode independently
        Returns:
            Dict[task_id, (B, N, D)] — task-specific embeddings
        """
        assert self.expert_models is not None, "Expert models not set. Call set_expert_models() first."

        # Each expert HSTU encodes the raw input independently
        expert_outputs = torch.stack(
            [
                expert_hstu(
                    past_lengths=past_lengths,
                    past_ids=past_ids,
                    past_embeddings=past_embeddings,
                    past_payloads=past_payloads,
                )
                for expert_hstu in self.expert_models
            ],
            dim=-2,
        )  # (B, N, num_experts, D)

        task_outputs = {}
        for tid in self.task_ids:
            gate_logits = self.gates[str(tid)](gate_embeddings)  # (B, N, num_experts)
            gate_weights = F.softmax(gate_logits, dim=-1)
            task_out = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
            task_outputs[tid] = task_out

        return task_outputs


class PLE(nn.Module):
    """
    Progressive Layered Extraction (single extraction layer).

    Each task has its own task-specific experts PLUS shared experts.
    Each task's gate selects from both its own experts and shared experts.

    Input:  (B, N, D) — HSTU sequence embeddings
    Output: Dict[task_id, (B, N, D)] — task-specific embeddings
    """

    def __init__(
        self,
        input_dim: int,
        num_shared_experts: int,
        num_task_experts: int,
        expert_hidden_dim: int,
        output_dim: int,
        task_ids: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task_ids = task_ids
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        total_experts_per_task = num_shared_experts + num_task_experts

        # Shared experts
        self.shared_experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden_dim, output_dim, dropout) for _ in range(num_shared_experts)]
        )

        # Task-specific experts
        self.task_experts = nn.ModuleDict(
            {
                str(tid): nn.ModuleList(
                    [Expert(input_dim, expert_hidden_dim, output_dim, dropout) for _ in range(num_task_experts)]
                )
                for tid in task_ids
            }
        )

        # Per-task gating (over shared + task-specific experts)
        self.gates = nn.ModuleDict({str(tid): nn.Linear(input_dim, total_experts_per_task) for tid in task_ids})

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        # x: (B, N, D)
        # Shared expert outputs
        shared_outs = [expert(x) for expert in self.shared_experts]

        task_outputs = {}
        for tid in self.task_ids:
            # Task-specific expert outputs
            task_outs = [expert(x) for expert in self.task_experts[str(tid)]]

            # Concatenate shared + task experts
            all_outs = torch.stack(shared_outs + task_outs, dim=-2)
            # all_outs: (B, N, num_shared + num_task, D_out)

            gate_logits = self.gates[str(tid)](x)  # (B, N, total_experts)
            gate_weights = F.softmax(gate_logits, dim=-1)

            task_out = (gate_weights.unsqueeze(-1) * all_outs).sum(dim=-2)
            task_outputs[tid] = task_out

        return task_outputs


# --- P15/P16 event-aware MMoE variants ---

class EventTypeGatedMMoE(nn.Module):
    """P15: MMoE with target event type conditioning in the router."""
    def __init__(self, input_dim:int, num_experts:int, expert_hidden_dim:int, output_dim:int, task_ids:List[int], num_event_types:int, condition_dim:int=32, dropout:float=0.1, expert_type:str="mlp", transformer_layers:int=2, transformer_heads:int=4, transformer_ffn_multiplier:int=4, gate_hidden_dim:int=0, load_balance_alpha:float=0.0, router_z_beta:float=0.0):
        super().__init__()
        self.task_ids = task_ids; self.num_experts = num_experts; self.load_balance_alpha = load_balance_alpha; self.router_z_beta = router_z_beta
        self.condition_emb = nn.Embedding(num_event_types + 1, condition_dim, padding_idx=0)
        nn.init.xavier_normal_(self.condition_emb.weight)
        with torch.no_grad(): self.condition_emb.weight[0].zero_()
        def build_expert():
            if expert_type == "transformer":
                return TransformerExpert(input_dim, output_dim, transformer_layers, transformer_heads, transformer_ffn_multiplier, dropout)
            return Expert(input_dim, expert_hidden_dim, output_dim, dropout)
        self.experts = nn.ModuleList([build_expert() for _ in range(num_experts)])
        gate_input_dim = input_dim + condition_dim
        if gate_hidden_dim and gate_hidden_dim > 0:
            self.gates = nn.ModuleDict({str(tid): nn.Sequential(nn.Linear(gate_input_dim, gate_hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(gate_hidden_dim, num_experts)) for tid in task_ids})
        else:
            self.gates = nn.ModuleDict({str(tid): nn.Linear(gate_input_dim, num_experts) for tid in task_ids})
    def forward(self, x: torch.Tensor, next_type_ids: torch.Tensor=None) -> Dict[int, torch.Tensor]:
        if next_type_ids is None:
            next_type_ids = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        cond = self.condition_emb(next_type_ids.long().clamp_min(0).clamp_max(self.condition_emb.num_embeddings - 1))
        gate_x = torch.cat([x, cond], dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        task_outputs = {}; self._last_gate_entropy = {}; gate_weights_per_task = {}; gate_logits_per_task = {}
        for tid in self.task_ids:
            logits = self.gates[str(tid)](gate_x); weights = F.softmax(logits, dim=-1)
            gate_logits_per_task[tid] = logits; gate_weights_per_task[tid] = weights
            self._last_gate_entropy[tid] = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().detach()
            task_outputs[tid] = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        if self.load_balance_alpha > 0:
            all_w = torch.stack([gate_weights_per_task[tid] for tid in self.task_ids]); f = all_w.mean(dim=[0,1,2]); P = all_w.mean(dim=[0,1,2]); self._load_balance_loss = self.num_experts * (f * P).sum()
        else: self._load_balance_loss = torch.tensor(0.0, device=x.device)
        if self.router_z_beta > 0:
            all_logits = torch.stack([gate_logits_per_task[tid] for tid in self.task_ids]); self._router_z_loss = torch.logsumexp(all_logits, dim=-1).square().mean()
        else: self._router_z_loss = torch.tensor(0.0, device=x.device)
        return task_outputs

class EventGroupSpecificMMoE(nn.Module):
    """P16: shared experts plus target-event-group-specific experts."""
    def __init__(self, input_dim:int, num_shared_experts:int, num_group_experts:int, expert_hidden_dim:int, output_dim:int, task_ids:List[int], num_event_types:int, condition_dim:int=32, dropout:float=0.1, gate_hidden_dim:int=0, load_balance_alpha:float=0.0, router_z_beta:float=0.0):
        super().__init__()
        from proposed12_group_residual.event_groups import build_event_type_to_group_tensor
        self.task_ids = task_ids; self.num_shared_experts = num_shared_experts; self.num_group_experts = num_group_experts; self.num_experts = num_shared_experts + num_group_experts; self.load_balance_alpha = load_balance_alpha; self.router_z_beta = router_z_beta
        self.shared_experts = nn.ModuleList([Expert(input_dim, expert_hidden_dim, output_dim, dropout) for _ in range(num_shared_experts)])
        self.group_experts = nn.ModuleList([nn.ModuleList([Expert(input_dim, expert_hidden_dim, output_dim, dropout) for _ in range(num_group_experts)]) for _ in range(6)])
        self.condition_emb = nn.Embedding(6, condition_dim, padding_idx=0); nn.init.xavier_normal_(self.condition_emb.weight)
        with torch.no_grad(): self.condition_emb.weight[0].zero_()
        self.register_buffer('event_type_to_group', build_event_type_to_group_tensor(num_event_types), persistent=False)
        gate_input_dim = input_dim + condition_dim; total = num_shared_experts + num_group_experts
        if gate_hidden_dim and gate_hidden_dim > 0:
            self.gates = nn.ModuleDict({str(tid): nn.Sequential(nn.Linear(gate_input_dim, gate_hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(gate_hidden_dim, total)) for tid in task_ids})
        else:
            self.gates = nn.ModuleDict({str(tid): nn.Linear(gate_input_dim, total) for tid in task_ids})
    def _group_ids(self, next_type_ids: torch.Tensor) -> torch.Tensor:
        safe = next_type_ids.long().clamp_min(0).clamp_max(self.event_type_to_group.numel() - 1); return self.event_type_to_group[safe]
    def forward(self, x: torch.Tensor, next_type_ids: torch.Tensor=None) -> Dict[int, torch.Tensor]:
        if next_type_ids is None: group_ids = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        else: group_ids = self._group_ids(next_type_ids.to(x.device))
        cond = self.condition_emb(group_ids); gate_x = torch.cat([x, cond], dim=-1)
        shared = torch.stack([e(x) for e in self.shared_experts], dim=-2) if self.num_shared_experts > 0 else None
        group_banks = [torch.stack([e(x) for e in bank], dim=-2) for bank in self.group_experts]
        all_group = torch.stack(group_banks, dim=2)
        gather_idx = group_ids.view(*group_ids.shape,1,1,1).expand(-1,-1,1,self.num_group_experts,x.size(-1))
        active_group = all_group.gather(2, gather_idx).squeeze(2)
        expert_outputs = torch.cat([shared, active_group], dim=-2) if shared is not None else active_group
        task_outputs = {}; self._last_gate_entropy = {}; gate_weights_per_task = {}; gate_logits_per_task = {}
        for tid in self.task_ids:
            logits = self.gates[str(tid)](gate_x); weights = F.softmax(logits, dim=-1)
            gate_logits_per_task[tid] = logits; gate_weights_per_task[tid] = weights
            self._last_gate_entropy[tid] = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().detach()
            task_outputs[tid] = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        if self.load_balance_alpha > 0:
            all_w = torch.stack([gate_weights_per_task[tid] for tid in self.task_ids]); f = all_w.mean(dim=[0,1,2]); P = all_w.mean(dim=[0,1,2]); self._load_balance_loss = self.num_experts * (f * P).sum()
        else: self._load_balance_loss = torch.tensor(0.0, device=x.device)
        if self.router_z_beta > 0:
            all_logits = torch.stack([gate_logits_per_task[tid] for tid in self.task_ids]); self._router_z_loss = torch.logsumexp(all_logits, dim=-1).square().mean()
        else: self._router_z_loss = torch.tensor(0.0, device=x.device)
        return task_outputs
