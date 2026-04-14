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
from typing import Dict, List, Optional


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
    ):
        super().__init__()
        self.task_ids = task_ids
        self.num_experts = num_experts
        
        # Shared experts
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Per-task gating networks
        self.gates = nn.ModuleDict({
            str(tid): nn.Linear(input_dim, num_experts)
            for tid in task_ids
        })

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        # x: (B, N, D)
        # Expert outputs: list of (B, N, D_out)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        # expert_outputs: (B, N, num_experts, D_out)
        
        task_outputs = {}
        for tid in self.task_ids:
            gate_logits = self.gates[str(tid)](x)  # (B, N, num_experts)
            gate_weights = F.softmax(gate_logits, dim=-1)  # (B, N, num_experts)
            # Weighted sum of experts: (B, N, num_experts, 1) * (B, N, num_experts, D_out) -> sum -> (B, N, D_out)
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
        self.shared_experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, output_dim, dropout)
            for _ in range(num_shared_experts)
        ])
        
        # Task-specific experts
        self.task_experts = nn.ModuleDict({
            str(tid): nn.ModuleList([
                Expert(input_dim, expert_hidden_dim, output_dim, dropout)
                for _ in range(num_task_experts)
            ])
            for tid in task_ids
        })
        
        # Per-task gating (over shared + task-specific experts)
        self.gates = nn.ModuleDict({
            str(tid): nn.Linear(input_dim, total_experts_per_task)
            for tid in task_ids
        })

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
