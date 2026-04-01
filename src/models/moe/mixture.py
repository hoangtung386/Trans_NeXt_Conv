"""Mixture of Experts with shared and routed experts."""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Tuple

from .experts import Expert, AdvancedExpert


class MixtureOfExperts(nn.Module):
    """Sparse Mixture of Experts with shared and routed experts.

    Uses low-rank router projection and top-k routing.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_routed_experts: int = 32,
        num_activated_experts: int = 4,
        num_shared_expert: int = 8,
        router_rank: int = 64,
    ):
        super().__init__()
        ffn_dim_routed = hidden_dim * 2
        ffn_dim_shared = hidden_dim * 3
        self.hidden_dim = hidden_dim
        self.num_routed_experts = num_routed_experts
        self.num_activated_experts = num_activated_experts
        self.num_shared_expert = num_shared_expert

        self.shared_experts = nn.ModuleList([
            AdvancedExpert(hidden_dim, ffn_dim_shared)
            for _ in range(num_shared_expert)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim_routed)
            for _ in range(num_routed_experts)
        ])
        self.router_down = nn.Linear(hidden_dim, router_rank, bias=False)
        self.router_up = nn.Linear(router_rank, num_routed_experts, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # Shared experts
        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_flat)
        shared_output = shared_output / self.num_shared_expert

        # Routing
        router_hidden = self.router_down(x_flat)
        router_logits = self.router_up(router_hidden)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_activated_experts, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Vectorized expert computation
        expert_outputs = torch.stack(
            [expert(x_flat) for expert in self.routed_experts]
        )  # [num_experts, B*S, D]

        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_routed_experts
        )  # [B*S, K, E]
        expert_mask = expert_mask.permute(2, 0, 1).float()  # [E, B*S, K]

        routed_output = torch.einsum(
            "ebd,ebk,bk->bd", expert_outputs, expert_mask, routing_weights
        )

        output = shared_output + routed_output
        output = output.view(batch_size, seq_len, hidden_dim)

        # Auxiliary losses
        expert_counts = torch.bincount(
            selected_experts.flatten(), minlength=self.num_routed_experts
        )
        load_balance_loss = expert_counts.float().var()

        routing_probs = F.softmax(router_logits, dim=-1)
        router_entropy = -(
            routing_probs * torch.log(routing_probs + 1e-10)
        ).sum(-1).mean()

        shared_output_norm = shared_output.norm(dim=-1).mean()
        routed_output_norm = routed_output.norm(dim=-1).mean()
        shared_routed_balance_loss = (
            shared_output_norm - routed_output_norm
        ).abs()

        aux_loss_dict = {
            "load_balance_loss": load_balance_loss,
            "router_entropy": router_entropy,
            "shared_routed_balance_loss": shared_routed_balance_loss,
        }
        return output, aux_loss_dict
