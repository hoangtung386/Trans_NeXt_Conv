"""Transformer encoder with Mixture of Experts."""

from typing import Dict, List, Tuple

from torch import nn, Tensor

from ..attention.gated_attention import MultiHeadAttention
from ..moe.mixture import MixtureOfExperts


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with MoE feed-forward."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_routed_experts: int = 32,
        num_activated_experts: int = 4,
        num_shared_expert: int = 8,
        router_rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads,
        )
        self.moe = MixtureOfExperts(
            hidden_dim=embed_dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            num_shared_expert=num_shared_expert,
            router_rank=router_rank,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        residual = x
        x = self.layer_norm_1(x)
        attn_output = self.attention(x, x, x, x)
        attn_output = self.dropout(attn_output)
        x = residual + attn_output

        residual = x
        x = self.layer_norm_2(x)
        moe_output, aux_loss_dict = self.moe(x)
        moe_output = self.dropout(moe_output)
        x = residual + moe_output
        return x, aux_loss_dict


class TransformerEncoder(nn.Module):
    """Stacked transformer encoder layers."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_routed_experts: int = 32,
        num_activated_experts: int = 4,
        num_shared_expert: int = 8,
        router_rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim, num_heads, num_routed_experts,
                num_activated_experts, num_shared_expert,
                router_rank, dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: Tensor,
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        all_aux_losses = []
        for layer in self.layers:
            x, aux_loss_dict = layer(x)
            all_aux_losses.append(aux_loss_dict)
        x = self.final_norm(x)
        return x, all_aux_losses
