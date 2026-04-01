"""Gated attention and multi-head attention modules."""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..nn import RMSNorm
from .scaled_attention import scaled_dot_product_attention


class AttentionGated(nn.Module):
    """Single-head gated attention with alpha/beta modulation."""

    def __init__(
        self,
        embed_dim_query: int,
        embed_dim_key: int,
        embed_dim_value: int,
        embed_dim_shortcut: int,
        head_dim: int,
    ):
        super().__init__()
        self.norm = RMSNorm(head_dim)
        self.q = nn.Sequential(
            nn.Linear(embed_dim_query, head_dim),
            nn.SiLU(),
        )
        self.k = nn.Sequential(
            nn.Linear(embed_dim_key, head_dim),
            nn.SiLU(),
        )
        self.v = nn.Sequential(
            nn.Linear(embed_dim_value, head_dim),
            nn.SiLU(),
        )
        self.alpha = nn.Sequential(
            nn.Linear(embed_dim_value, head_dim // 2),
            nn.Linear(head_dim // 2, head_dim),
            nn.Sigmoid(),
        )
        self.beta = nn.Sequential(
            nn.Linear(embed_dim_value, head_dim),
            nn.Sigmoid(),
        )
        self.shortcut = nn.Sequential(
            nn.Linear(embed_dim_shortcut, head_dim // 2),
            nn.Linear(head_dim // 2, head_dim),
            nn.Sigmoid(),
        )
        self.linear1 = nn.Linear(head_dim, head_dim)

    def forward(
        self,
        hidden_query: Tensor,
        hidden_key: Tensor,
        hidden_value: Tensor,
        hidden_shortcut: Tensor,
    ) -> Tensor:
        query = F.normalize(self.q(hidden_query), p=2, dim=-1)
        key = F.normalize(self.k(hidden_key), p=2, dim=-1)
        value = self.v(hidden_value)

        alpha = self.alpha(hidden_value)
        beta = self.beta(hidden_value)

        shortcut = self.shortcut(hidden_shortcut)
        if shortcut.shape[1] != query.shape[1]:
            shortcut = shortcut.transpose(1, 2)
            shortcut = F.interpolate(
                shortcut, size=query.shape[1], mode="linear", align_corners=False
            )
            shortcut = shortcut.transpose(1, 2)

        value = value * alpha + beta

        attn_outputs = scaled_dot_product_attention(
            query=query, key=key, value=value
        )
        attn_outputs = self.norm(attn_outputs) * shortcut
        attn_outputs = self.linear1(attn_outputs)
        return attn_outputs


class MultiHeadAttention(nn.Module):
    """Multi-head gated attention."""

    def __init__(
        self,
        embed_dim_query: int,
        embed_dim_key: int,
        embed_dim_value: int,
        embed_dim_shortcut: int,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.heads = nn.ModuleList([
            AttentionGated(
                embed_dim_query, embed_dim_key, embed_dim_value,
                embed_dim_shortcut, head_dim,
            )
            for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_query: Tensor,
        hidden_key: Tensor,
        hidden_value: Tensor,
        hidden_shortcut: Tensor,
    ) -> Tensor:
        x = torch.cat(
            [h(hidden_query, hidden_key, hidden_value, hidden_shortcut)
             for h in self.heads],
            dim=-1,
        )
        x = self.output_linear(x)
        return x
