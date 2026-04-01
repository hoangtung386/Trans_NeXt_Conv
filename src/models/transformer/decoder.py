"""Transformer decoder layer with 4-stream gated interaction."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..attention.gated_attention import MultiHeadAttention
from ..moe.mixture import MixtureOfExperts


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with 4-stream gated interaction.

    Inputs:
        1. Gate Input (low-rank source) -> reduced to gate_dim
        2. Query (transformer encoder output)
        3. Key Source (final upsample) -> reduced to embed_dim
        4. Value Source (decoder block) -> reduced to embed_dim

    Mechanism:
        - Value is modulated by Gate (Value * Gate)
        - Cross Attention(Q, K, Modulated_V)
        - MoE Processing
        - Generate new gate
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_routed_experts: int = 32,
        num_activated_experts: int = 4,
        num_shared_expert: int = 8,
        router_rank: int = 64,
        dropout: float = 0.1,
        gate_input_dim: int = None,
        key_input_dim: int = None,
        value_input_dim: int = None,
        gate_dim: int = 64,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.gate_dim = gate_dim

        # Gate reduction & projection
        assert gate_input_dim is not None, "gate_input_dim required"
        self.gate_reduction = nn.Linear(gate_input_dim, gate_dim)
        self.gate_to_embed = nn.Linear(gate_dim, embed_dim)

        # Key reduction
        assert key_input_dim is not None, "key_input_dim required"
        self.key_reduction = nn.Linear(key_input_dim, embed_dim)

        # Value reduction
        assert value_input_dim is not None, "value_input_dim required"
        self.value_reduction = nn.Linear(value_input_dim, embed_dim)

        # Output gate generation
        self.output_gate_gen = nn.Sequential(
            nn.Linear(embed_dim, gate_dim),
            nn.Sigmoid(),
        )

        # Layer norms
        self.layer_norm_q = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_k = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_v = nn.LayerNorm(normalized_shape=embed_dim)

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)

        # Attention
        self.self_attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads,
        )
        self.cross_attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads,
        )

        # MoE
        self.moe = MixtureOfExperts(
            hidden_dim=embed_dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            num_shared_expert=num_shared_expert,
            router_rank=router_rank,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gate_input: Tensor,
        query: Tensor,
        key_source: Tensor,
        value_source: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """Forward pass with explicit 4-stream interaction.

        Args:
            gate_input: Gate source (bottleneck or previous gate).
            query: Transformer encoder output [B, N, D].
            key_source: CNN final upsample [B, C, H, W].
            value_source: CNN decoder block [B, C, H, W].

        Returns:
            Tuple of (output, aux_loss_dict, output_gate).
        """
        # 1. Process gate
        if gate_input.dim() == 4:
            gate_pooled = F.adaptive_avg_pool2d(
                gate_input, (1, 1)
            ).view(gate_input.size(0), -1)
        else:
            gate_pooled = gate_input

        gate_lowrank = self.gate_reduction(gate_pooled)
        gate_embed = self.gate_to_embed(gate_lowrank)
        gate_modulator = gate_embed.unsqueeze(1)

        # 2. Process key
        b_k, c_k, h_k, w_k = key_source.shape
        key_flat = key_source.view(b_k, c_k, -1).transpose(1, 2)
        key = self.key_reduction(key_flat)
        key = self.layer_norm_k(key)

        # 3. Process value (resize to match key spatial dims)
        if (value_source.shape[2] != h_k) or (value_source.shape[3] != w_k):
            value_resized = F.interpolate(
                value_source, size=(h_k, w_k),
                mode="bilinear", align_corners=False,
            )
        else:
            value_resized = value_source

        value_flat = value_resized.view(
            b_k, value_source.shape[1], -1
        ).transpose(1, 2)
        value = self.value_reduction(value_flat)

        # 4. Modulation: Value * Gate
        value_modulated = value * gate_modulator
        value_modulated = self.layer_norm_v(value_modulated)

        # 5. Self attention on query
        residual = query
        query = self.layer_norm_q(query)
        attn_out = self.self_attention(query, query, query, query)
        query = residual + self.dropout(attn_out)
        query = self.layer_norm_1(query)

        # 6. Cross attention
        residual = query
        cross_out = self.cross_attention(
            query, key, value_modulated, residual
        )
        query = residual + self.dropout(cross_out)
        query = self.layer_norm_2(query)

        # 7. MoE
        residual = query
        moe_out, aux_loss = self.moe(query)
        query = residual + self.dropout(moe_out)

        # 8. New gate generation
        query_pooled = query.mean(dim=1)
        output_gate = self.output_gate_gen(query_pooled)

        return query, aux_loss, output_gate
