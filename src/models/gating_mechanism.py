"""
Gating Mechanisms for CNN-Transformer Interaction
New implementation for proper Gated-Interaction architecture
"""

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# Import attention mechanisms and MoE from segformer
from .segformer import MultiHeadAttention, MixtureOfExperts


class GatingMechanism(nn.Module):
    """
    Low-rank Gating Mechanism for CNN-Transformer interaction.
    Transformer Decoder sends Gate to modulate CNN features.
    """
    def __init__(self, embed_dim: int, gate_dim: int = 64):
        super().__init__()
        self.gate_dim = gate_dim
        
        # Low-rank projection for Gate
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim, gate_dim, bias=False),
            nn.Sigmoid()
        )
        
        # Feature modulation
        self.feature_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, trans_features: Tensor, cnn_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            trans_features: Features from Transformer Decoder [B, N, D]
            cnn_features: Features from CNN Decoder [B, C, H, W]
        Returns:
            gate: Low-rank gate for CNN modulation [B, gate_dim]
            query: Query for next attention [B, N, D]
            modulated_cnn: CNN features modulated by gate [B, C, H, W]
        """
        # Generate low-rank gate from transformer features
        gate = self.gate_proj(trans_features.mean(dim=1))  # [B, gate_dim]
        
        # Query for attention
        query = self.feature_proj(trans_features)  # [B, N, D]
        
        # Modulate CNN features (apply gate in a learnable way)
        B, C, H, W = cnn_features.shape
        gate_expanded = gate.view(B, 1, 1, 1).expand(-1, C, H, W)
        modulated_cnn = cnn_features * (1 + gate_expanded)  # Multiplicative modulation
        
        return gate, query, modulated_cnn


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with Gated-Interaction for CNN-Transformer architecture.
    
    Key features:
    - Self-attention on transformer features
    - Cross-attention with CNN features (as Key/Value)
    - Low-rank gating mechanism to modulate CNN features
    - Mixture of Experts for enhanced representation
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 num_routed_experts: int = 32, num_activated_experts: int = 4,
                 num_shared_expert: int = 8, router_rank: int = 64,
                 dropout: float = 0.1, cnn_feature_channels: int = None,
                 use_gating: bool = True, gate_dim: int = 64):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_gating = use_gating
        
        # Layer normalization
        self.layer_norm_q = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_k = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_v = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        
        # Gating mechanism
        if use_gating:
            assert cnn_feature_channels is not None, "cnn_feature_channels required for gating"
            self.gating = GatingMechanism(embed_dim, gate_dim)
            self.cnn_proj = nn.Linear(cnn_feature_channels, embed_dim)
        
        # Multi-head self-attention (for processing transformer features)
        self.self_attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads
        )
        
        # Cross-attention (Transformer Query with CNN Key/Value)
        self.cross_attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads
        )
        
        # Mixture of Experts
        self.MoE = MixtureOfExperts(
            hidden_dim=embed_dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            num_shared_expert=num_shared_expert,
            router_rank=router_rank
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trans_query: Tensor, cnn_features: Tensor) -> tuple[Tensor, dict, Tensor]:
        """
        Args:
            trans_query: Query from previous Transformer layer [B, N, D]
            cnn_features: Features from CNN Decoder [B, C, H, W]
        Returns:
            output: Updated transformer features [B, N, D]
            aux_loss_dict: Auxiliary losses from MoE
            gate: Low-rank gate for CNN modulation
        """
        B, C, H, W = cnn_features.shape
        
        # Project CNN features to token format
        cnn_tokens = cnn_features.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        cnn_tokens = self.cnn_proj(cnn_tokens)  # [B, H*W, D]
        
        # Self-attention on transformer features
        residual = trans_query
        trans_query = self.layer_norm_q(trans_query)
        attn_output = self.self_attention(trans_query, trans_query, trans_query, trans_query)
        attn_output = self.dropout(attn_output)
        trans_query = residual + attn_output
        trans_query = self.layer_norm_1(trans_query)
        
        # Cross-attention: Transformer Query with CNN Key/Value
        residual = trans_query
        # CNN features serve as Key andValue
        key = self.layer_norm_k(cnn_tokens)
        value = self.layer_norm_v(cnn_tokens)
        
        cross_attn_output = self.cross_attention(
            query=trans_query,
            key=key,
            value=value,
            hidden_shortcut=residual  # Use residual as shortcut
        )
        cross_attn_output = self.dropout(cross_attn_output)
        trans_query = residual + cross_attn_output
        trans_query = self.layer_norm_2(trans_query)
        
        # Mixture of Experts
        residual = trans_query
        moe_output, aux_loss_dict = self.MoE(trans_query)
        moe_output = self.dropout(moe_output)
        trans_query = residual + moe_output
        
        # Gating mechanism
        if self.use_gating:
            gate, trans_output, modulated_cnn = self.gating(trans_query, cnn_features)
            return trans_output, aux_loss_dict, gate
        else:
            return trans_query, aux_loss_dict, None
