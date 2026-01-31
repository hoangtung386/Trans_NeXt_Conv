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
    Transformer Decoder Layer with 4-Stream Gated-Interaction.
    Strict implementation of user requirements.
    
    Inputs:
    1. Gate Input (Low-rank source) -> Reduced to gate_dim
    2. Query (Transformer Encoder Output) -> Used as Query
    3. Key Source (Final Upsample) -> Reduced to embed_dim
    4. Value Source (Decoder Block) -> Reduced to embed_dim
    
    Mechanism:
    - Value is modulated by Gate (Value * Gate)
    - Cross Attention(Q, K, Modulated_V)
    - MoE Processing
    - Generate New Gate
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 num_routed_experts: int = 32, num_activated_experts: int = 4,
                 num_shared_expert: int = 8, router_rank: int = 64,
                 dropout: float = 0.1, 
                 gate_input_dim: int = None,
                 key_input_dim: int = None,
                 value_input_dim: int = None,
                 gate_dim: int = 64):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.gate_dim = gate_dim
        
        # --- 1. Gate Reduction & Projection ---
        assert gate_input_dim is not None, "gate_input_dim required"
        # Learnable dimensionality reduction: Linear(C -> gate_dim)
        # Note: If input is spatial [B, C, H, W], we pool first then project? 
        # User said "Bottleneck @ Learnable...". 
        # We will assume pooling happens before or inside.
        self.gate_reduction = nn.Linear(gate_input_dim, gate_dim)
        
        # Gate to Embed (for modulation)
        # We need to broadcast gate [B, gate_dim] to [B, 1, embed_dim] or similar to multiply with Value
        self.gate_to_embed = nn.Linear(gate_dim, embed_dim)
        
        # --- 2. Key Reduction ---
        assert key_input_dim is not None, "key_input_dim required"
        self.key_reduction = nn.Linear(key_input_dim, embed_dim)
        
        # --- 3. Value Reduction ---
        assert value_input_dim is not None, "value_input_dim required"
        self.value_reduction = nn.Linear(value_input_dim, embed_dim)

        # --- 4. Output Gate Generation ---
        self.output_gate_gen = nn.Sequential(
            nn.Linear(embed_dim, gate_dim),
            nn.Sigmoid() # Normalize gate? Or generally just Linear/Non-linear
        )

        # Layer Norms
        self.layer_norm_q = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_k = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_v = nn.LayerNorm(normalized_shape=embed_dim)
        
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        
        # Attention
        self.self_attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads
        )
        
        self.cross_attention = MultiHeadAttention(
            embed_dim_query=embed_dim, embed_dim_key=embed_dim,
            embed_dim_value=embed_dim, embed_dim_shortcut=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads
        )
        
        # MoE
        self.MoE = MixtureOfExperts(
            hidden_dim=embed_dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            num_shared_expert=num_shared_expert,
            router_rank=router_rank
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                gate_input: Tensor,  # Source 1
                query: Tensor,       # Source 2
                key_source: Tensor,  # Source 3
                value_source: Tensor # Source 4
               ) -> tuple[Tensor, dict, Tensor]:
        """
        Forward pass with explicit 4-stream interaction.
        """
        B_q, N_q, D_q = query.shape
        
        # ================== 1. PROCESS GATE ==================
        # Flatten spatial dims if 4D tensor
        if gate_input.dim() == 4:
             # Avg Pool to get context vector per sample
             gate_pooled = F.adaptive_avg_pool2d(gate_input, (1, 1)).view(gate_input.size(0), -1)
        else:
             gate_pooled = gate_input
             
        # Low-rank reduction
        gate_lowrank = self.gate_reduction(gate_pooled) # [B, gate_dim]
        
        # Prepare modulator
        gate_embed = self.gate_to_embed(gate_lowrank)   # [B, embed_dim]
        gate_modulator = gate_embed.unsqueeze(1)        # [B, 1, embed_dim] (Broadcastable)
        
        # ================== 2. PROCESS KEY ==================
        # Assume key_source is [B, C, H, W]
        # We process key to be [B, N_spatial, embed_dim]
        B_k, C_k, H_k, W_k = key_source.shape
        key_flat = key_source.view(B_k, C_k, -1).transpose(1, 2) # [B, H*W, C_k]
        key = self.key_reduction(key_flat)                       # [B, N_key, embed_dim]
        key = self.layer_norm_k(key)
        
        # ================== 3. PROCESS VALUE ==================
        # Assume value_source is [B, C, H, W]
        # WARNING: Value length must match Key length for dot product attention? 
        # Actually, in generic attention: K [B, M, D], V [B, M, D]. 
        # Source 3 (Key) is Upsample (H, W). Source 4 (Value) is DecBlock (H/k, W/k).
        # They naturally have DIFFERENT spatial sizes.
        # To make Attention work, K and V MUST have same sequence length M.
        # Solution: We must interpolate one to match the other. 
        # High-res Key (Upsample) contains strict spatial geometry.
        # Low-res Value (DecBlock) contains semantic features.
        # We should align them. 
        # Let's resize VALUE to match KEY (Upsample resolution).
        # Upsample is [B, 64, H_in, W_in].
        
        if (value_source.shape[2] != H_k) or (value_source.shape[3] != W_k):
            value_resized = F.interpolate(value_source, size=(H_k, W_k), mode='bilinear', align_corners=False)
        else:
            value_resized = value_source
            
        value_flat = value_resized.view(B_k, value_source.shape[1], -1).transpose(1, 2) # [B, N_key, C_v]
        value = self.value_reduction(value_flat) # [B, N_key, embed_dim]
        
        # ================== 4. MODULATION ==================
        # Value * Gate
        value_modulated = value * gate_modulator # [B, N_key, embed_dim]
        value_modulated = self.layer_norm_v(value_modulated)
        
        # ================== 5. SELF ATTENTION (Query) ==================
        residual = query
        query = self.layer_norm_q(query)
        attn_out = self.self_attention(query, query, query, query)
        query = residual + self.dropout(attn_out)
        query = self.layer_norm_1(query)
        
        # ================== 6. CROSS ATTENTION ==================
        residual = query
        # Query, Key, Value_Modulated
        cross_out = self.cross_attention(query, key, value_modulated, residual)
        query = residual + self.dropout(cross_out)
        query = self.layer_norm_2(query)
        
        # ================== 7. MOE ==================
        residual = query
        moe_out, aux_loss = self.MoE(query)
        query = residual + self.dropout(moe_out)
        
        # ================== 8. NEW GATE GENERATION ==================
        # Average pool query features to get global gate signal
        query_pooled = query.mean(dim=1) # [B, embed_dim]
        output_gate = self.output_gate_gen(query_pooled) # [B, gate_dim]
        
        return query, aux_loss, output_gate
