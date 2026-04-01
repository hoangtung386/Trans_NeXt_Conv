"""Transformer and multi-scale encoder for CrossViT."""

from torch import nn, Tensor

from ..attention.vit_attention import Attention, FeedForward
from .cross_transformer import CrossTransformer


class Transformer(nn.Module):
    """Standard transformer encoder block."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim, heads=heads, dim_head=dim_head, dropout=dropout,
                ),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MultiScaleEncoder(nn.Module):
    """Multi-scale encoder with cross-attention between scales."""

    def __init__(
        self,
        *,
        depth: int,
        sm_dim: int,
        lg_dim: int,
        sm_enc_params: dict,
        lg_enc_params: dict,
        cross_attn_heads: int,
        cross_attn_depth: int,
        cross_attn_dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                CrossTransformer(
                    sm_dim=sm_dim, lg_dim=lg_dim,
                    depth=cross_attn_depth, heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head, dropout=dropout,
                ),
            ]))

    def forward(self, sm_tokens: Tensor, lg_tokens: Tensor) -> tuple:
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)
        return sm_tokens, lg_tokens
