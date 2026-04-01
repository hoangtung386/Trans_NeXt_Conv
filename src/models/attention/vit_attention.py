"""Vision Transformer style attention and feed-forward modules."""

import torch
from torch import nn, Tensor
from einops import rearrange

from ..nn import default


class FeedForward(nn.Module):
    """Feed-forward network with layer norm."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Standard multi-head attention for Vision Transformer."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor = None,
        kv_include_self: bool = False,
    ) -> Tensor:
        b, n, _ = x.shape
        h = self.heads
        x = self.norm(x)
        context = default(context, x)
        if kv_include_self:
            context = torch.cat((x, context), dim=1)
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv
        )
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
