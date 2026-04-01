"""Utility layers and helper functions for neural networks."""

import torch
from torch import nn, Tensor


def exists(val):
    """Check if value is not None."""
    return val is not None


def default(val, d):
    """Return val if it exists, otherwise return d."""
    return val if exists(val) else d


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.scale


class LayerScaler(nn.Module):
    """Learnable per-channel scaling (used in ConvNeXt)."""

    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(
            init_value * torch.ones(dimensions), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.gamma[None, ..., None, None] * x
