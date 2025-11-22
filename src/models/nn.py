"""
Various utilities for neural networks.
"""

import torch
from torch import nn
from torch import Tensor


# HELPER FUNCTIONS
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# NORMALIZATION
class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.scale

# CONVNEXT COMPONENTS
class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), requires_grad=True)

    def forward(self, x):
        return self.gamma[None, ..., None, None] * x
