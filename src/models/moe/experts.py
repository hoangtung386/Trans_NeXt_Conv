"""Expert network modules for Mixture of Experts."""

import torch.nn.functional as F
from torch import nn, Tensor

from ..nn import RMSNorm


class Expert(nn.Module):
    """Standard SwiGLU expert."""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.w1(x))
        val = self.w3(x)
        x = self.w2(gate * val)
        return x


class AdvancedExpert(nn.Module):
    """SwiGLU expert with RMSNorm and residual connection."""

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        gate = F.silu(self.w1(x))
        val = self.w3(x)
        out = self.w2(gate * val)
        return out + residual
