"""Scaled dot-product attention implementation."""

import torch
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor = None,
) -> Tensor:
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape [B, N, D].
        key: Key tensor of shape [B, M, D].
        value: Value tensor of shape [B, M, D].
        mask: Optional attention mask.

    Returns:
        Attention output of shape [B, N, D].
    """
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(
        torch.tensor(dim_k, dtype=torch.float32)
    )
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)
