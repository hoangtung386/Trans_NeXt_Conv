"""Cross-attention transformer for multi-scale feature fusion."""

import torch
from torch import nn, Tensor

from ..attention.vit_attention import Attention


class ProjectInOut(nn.Module):
    """Project input/output dimensions around a function."""

    def __init__(self, dim_in: int, dim_out: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        need_projection = dim_in != dim_out
        self.project_in = (
            nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


class CrossTransformer(nn.Module):
    """Cross-attention between small and large scale tokens."""

    def __init__(
        self,
        sm_dim: int,
        lg_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(
                    sm_dim, lg_dim,
                    Attention(
                        lg_dim, heads=heads,
                        dim_head=dim_head, dropout=dropout,
                    ),
                ),
                ProjectInOut(
                    lg_dim, sm_dim,
                    Attention(
                        sm_dim, heads=heads,
                        dim_head=dim_head, dropout=dropout,
                    ),
                ),
            ]))

    def forward(
        self, sm_tokens: Tensor, lg_tokens: Tensor,
    ) -> tuple:
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(
            lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens)
        )
        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = (
                sm_attend_lg(
                    sm_cls, context=lg_patch_tokens, kv_include_self=True
                ) + sm_cls
            )
            lg_cls = (
                lg_attend_sm(
                    lg_cls, context=sm_patch_tokens, kv_include_self=True
                ) + lg_cls
            )
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens
