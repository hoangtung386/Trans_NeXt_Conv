"""CrossViT: Multi-scale vision transformer."""

import torch
from torch import nn, Tensor

from .embedder import ImageEmbedder
from .encoder import MultiScaleEncoder


class CrossViT(nn.Module):
    """Cross-attention Vision Transformer for multi-scale feature extraction.

    Processes image at two patch scales (small and large) with
    cross-attention between the two branches.
    """

    def __init__(
        self,
        *,
        image_size: int,
        sm_dim: int,
        lg_dim: int,
        sm_patch_size: int = 12,
        sm_enc_depth: int = 1,
        sm_enc_heads: int = 8,
        sm_enc_mlp_dim: int = 1024,
        sm_enc_dim_head: int = 64,
        lg_patch_size: int = 16,
        lg_enc_depth: int = 4,
        lg_enc_heads: int = 8,
        lg_enc_mlp_dim: int = 1024,
        lg_enc_dim_head: int = 64,
        cross_attn_depth: int = 2,
        cross_attn_heads: int = 8,
        cross_attn_dim_head: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        channels: int = 3,
        use_projection: bool = False,
        output_dim: int = None,
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(
            dim=sm_dim, channels=channels, image_size=image_size,
            patch_size=sm_patch_size, dropout=emb_dropout,
        )
        self.lg_image_embedder = ImageEmbedder(
            dim=lg_dim, channels=channels, image_size=image_size,
            patch_size=lg_patch_size, dropout=emb_dropout,
        )
        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth, sm_dim=sm_dim, lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth, heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim, dim_head=sm_enc_dim_head,
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth, heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim, dim_head=lg_enc_dim_head,
            ),
            dropout=dropout,
        )
        self.use_projection = use_projection
        if use_projection:
            assert output_dim is not None, (
                "output_dim must be specified when use_projection=True"
            )
            self.sm_projection = nn.Linear(sm_dim, output_dim)
            self.lg_projection = nn.Linear(lg_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.output_dim = None

    def forward(
        self, img: Tensor, return_concat: bool = False,
    ) -> Tensor:
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)
        sm_patch_tokens = sm_tokens[:, 1:, :]
        lg_patch_tokens = lg_tokens[:, 1:, :]

        if return_concat:
            if self.use_projection:
                sm_projected = self.sm_projection(sm_patch_tokens)
                lg_projected = self.lg_projection(lg_patch_tokens)
                return torch.cat([sm_projected, lg_projected], dim=1)
            else:
                raise ValueError(
                    "Cannot concat tokens with different dimensions."
                )
        return sm_patch_tokens, lg_patch_tokens
