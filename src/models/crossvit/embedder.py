"""Image patch embedding for CrossViT."""

import torch
from torch import nn, Tensor
from einops import repeat
from einops.layers.torch import Rearrange


class ImageEmbedder(nn.Module):
    """Embed image patches with positional encoding and CLS token."""

    def __init__(
        self,
        *,
        dim: int,
        image_size: int,
        patch_size: int,
        dropout: float = 0.0,
        channels: int = 3,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            "Image size must be divisible by patch size"
        )
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img: Tensor) -> Tensor:
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return self.dropout(x)
