"""TransNextConv: Hybrid CNN-Transformer architecture for segmentation."""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .convnext import (
    BottleNeckBlock,
    ConvNextStem,
    ConvNextEncoder,
    ConvNextDecoder,
)
from .transformer.encoder import TransformerEncoder
from .crossvit import CrossViT
from .transformer.decoder import TransformerDecoderLayer


class TransNextConv(nn.Module):
    """TransNextConv: Hybrid CNN-Transformer with gated interaction.

    Architecture:
        - CNN Encoder-Decoder stream for local feature extraction
        - Transformer stream (CrossViT -> Encoder -> Decoder) for global context
        - Gating mechanism: transformer decoder modulates CNN decoder
        - Fusion: combines CNN and transformer outputs
    """

    def __init__(
        self,
        image_size: int = 512,
        n_channels: int = 3,
        n_classes: int = 1,
        stem_features: int = 64,
        depths: List[int] = None,
        widths: List[int] = None,
        drop_p: float = 0.0,
        embed_dim: int = 1024,
        transformer_num_heads: int = 32,
        transformer_num_experts: int = 16,
        transformer_activated_experts: int = 2,
        transformer_shared_experts: int = 4,
        gate_dim: int = 64,
    ):
        super().__init__()
        if depths is None:
            depths = [3, 4, 6, 2, 2, 2]
        if widths is None:
            widths = [256, 512, 1024]

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.image_size = image_size

        enc_depths = depths[:3]
        dec_depths = depths[3:] if len(depths) > 3 else [2, 2, 2]

        # ---- CNN Path ----
        self.in_conv = ConvNextStem(n_channels, stem_features)

        self.enc_1 = ConvNextEncoder(
            stem_features, widths[0], enc_depths[0], drop_p,
        )
        self.enc_2 = ConvNextEncoder(
            widths[0], widths[1], enc_depths[1], drop_p,
        )
        self.enc_3 = ConvNextEncoder(
            widths[1], widths[2], enc_depths[2], drop_p,
        )

        self.bottleneck = BottleNeckBlock(widths[2], widths[2])

        self.dec_1 = ConvNextDecoder(
            widths[2], widths[1], dec_depths[0], drop_p,
        )
        self.dec_2 = ConvNextDecoder(
            widths[1], widths[0], dec_depths[1], drop_p,
        )
        self.dec_3 = ConvNextDecoder(
            widths[0], stem_features, dec_depths[2], drop_p,
        )

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                stem_features, stem_features, kernel_size=4, stride=4,
            ),
            nn.GroupNorm(num_groups=1, num_channels=stem_features),
            nn.GELU(),
        )

        # ---- Transformer Path ----
        self.encode_layer_transformer = CrossViT(
            image_size=image_size,
            sm_dim=embed_dim // 2,
            lg_dim=embed_dim // 2,
            sm_patch_size=8,
            lg_patch_size=16,
            channels=n_channels,
            use_projection=True,
            output_dim=embed_dim,
        )

        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=transformer_num_heads,
            num_layers=2,
            num_routed_experts=transformer_num_experts,
            num_activated_experts=transformer_activated_experts,
            num_shared_expert=transformer_shared_experts,
            router_rank=64,
            dropout=0.1,
        )

        # Transformer decoder layers (4-stream gated interaction)
        self.decoder_layer_1 = TransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=transformer_num_heads,
            num_routed_experts=transformer_num_experts,
            num_activated_experts=transformer_activated_experts,
            num_shared_expert=transformer_shared_experts,
            router_rank=64,
            dropout=0.1,
            gate_input_dim=widths[2],
            key_input_dim=stem_features,
            value_input_dim=widths[1],
            gate_dim=gate_dim,
        )

        self.decoder_layer_2 = TransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=transformer_num_heads,
            num_routed_experts=transformer_num_experts,
            num_activated_experts=transformer_activated_experts,
            num_shared_expert=transformer_shared_experts,
            router_rank=64,
            dropout=0.1,
            gate_input_dim=gate_dim,
            key_input_dim=stem_features,
            value_input_dim=widths[0],
            gate_dim=gate_dim,
        )

        self.decoder_layer_3 = TransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=transformer_num_heads,
            num_routed_experts=transformer_num_experts,
            num_activated_experts=transformer_activated_experts,
            num_shared_expert=transformer_shared_experts,
            router_rank=64,
            dropout=0.1,
            gate_input_dim=gate_dim,
            key_input_dim=stem_features,
            value_input_dim=stem_features,
            gate_dim=gate_dim,
        )

        # ---- Fusion ----
        self.spatial_size = image_size

        self.transformer_to_spatial = nn.Sequential(
            nn.Linear(embed_dim, stem_features * 4),
            nn.GELU(),
            nn.Linear(stem_features * 4, stem_features),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                stem_features * 2, stem_features, kernel_size=3, padding=1,
            ),
            nn.GroupNorm(num_groups=1, num_channels=stem_features),
            nn.GELU(),
            BottleNeckBlock(stem_features, stem_features),
        )

        self.out_conv = nn.Conv2d(stem_features, n_classes, kernel_size=1)

    def forward(
        self, x: Tensor,
    ) -> Tuple[Tensor, List, Dict, Dict, Dict, Dict]:
        """Forward pass.

        Returns:
            Tuple of (output, all_aux_losses, aux_loss_1, aux_loss_2,
                       aux_loss_3, gates).
        """
        batch_size = x.shape[0]
        gates = {}

        # 1. CNN encoder
        x1 = self.in_conv(x)
        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        cnn_embed = self.bottleneck(x4)

        # 2. CNN decoder
        x5 = self.dec_1(cnn_embed, x3)
        x6 = self.dec_2(x5, x2)
        x7 = self.dec_3(x6, x1)
        cnn_output = self.final_upsample(x7)

        # 3. Transformer encoder
        img_embed_transformer = self.encode_layer_transformer(
            x, return_concat=True,
        )
        trans_embed, all_aux_losses = self.transformer_encoder(
            img_embed_transformer,
        )

        # 4. Transformer decoder (gated interaction)
        y1, aux_loss_1, gate_1 = self.decoder_layer_1(
            gate_input=cnn_embed,
            query=trans_embed,
            key_source=cnn_output,
            value_source=x5,
        )
        gates["layer1"] = gate_1

        y2, aux_loss_2, gate_2 = self.decoder_layer_2(
            gate_input=gate_1,
            query=y1,
            key_source=cnn_output,
            value_source=x6,
        )
        gates["layer2"] = gate_2

        y3, aux_loss_3, gate_3 = self.decoder_layer_3(
            gate_input=gate_2,
            query=y2,
            key_source=cnn_output,
            value_source=x7,
        )
        gates["layer3"] = gate_3

        # 5. Fusion
        y3_projected = self.transformer_to_spatial(y3)
        spatial_h = spatial_w = self.spatial_size
        y3_spatial = y3_projected.permute(0, 2, 1).contiguous()
        y3_spatial = F.adaptive_avg_pool1d(
            y3_spatial, spatial_h * spatial_w,
        )
        y3_spatial = y3_spatial.view(batch_size, -1, spatial_h, spatial_w)

        fused = torch.cat([cnn_output, y3_spatial], dim=1)
        fused = self.fusion(fused)

        output = self.out_conv(fused)

        return output, all_aux_losses, aux_loss_1, aux_loss_2, aux_loss_3, gates
