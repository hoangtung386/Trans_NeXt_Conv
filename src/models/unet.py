import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import List

from .attention_cnn import (
    BottleNeckBlock,
    ConvNextStem,
    ConvNextEncoder,
    ConvNextDecoder
)
from .segformer import ( 
    TransformerEncoder, 
    CrossViT, 
    TransformerDecoderLayer
)


# MAIN MODEL
class TransNextConv(nn.Module):
    def __init__(self,
                 image_size: int = 512,
                 n_channels: int = 3,
                 n_classes: int = 1,
                 stem_features: int = 64,
                 depths: List[int] = [3, 4, 6, 2, 2, 2],
                 widths: List[int] = [256, 512, 1024],
                 drop_p: float = 0.0,
                 embed_dim: int = 1024):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        # Split depths into encoder and decoder
        enc_depths = depths[:3]
        dec_depths = depths[3:] if len(depths) > 3 else [2, 2, 2]

        ''' -------------------- CNN PATH -------------------- '''
        # Initial convolution (Stem)
        self.in_conv = ConvNextStem(n_channels, stem_features)

        # Encoder path
        self.enc_1 = ConvNextEncoder(stem_features, widths[0], enc_depths[0], drop_p)
        self.enc_2 = ConvNextEncoder(widths[0], widths[1], enc_depths[1], drop_p)
        self.enc_3 = ConvNextEncoder(widths[1], widths[2], enc_depths[2], drop_p)

        # Bottleneck
        self.bottleneck = BottleNeckBlock(widths[2], widths[2])

        # Decoder path
        self.dec_1 = ConvNextDecoder(widths[2], widths[1], dec_depths[0], drop_p)
        self.dec_2 = ConvNextDecoder(widths[1], widths[0], dec_depths[1], drop_p)
        self.dec_3 = ConvNextDecoder(widths[0], stem_features, dec_depths[2], drop_p)

        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(stem_features, stem_features, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=1, num_channels=stem_features),
            nn.GELU(),
        )

        ''' -------------------- TRANSFORMER PATH -------------------- '''
        # Feature extraction from original image
        self.encode_layer_transformer = CrossViT(
            image_size=image_size,
            sm_dim=embed_dim//2,
            lg_dim=embed_dim//2,
            sm_patch_size=8,
            lg_patch_size=16,
            channels=n_channels,
            use_projection=True,
            output_dim=embed_dim
        )

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(embed_dim=embed_dim)

        # Transformer Decoder Layers
        # Calculate feature map size after enc_3: image_size / 4 / 2 / 2 / 2 = image_size / 32
        encoder_feature_size = image_size // 32

        self.decoder_layer_1 = TransformerDecoderLayer(
            image_size=image_size, channels=n_channels,
            embed_dim=embed_dim, num_heads=32,
            num_routed_experts=16, num_activated_experts=2,
            num_shared_expert=4, router_rank=64, dropout=0.1,
            first_layer=True,
            encoder_feature_channels=widths[2],  # 1024
            encoder_feature_size=encoder_feature_size,  # H/32
            decoder_feature_channels=widths[1],  # 512 from dec_1 output
            decoder_feature_size=image_size // 16  # H/16
        )

        self.decoder_layer_2 = TransformerDecoderLayer(
            image_size=image_size, channels=n_channels,
            embed_dim=embed_dim, num_heads=32,
            num_routed_experts=16, num_activated_experts=2,
            num_shared_expert=4, router_rank=64, dropout=0.1,
            first_layer=False,
            decoder_feature_channels=widths[0],  # 256 from dec_2 output
            decoder_feature_size=image_size // 8  # H/8
        )

        self.decoder_layer_3 = TransformerDecoderLayer(
            image_size=image_size, channels=n_channels,
            embed_dim=embed_dim, num_heads=32,
            num_routed_experts=16, num_activated_experts=2,
            num_shared_expert=4, router_rank=64, dropout=0.1,
            first_layer=False,
            decoder_feature_channels=stem_features,  # 64 from dec_3 output
            decoder_feature_size=image_size // 4  # H/4
        )

        ''' -------------------- FUSION -------------------- '''
        # Project transformer output back to spatial dimensions
        num_patches = (image_size // 8) ** 2 + (image_size // 16) ** 2
        self.spatial_size = image_size

        # Adaptive projection to match CNN output size
        self.transformer_to_spatial = nn.Sequential(
            nn.Linear(embed_dim, stem_features * 4),
            nn.GELU(),
            nn.Linear(stem_features * 4, stem_features)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(stem_features * 2, stem_features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=stem_features),
            nn.GELU(),
            BottleNeckBlock(stem_features, stem_features)
        )

        # Output convolution
        self.out_conv = nn.Conv2d(stem_features, n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, list, dict, dict, dict]:
        batch_size = x.shape[0]

        ''' -------------------- TRANSFORMER ENCODER PATH -------------------- '''
        img_embed_transformer = self.encode_layer_transformer(x, return_concat=True)
        img_embed_transformer, all_aux_losses = self.transformer_encoder(img_embed_transformer)

        ''' -------------------- CNN ENCODER PATH -------------------- '''
        x1 = self.in_conv(x)      # [B, 64, H/4, W/4]
        x2 = self.enc_1(x1)       # [B, 256, H/8, W/8]
        x3 = self.enc_2(x2)       # [B, 512, H/16, W/16]
        x4 = self.enc_3(x3)       # [B, 1024, H/32, W/32]

        # Bottleneck
        img_embed_cnn = self.bottleneck(x4)  # [B, 1024, H/32, W/32]

        ''' -------------------- CNN DECODER PATH -------------------- '''
        ''' -------------------- CNN DECODER PATH -------------------- '''
        x5 = self.dec_1(img_embed_cnn, x3)  # [B, 512, H/16, W/16]
        x6 = self.dec_2(x5, x2)              # [B, 256, H/8, W/8]
        x7 = self.dec_3(x6, x1)              # [B, 64, H/4, W/4]
        x_cnn = self.final_upsample(x7)      # [B, 64, H, W]

        # Verify dimensions
        if x5.shape[1] != 512: print(f"Warning: x5 channels expected 512, got {x5.shape[1]}")
        if x6.shape[1] != 256: print(f"Warning: x6 channels expected 256, got {x6.shape[1]}")
        if x7.shape[1] != 64: print(f"Warning: x7 channels expected 64, got {x7.shape[1]}")

        ''' -------------------- TRANSFORMER DECODER PATH -------------------- '''
        # Layer 1: query=transformer_embed, key=original_image, value=x5, shortcut=cnn_embed
        y1, aux_loss_1 = self.decoder_layer_1(x, img_embed_transformer, x5, img_embed_cnn)

        # Layer 2: query=y1, key=original_image, value=x6, shortcut=y1
        y2, aux_loss_2 = self.decoder_layer_2(x, y1, x6, y1)

        # Layer 3: query=y2, key=original_image, value=x7, shortcut=y2
        y3, aux_loss_3 = self.decoder_layer_3(x, y2, x7, y2)
            
        ''' -------------------- FUSION -------------------- '''
        # Project transformer tokens to spatial
        y3_projected = self.transformer_to_spatial(y3)  # [B, num_patches, 64]

        # Reshape to spatial dimensions using adaptive pooling
        # Calculate target spatial size based on patches
        spatial_h = spatial_w = self.spatial_size
        num_tokens = y3_projected.shape[1]

        # Simpler version using adaptive pooling (User suggested)
        y3_spatial = y3_projected.permute(0, 2, 1).contiguous()  # [B, 64, num_patches]
        y3_spatial = F.adaptive_avg_pool1d(y3_spatial, spatial_h * spatial_w)
        y3_spatial = y3_spatial.view(batch_size, -1, spatial_h, spatial_w)

        # Concatenate CNN and Transformer features
        fused = torch.cat([x_cnn, y3_spatial], dim=1)  # [B, 128, H, W]
        fused = self.fusion(fused)  # [B, 64, H, W]

        # Final output
        output = self.out_conv(fused)  # [B, n_classes, H, W]

        return output, all_aux_losses, aux_loss_1, aux_loss_2, aux_loss_3
        
