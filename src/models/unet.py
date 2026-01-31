import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple

from .attention_cnn import (
    BottleNeckBlock,
    ConvNextStem,
    ConvNextEncoder,
    ConvNextDecoder
)
from .segformer import ( 
    TransformerEncoder, 
    CrossViT
)
from .gating_mechanism import (
    TransformerDecoderLayer,
    GatingMechanism
)


# MAIN MODEL
class TransNextConv(nn.Module):
    """
    TransNextConv: Hybrid CNN-Transformer Architecture with Gated-Interaction
    
    Architecture:
    - CNN Encoder-Decoder stream for local feature extraction and spatial reconstruction
    - Transformer stream (CrossViT -> Transformer Encoder -> Transformer Decoder) for global context
    - Gating Mechanism: Transformer Decoder modulates CNN Decoder through Low-rank Gates
    - Fusion: Combines CNN and Transformer outputs at the end
    
    Key interaction:
    - Transformer Decoder sends Query & Low-rank Gate to CNN Decoder
    - CNN Decoder provides Key & Value to Transformer Decoder
    - This allows Transformer to "guide" CNN during decoding
    """
    def __init__(self,
                 image_size: int = 512,
                 n_channels: int = 3,
                 n_classes: int = 1,
                 stem_features: int = 64,
                 depths: List[int] = [3, 4, 6, 2, 2, 2],
                 widths: List[int] = [256, 512, 1024],
                 drop_p: float = 0.0,
                 embed_dim: int = 1024,
                 transformer_num_heads: int = 32,
                 transformer_num_experts: int = 16,
                 transformer_activated_experts: int = 2,
                 transformer_shared_experts: int = 4,
                 gate_dim: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.image_size = image_size
        
        # Split depths into encoder and decoder
        enc_depths = depths[:3]
        dec_depths = depths[3:] if len(depths) > 3 else [2, 2, 2]

        ''' -------------------- CNN PATH (Part I & II) -------------------- '''
        # Initial convolution (Stem)
        self.in_conv = ConvNextStem(n_channels, stem_features)
        
        # Encoder path - extracts multi-scale local features
        self.enc_1 = ConvNextEncoder(stem_features, widths[0], enc_depths[0], drop_p)
        self.enc_2 = ConvNextEncoder(widths[0], widths[1], enc_depths[1], drop_p)
        self.enc_3 = ConvNextEncoder(widths[1], widths[2], enc_depths[2], drop_p)
        
        # Bottleneck - bridge between encoder and decoder
        self.bottleneck = BottleNeckBlock(widths[2], widths[2])
        
        # Decoder path - reconstructs spatial resolution with skip connections
        self.dec_1 = ConvNextDecoder(widths[2], widths[1], dec_depths[0], drop_p)
        self.dec_2 = ConvNextDecoder(widths[1], widths[0], dec_depths[1], drop_p)
        self.dec_3 = ConvNextDecoder(widths[0], stem_features, dec_depths[2], drop_p)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(stem_features, stem_features, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=1, num_channels=stem_features),
            nn.GELU(),
        )

        ''' -------------------- TRANSFORMER PATH (Part III, IV, V) -------------------- '''
        # Part III: CrossViT - multi-scale feature extraction
        self.encode_layer_transformer = CrossViT(
            image_size=image_size,
            sm_dim=embed_dim // 2,
            lg_dim=embed_dim // 2,
            sm_patch_size=8,
            lg_patch_size=16,
            channels=n_channels,
            use_projection=True,
            output_dim=embed_dim
        )
        
        # Part IV: Transformer Encoder - processes global context
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=transformer_num_heads,
            num_layers=2,
            num_routed_experts=transformer_num_experts,
            num_activated_experts=transformer_activated_experts,
            num_shared_expert=transformer_shared_experts,
            router_rank=64,
            dropout=0.1
        )
        
        # Part V: Transformer Decoder with Gating Mechanism
        # 3 decoder layers that interact with CNN decoder features
        self.decoder_layer_1 = TransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=transformer_num_heads,
            num_routed_experts=transformer_num_experts,
            num_activated_experts=transformer_activated_experts,
            num_shared_expert=transformer_shared_experts,
            router_rank=64,
            dropout=0.1,
            cnn_feature_channels=widths[1],  # 512 channels from dec_1
            use_gating=True,
            gate_dim=gate_dim
        )
        
        self.decoder_layer_2 = TransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=transformer_num_heads,
            num_routed_experts=transformer_num_experts,
            num_activated_experts=transformer_activated_experts,
            num_shared_expert=transformer_shared_experts,
            router_rank=64,
            dropout=0.1,
            cnn_feature_channels=widths[0],  # 256 channels from dec_2
            use_gating=True,
            gate_dim=gate_dim
        )
        
        self.decoder_layer_3 = TransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=transformer_num_heads,
            num_routed_experts=transformer_num_experts,
num_activated_experts=transformer_activated_experts,
            num_shared_expert=transformer_shared_experts,
            router_rank=64,
            dropout=0.1,
            cnn_feature_channels=stem_features,  # 64 channels from dec_3
            use_gating=True,
            gate_dim=gate_dim
        )

        ''' -------------------- FUSION -------------------- '''
        # Project transformer output back to spatial dimensions
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
        
    def forward(self, x: Tensor) -> Tuple[Tensor, list, dict, dict, dict, dict]:
        """
        Forward pass implementing the Gated-Interaction architecture.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            output: Final prediction [B, n_classes, H, W]
            all_aux_losses: List of auxiliary losses from transformer encoder
            aux_loss_1: Auxiliary losses from decoder layer 1
            aux_loss_2: Auxiliary losses from decoder layer 2
            aux_loss_3: Auxiliary losses from decoder layer 3
            gates: Dictionary of gates from each decoder layer
        """
        batch_size = x.shape[0]
        gates = {}
        
        ''' ==================== CNN ENCODER PATH (Part I) ==================== '''
        x1 = self.in_conv(x)      # [B, 64, H/4, W/4]
        x2 = self.enc_1(x1)       # [B, 256, H/8, W/8]
        x3 = self.enc_2(x2)       # [B, 512, H/16, W/16]
        x4 = self.enc_3(x3)       # [B, 1024, H/32, W/32]
        
        # Bottleneck
        cnn_embed = self.bottleneck(x4)  # [B, 1024, H/32, W/32]
        
        ''' ==================== CNN DECODER PATH (Part II) ==================== '''
        # Start decoding with bottleneck features
        x5 = self.dec_1(cnn_embed, x3)  # [B, 512, H/16, W/16]
        x6 = self.dec_2(x5, x2)         # [B, 256, H/8, W/8]
        x7 = self.dec_3(x6, x1)         # [B, 64, H/4, W/4]
        cnn_output = self.final_upsample(x7)  # [B, 64, H, W]
        
        ''' ==================== TRANSFORMER PATH ==================== '''
        # Part III: CrossViT - multi-scale embedding
        img_embed_transformer = self.encode_layer_transformer(x, return_concat=True)
        
        # Part IV: Transformer Encoder - global context
        trans_embed, all_aux_losses = self.transformer_encoder(img_embed_transformer)
        
        ''' ==================== GATED INTERACTION ==================== '''
        # Transformer Decoder layers interact with CNN Decoder features
        # Each layer receives CNN features and returns modulated features + gate
        
        # Layer 1: Interact with x5 (512 channels, H/16 x W/16)
        y1, aux_loss_1, gate_1 = self.decoder_layer_1(trans_embed, x5)
        gates['layer1'] = gate_1
        
        # Layer 2: Interact with x6 (256 channels, H/8 x W/8)
        y2, aux_loss_2, gate_2 = self.decoder_layer_2(y1, x6)
        gates['layer2'] = gate_2
        
        # Layer 3: Interact with x7 (64 channels, H/4 x H/4) then upsample
        y3, aux_loss_3, gate_3 = self.decoder_layer_3(y2, x7)
        gates['layer3'] = gate_3
        
        ''' ==================== FUSION ==================== '''
        # Project transformer features to spatial dimensions
        y3_projected = self.transformer_to_spatial(y3)  # [B, num_patches, 64]
        
        # Reshape to spatial dimensions using adaptive pooling
        spatial_h = spatial_w = self.spatial_size
        y3_spatial = y3_projected.permute(0, 2, 1).contiguous()  # [B, 64, num_patches]
        y3_spatial = F.adaptive_avg_pool1d(y3_spatial, spatial_h * spatial_w)
        y3_spatial = y3_spatial.view(batch_size, -1, spatial_h, spatial_w)
        
        # Concatenate CNN and Transformer features
        fused = torch.cat([cnn_output, y3_spatial], dim=1)  # [B, 128, H, W]
        fused = self.fusion(fused)  # [B, 64, H, W]
        
        # Final output
        output = self.out_conv(fused)  # [B, n_classes, H, W]
        
        return output, all_aux_losses, aux_loss_1, aux_loss_2, aux_loss_3, gates
