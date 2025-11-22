import torch
from torch import nn
from torch import Tensor
from torchvision.ops import StochasticDepth
from .nn import LayerScaler

class BottleNeckBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 expansion: int = 4, drop_p: float = 0.0,
                 layer_scaler_init_value: float = 1e-6):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features),
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x

class ConvNextStem(nn.Sequential):
    """Initial downsampling stem"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_features)
        )

class ConvNextEncoder(nn.Module):
    """Encoder block: Downsample + BottleNeck blocks"""
    def __init__(self, in_features: int, out_features: int, depth: int, drop_p: float = 0.0):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
        )
        self.blocks = nn.Sequential(
            *[BottleNeckBlock(out_features, out_features, drop_p=drop_p) for _ in range(depth)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x

class ConvNextDecoder(nn.Module):
    """Decoder block: Upsample + Skip Connection + BottleNeck blocks"""
    def __init__(self, in_features: int, out_features: int, depth: int = 2, drop_p: float = 0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_features, out_features, kernel_size=2, stride=2)
        self.fusion = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=out_features * 2),
            nn.Conv2d(out_features * 2, out_features, kernel_size=1),
        )
        self.blocks = nn.Sequential(
            *[BottleNeckBlock(out_features, out_features, drop_p=drop_p) for _ in range(depth)]
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        x = self.blocks(x)
        return x
