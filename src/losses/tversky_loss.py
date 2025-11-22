import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.losses.utils import one_hot_encode

class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    
    Controls trade-off between FP and FN
    α = β = 0.5 → Dice Loss
    α > β → Penalize FP more (precision focus)
    α < β → Penalize FN more (recall focus)
    
    Good for: When you need to control FP/FN trade-off
    """
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        eps: float = 1e-7
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply activation
        if output.shape[1] > 1:
            output = F.softmax(output, dim=1)
        else:
            output = torch.sigmoid(output)
        
        # One-hot
        if target.dim() == 3:
            num_classes = output.shape[1]
            target = one_hot_encode(target, num_classes, output.device)
        
        # TP, FP, FN
        tp = (output * target).sum(dim=(0, 2, 3))
        fp = (output * (1 - target)).sum(dim=(0, 2, 3))
        fn = ((1 - output) * target).sum(dim=(0, 2, 3))
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps)
        
        # Loss
        tversky_loss = 1.0 - tversky
        
        return tversky_loss.mean()
