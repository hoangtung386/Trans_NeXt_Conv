"""Focal Loss implementations for segmentation tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .utils import one_hot_encode

class FocalLoss(nn.Module):
    """
    Focal Loss - Focuses on hard examples
    
    Formula: -α * (1 - p_t)^γ * log(p_t)
    
    Good for: Extreme class imbalance
    
    Args:
        alpha: weighting factor [0, 1] or list for multi-class
        gamma: focusing parameter (γ >= 0)
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: [B, C, H, W] - logits
            target: [B, H, W] - class indices
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(output, target, reduction='none', ignore_index=self.ignore_index or -100)
        
        # Get probabilities
        p = F.softmax(output, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = self.alpha * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Combines Focal and Tversky
    
    Good for: Small ROIs, highly imbalanced segmentation
    """
    def __init__(
        self,
        alpha: float = 0.7,  # Weight for False Positives
        beta: float = 0.3,   # Weight for False Negatives
        gamma: float = 0.75, # Focal parameter
        smooth: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax
        if output.shape[1] > 1:
            output = F.softmax(output, dim=1)
        else:
            output = torch.sigmoid(output)
        
        # One-hot
        if target.dim() == 3:
            num_classes = output.shape[1]
            target = one_hot_encode(target, num_classes, output.device)
        
        # True Positives, False Positives, False Negatives
        tp = (output * target).sum(dim=(0, 2, 3))
        fp = (output * (1 - target)).sum(dim=(0, 2, 3))
        fn = ((1 - output) * target).sum(dim=(0, 2, 3))
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal Tversky Loss
        ftl = (1 - tversky) ** self.gamma
        
        return ftl.mean()
