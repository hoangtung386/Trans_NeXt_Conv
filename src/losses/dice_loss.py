"""Dice Loss and Generalized Dice Loss Implementation for Segmentation Tasks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .utils import soft_dice_score, one_hot_encode

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    
    Formula: 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    
    Good for: Imbalanced datasets, medical image segmentation
    """
    def __init__(
        self,
        smooth: float = 1.0,
        eps: float = 1e-7,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None
    ):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: [B, C, H, W] - logits or probabilities
            target: [B, H, W] - class indices OR [B, C, H, W] - one-hot
        """
        # Apply softmax if needed
        if output.shape[1] > 1:  # Multi-class
            output = F.softmax(output, dim=1)
        else:  # Binary
            output = torch.sigmoid(output)
        
        # Convert target to one-hot if needed
        if target.dim() == 3:
            num_classes = output.shape[1]
            target = one_hot_encode(target, num_classes, output.device)
        
        # Ignore specific index
        if self.ignore_index is not None:
            mask = target[:, self.ignore_index:self.ignore_index+1, :, :].bool()
            output = output * (~mask)
            target = target * (~mask)
        
        # Compute dice score per class
        dice_score = soft_dice_score(output, target, self.smooth, self.eps, dims=(0, 2, 3))
        dice_loss = 1.0 - dice_score
        
        # Reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        return dice_loss


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss - Better for highly imbalanced classes
    
    Weights each class by 1 / (frequency^2)
    """
    def __init__(
        self,
        smooth: float = 1.0,
        eps: float = 1e-7,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax
        if output.shape[1] > 1:
            output = F.softmax(output, dim=1)
        else:
            output = torch.sigmoid(output)
        
        # One-hot encoding
        if target.dim() == 3:
            num_classes = output.shape[1]
            target = one_hot_encode(target, num_classes, output.device)
        
        # Calculate weights (1 / frequency^2)
        target_sum = target.sum(dim=(0, 2, 3))
        class_weights = 1.0 / (target_sum ** 2 + self.eps)
        
        # Weighted intersection and cardinality
        intersection = torch.sum(output * target, dim=(0, 2, 3))
        cardinality = torch.sum(output + target, dim=(0, 2, 3))
        
        weighted_intersection = (class_weights * intersection).sum()
        weighted_cardinality = (class_weights * cardinality).sum()
        
        gdl = 1.0 - (2.0 * weighted_intersection + self.smooth) / (weighted_cardinality + self.smooth)
        
        return gdl
