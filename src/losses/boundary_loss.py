"""Boundary Loss Implementation for Segmentation Tasks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import one_hot_encode

class BoundaryLoss(nn.Module):
    """
    Boundary Loss - Focuses on object boundaries
    
    Uses distance transform to weight pixels near boundaries more
    
    Good for: Tasks requiring precise boundary delineation
    """
    def __init__(self, theta0: float = 3, theta: float = 5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: [B, C, H, W] - predictions
            target: [B, H, W] - class indices
        """
        from scipy.ndimage import distance_transform_edt
        import numpy as np
        
        # Apply softmax
        output = F.softmax(output, dim=1)
        
        # One-hot
        if target.dim() == 3:
            num_classes = output.shape[1]
            target = one_hot_encode(target, num_classes, output.device)
        
        batch_size, num_classes = output.shape[0], output.shape[1]
        
        # Compute boundary loss for each sample
        boundary_losses = []
        
        for b in range(batch_size):
            for c in range(num_classes):
                # Get binary mask
                gt_mask = target[b, c].cpu().numpy()
                
                # Compute distance transform
                if gt_mask.sum() > 0:
                    # Distance from boundary
                    pos_dist = distance_transform_edt(gt_mask)
                    neg_dist = distance_transform_edt(1 - gt_mask)
                    
                    # Combine distances
                    dist_map = neg_dist - pos_dist
                    dist_map = torch.from_numpy(dist_map).float().to(output.device)
                    
                    # Compute boundary loss
                    pred = output[b, c]
                    bl = (pred * dist_map).sum()
                    boundary_losses.append(bl)
        
        if len(boundary_losses) > 0:
            return torch.stack(boundary_losses).mean()
        return torch.tensor(0.0, device=output.device)
