"""Compute evaluation metrics for segmentation tasks."""

import torch

def compute_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    target = target > 0.5
    
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.float().sum() + target.float().sum() + 1e-6)
    
    return {'iou': iou.item(), 'dice': dice.item()}
