"""Utility functions for loss calculations in segmentation tasks."""

import torch
from typing import Optional


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-7,
    dims: Optional[tuple] = None,
) -> torch.Tensor:
    """Compute soft dice score.

    Args:
        output: [B, C, H, W] predictions (after softmax/sigmoid).
        target: [B, C, H, W] one-hot encoded targets.
        smooth: Smoothing constant.
        eps: Epsilon for numerical stability.
        dims: Dimensions to reduce.
    """
    if dims is None:
        dims = (0, 2, 3)

    intersection = torch.sum(output * target, dim=dims)
    cardinality = torch.sum(output + target, dim=dims)

    dice_score = (2.0 * intersection + smooth) / (
        cardinality + smooth + eps
    )
    return dice_score


def iou_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-7,
    dims: Optional[tuple] = None,
) -> torch.Tensor:
    """Compute IoU (Jaccard) score."""
    if dims is None:
        dims = (0, 2, 3)

    intersection = torch.sum(output * target, dim=dims)
    union = torch.sum(output + target, dim=dims) - intersection

    iou = (intersection + smooth) / (union + smooth + eps)
    return iou


def one_hot_encode(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Convert class indices to one-hot encoding.

    Args:
        labels: [B, H, W] class indices.
        num_classes: Number of classes.
        device: Target device.

    Returns:
        [B, C, H, W] one-hot encoded tensor.
    """
    if device is None:
        device = labels.device

    batch_size, height, width = labels.shape
    one_hot = torch.zeros(
        batch_size, num_classes, height, width, device=device,
    )
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot
