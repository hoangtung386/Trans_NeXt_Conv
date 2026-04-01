"""Compute evaluation metrics for multi-class segmentation tasks."""

import torch


def compute_metrics(pred, target, num_classes=6):
    """Compute IoU and Dice for multi-class segmentation.

    Args:
        pred: Logits tensor [B, C, H, W].
        target: Ground truth [B, H, W] or [B, 1, H, W].
        num_classes: Number of segmentation classes.

    Returns:
        Dict with 'iou' and 'dice' scores (averaged over organ classes).
    """
    pred = torch.argmax(pred, dim=1)  # [B, H, W]

    if target.dim() == 4:
        target = target.squeeze(1)

    intersection_sum = 0.0
    union_sum = 0.0
    dice_sum = 0.0
    valid_classes = 0

    for c in range(1, num_classes):  # Skip background
        p = (pred == c).float()
        t = (target == c).float()

        if t.sum() == 0 and p.sum() == 0:
            continue

        inter = (p * t).sum()
        dice_sum += (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
        intersection_sum += inter
        union_sum += (p + t - p * t).sum()
        valid_classes += 1

    if valid_classes == 0:
        return {"iou": 0.0, "dice": 0.0}

    iou = (intersection_sum + 1e-6) / (union_sum + 1e-6)
    dice = dice_sum / valid_classes

    return {"iou": iou.item(), "dice": dice.item()}
