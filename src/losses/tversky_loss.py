"""Tversky Loss for segmentation tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import one_hot_encode


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss.

    Controls trade-off between FP and FN:
        alpha = beta = 0.5 -> Dice Loss
        alpha > beta -> Penalize FP more (precision focus)
        alpha < beta -> Penalize FN more (recall focus)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps

    def forward(
        self, output: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        if output.shape[1] > 1:
            output = F.softmax(output, dim=1)
        else:
            output = torch.sigmoid(output)

        if target.dim() == 3:
            num_classes = output.shape[1]
            target = one_hot_encode(target, num_classes, output.device)

        tp = (output * target).sum(dim=(0, 2, 3))
        fp = (output * (1 - target)).sum(dim=(0, 2, 3))
        fn = ((1 - output) * target).sum(dim=(0, 2, 3))

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn
            + self.smooth + self.eps
        )

        tversky_loss = 1.0 - tversky

        return tversky_loss.mean()
