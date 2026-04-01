"""Combined and adaptive loss functions for segmentation tasks."""

import torch
import torch.nn as nn
from typing import Optional


class CombinedLoss(nn.Module):
    """Weighted combination of multiple loss functions.

    Example: 0.5 * DiceLoss + 0.3 * FocalLoss + 0.2 * TverskyLoss
    """

    def __init__(
        self,
        losses: dict,
        weights: dict,
        aux_loss_weight: float = 0.1,
    ):
        """Initialize combined loss.

        Args:
            losses: Dict of loss name -> loss module.
            weights: Dict of loss name -> weight (should sum to 1.0).
            aux_loss_weight: Weight for MoE auxiliary losses.
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        self.aux_loss_weight = aux_loss_weight

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-3:
            print(
                f"Warning: Loss weights sum to {total_weight}, not 1.0"
            )

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        aux_losses: Optional[list] = None,
    ) -> tuple:
        """Compute combined loss.

        Args:
            output: [B, C, H, W] predictions.
            target: [B, H, W] ground truth.
            aux_losses: List of auxiliary loss dicts from MoE layers.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(output, target)
            weight = self.weights[name]
            total_loss += weight * loss_value
            loss_dict[name] = loss_value.item()

        if aux_losses is not None and len(aux_losses) > 0:
            aux_total = 0.0
            for aux_dict in aux_losses:
                for key, value in aux_dict.items():
                    aux_total += value
                    if key not in loss_dict:
                        loss_dict[key] = 0.0
                    loss_dict[key] += value.item()

            aux_total = aux_total / len(aux_losses)
            total_loss += self.aux_loss_weight * aux_total
            loss_dict["aux_loss"] = aux_total.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


class AdaptiveLoss(nn.Module):
    """Adaptive loss that learns optimal weights during training.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al.).
    """

    def __init__(self, losses: dict, init_weights: dict = None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)

        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in losses
        })

        if init_weights is not None:
            for name, weight in init_weights.items():
                if name in self.log_vars:
                    self.log_vars[name].data.fill_(
                        -torch.log(torch.tensor(weight))
                    )

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        aux_losses: Optional[list] = None,
    ) -> tuple:
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(output, target)
            log_var = self.log_vars[name]

            precision = torch.exp(-log_var)
            weighted_loss = precision * loss_value + log_var

            total_loss += weighted_loss
            loss_dict[name] = loss_value.item()
            loss_dict[f"{name}_weight"] = precision.item()

        if aux_losses is not None and len(aux_losses) > 0:
            aux_total = 0.0
            for aux_dict in aux_losses:
                for key, value in aux_dict.items():
                    aux_total += value
            aux_total = aux_total / len(aux_losses)
            total_loss += 0.1 * aux_total
            loss_dict["aux_loss"] = aux_total.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict
