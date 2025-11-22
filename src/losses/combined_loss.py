"""Combined Loss Functions for Segmentation Tasks"""

import torch
import torch.nn as nn
from typing import Optional

class CombinedLoss(nn.Module):
    """
    Combined Loss - Weighted combination of multiple losses
    
    Example: 0.5 * DiceLoss + 0.3 * FocalLoss + 0.2 * BoundaryLoss
    
    Good for: Leveraging strengths of different loss functions
    """
    def __init__(
        self,
        losses: dict,
        weights: dict,
        aux_loss_weight: float = 0.1
    ):
        """
        Args:
            losses: {'dice': DiceLoss(), 'focal': FocalLoss(), ...}
            weights: {'dice': 0.5, 'focal': 0.3, ...}
            aux_loss_weight: Weight for auxiliary losses (MoE losses)
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        self.aux_loss_weight = aux_loss_weight
        
        # Validate weights sum to 1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-5, f"Weights must sum to 1, got {total_weight}"
    
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        aux_losses: Optional[list] = None
    ) -> tuple:
        """
        Args:
            output: [B, C, H, W] - predictions
            target: [B, H, W] - ground truth
            aux_losses: List of auxiliary loss dicts from MoE layers
        
        Returns:
            total_loss, loss_dict
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Compute main losses
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(output, target)
            weight = self.weights[name]
            total_loss += weight * loss_value
            loss_dict[name] = loss_value.item()
        
        # Add auxiliary losses (from MoE)
        if aux_losses is not None and len(aux_losses) > 0:
            aux_total = 0.0
            for aux_dict in aux_losses:
                for key, value in aux_dict.items():
                    aux_total += value
                    if key not in loss_dict:
                        loss_dict[key] = 0.0
                    loss_dict[key] += value.item()
            
            # Average and weight auxiliary losses
            aux_total = aux_total / len(aux_losses)
            total_loss += self.aux_loss_weight * aux_total
            loss_dict['aux_loss'] = aux_total.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class AdaptiveLoss(nn.Module):
    """
    Adaptive Loss - Learns optimal weights during training
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    """
    def __init__(self, losses: dict, init_weights: dict = None):
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        
        # Learnable log-variance parameters
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in losses.keys()
        })
        
        if init_weights is not None:
            for name, weight in init_weights.items():
                if name in self.log_vars:
                    # Convert weight to log-variance
                    self.log_vars[name].data.fill_(-torch.log(torch.tensor(weight)))
    
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        aux_losses: Optional[list] = None
    ) -> tuple:
        total_loss = 0.0
        loss_dict = {}
        
        # Compute weighted losses
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(output, target)
            log_var = self.log_vars[name]
            
            # Weighted loss: loss / (2 * exp(log_var)) + log_var / 2
            precision = torch.exp(-log_var)
            weighted_loss = precision * loss_value + log_var
            
            total_loss += weighted_loss
            loss_dict[name] = loss_value.item()
            loss_dict[f'{name}_weight'] = precision.item()
        
        # Auxiliary losses
        if aux_losses is not None and len(aux_losses) > 0:
            aux_total = 0.0
            for aux_dict in aux_losses:
                for key, value in aux_dict.items():
                    aux_total += value
            aux_total = aux_total / len(aux_losses)
            total_loss += 0.1 * aux_total
            loss_dict['aux_loss'] = aux_total.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
