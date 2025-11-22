import torch
import torch.nn.functional as F
from .segformer import MixtureOfExperts

def train_step(model, batch, optimizer, loss_weights, max_grad_norm=1.0):
    """
    Single training step with MoE auxiliary losses
    """
    input_ids = batch['input_ids']
    labels = batch['labels']

    # Forward pass
    logits, all_aux_losses = model(input_ids)

    # 1. Main task loss
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # 2. Aggregate auxiliary losses across layers
    if all_aux_losses:
        load_balance_loss = torch.stack([aux['load_balance_loss'] for aux in all_aux_losses]).mean()
        router_entropy = torch.stack([aux['router_entropy'] for aux in all_aux_losses]).mean()
        shared_routed_balance_loss = torch.stack([aux['shared_routed_balance_loss'] for aux in all_aux_losses]).mean()
    else:
        load_balance_loss = torch.tensor(0.0, device=input_ids.device)
        router_entropy = torch.tensor(0.0, device=input_ids.device)
        shared_routed_balance_loss = torch.tensor(0.0, device=input_ids.device)

    # 3. Total loss
    total_loss = (
        ce_loss +
        loss_weights.get('load_balance', 0.01) * load_balance_loss +
        loss_weights.get('shared_routed_balance', 0.01) * shared_routed_balance_loss -
        loss_weights.get('entropy', 0.001) * router_entropy
    )

    # 4. Backward pass with gradient clipping
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    # Return losses for logging
    return {
        'total_loss': total_loss.item(),
        'ce_loss': ce_loss.item(),
        'load_balance_loss': load_balance_loss.item(),
        'shared_routed_balance_loss': shared_routed_balance_loss.item(),
        'router_entropy': router_entropy.item()
    }