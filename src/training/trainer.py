"""Fixed Trainer module for medical image segmentation"""

import os
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

# ✅ Fixed imports
from src.losses import CombinedLoss, DiceLoss, FocalLoss, TverskyLoss
from src.metrics.compute_metric import compute_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device,
                 checkpoint_dir='./output', use_wandb=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        self.device = device
        self.model.to(self.device)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = CombinedLoss(
            losses={
                'dice': DiceLoss(smooth=1.0),
                'focal': FocalLoss(alpha=0.25, gamma=2.0),
                'tversky': TverskyLoss(alpha=0.3, beta=0.7)
            },
            weights={
                'dice': 0.5,
                'focal': 0.3,
                'tversky': 0.2
            },
            aux_loss_weight=config.get('aux_loss_weight', 0.1)
        )
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.accum_steps = config.get('gradient_accumulation_steps', 1)
        
        # Training state
        self.start_epoch = 0
        self.best_dice = 0.0
        self.history = []
        self.wandb_run_id = None
        
        # Paths
        self.checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        self.best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        self.history_csv_path = os.path.join(checkpoint_dir, 'training_history.csv')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_losses = {'total': 0, 'dice': 0, 'focal': 0, 'tversky': 0, 'aux': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        self.optimizer.zero_grad()
        
        # ✅ Fixed: Proper batch unpacking
        for step, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['seg'].to(self.device)
            
            # Remove channel dimension from masks if present
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    # ✅ Fixed: Proper model output unpacking
                    output, all_aux, aux1, aux2, aux3 = self.model(images)
                    
                    # Combine auxiliary losses
                    aux_losses = []
                    if isinstance(all_aux, list):
                        aux_losses.extend(all_aux)
                    if aux1: aux_losses.append(aux1)
                    if aux2: aux_losses.append(aux2)
                    if aux3: aux_losses.append(aux3)
                    
                    loss, loss_dict = self.criterion(output, masks, aux_losses=aux_losses)
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output, all_aux, aux1, aux2, aux3 = self.model(images)
                
                aux_losses = []
                if isinstance(all_aux, list):
                    aux_losses.extend(all_aux)
                if aux1: aux_losses.append(aux1)
                if aux2: aux_losses.append(aux2)
                if aux3: aux_losses.append(aux3)
                
                loss, loss_dict = self.criterion(output, masks, aux_losses=aux_losses)
                loss.backward()
                
                if (step + 1) % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Accumulate losses
            for key in total_losses.keys():
                if key in loss_dict:
                    total_losses[key] += loss_dict[key]
            
            pbar.set_postfix({
                'loss': f"{loss_dict.get('total_loss', 0):.4f}",
                'dice': f"{loss_dict.get('dice', 0):.4f}"
            })
        
        # Average losses
        n = len(self.train_loader)
        return {k: v / n for k, v in total_losses.items()}
    
    @torch.no_grad()
    def validate(self):  # ✅ Fixed: Removed epoch parameter
        """Validate the model"""
        self.model.eval()
        total_losses = {'total': 0, 'dice': 0, 'focal': 0, 'tversky': 0, 'aux': 0}
        metrics = {'iou': 0, 'dice': 0}
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            images = batch['image'].to(self.device)
            masks = batch['seg'].to(self.device)
            
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            
            if self.use_amp:
                with autocast():
                    output, all_aux, aux1, aux2, aux3 = self.model(images)
                    
                    aux_losses = []
                    if isinstance(all_aux, list):
                        aux_losses.extend(all_aux)
                    if aux1: aux_losses.append(aux1)
                    if aux2: aux_losses.append(aux2)
                    if aux3: aux_losses.append(aux3)
                    
                    loss, loss_dict = self.criterion(output, masks, aux_losses=aux_losses)
            else:
                output, all_aux, aux1, aux2, aux3 = self.model(images)
                
                aux_losses = []
                if isinstance(all_aux, list):
                    aux_losses.extend(all_aux)
                if aux1: aux_losses.append(aux1)
                if aux2: aux_losses.append(aux2)
                if aux3: aux_losses.append(aux3)
                
                loss, loss_dict = self.criterion(output, masks, aux_losses=aux_losses)
            
            for key in total_losses.keys():
                if key in loss_dict:
                    total_losses[key] += loss_dict[key]
            
            batch_metrics = compute_metrics(output, masks)
            for k in metrics:
                metrics[k] += batch_metrics[k]
        
        n = len(self.val_loader)
        avg_loss = {k: v / n for k, v in total_losses.items()}
        avg_metrics = {k: v / n for k, v in metrics.items()}
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'history': self.history,
            'config': self.config,
            'wandb_run_id': self.wandb_run_id
        }
        
        torch.save(checkpoint, self.checkpoint_path)
        print(f"✓ Checkpoint saved at epoch {epoch}")
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"✓ Best model saved! Dice: {val_metrics['dice']:.4f}")
    
    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_dice = checkpoint['best_dice']
            self.history = checkpoint.get('history', [])
            self.wandb_run_id = checkpoint.get('wandb_run_id', None)
            
            print(f"✓ Resumed from epoch {self.start_epoch}, best dice: {self.best_dice:.4f}")
            return True
        return False
    
    def save_history_csv(self):
        """Save training history to CSV"""
        if self.history:
            df = pd.DataFrame(self.history)
            df.to_csv(self.history_csv_path, index=False)
            print(f"✓ History saved to {self.history_csv_path}")
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        # Try to resume from checkpoint
        resumed = self.load_checkpoint()
        
        # Initialize W&B
        if self.use_wandb:
            if resumed and self.wandb_run_id:
                wandb.init(
                    project=self.config.get('wandb_project', 'Trans_NeXt_Conv'),
                    config=self.config,
                    resume="allow",
                    id=self.wandb_run_id
                )
            else:
                wandb.init(
                    project=self.config.get('wandb_project', 'Trans_NeXt_Conv'),
                    config=self.config,
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.wandb_run_id = wandb.run.id
            
            wandb.watch(self.model, log='all', log_freq=100)
        
        print(f"\n{'='*60}")
        print(f"Starting training from epoch {self.start_epoch} to {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_losses = self.train_epoch(epoch + 1)
            
            # Validate
            val_losses, val_metrics = self.validate()  # ✅ Fixed: No epoch parameter
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_losses['total'],
                'val_loss': val_losses['total'],
                'val_dice': val_metrics['dice'],
                'val_iou': val_metrics['iou'],
                'learning_rate': current_lr,
                'best_dice': self.best_dice
            }
            
            self.history.append(log_dict)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Val Dice: {val_metrics['dice']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            if self.use_wandb:
                wandb.log(log_dict)
            
            # Save best model
            is_best = val_metrics['dice'] > self.best_dice
            if is_best:
                self.best_dice = val_metrics['dice']
            
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Save history periodically
            if (epoch + 1) % 5 == 0:
                self.save_history_csv()
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best Dice: {self.best_dice:.4f}")
        print(f"{'='*60}\n")
        
        self.save_history_csv()
        
        if self.use_wandb:
            wandb.finish()
            