"""Trainer module for medical image segmentation using MONAI and W&B integration."""

import os
import pandas as pd
import torch
import torch.optim as optim
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from tqdm import tqdm
import wandb
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from losses import CombinedLoss, DiceLoss, FocalLoss, TverskyLoss
from metrics.compute_metric import compute_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device,
                 checkpoint_dir='../experiments', use_wandb=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb

        self.device = device
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = CombinedLoss(
            losses={
                'dice': DiceLoss(smooth=1.0),
                'focal': FocalLoss(alpha=0.25, gamma=2.0),
                'tversky': TverskyLoss(alpha=0.3, beta=0.7)  # Favor recall
            },
            weights={
                'dice': 0.5,    # Main loss
                'focal': 0.3,   # Handle imbalance
                'tversky': 0.2  # Fine-tune FP/FN
            },
            aux_loss_weight=0.1  # Weight for MoE losses
        )

        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        self.dice_metric = DiceMetric(include_background=False, reduction='mean')
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        self.use_amp = config.get('use_amp', True)
        
        # Gradient accumulation
        self.accum_steps = config.get('gradient_accumulation_steps', 1)
        
        # Best metric tracking
        self.best_dice = 0.0

        # Training state
        self.start_epoch = 0
        self.best_dice = 0.0
        self.history = []
        self.wandb_run_id = None

        # Paths
        self.checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_2d.pth')
        self.best_model_path = os.path.join(checkpoint_dir, 'best_model_2d.pth')
        self.history_csv_path = os.path.join(checkpoint_dir, 'training_history_2d.csv')

    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """Save checkpoint"""
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
        print(f"Checkpoint saved at epoch {epoch}")

        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"Best model saved! Dice: {val_dice:.4f}")

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
            self.history = checkpoint['history']

            if 'wandb_run_id' in checkpoint:
                self.wandb_run_id = checkpoint['wandb_run_id']

            print(f"Resumed from epoch {self.start_epoch}, best dice: {self.best_dice:.4f}")
            return True
        return False

    def save_history_csv(self):
        """Save training history to CSV"""
        if self.history:
            df = pd.DataFrame(self.history)
            df.to_csv(self.history_csv_path, index=False)

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = {'total': 0.0, 'bce': 0.0, 'dice': 0.0, 'focal': 0.0, 'tversky': 0.0, 'aux': 0.0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for step, (images, masks) in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            #########################################################################
            """all_aux_losses: ở đây là điền tạm cho đỡ lỗi thôi chứ del đúng đâu"""
            #########################################################################
            outputs, all_aux_losses = self.model(images)
            loss, loss_dict = self.criterion(outputs, masks, aux_losses=all_aux_losses)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = {'total': 0, 'bce': 0, 'dice': 0, 'aux': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        self.optimizer.zero_grad()
        
        for step, (images, masks) in enumerate(pbar):
            images = images.to(self.device).half()
            masks = masks.to(self.device).half()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    output, all_aux, aux1, aux2, aux3 = self.model(images)
                    losses = self.criterion(
                        output, masks, all_aux, aux1, aux2, aux3
                    )
                    loss = losses['total'] / self.accum_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output, all_aux, aux1, aux2, aux3 = self.model(images)
                losses = self.criterion(output, masks, all_aux, aux1, aux2, aux3)
                loss = losses['total'] / self.accum_steps
                loss.backward()
                
                if (step + 1) % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Accumulate losses
            for k in total_loss:
                total_loss[k] += losses[k].item()
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'dice': f"{losses['dice'].item():.4f}",
                'aux': f"{losses['aux'].item():.4f}"
            })
        
        # Average losses
        n = len(self.train_loader)
        return {k: v / n for k, v in total_loss.items()}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = {'total': 0, 'bce': 0, 'dice': 0, 'aux': 0}
        metrics = {'iou': 0, 'dice': 0}
        
        for images, masks in tqdm(self.val_loader, desc='Validating'):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            if self.use_amp:
                with autocast():
                    output, all_aux, aux1, aux2, aux3 = self.model(images)
                    losses = self.criterion(output, masks, all_aux, aux1, aux2, aux3)
            else:
                output, all_aux, aux1, aux2, aux3 = self.model(images)
                losses = self.criterion(output, masks, all_aux, aux1, aux2, aux3)
            
            for k in total_loss:
                total_loss[k] += losses[k].item()
            
            batch_metrics = compute_metrics(output, masks)
            for k in metrics:
                metrics[k] += batch_metrics[k]
        
        n = len(self.val_loader)
        avg_loss = {k: v / n for k, v in total_loss.items()}
        avg_metrics = {k: v / n for k, v in metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']

        resumed = self.load_checkpoint()

        if self.use_wandb:
            if resumed and self.wandb_run_id:
                wandb.init(
                    project="my-2D-Unet-segment-RSNA",
                    config=self.config,
                    resume="allow",
                    id=self.wandb_run_id,
                    name=f"unet2d_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                print(f"Resumed W&B run: {self.wandb_run_id}")
            else:
                run_name = f"unet2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                wandb.init(
                    project="my-2D-Unet-segment-RSNA",
                    config=self.config,
                    name=run_name,
                    tags=["2d-unet", "organ-segmentation", "rsna"]
                )
                self.wandb_run_id = wandb.run.id
                print(f"Created new W&B run: {self.wandb_run_id}")

            wandb.watch(self.model, log='all', log_freq=100)

        print(f"\nStarting training from epoch {self.start_epoch} to {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

        for epoch in range(self.start_epoch, num_epochs):
            train_loss = self.train_epoch(epoch + 1)
            print(f"\nTrain - Loss: {train_loss['total']:.4f}, "
                  f"BCE: {train_loss['bce']:.4f}, "
                  f"Dice: {train_loss['dice']:.4f}, "
                  f"Aux: {train_loss['aux']:.4f}")
            
            # Validation
            val_loss, val_metrics = self.validate(epoch + 1)
            print(f"Val - Loss: {val_loss['total']:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}, "
                  f"Dice: {val_metrics['dice']:.4f}")
            
            # Update scheduler
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'best_dice': self.best_dice
            }

            self.history.append(metrics)

            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Best Dice: {self.best_dice:.4f}")

            if self.use_wandb:
                wandb.log(metrics)

            # Save best model
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.save_checkpoint(epoch, val_metrics, 'best_model.pth')
                print(f"New best model saved! Dice: {self.best_dice:.4f}")
                self.save_history_csv()
        
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_metrics, f'checkpoint_epoch_{epoch}.pth')

        print(f"\nTraining complete! Best Dice: {self.best_dice:.4f}")

        if self.use_wandb:
            wandb.finish()
