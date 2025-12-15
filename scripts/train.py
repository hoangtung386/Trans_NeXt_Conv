"""File này đảm nhiệm việc truyền tham số và kiểm tra mô hình TransNextConv."""

import argparse
import os
import torch
from tqdm import tqdm
from configs.config import CONFIG
from training.trainer import Trainer
from src.models.initialize_model import get_model
from data.dataloader import get_loaders
from .evaluate import SegmentationEvaluator
import wandb
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Train TransNextConv Model")
    
    # Data settings
    parser.add_argument("--base_path", type=str, default=CONFIG['base_path'],
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default=CONFIG['output_dir'],
                        help="Directory to save outputs")
    parser.add_argument("--train_split", type=float, default=CONFIG['train_split'],
                        help="Train/val split ratio")
    
    # Model settings
    parser.add_argument("--n_channels", type=int, default=CONFIG['n_channels'],
                        help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=CONFIG['num_classes'],
                        help="Number of segmentation classes")
    parser.add_argument("--init_features", type=int, default=CONFIG['init_features'],
                        help="Initial feature size")
    parser.add_argument("--spatial_size", type=int, nargs=2, default=CONFIG['spatial_size'],
                        help="Spatial size [H, W]")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=CONFIG['batch_size'],
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=CONFIG['num_epochs'],
                        help="Number of training epochs")
    parser.add_argument("--lr", "--learning_rate", type=float, default=CONFIG['learning_rate'],
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=CONFIG['weight_decay'],
                        help="Weight decay")
    parser.add_argument("--aux_loss_weight", type=float, default=CONFIG['aux_loss_weight'],
                        help="Auxiliary loss weight")
    parser.add_argument("--load_balance_weight", type=float, default=CONFIG['load_balance_weight'],
                        help="Load balance weight")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=CONFIG['gradient_accumulation_steps'],
                        help="Gradient accumulation steps")
    
    # Performance settings
    parser.add_argument("--use_amp", action="store_true", default=CONFIG['use_amp'],
                        help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="Disable automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=CONFIG['num_workers'],
                        help="Number of data loader workers")
    parser.add_argument("--cache_rate", type=float, default=CONFIG['cache_rate'],
                        help="Cache rate for data loading")
    parser.add_argument("--device", type=str, default=CONFIG['device'],
                        choices=['cuda', 'cpu'], help="Device to use")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=CONFIG['seed'],
                        help="Random seed")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", default=CONFIG['use_wandb'],
                        help="Enable W&B logging")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default=CONFIG['wandb_project'],
                        help="W&B project name")
    parser.add_argument("--wandb_api_key", type=str, default=CONFIG['wandb_api_key'],
                        help="W&B API key")
    
    return parser.parse_args()


def update_config(args):
    """Update CONFIG with command line arguments."""
    CONFIG.update({
        'seed': args.seed,
        'n_channels': args.n_channels,
        'train_split': args.train_split,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'aux_loss_weight': args.aux_loss_weight,
        'load_balance_weight': args.load_balance_weight,
        'use_amp': args.use_amp,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'spatial_size': args.spatial_size,
        'init_features': args.init_features,
        'num_classes': args.num_classes,
        'cache_rate': args.cache_rate,
        'num_workers': args.num_workers,
        'base_path': args.base_path,
        'output_dir': args.output_dir,
        'device': args.device,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_api_key': args.wandb_api_key,
    })
    return CONFIG


if __name__ == "__main__":
    args = parse_args()
    config = update_config(args)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # W&B Authentication
    if config['use_wandb']:
        try:
            wandb.login(key=config['wandb_api_key'])
            print("W&B authentication successful!")
        except Exception as e:
            print(f"W&B authentication failed: {e}")
            print("Continuing without W&B logging")
            config['use_wandb'] = False

    # Initialize model and data loaders
    model = get_model(config)
    train_loader, val_loader = get_loaders(config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device'],
        checkpoint_dir=config['output_dir'],
        use_wandb=config['use_wandb']
    )

    gc.collect()
    torch.cuda.empty_cache()

    trainer.train(num_epochs=config['num_epochs'])

    # Load Best Model for Inference
    best_checkpoint = torch.load(trainer.best_model_path, map_location=config['device'])
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"Best model loaded from epoch {best_checkpoint['epoch']} with Dice: {best_checkpoint['best_dice']:.4f}")
    
    model.eval()

    evaluator = SegmentationEvaluator(
        model, val_loader, 
        config['device'], 
        num_classes=config['num_classes']
    )

    # Compute metrics
    print("\nComputing metrics")
    results = evaluator.compute_metrics()
    
    # Plot metrics comparison
    print("\nPlotting metrics comparison")
    evaluator.plot_metrics(results)
    
    # Visualize overlay masks
    print("\nCreating overlay visualizations")
    evaluator.visualize_predictions(num_samples=5)

    # Per-class comparison
    print("\nCreating per-class comparison")
    evaluator.plot_per_class_comparison(num_samples=3)
    
    # Confusion matrix analysis
    print("\nCreating confusion matrix")
    evaluator.plot_confusion_analysis()
    
    # Create summary report
    evaluator.create_summary_report(results)
