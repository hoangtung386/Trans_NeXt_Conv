import os
import torch
import set_determinism

random_seed = 42
set_determinism(seed=random_seed)

CONFIG = {
    'seed': random_seed,
    'n_channels': 1,
    'train_split': 0.8,
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 1.0e-4,
    'weight_decay': 1.0e-4,
    'aux_loss_weight': 0.01,
    'load_balance_weight': 0.1,
    'use_amp': True,
    'gradient_accumulation_steps': 4,
    'spatial_size': [256, 256],
    'init_features': 32,
    'num_classes': 6,
    'cache_rate': 0,
    'num_workers': 2,
    'base_path': "/path/to/your/data",
    'output_dir': "./output",
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_wandb': True,
    'wandb_project': None, 
    'wandb_api_key': None,
} 
