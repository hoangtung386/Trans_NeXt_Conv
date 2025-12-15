"""Configuration file for Trans_NeXt_Conv project"""
import os
import torch
from monai.utils import set_determinism

random_seed = 42
set_determinism(seed=random_seed)

CONFIG = {
    # Reproducibility
    'seed': random_seed,
    
    # Data paths
    'base_path': os.environ.get('DATA_PATH', "/path/to/your/data"),
    'output_dir': "./output",
    
    # Model architecture
    'n_channels': 1,
    'num_classes': 6,
    'spatial_size': [256, 256],  # [H, W]
    'init_features': 32,
    
    # Training hyperparameters
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1.0e-4,
    'weight_decay': 1.0e-4,
    'train_split': 0.8,
    
    # Loss weights
    'aux_loss_weight': 0.01,
    'load_balance_weight': 0.1,
    
    # Performance settings
    'use_amp': True,
    'gradient_accumulation_steps': 4,
    'num_workers': 2,
    'cache_rate': 0.2,
    
    # Device
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    
    # Logging
    'use_wandb': True,
    'wandb_project': 'Trans_NeXt_Conv',
    'wandb_api_key': os.environ.get('WANDB_API_KEY', None),
    
    # Checkpoint
    'save_every_n_epochs': 5,
    'early_stopping_patience': 20,
}

def validate_config(config):
    """Validate configuration parameters"""
    assert config['num_classes'] > 0, "num_classes must be positive"
    assert 0 < config['train_split'] < 1, "train_split must be between 0 and 1"
    assert len(config['spatial_size']) == 2, "spatial_size must be [H, W]"
    
    if config['use_wandb'] and config['wandb_api_key'] is None:
        print("Warning: W&B enabled but no API key provided")
    
    if not os.path.exists(config['base_path']):
        print(f"Warning: Data path does not exist: {config['base_path']}")
    
    print("Configuration validated")
    return True

# Validate on import
validate_config(CONFIG)