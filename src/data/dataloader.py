"""Module for data processing and DataLoader creation."""

from torch.utils.data import DataLoader
from configs.config import CONFIG
from .preprocessing import RSNA2DDataset
from .dataset import prepare_data

def get_loaders(config):
    """
    Create and return train and validation data loaders.
    """
    train_list, val_list = prepare_data(config)
    
    # Check if lists are empty
    if not train_list:
        print("Warning: Training set is empty!")
        # We might want to handle this gracefully or let DataLoader handle it (it will be empty)
    
    spatial_size = config['spatial_size'] if isinstance(config, dict) else config.spatial_size
    batch_size = config['batch_size'] if isinstance(config, dict) else config.batch_size
    num_workers = config['num_workers'] if isinstance(config, dict) else config.num_workers
    
    train_ds = RSNA2DDataset(
        train_list, 
        transforms=True, 
        spatial_size=spatial_size
    )
    
    val_ds = RSNA2DDataset(
        val_list, 
        transforms=True, 
        spatial_size=spatial_size
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=1, # Usually validation batch size is 1 or small
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test execution
    loaders = get_loaders(CONFIG)
    if loaders:
        train_l, val_l = loaders
        print(f"Train batches: {len(train_l)}, Val batches: {len(val_l)}")
