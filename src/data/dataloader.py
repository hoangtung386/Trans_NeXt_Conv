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
        transforms=False, # Usually validation has no auth except resize/normalization which might be in transforms=True logic of RSNA2DDataset?
        # Looking at preprocessing.py: if self.transforms: ...
        # It seems 'transforms' flag controls resizing and normalization too?
        # Yes, line 60: if self.transforms: ... resize ... 
        # Wait, if transforms=False, then it returns raw images? 
        # Line 87 just adds channel dim.
        # But `train_ds` in original line 98 used transforms=True.
        # `val_ds` in original line 99 used transforms=True.
        # So I should use True for both, maybe?
        # Usually valid set needs resize and normalization but NOT augmentation (like crop).
        # But RSNA2DDataset calculates crop based on mask? Line 66: mask = image_2d > 0.1
        # This seems to be a cropping strategy to find ROI. This should probably be done for Validation too if we want same input size.
        # So I will set transforms=True for both as per original code.
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
