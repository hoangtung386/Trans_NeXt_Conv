"""Module for data processing and DataLoader creation."""

from torch.utils.data import DataLoader

from .preprocessing import RSNA2DDataset
from .dataset import prepare_data


def get_loaders(config):
    """Create and return train and validation data loaders.

    Args:
        config: Configuration dict or object.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_list, val_list = prepare_data(config)

    if not train_list:
        raise ValueError(
            "Training set is empty! Check your data path and preprocessing."
        )

    if isinstance(config, dict):
        spatial_size = config["spatial_size"]
        batch_size = config["batch_size"]
        num_workers = config["num_workers"]
    else:
        spatial_size = config.spatial_size
        batch_size = config.batch_size
        num_workers = config.num_workers

    train_ds = RSNA2DDataset(
        train_list,
        transforms=True,
        spatial_size=spatial_size,
    )

    val_ds = RSNA2DDataset(
        val_list,
        transforms=True,
        spatial_size=spatial_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
