"""Module xử lý dữ liệu và DataLoader cho model training and evaluation."""

from torch.utils.data import DataLoader
from configs.config import CONFIG
from .preprocessing import train_ds, val_ds

if __name__ == "__main__":
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
