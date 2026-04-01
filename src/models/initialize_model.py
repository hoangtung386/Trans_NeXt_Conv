"""Module to initialize the TransNextConv model."""

from src.models.trans_next_conv import TransNextConv


def get_model(config):
    """Initialize and return the TransNextConv model based on config.

    Args:
        config: Configuration dict or object.

    Returns:
        TransNextConv model on the specified device.
    """
    if isinstance(config, dict):
        spatial_size = config.get("spatial_size", [256, 256])
        image_size = spatial_size[0]
        n_channels = config.get("n_channels", 1)
        n_classes = config.get("num_classes", 6)
        device = config.get("device", "cpu")
    else:
        spatial_size = getattr(config, "spatial_size", [256, 256])
        image_size = spatial_size[0]
        n_channels = getattr(config, "n_channels", 1)
        n_classes = getattr(config, "num_classes", 6)
        device = getattr(config, "device", "cpu")

    model = TransNextConv(
        image_size=image_size,
        n_channels=n_channels,
        n_classes=n_classes,
        stem_features=64,
        depths=[3, 4, 6, 2, 2, 2],
        widths=[256, 512, 1024],
        drop_p=0.0,
        embed_dim=1024,
    )

    model = model.to(device)

    return model
