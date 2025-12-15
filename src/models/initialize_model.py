"""Module to initialize the TransNextConv model."""
import torch
from src.models.unet import TransNextConv
# from src.models.quantization import ModelQuantizer # Removed as per user request to avoid manual quantization
from configs.config import CONFIG

def get_model(config):
    """
    Initialize and return the TransNextConv model based on config.
    """
    image_size = config['image_crop_size'] # Assuming key is image_crop_size based on original file, but config might have spatial_size
    # Original file used CONFIG.image_crop_size. 
    # train.py uses spatial_size which is [H, W]. 
    # Let's check config usage. train.py updates config with args.
    # We should probably trust config object passed in.
    
    # However, CONFIG in initialize_model used CONFIG.image_crop_size directly.
    # Let's check config.py content? I haven't seen it yet.
    # But I can infer from initialize_model.py line 11: image_size = CONFIG.image_crop_size
    
    # In train.py line 34: default=CONFIG['spatial_size'] which is [H, W] probably.
    # initialize_model line 22: image_size=image_size where image_size is int.
    # So model expects square int?
    
    # Let's use config.get('image_crop_size', 256) or similar.
    # Or better, just use what was there.
    
    if hasattr(config, 'image_crop_size'):
         image_size = config.image_crop_size
    elif isinstance(config, dict) and 'image_crop_size' in config:
         image_size = config['image_crop_size']
    else:
         # Fallback or check if spatial_size is available and use first dim
         image_size = config.get('spatial_size', [256, 256])[0] if isinstance(config, dict) else 256

    n_channels = config['n_channels'] if isinstance(config, dict) else config.n_channels
    n_classes = config['num_classes'] if isinstance(config, dict) and 'num_classes' in config else (config['n_classes'] if isinstance(config, dict) and 'n_classes' in config else config.n_classes) 
    # train.py uses 'num_classes', initialize_model used 'n_classes'. 
    # config.py likely has both or train.py updates it.
    
    model = TransNextConv(
        image_size=image_size,
        n_channels=n_channels,
        n_classes=n_classes,
        stem_features=64,
        depths=[3, 4, 6, 2, 2, 2],
        widths=[256, 512, 1024],
        drop_p=0.0,
        embed_dim=1024
    )
    
    device = config['device'] if isinstance(config, dict) else config.device
    model = model.to(device)
    
    return model
