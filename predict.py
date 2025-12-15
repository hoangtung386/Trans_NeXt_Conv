"""Inference script for TransNextConv model."""
import argparse
import os
import torch
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from configs.config import CONFIG
from src.models.initialize_model import get_model
from skimage.transform import resize

def parse_args():
    parser = argparse.ArgumentParser(description="Predict using TransNextConv")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image (NIfTI .nii or standard image)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, default="prediction.png", help="Path to save prediction image")
    parser.add_argument("--slice_idx", type=int, default=None, help="Slice index for 3D volumes. Default: middle slice")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def preprocess_image(image_path, slice_idx=None, target_size=(256, 256)):
    """Load and preprocess image."""
    if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
        nii = nib.load(image_path)
        data = nii.get_fdata()
        
        if len(data.shape) == 3:
            if slice_idx is None:
                slice_idx = data.shape[2] // 2
            image_2d = data[:, :, slice_idx]
        elif len(data.shape) == 4:
            if slice_idx is None:
                slice_idx = data.shape[2] // 2
            image_2d = data[0, :, :, slice_idx] # Assuming channel first 3D or similar
        else:
             image_2d = data
             
        # Normalize
        image_2d = np.clip(image_2d, -125, 275)
        image_2d = (image_2d + 125) / 400.0
        
    else:
        # Standard image
        img = Image.open(image_path).convert('L') # Grayscale
        image_2d = np.array(img)
        image_2d = image_2d / 255.0
    
    # Resize
    image_2d = resize(image_2d, target_size, preserve_range=True)
    
    # To Tensor: (1, C, H, W)
    image_tensor = torch.from_numpy(image_2d).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0) # [B, C, H, W]
    
    return image_tensor, image_2d

def main():
    args = parse_args()
    
    # Load Config
    config = CONFIG
    config['device'] = args.device
    
    # Load Model
    model = get_model(config)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Preprocess
    print(f"Processing image {args.image_path}")
    image_tensor, image_orig = preprocess_image(args.image_path, args.slice_idx, 
                                                tuple(config['spatial_size']))
    image_tensor = image_tensor.to(args.device)
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)[0] # Output is tuple
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
        
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_orig, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    axes[1].imshow(image_orig, cmap='gray')
    axes[1].imshow(pred_mask, cmap='jet', alpha=0.5)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output_path)
    print(f"Prediction saved to {args.output_path}")

if __name__ == "__main__":
    main()
