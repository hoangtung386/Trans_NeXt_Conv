"""Data preprocessing using MONAI."""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from configs.config import CONFIG


class RSNA2DDataset(Dataset):
    def __init__(self, data_list, transforms=None, spatial_size=(256, 256)):
        self.data_list = data_list
        self.transforms = transforms
        self.spatial_size = spatial_size
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        image_dir = data_dict['image']
        seg_path = data_dict['seg']
        slice_idx = data_dict['slice_idx']
        
        # Load 3D volumes
        # Load image (DICOM series)
        import os
        if os.path.isdir(image_dir):
            dicom_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.dcm')])
            if not dicom_files:
                # Fallback or error if empty/no dcm
                # Maybe it is just a path to a file if not dir?
                # But problem said it is a folder.
                pass 
            # Use LoadImage on list of files? Or stack?
            # monai LoadImage can handle directory if it contains dicom series usually, but explicit is better.
            # actually LoadImage(image_only=True)(image_dir) *should* work if it is a directory of DICOMs for some versions,
            # but User said "LoadImage does not automatically load DICOM folder".
            # So let's do the manual listing and stacking or pass list to LoadImage.
            # Better: use pydicom/monai to load series properly.
            # Simplest fix as per user suggestion:
            from monai.transforms import LoadImage
            loader = LoadImage(image_only=True)
            # Check if it is a dir
            if os.path.isdir(image_dir):
                 dicom_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                                      if f.endswith('.dcm') or f.endswith('.DCM')])
                 if dicom_files:
                     # Load each slice
                     slices = []
                     for f in dicom_files:
                         s = loader(f)
                         # s might be (H, W) or (1, H, W)
                         if isinstance(s, torch.Tensor):
                             s = s.numpy()
                         slices.append(s)
                     image_3d = np.stack(slices, axis=-1) # Stack along depth
                 else:
                     # Fallback if no dcm found
                     image_3d = loader(image_dir)
            else:
                 image_3d = loader(image_dir)

        # Load segmentation
        # seg_path is likely a nifti file or single file, so loader(seg_path) is fine.
        seg_3d = loader(seg_path)
        
        # Convert to numpy if torch tensor
        if isinstance(image_3d, torch.Tensor):
            image_3d = image_3d.numpy()
        if isinstance(seg_3d, torch.Tensor):
            seg_3d = seg_3d.numpy()
        
        # Extract 2D slice
        if len(image_3d.shape) == 4:
            image_2d = image_3d[0, :, :, slice_idx]
        else:
            image_2d = image_3d[:, :, slice_idx]
            
        if len(seg_3d.shape) == 4:
            seg_2d = seg_3d[0, :, :, slice_idx]
        else:
            seg_2d = seg_3d[:, :, slice_idx]
        
        # Ensure numpy array
        image_2d = np.array(image_2d, dtype=np.float32)
        seg_2d = np.array(seg_2d, dtype=np.float32)
        
        # Ensure 2D shape (H, W)
        if len(image_2d.shape) == 3:
            image_2d = image_2d[0]
        if len(seg_2d.shape) == 3:
            seg_2d = seg_2d[0]
        
        # Apply transforms
        if self.transforms:
            # Intensity scaling
            image_2d = np.clip(image_2d, -125, 275)
            image_2d = (image_2d + 125) / 400.0  # Normalize to [0, 1]
            
            # Crop foreground
            mask = image_2d > 0.1
            if mask.sum() > 0:
                coords = np.argwhere(mask)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Add padding
                pad = 10
                y_min = max(0, y_min - pad)
                x_min = max(0, x_min - pad)
                y_max = min(image_2d.shape[0], y_max + pad)
                x_max = min(image_2d.shape[1], x_max + pad)
                
                image_2d = image_2d[y_min:y_max, x_min:x_max]
                seg_2d = seg_2d[y_min:y_max, x_min:x_max]
            
            # Resize to fixed size
            from skimage.transform import resize
            image_2d = resize(image_2d, self.spatial_size, order=1, preserve_range=True)
            seg_2d = resize(seg_2d, self.spatial_size, order=0, preserve_range=True)
        
        # Add channel dimension (C, H, W)
        image_2d = image_2d[np.newaxis, ...].astype(np.float32)
        seg_2d = seg_2d[np.newaxis, ...].astype(np.int64)
        
        # Convert to torch tensors
        image_2d = torch.from_numpy(image_2d)
        seg_2d = torch.from_numpy(seg_2d)
        
        return {'image': image_2d, 'seg': seg_2d}


