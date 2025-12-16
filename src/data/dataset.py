"""Custom Dataset class for loading medical imaging data and corresponding segmentations."""
import os
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(config):
    """
    Prepare data lists for training and validation.
    Scans the directory structure and creates a list of samples.
    """
    base_path = config['base_path'] if isinstance(config, dict) else config.base_path
    train_split = config['train_split'] if isinstance(config, dict) else config.train_split
    seed = config['seed'] if isinstance(config, dict) else config.seed
    
    csv_meta_path = os.path.join(base_path, 'train_series_meta.csv')
    csv_labels_path = os.path.join(base_path, 'train_2024.csv')
    
    if not os.path.exists(csv_meta_path) or not os.path.exists(csv_labels_path):
         print(f"Warning: CSV files not found at {base_path}")
         return [], []

    series_meta = pd.read_csv(csv_meta_path)
    # labels = pd.read_csv(csv_labels_path) # labels might be unused in this function logic actually, based on previous code.
    
    seg_path = os.path.join(base_path, 'segmentations')
    if not os.path.exists(seg_path):
         print(f"Warning: Segmentation folder not found at {seg_path}")
         return [], []
         
    seg_files = [f for f in os.listdir(seg_path) if f.endswith('.nii')]
    
    print("Extracting 2D slices from 3D volumes")
    data_list = []
    
    for seg_file in tqdm(seg_files, desc="Processing volumes"):
        try:
            series_id = int(seg_file.split('.')[0])
        except ValueError:
            continue
            
        patient_row = series_meta[series_meta['series_id'] == series_id]
        
        if not patient_row.empty:
            patient_id = int(patient_row['patient_id'].values[0])
            image_dir = os.path.join(base_path, f'train_images/{patient_id}/{series_id}')
            seg_file_path = os.path.join(seg_path, seg_file)
            
            # Load segmentation to get number of slices
            try:
                seg_nii = nib.load(seg_file_path)
                seg_data = seg_nii.get_fdata()
                
                # Get slices that contain organs (not all background)
                num_slices = seg_data.shape[2]
                for slice_idx in range(num_slices):
                    slice_seg = seg_data[:, :, slice_idx]
                    # Only include slices with at least some organ segmentation
                    total_pixels = slice_seg.shape[0] * slice_seg.shape[1]
                    min_ratio = config.get('min_organ_ratio', 0.01) if isinstance(config, dict) else getattr(config, 'min_organ_ratio', 0.01)
                    if np.sum(slice_seg > 0) > total_pixels * min_ratio:  # At least 1% (default) has organs
                        data_list.append({
                            'image': image_dir,
                            'seg': seg_file_path,
                            'slice_idx': slice_idx
                        })
            except Exception as e:
                print(f"Error processing {seg_file}: {e}")
                continue
    
    print(f"Total 2D slices extracted: {len(data_list)}")
    
    if len(data_list) == 0:
        return [], []

    train_list, val_list = train_test_split(
        data_list,
        test_size=1-train_split,
        random_state=seed,
        shuffle=True
    )
    
    print(f"Train slices: {len(train_list)}, Val slices: {len(val_list)}")
    return train_list, val_list
