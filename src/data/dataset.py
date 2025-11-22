"""Custom Dataset class for loading medical imaging data and corresponding segmentations."""
import os
import pandas as pd
from configs.config import CONFIG
from tqdm import tqdm
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split

series_meta = pd.read_csv(os.path.join(CONFIG.base_path, 'train_series_meta.csv'))
labels = pd.read_csv(os.path.join(CONFIG.base_path, 'train_2024.csv'))

seg_path = os.path.join(CONFIG.base_path, 'segmentations')
seg_files = [f for f in os.listdir(seg_path) if f.endswith('.nii')]


print("Extracting 2D slices from 3D volumes")
data_list = []

for seg_file in tqdm(seg_files, desc="Processing volumes"):
    series_id = int(seg_file.split('.')[0])
    patient_row = series_meta[series_meta['series_id'] == series_id]
    
    if not patient_row.empty:
        patient_id = int(patient_row['patient_id'].values[0])
        image_dir = os.path.join(CONFIG.base_path, f'train_images/{patient_id}/{series_id}')
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
                if np.sum(slice_seg > 0) > 100:  # At least 100 pixels with organs
                    data_list.append({
                        'image': image_dir,
                        'seg': seg_file_path,
                        'slice_idx': slice_idx
                    })
        except Exception as e:
            print(f"Error processing {seg_file}: {e}")
            continue

print(f"Total 2D slices extracted: {len(data_list)}")

train_list, val_list = train_test_split(
    data_list,
    test_size=1-CONFIG['train_split'],
    random_state=CONFIG['seed'],
    shuffle=True
)

print(f"Train slices: {len(train_list)}, Val slices: {len(val_list)}")
