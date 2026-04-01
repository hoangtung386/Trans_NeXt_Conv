"""Dataset preparation: scan DICOM directories and create sample lists."""

import os

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare_data(config):
    """Prepare data lists for training and validation.

    Scans the directory structure and creates a list of 2D slice samples
    from 3D CT volumes with corresponding segmentation masks.

    Args:
        config: Configuration dict or object.

    Returns:
        Tuple of (train_list, val_list).
    """
    if isinstance(config, dict):
        base_path = config["base_path"]
        train_split = config["train_split"]
        seed = config["seed"]
        min_organ_ratio = config.get("min_organ_ratio", 0.01)
    else:
        base_path = config.base_path
        train_split = config.train_split
        seed = config.seed
        min_organ_ratio = getattr(config, "min_organ_ratio", 0.01)

    csv_meta_path = os.path.join(base_path, "train_series_meta.csv")
    csv_labels_path = os.path.join(base_path, "train_2024.csv")

    if not os.path.exists(csv_meta_path) or not os.path.exists(
        csv_labels_path
    ):
        print(f"Warning: CSV files not found at {base_path}")
        return [], []

    series_meta = pd.read_csv(csv_meta_path)

    seg_path = os.path.join(base_path, "segmentations")
    if not os.path.exists(seg_path):
        print(f"Warning: Segmentation folder not found at {seg_path}")
        return [], []

    seg_files = [f for f in os.listdir(seg_path) if f.endswith(".nii")]

    print("Extracting 2D slices from 3D volumes")
    data_list = []

    for seg_file in tqdm(seg_files, desc="Processing volumes"):
        try:
            series_id = int(seg_file.split(".")[0])
        except ValueError:
            continue

        patient_row = series_meta[series_meta["series_id"] == series_id]

        if patient_row.empty:
            continue

        patient_id = int(patient_row["patient_id"].values[0])
        image_dir = os.path.join(
            base_path, f"train_images/{patient_id}/{series_id}",
        )
        seg_file_path = os.path.join(seg_path, seg_file)

        try:
            seg_nii = nib.load(seg_file_path)
            seg_data = seg_nii.get_fdata()

            num_slices = seg_data.shape[2]
            for slice_idx in range(num_slices):
                slice_seg = seg_data[:, :, slice_idx]
                total_pixels = slice_seg.shape[0] * slice_seg.shape[1]
                if np.sum(slice_seg > 0) > total_pixels * min_organ_ratio:
                    data_list.append({
                        "image": image_dir,
                        "seg": seg_file_path,
                        "slice_idx": slice_idx,
                    })
        except Exception as e:
            print(f"Error processing {seg_file}: {e}")
            continue

    print(f"Total 2D slices extracted: {len(data_list)}")

    if len(data_list) == 0:
        return [], []

    train_list, val_list = train_test_split(
        data_list,
        test_size=1 - train_split,
        random_state=seed,
        shuffle=True,
    )

    print(f"Train slices: {len(train_list)}, Val slices: {len(val_list)}")
    return train_list, val_list
