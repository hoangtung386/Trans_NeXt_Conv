"""Data preprocessing: RSNA 2D dataset with DICOM loading."""

import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class RSNA2DDataset(Dataset):
    """Dataset for loading 2D slices from 3D CT volumes.

    Loads DICOM series, extracts individual slices, applies intensity
    normalization and optional foreground cropping.
    """

    def __init__(self, data_list, transforms=None, spatial_size=(256, 256)):
        self.data_list = data_list
        self.transforms = transforms
        self.spatial_size = spatial_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        image_dir = data_dict["image"]
        seg_path = data_dict["seg"]
        slice_idx = data_dict["slice_idx"]

        # Load image (DICOM series)
        image_3d = self._load_image(image_dir)

        # Load segmentation (NIfTI)
        seg_nii = nib.load(seg_path)
        seg_3d = seg_nii.get_fdata()

        # Convert to numpy if torch tensor
        if isinstance(image_3d, torch.Tensor):
            image_3d = image_3d.numpy()
        if isinstance(seg_3d, torch.Tensor):
            seg_3d = seg_3d.numpy()

        # Extract 2D slice
        image_2d = self._extract_slice(image_3d, slice_idx)
        seg_2d = self._extract_slice(seg_3d, slice_idx)

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
            image_2d, seg_2d = self._apply_transforms(image_2d, seg_2d)

        # Add channel dimension (C, H, W)
        image_2d = image_2d[np.newaxis, ...].astype(np.float32)
        seg_2d = seg_2d[np.newaxis, ...].astype(np.int64)

        # Convert to torch tensors
        image_2d = torch.from_numpy(image_2d)
        seg_2d = torch.from_numpy(seg_2d)

        return {"image": image_2d, "seg": seg_2d}

    def _load_image(self, image_dir):
        """Load 3D image from DICOM directory or NIfTI file."""
        if not os.path.isdir(image_dir):
            nii = nib.load(image_dir)
            return nii.get_fdata()

        import pydicom

        dicom_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(".dcm")
        ]

        if not dicom_files:
            nii = nib.load(image_dir)
            return nii.get_fdata()

        # Sort by InstanceNumber
        def get_instance_number(filepath):
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                return int(dcm.InstanceNumber)
            except (AttributeError, ValueError):
                return 0

        dicom_files.sort(key=get_instance_number)

        # Load slices
        slices = []
        for f in dicom_files:
            dcm = pydicom.dcmread(f)
            pixel_array = dcm.pixel_array.astype(np.float32)

            if hasattr(dcm, "RescaleIntercept") and hasattr(
                dcm, "RescaleSlope"
            ):
                intercept = dcm.RescaleIntercept
                slope = dcm.RescaleSlope
                pixel_array = pixel_array * slope + intercept

            slices.append(pixel_array)

        return np.stack(slices, axis=-1)

    @staticmethod
    def _extract_slice(volume, slice_idx):
        """Extract 2D slice from 3D/4D volume."""
        if len(volume.shape) == 4:
            return volume[0, :, :, slice_idx]
        return volume[:, :, slice_idx]

    def _apply_transforms(self, image_2d, seg_2d):
        """Apply intensity normalization, cropping, and resizing."""
        from skimage.transform import resize

        # Intensity scaling
        image_2d = np.clip(image_2d, -125, 275)
        image_2d = (image_2d + 125) / 400.0

        # Crop foreground
        mask = image_2d > 0.1
        if mask.sum() > 0:
            coords = np.argwhere(mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            pad = 10
            y_min = max(0, y_min - pad)
            x_min = max(0, x_min - pad)
            y_max = min(image_2d.shape[0], y_max + pad)
            x_max = min(image_2d.shape[1], x_max + pad)

            image_2d = image_2d[y_min:y_max, x_min:x_max]
            seg_2d = seg_2d[y_min:y_max, x_min:x_max]

        # Resize to fixed size
        image_2d = resize(
            image_2d, self.spatial_size, order=1, preserve_range=True,
        )
        seg_2d = resize(
            seg_2d, self.spatial_size, order=0, preserve_range=True,
        )

        return image_2d, seg_2d
