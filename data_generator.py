import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from data_preparation import normalize_scan


class BraTSDataset(Dataset):
    """Dataset for BraTS 2020 dataset."""

    def __init__(
        self,
        data_path,
        split_file,
        split_type="train",
        patch_size=(128, 128, 128),
        augment=False,
        random_seed=42,
        cache_size=10,  # Cache up to 10 patients in memory
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the BraTS dataset
            split_file: CSV file containing dataset splits
            split_type: 'train', 'validation', or 'test'
            patch_size: Size of patches to extract (depth, height, width)
            augment: Whether to apply data augmentation
            random_seed: Random seed for reproducibility
            cache_size: Number of patients to cache in memory
        """
        self.data_path = data_path
        self.patch_size = patch_size
        self.augment = augment
        self.cache_size = cache_size
        self.data_cache = {}  # Patient ID -> (input_data, seg_one_hot)

        # Load the split file
        self.df = pd.read_csv(split_file)
        self.df = self.df[self.df["split"] == split_type]
        self.patient_ids = self.df["patient_id"].tolist()

        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        print(f"Initialized {split_type} dataset with {len(self.patient_ids)} patients")

    def __len__(self):
        """Return the number of patients in the dataset."""
        return len(self.patient_ids)

    def __getitem__(self, index):
        """Get a data sample."""
        patient_id = self.patient_ids[index]

        # Load and preprocess the MRI scans and segmentation
        x, y = self._load_patient_data(patient_id)

        # Extract a random patch
        x_patch, y_patch = self._extract_random_patch(x, y)

        # Apply data augmentation if specified
        if self.augment:
            x_patch, y_patch = self._augment_data(x_patch, y_patch)

        # Convert to PyTorch tensors and adjust dimensions to (C, D, H, W)
        x_tensor = torch.from_numpy(x_patch.copy()).permute(3, 0, 1, 2).float()
        y_tensor = torch.from_numpy(y_patch.copy()).permute(3, 0, 1, 2).float()

        return x_tensor, y_tensor

    def _load_patient_data(self, patient_id):
        """Load and preprocess the MRI scans and segmentation for a patient."""
        # Check if patient data is in cache
        if patient_id in self.data_cache:
            return self.data_cache[patient_id]

        # Load the four MRI modalities
        flair = nib.load(
            os.path.join(self.data_path, patient_id, f"{patient_id}_flair.nii")
        ).get_fdata()
        t1 = nib.load(
            os.path.join(self.data_path, patient_id, f"{patient_id}_t1.nii")
        ).get_fdata()
        t1ce = nib.load(
            os.path.join(self.data_path, patient_id, f"{patient_id}_t1ce.nii")
        ).get_fdata()
        t2 = nib.load(
            os.path.join(self.data_path, patient_id, f"{patient_id}_t2.nii")
        ).get_fdata()

        # Load segmentation mask
        seg = nib.load(
            os.path.join(self.data_path, patient_id, f"{patient_id}_seg.nii")
        ).get_fdata()

        # Normalize each modality
        flair_norm = normalize_scan(flair)
        t1_norm = normalize_scan(t1)
        t1ce_norm = normalize_scan(t1ce)
        t2_norm = normalize_scan(t2)

        # Stack the modalities to create a 4-channel input
        input_data = np.stack([flair_norm, t1_norm, t1ce_norm, t2_norm], axis=-1)

        # Convert segmentation to one-hot encoding
        seg_one_hot = np.zeros((*seg.shape, 4), dtype=np.float32)
        seg_one_hot[..., 0] = (seg == 0).astype(np.float32)
        seg_one_hot[..., 1] = (seg == 1).astype(np.float32)
        seg_one_hot[..., 2] = (seg == 2).astype(np.float32)
        seg_one_hot[..., 3] = (seg == 4).astype(np.float32)

        # Add to cache if we have space and cache is enabled
        if self.cache_size > 0:
            if len(self.data_cache) < self.cache_size:
                # Cache has space
                self.data_cache[patient_id] = (input_data, seg_one_hot)
            elif np.random.random() < 0.1:
                # Cache is full but we'll randomly replace with 10% probability
                cache_keys = list(self.data_cache.keys())
                if len(cache_keys) > 0:  # Safety check
                    key_to_remove = cache_keys[np.random.randint(0, len(cache_keys))]
                    del self.data_cache[key_to_remove]
                    self.data_cache[patient_id] = (input_data, seg_one_hot)

        return input_data, seg_one_hot

    def _extract_random_patch(self, x, y):
        """Extract a random patch from the volume."""
        # Get the original volume shape
        orig_shape = x.shape[:-1]  # Exclude the channel dimension

        # Calculate valid starting indices
        valid_start_idx = [max(0, s - p) for s, p in zip(orig_shape, self.patch_size)]

        # Generate random starting point
        start_idx = [
            np.random.randint(0, v + 1) if v > 0 else 0 for v in valid_start_idx
        ]

        # Extract patches
        x_patch = x[
            start_idx[0] : start_idx[0] + self.patch_size[0],
            start_idx[1] : start_idx[1] + self.patch_size[1],
            start_idx[2] : start_idx[2] + self.patch_size[2],
            :,
        ]

        y_patch = y[
            start_idx[0] : start_idx[0] + self.patch_size[0],
            start_idx[1] : start_idx[1] + self.patch_size[1],
            start_idx[2] : start_idx[2] + self.patch_size[2],
            :,
        ]

        # Pad if necessary
        if x_patch.shape[:-1] != self.patch_size:
            padding = [
                (0, max(0, p - s)) for p, s in zip(self.patch_size, x_patch.shape[:-1])
            ]
            padding.append((0, 0))  # No padding for channels
            x_patch = np.pad(x_patch, padding, mode="constant")
            y_patch = np.pad(y_patch, padding, mode="constant")

        return x_patch, y_patch

    def _augment_data(self, x, y):
        """Apply data augmentation to the patch."""
        # Flip along random axes
        for axis in range(3):
            if np.random.random() > 0.5:
                x = np.flip(x, axis=axis).copy()  # Add copy to avoid negative strides
                y = np.flip(y, axis=axis).copy()

        # Rotate 90 degrees around a random axis
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)  # 1, 2, or 3 times 90 degrees
            axis = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
            x = np.rot90(x, k=k, axes=axis).copy()  # Add copy to avoid negative strides
            y = np.rot90(y, k=k, axes=axis).copy()

        # Add random noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.05, x.shape)
            x = x + noise

        # Apply random intensity shift
        if np.random.random() > 0.7:
            shift = np.random.uniform(-0.1, 0.1, size=4)
            for c in range(4):
                x[..., c] = x[..., c] + shift[c]

        return x, y


def get_dataloader(
    data_path,
    split_file,
    split_type="train",
    batch_size=2,
    patch_size=(128, 128, 128),
    augment=False,
    num_workers=4,
    pin_memory=True,
    random_seed=42,
    cache_size=10,
):
    """
    Create a PyTorch DataLoader for the BraTS dataset.

    Args:
        data_path: Path to the BraTS dataset
        split_file: CSV file containing dataset splits
        split_type: 'train', 'validation', or 'test'
        batch_size: Batch size
        patch_size: Size of patches to extract (depth, height, width)
        augment: Whether to apply data augmentation
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in GPU
        random_seed: Random seed for reproducibility
        cache_size: Number of patients to cache in memory

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    print(f"Creating {split_type} dataloader with {num_workers} workers")

    dataset = BraTSDataset(
        data_path=data_path,
        split_file=split_file,
        split_type=split_type,
        patch_size=patch_size,
        augment=augment,
        random_seed=random_seed,
        cache_size=cache_size,
    )

    # Use persistent workers to keep workers alive between epochs
    persistent_workers = num_workers > 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split_type == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return dataloader
