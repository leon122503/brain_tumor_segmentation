import os
import random
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def get_patient_list(data_path):
    """Get a list of all patient folders in the dataset."""
    return [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]


def split_dataset(
    patient_list, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42
):
    """Split the patient list into training, validation, and test sets."""
    # First split into train and temp (validation + test)
    train_patients, temp_patients = train_test_split(
        patient_list, train_size=train_size, random_state=random_state
    )

    # Then split temp into validation and test
    relative_val_size = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients, train_size=relative_val_size, random_state=random_state
    )

    return {"train": train_patients, "validation": val_patients, "test": test_patients}


def save_dataset_split(splits, output_file="dataset_splits.csv"):
    """Save the dataset splits to a CSV file."""
    data = []
    for split_name, patients in splits.items():
        for patient in patients:
            data.append({"patient_id": patient, "split": split_name})

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Dataset splits saved to {output_file}")

    # Print split statistics
    print("\nDataset Split Statistics:")
    for split_name, patients in splits.items():
        print(
            f"{split_name}: {len(patients)} patients ({len(patients)/len(sum(splits.values(), []))*100:.1f}%)"
        )


def normalize_scan(scan):
    """Normalize scan to have zero mean and unit variance."""
    scan = scan.astype(np.float32)
    mean = np.mean(scan)
    std = np.std(scan)
    return (scan - mean) / (std + 1e-8)


def process_patient_data(patient_folder, data_path):
    """Load and preprocess the MRI scans and segmentation for a patient."""
    patient_id = os.path.basename(patient_folder)

    # Load the four MRI modalities
    flair = nib.load(
        os.path.join(data_path, patient_folder, f"{patient_id}_flair.nii")
    ).get_fdata()
    t1 = nib.load(
        os.path.join(data_path, patient_folder, f"{patient_id}_t1.nii")
    ).get_fdata()
    t1ce = nib.load(
        os.path.join(data_path, patient_folder, f"{patient_id}_t1ce.nii")
    ).get_fdata()
    t2 = nib.load(
        os.path.join(data_path, patient_folder, f"{patient_id}_t2.nii")
    ).get_fdata()

    # Load segmentation mask
    seg = nib.load(
        os.path.join(data_path, patient_folder, f"{patient_id}_seg.nii")
    ).get_fdata()

    # Normalize each modality
    flair_norm = normalize_scan(flair)
    t1_norm = normalize_scan(t1)
    t1ce_norm = normalize_scan(t1ce)
    t2_norm = normalize_scan(t2)

    # Stack the modalities to create a 4-channel input
    input_data = np.stack([flair_norm, t1_norm, t1ce_norm, t2_norm], axis=-1)

    # Convert segmentation to one-hot encoding (background, necrotic/non-enhancing, edema, enhancing)
    # 0: Background, 1: Necrotic/Non-enhancing, 2: Edema, 4: Enhancing
    seg_one_hot = np.zeros((*seg.shape, 4), dtype=np.uint8)
    seg_one_hot[..., 0] = (seg == 0).astype(np.uint8)
    seg_one_hot[..., 1] = (seg == 1).astype(np.uint8)
    seg_one_hot[..., 2] = (seg == 2).astype(np.uint8)
    seg_one_hot[..., 3] = (seg == 4).astype(np.uint8)

    return input_data, seg_one_hot


def main():
    # Data path
    data_path = "data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

    # Create output directories if they don't exist
    os.makedirs("processed_data", exist_ok=True)

    # Get list of patient folders
    patient_list = get_patient_list(data_path)
    print(f"Found {len(patient_list)} patients in the dataset.")

    # Split dataset
    splits = split_dataset(patient_list)

    # Save splits to CSV
    save_dataset_split(splits, output_file="dataset_splits.csv")


if __name__ == "__main__":
    main()
