import os
import argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from data_generator import get_dataloader
from model import get_model


def dice_coefficient(y_true, y_pred):
    """
    Calculate Dice coefficient.

    Args:
        y_true: Ground truth segmentation mask
        y_pred: Predicted segmentation mask

    Returns:
        Dice coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-8)


def evaluate_model(model, data_loader, device, output_dir="results"):
    """
    Evaluate the model on the test set.

    Args:
        model: PyTorch model
        data_loader: Test data loader
        device: Device to evaluate on
        output_dir: Directory to save results

    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    results = {
        "patient_id": [],
        "dice_no_tumor": [],
        "dice_whole": [],
        "dice_core": [],
        "dice_enhancing": [],
        "precision_no_tumor": [],
        "precision_whole": [],
        "precision_core": [],
        "precision_enhancing": [],
        "recall_no_tumor": [],
        "recall_whole": [],
        "recall_core": [],
        "recall_enhancing": [],
    }

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Convert to probabilities and get predicted class
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Convert targets to class indices
            target_classes = torch.argmax(targets, dim=1)

            # Move data to CPU for numpy operations
            preds = preds.cpu().numpy()
            target_classes = target_classes.cpu().numpy()

            # Process each sample in the batch
            for j in range(preds.shape[0]):
                patient_id = f"sample_{i*data_loader.batch_size + j}"
                results["patient_id"].append(patient_id)

                # Calculate Dice scores
                # No tumor (class 0)
                no_tumor_pred = (preds[j] == 0).astype(np.uint8)
                no_tumor_true = (target_classes[j] == 0).astype(np.uint8)
                dice_no_tumor = dice_coefficient(no_tumor_true, no_tumor_pred)

                # Whole tumor (all classes > 0)
                whole_pred = (preds[j] > 0).astype(np.uint8)
                whole_true = (target_classes[j] > 0).astype(np.uint8)
                dice_whole = dice_coefficient(whole_true, whole_pred)

                # Tumor core (classes 1 and 3)
                core_pred = ((preds[j] == 1) | (preds[j] == 3)).astype(np.uint8)
                core_true = (
                    (target_classes[j] == 1) | (target_classes[j] == 3)
                ).astype(np.uint8)
                dice_core = dice_coefficient(core_true, core_pred)

                # Enhancing tumor (class 3)
                enhancing_pred = (preds[j] == 3).astype(np.uint8)
                enhancing_true = (target_classes[j] == 3).astype(np.uint8)
                dice_enhancing = dice_coefficient(enhancing_true, enhancing_pred)

                # Calculate precision and recall
                # No tumor (class 0)
                precision_no_tumor = precision_score(
                    no_tumor_true.flatten(), no_tumor_pred.flatten(), zero_division=0
                )
                recall_no_tumor = recall_score(
                    no_tumor_true.flatten(), no_tumor_pred.flatten(), zero_division=0
                )

                # Whole tumor
                precision_whole = precision_score(
                    whole_true.flatten(), whole_pred.flatten(), zero_division=0
                )
                recall_whole = recall_score(
                    whole_true.flatten(), whole_pred.flatten(), zero_division=0
                )

                # Tumor core
                precision_core = precision_score(
                    core_true.flatten(), core_pred.flatten(), zero_division=0
                )
                recall_core = recall_score(
                    core_true.flatten(), core_pred.flatten(), zero_division=0
                )

                # Enhancing tumor
                precision_enhancing = precision_score(
                    enhancing_true.flatten(), enhancing_pred.flatten(), zero_division=0
                )
                recall_enhancing = recall_score(
                    enhancing_true.flatten(), enhancing_pred.flatten(), zero_division=0
                )

                # Store results
                results["dice_no_tumor"].append(dice_no_tumor)
                results["dice_whole"].append(dice_whole)
                results["dice_core"].append(dice_core)
                results["dice_enhancing"].append(dice_enhancing)
                results["precision_no_tumor"].append(precision_no_tumor)
                results["precision_whole"].append(precision_whole)
                results["precision_core"].append(precision_core)
                results["precision_enhancing"].append(precision_enhancing)
                results["recall_no_tumor"].append(recall_no_tumor)
                results["recall_whole"].append(recall_whole)
                results["recall_core"].append(recall_core)
                results["recall_enhancing"].append(recall_enhancing)

    # Calculate average metrics
    avg_results = {
        "dice_no_tumor": np.mean(results["dice_no_tumor"]),
        "dice_whole": np.mean(results["dice_whole"]),
        "dice_core": np.mean(results["dice_core"]),
        "dice_enhancing": np.mean(results["dice_enhancing"]),
        "precision_no_tumor": np.mean(results["precision_no_tumor"]),
        "precision_whole": np.mean(results["precision_whole"]),
        "precision_core": np.mean(results["precision_core"]),
        "precision_enhancing": np.mean(results["precision_enhancing"]),
        "recall_no_tumor": np.mean(results["recall_no_tumor"]),
        "recall_whole": np.mean(results["recall_whole"]),
        "recall_core": np.mean(results["recall_core"]),
        "recall_enhancing": np.mean(results["recall_enhancing"]),
    }

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)

    # Print average metrics
    print("\nEvaluation Results:")
    print(f"Dice No Tumor: {avg_results['dice_no_tumor']:.4f}")
    print(f"Dice Whole Tumor: {avg_results['dice_whole']:.4f}")
    print(f"Dice Tumor Core: {avg_results['dice_core']:.4f}")
    print(f"Dice Enhancing Tumor: {avg_results['dice_enhancing']:.4f}")
    print(f"Precision No Tumor: {avg_results['precision_no_tumor']:.4f}")
    print(f"Precision Whole Tumor: {avg_results['precision_whole']:.4f}")
    print(f"Precision Tumor Core: {avg_results['precision_core']:.4f}")
    print(f"Precision Enhancing Tumor: {avg_results['precision_enhancing']:.4f}")
    print(f"Recall No Tumor: {avg_results['recall_no_tumor']:.4f}")
    print(f"Recall Whole Tumor: {avg_results['recall_whole']:.4f}")
    print(f"Recall Tumor Core: {avg_results['recall_core']:.4f}")
    print(f"Recall Enhancing Tumor: {avg_results['recall_enhancing']:.4f}")

    return avg_results


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate 3D U-Net for BraTS2020 segmentation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        help="Path to BraTS2020 dataset",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="dataset_splits.csv",
        help="Path to dataset splits CSV file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,  # Will be set dynamically based on model_type
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Size of patches to extract",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to evaluate on (cuda or cpu)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unet",
        choices=["unet", "attention_residual_unet", "depthwise_unet"],
        help="Type of model to use: unet, attention_residual_unet, or depthwise_unet",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of worker processes"
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=0,
        help="Number of patients to cache in memory (0 to disable caching)",
    )
    args = parser.parse_args()

    # Set default checkpoint path if not provided
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.model_type}_best_model.pth"

    # Check device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")

    # Create test dataloader
    test_loader = get_dataloader(
        data_path=args.data_path,
        split_file=args.split_file,
        split_type="test",
        batch_size=args.batch_size,
        patch_size=tuple(args.patch_size),
        augment=False,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
    )

    # Load model with the specified type
    model = get_model(
        in_channels=4, out_channels=4, device=device, model_type=args.model_type
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model from {args.checkpoint}")
    print(
        f"Validation loss: {checkpoint['val_loss']:.4f}, Validation Dice: {checkpoint['val_dice']:.4f}"
    )

    # Evaluate model
    results = evaluate_model(
        model=model, data_loader=test_loader, device=device, output_dir=args.output_dir
    )

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
