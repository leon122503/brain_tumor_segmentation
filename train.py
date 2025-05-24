import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging  # Add logging importlr
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_generator import get_dataloader
from model import get_model, combined_loss
from data_preparation import split_dataset, get_patient_list, save_dataset_split


def dice_coefficient_torch(y_pred, y_true, smooth=1e-8):
    """
    Calculate Dice coefficient using PyTorch tensors.
    Assumes y_pred are class indices (argmax) and y_true are class indices.
    Calculates Dice for the whole tumor region (class > 0).

    Args:
        y_pred: Predicted segmentation map (N, D, H, W)
        y_true: Ground truth segmentation map (N, D, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient for the whole tumor
    """
    y_true_f = (y_true > 0).float().view(y_true.shape[0], -1)
    y_pred_f = (y_pred > 0).float().view(y_pred.shape[0], -1)
    intersection = torch.sum(y_true_f * y_pred_f, dim=1)
    score = (2.0 * intersection + smooth) / (torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1) + smooth)
    return score.mean() # Return average Dice over the batch


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs=100,
    patience=10,
    save_path="checkpoints",
    model_type="unet",
):
    """
    Train the model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        save_path: Directory to save model checkpoints
        model_type: Type of model being trained

    Returns:
        Trained model and training history
    """
    # Set up logging
    log_file = os.path.join(save_path, "training_log.txt")
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_path, "logs"))

    # Initialize variables for early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Using batch size: {train_loader.batch_size}")
    print(f"Using patch size: {train_loader.dataset.patch_size}")

    logging.info(f"Starting training for {num_epochs} epochs...")
    logging.info(f"Training on {len(train_loader.dataset)} samples")
    logging.info(f"Validating on {len(val_loader.dataset)} samples")
    logging.info(f"Using batch size: {train_loader.batch_size}")
    logging.info(f"Using patch size: {train_loader.dataset.patch_size}")

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        batch_count = 0

        print(f"\nEpoch {epoch+1}/{num_epochs} [Training]")
        logging.info(f"\nEpoch {epoch+1}/{num_epochs} [Training]")

        # Setup progress bar with time info
        train_progress = tqdm(train_loader, desc=f"Training", leave=True)

        # Track and report time for data loading and processing
        data_time = 0
        process_time = 0

        # Also log periodically to file
        log_interval = 5

        for i, (inputs, targets) in enumerate(train_progress):
            batch_start = time.time()

            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            data_time += time.time() - batch_start
            forward_start = time.time()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate Dice score using the correct function
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1) # Get predicted class indices
                target_classes = torch.argmax(targets, dim=1) # Get ground truth class indices
                dice = dice_coefficient_torch(preds, target_classes)

            process_time += time.time() - forward_start

            # Update statistics
            train_loss += loss.item()
            train_dice += dice.item() # Accumulate the correctly calculated Dice
            batch_count += 1

            # Update progress bar every batch
            train_progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "dice": f"{dice.item():.4f}",
                    "data_time": f"{data_time/batch_count:.3f}s",
                    "process_time": f"{process_time/batch_count:.3f}s",
                }
            )

            # Also log to file periodically
            if i % log_interval == 0:
                avg_loss = train_loss / batch_count if batch_count > 0 else 0
                avg_dice = train_dice / batch_count if batch_count > 0 else 0
                batch_data_time = data_time / batch_count if batch_count > 0 else 0
                batch_process_time = (
                    process_time / batch_count if batch_count > 0 else 0
                )

                progress_msg = (
                    f"Batch {i}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Dice: {dice.item():.4f}, "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"Avg Dice: {avg_dice:.4f}, "
                    f"Data time: {batch_data_time:.3f}s, "
                    f"Process time: {batch_process_time:.3f}s"
                )
                logging.info(progress_msg)

        # Calculate average training loss and Dice score
        avg_train_loss = train_loss / batch_count
        avg_train_dice = train_dice / batch_count

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        batch_count = 0

        print(f"\nEpoch {epoch+1}/{num_epochs} [Validation]")
        logging.info(f"\nEpoch {epoch+1}/{num_epochs} [Validation]")

        # Setup progress bar
        val_progress = tqdm(val_loader, desc=f"Validation", leave=True)

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_progress):
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = criterion(outputs, targets)

                # Calculate Dice score using the correct function
                preds = torch.argmax(outputs, dim=1) # Get predicted class indices
                target_classes = torch.argmax(targets, dim=1) # Get ground truth class indices
                dice = dice_coefficient_torch(preds, target_classes)

                # Update statistics
                val_loss += loss.item()
                val_dice += dice.item() # Accumulate the correctly calculated Dice
                batch_count += 1

                # Update progress bar
                val_progress.set_postfix(
                    {"loss": f"{loss.item():.4f}", "dice": f"{dice.item():.4f}"}
                )

                # Also log to file periodically
                if i % log_interval == 0:
                    progress_msg = (
                        f"Validation Batch {i}/{len(val_loader)}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Dice: {dice.item():.4f}"
                    )
                    logging.info(progress_msg)        # Calculate average validation loss and Dice score
        avg_val_loss = val_loss / batch_count
        avg_val_dice = val_dice / batch_count # Now this uses the correct Dice calculation
        
        # Step the learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_dice"].append(avg_train_dice)
        history["val_dice"].append(avg_val_dice)

        # Write to TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Dice/train", avg_train_dice, epoch)
        writer.add_scalar("Dice/val", avg_val_dice, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)        # Print statistics
        time_elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {time_elapsed:.1f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        print(f"Current Learning Rate: {current_lr:.6f}")

        logging.info(f"\nEpoch {epoch+1}/{num_epochs} completed in {time_elapsed:.1f}s")
        logging.info(
            f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}"
        )
        logging.info(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        logging.info(f"Current Learning Rate: {current_lr:.6f}")

        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save the model
            model_filename = f"{model_type}_best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_dice": avg_val_dice,
                },
                os.path.join(save_path, model_filename),
            )

            print(
                f"New best model saved as {model_filename} with validation loss: {avg_val_loss:.4f}"
            )
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        print("-" * 60)

    # Close TensorBoard writer
    writer.close()

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_path, "training_history.csv"), index=False)

    # Load best model
    checkpoint = torch.load(os.path.join(save_path, f"{model_type}_best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, history


def setup_dataloaders(
    data_path,
    patch_size=(32, 32, 32),
    batch_size=2,
    num_workers=4,
    split_file="dataset_splits.csv",
    cache_size=0,
):
    """
    Set up the data loaders for training, validation, and testing.

    Args:
        data_path: Path to the BraTS dataset
        patch_size: Size of patches to extract
        batch_size: Batch size
        num_workers: Number of worker processes
        split_file: CSV file containing dataset splits
        cache_size: Number of patients to cache in memory (0 to disable caching)

    Returns:
        Dictionary containing the data loaders
    """
    # Create or load dataset splits
    if not os.path.exists(split_file):
        print(f"Split file {split_file} not found. Creating new splits...")
        patient_list = get_patient_list(data_path)
        splits = split_dataset(patient_list)
        save_dataset_split(splits, output_file=split_file)
    print("Creating train loader...")
    # Create dataloaders
    train_loader = get_dataloader(
        data_path=data_path,
        split_file=split_file,
        split_type="train",
        batch_size=batch_size,
        patch_size=patch_size,
        augment=True,
        num_workers=num_workers,
        cache_size=cache_size,
    )
    print("Creating validation loader...")
    val_loader = get_dataloader(
        data_path=data_path,
        split_file=split_file,
        split_type="validation",
        batch_size=batch_size,
        patch_size=patch_size,
        augment=False,
        num_workers=num_workers,
        cache_size=cache_size,
    )
    print("Creating test loader...")
    test_loader = get_dataloader(
        data_path=data_path,
        split_file=split_file,
        split_type="test",
        batch_size=batch_size,
        patch_size=patch_size,
        augment=False,
        num_workers=num_workers,
        cache_size=cache_size,
    )

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    return loaders


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net for BraTS2020 segmentation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        help="Path to BraTS2020 dataset",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="Size of patches to extract",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of worker processes"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to train on (cuda or cpu)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unet",
        choices=["unet", "attention_residual_unet", "depthwise_unet"],
        help="Type of model to use: unet, attention_residual_unet, or depthwise_unet",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=0,
        help="Number of patients to cache in memory (0 to disable caching)",
    )
    args = parser.parse_args()

    # Check device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Patch size: {args.patch_size}")
    print(f"Model type: {args.model_type}")

    # Setup dataloaders
    loaders = setup_dataloaders(
        data_path=args.data_path,
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
    )

    print("Creating model...")
    # Create model with specified type
    model = get_model(
        in_channels=4, out_channels=4, device=device, model_type=args.model_type
    )    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,  # Reduce LR by half when plateau is reached
        patience=10,  # Number of epochs with no improvement after which LR will be reduced
        verbose=True,
        min_lr=1e-6  # Minimum LR that the scheduler will decay to
    )
    
    # Set loss function
    criterion = combined_loss
    print("Training model...")
    # Train the model
    trained_model, history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        patience=args.patience,
        save_path=args.checkpoint_dir,
        model_type=args.model_type,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
