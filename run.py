#!/usr/bin/env python
import os
import sys
import argparse
import subprocess


def run_command(command):
    """Run a command."""
    # Execute the command
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print("Error running command:")
        for line in process.stderr:
            print(line, end="")
        sys.exit(process.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run BraTS2020 segmentation scripts")
    parser.add_argument(
        "command",
        choices=["prepare", "train", "evaluate", "visualize", "all"],
        help="Command to run",
    )

    # Add arguments for prepare data
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        help="Path to BraTS2020 dataset",
    )

    # Add arguments for training
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[64, 64, 64],  # Reduced patch size for faster training
        help="Size of patches to extract",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
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

    # Add arguments for visualization
    parser.add_argument(
        "--visualize_data", action="store_true", help="Visualize dataset examples"
    )
    parser.add_argument(
        "--visualize_predictions",
        action="store_true",
        help="Visualize model predictions",
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to visualize"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    # Run the specified command
    if args.command == "prepare" or args.command == "all":
        run_command(f"python data_preparation.py --data_path {args.data_path}")

    if args.command == "train" or args.command == "all":
        patch_size = " ".join(map(str, args.patch_size))
        run_command(
            f"python train.py --data_path {args.data_path} --patch_size {patch_size} "
            f"--batch_size {args.batch_size} --num_epochs {args.num_epochs} "
            f"--learning_rate {args.learning_rate} --num_workers {args.num_workers} "
            f"--model_type {args.model_type} --cache_size {args.cache_size}"
        )

    if args.command == "evaluate" or args.command == "all":
        patch_size = " ".join(map(str, args.patch_size))
        run_command(
            f"python evaluate.py --data_path {args.data_path} "
            f"--checkpoint checkpoints/{args.model_type}_best_model.pth --batch_size 1 "
            f"--patch_size {patch_size} --num_workers {args.num_workers} "
            f"--model_type {args.model_type} --cache_size {args.cache_size}"
        )

    if args.command == "visualize" or args.command == "all":
        vis_args = ""
        if args.visualize_data or args.command == "all":
            vis_args += " --visualize_data"
        if args.visualize_predictions or args.command == "all":
            vis_args += " --visualize_predictions"

        patch_size = " ".join(map(str, args.patch_size))
        run_command(
            f"python visualize.py --data_path {args.data_path} "
            f"--checkpoint checkpoints/{args.model_type}_best_model.pth "
            f"--num_samples {args.num_samples} {vis_args} "
            f"--patch_size {patch_size} --num_workers {args.num_workers} "
            f"--model_type {args.model_type} --cache_size {args.cache_size}"
        )


if __name__ == "__main__":
    main()
