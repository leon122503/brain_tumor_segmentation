import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import nibabel as nib
from tqdm import tqdm
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_generator import get_dataloader
from model import get_model
from data_preparation import normalize_scan

def plot_sample_slices(
    image, mask, prediction=None, slice_indices=None, save_path=None
):
    """
    Plot sample slices from the 3D volume along different axes.
    Args:
        image: 3D image volume (D, H, W, C)
        mask: Ground truth segmentation mask (D, H, W)
        prediction: Predicted segmentation mask (D, H, W), optional
        slice_indices: List of indices to plot for each axis, optional
        save_path: Path to save the figure, optional
    """
    depth, height, width, channels = image.shape
    # If no slice indices are provided, use multiple slices along each axis
    if slice_indices is None:
        slice_indices = {
            "axial": [depth // 4, depth // 2, 3 * depth // 4],
            "coronal": [height // 4, height // 2, 3 * height // 4],
            "sagittal": [width // 4, width // 2, 3 * width // 4],
        }
    
    # Create color maps for visualization
    cmap_dict = {
        "image": "gray",
        "mask": matplotlib.colormaps.get_cmap("viridis"),
    }
    
    # Define the number of columns based on the maximum number of slices in any direction
    n_cols = max(len(slice_indices["axial"]), 
                 len(slice_indices["coronal"]), 
                 len(slice_indices["sagittal"]))
    
    # Create a single-row layout for each slice type
    # Calculate the number of rows based on slice types and whether prediction is available
    rows_per_slice_type = 2 if prediction is None else 3  # Image + mask (+ prediction if available)
    n_rows = rows_per_slice_type * 3  # 3 types: axial, coronal, sagittal
    
    # Create a more compact figure with all slots filled
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D
    
    # Hide all axes initially - we'll only show those we use
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    
    # Plot axial slices (top row)
    for i, idx in enumerate(slice_indices["axial"]):
        if i < n_cols:
            # Plot MRI image (FLAIR)
            ax = axes[0, i] if axes.shape[0] > 0 and axes.shape[1] > i else None
            if ax is not None:
                ax.imshow(image[idx, :, :, 0], cmap=cmap_dict["image"])
                ax.set_title(f"Axial Slice {idx}")
                ax.axis("off")
            
            # Plot ground truth mask
            ax = axes[1, i] if axes.shape[0] > 1 and axes.shape[1] > i else None
            if ax is not None:
                ax.imshow(mask[idx, :, :], cmap=cmap_dict["mask"], vmin=0, vmax=3)
                ax.set_title(f"Ground Truth Mask")
                ax.axis("off")
            
            # Plot prediction if available
            if prediction is not None:
                ax = axes[2, i] if axes.shape[0] > 2 and axes.shape[1] > i else None
                if ax is not None:
                    ax.imshow(prediction[idx, :, :], cmap=cmap_dict["mask"], vmin=0, vmax=3)
                    ax.set_title(f"Predicted Mask")
                    ax.axis("off")
    
    # Plot coronal slices (middle row)
    row_offset = 3 if prediction is not None else 2
    for i, idx in enumerate(slice_indices["coronal"]):
        if i < n_cols:
            # Plot MRI image (FLAIR)
            ax = axes[row_offset, i] if axes.shape[0] > row_offset and axes.shape[1] > i else None
            if ax is not None:
                ax.imshow(image[:, idx, :, 0], cmap=cmap_dict["image"])
                ax.set_title(f"Coronal Slice {idx}")
                ax.axis("off")
            
            # Plot ground truth mask
            ax = axes[row_offset + 1, i] if axes.shape[0] > (row_offset + 1) and axes.shape[1] > i else None
            if ax is not None:
                ax.imshow(mask[:, idx, :], cmap=cmap_dict["mask"], vmin=0, vmax=3)
                ax.set_title(f"Ground Truth Mask")
                ax.axis("off")
            
            # Plot prediction if available
            if prediction is not None:
                ax = axes[row_offset + 2, i] if axes.shape[0] > (row_offset + 2) and axes.shape[1] > i else None
                if ax is not None:
                    ax.imshow(prediction[:, idx, :], cmap=cmap_dict["mask"], vmin=0, vmax=3)
                    ax.set_title(f"Predicted Mask")
                    ax.axis("off")
    
    # Plot sagittal slices (bottom row)
    row_offset = 6 if prediction is not None else 4
    for i, idx in enumerate(slice_indices["sagittal"]):
        if i < n_cols:
            # Plot MRI image (FLAIR)
            ax = axes[row_offset, i] if axes.shape[0] > row_offset and axes.shape[1] > i else None
            if ax is not None:
                ax.imshow(image[:, :, idx, 0], cmap=cmap_dict["image"])
                ax.set_title(f"Sagittal Slice {idx}")
                ax.axis("off")
            
            # Plot ground truth mask
            ax = axes[row_offset + 1, i] if axes.shape[0] > (row_offset + 1) and axes.shape[1] > i else None
            if ax is not None:
                ax.imshow(mask[:, :, idx], cmap=cmap_dict["mask"], vmin=0, vmax=3)
                ax.set_title(f"Ground Truth Mask")
                ax.axis("off")
            
            # Plot prediction if available
            if prediction is not None:
                ax = axes[row_offset + 2, i] if axes.shape[0] > (row_offset + 2) and axes.shape[1] > i else None
                if ax is not None:
                    ax.imshow(prediction[:, :, idx], cmap=cmap_dict["mask"], vmin=0, vmax=3)
                    ax.set_title(f"Predicted Mask")
                    ax.axis("off")
    
    # Add legend for tumor types
    class_labels = ['Background', 'Whole Tumor (Edema)', 'Enhancing Tumor', 'Tumor Core (Necrosis)']
    class_colors = [cmap_dict["mask"](i/3) for i in range(4)]
    
    # Create custom legend patches
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=class_colors[i]) for i in range(4)]
    
    # First apply tight layout to get proper spacing for the plots
    plt.tight_layout()
    
    # Then add the legend
    legend = fig.legend(legend_patches, class_labels, loc='lower center', ncol=4, 
                      bbox_to_anchor=(0.5, 0.0), fontsize=20, frameon=True, 
                      facecolor='white', edgecolor='black')
    
    # Make the legend items larger
    for handle in legend.legendHandles:
        handle.set_height(20)
        handle.set_width(20)
    
    # Adjust bottom margin to eliminate white space above legend
    # Use the legend height to calculate the right amount of space
    legend_height = legend.get_window_extent().height/fig.dpi
    fig.subplots_adjust(bottom=legend_height/fig.get_figheight()+0.02)
    
    # Save or show figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def visualize_dataset_examples(data_loader, num_samples=3, output_dir="visualizations"):
    """
    Visualize examples from the dataset.
    Args:
        data_loader: Data loader for the dataset
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples from the data loader
    samples = []
    for inputs, targets in data_loader:
        # Add samples to the list
        for i in range(inputs.shape[0]):
            samples.append((inputs[i].numpy(), targets[i].numpy()))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    # Visualize each sample
    for i, (input_sample, target_sample) in enumerate(samples):
        # Convert from (C, D, H, W) to (D, H, W, C)
        input_sample = np.transpose(input_sample, (1, 2, 3, 0))
        target_sample = np.transpose(target_sample, (1, 2, 3, 0))
        
        # Convert one-hot encoded mask to class indices
        target_mask = np.argmax(target_sample, axis=-1)
        
        # Choose slice indices for visualization
        depth, height, width = input_sample.shape[:3]
        slice_indices = {
            "axial": [depth // 4, depth // 2, 3 * depth // 4],
            "coronal": [height // 4, height // 2, 3 * height // 4],
            "sagittal": [width // 4, width // 2, 3 * width // 4],
        }
        
        # Plot slices
        save_path = os.path.join(output_dir, f"sample_{i+1}.png")
        plot_sample_slices(
            image=input_sample,
            mask=target_mask,
            slice_indices=slice_indices,
            save_path=save_path,
        )
        print(f"Saved visualization for sample {i+1} to {save_path}")

def visualize_predictions(
    model, data_loader, device, num_samples=3, output_dir="visualizations"
):
    """
    Visualize model predictions on the dataset.
    Args:
        model: PyTorch model
        data_loader: Data loader for the dataset
        device: Device to run the model on
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    samples = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move inputs to device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Convert to class probabilities and predicted classes
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Move data back to CPU and convert to numpy
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            # Add samples to the list
            for i in range(inputs_np.shape[0]):
                samples.append((inputs_np[i], targets_np[i], preds_np[i]))
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break
    
    # Visualize each sample
    for i, (input_sample, target_sample, pred_sample) in enumerate(samples):
        # Convert from (C, D, H, W) to (D, H, W, C)
        input_sample = np.transpose(input_sample, (1, 2, 3, 0))
        target_sample = np.transpose(target_sample, (1, 2, 3, 0))
        
        # Convert one-hot encoded mask to class indices
        target_mask = np.argmax(target_sample, axis=-1)
        
        # Choose slice indices for visualization
        depth, height, width = input_sample.shape[:3]
        slice_indices = {
            "axial": [depth // 4, depth // 2, 3 * depth // 4],
            "coronal": [height // 4, height // 2, 3 * height // 4],
            "sagittal": [width // 4, width // 2, 3 * width // 4],
        }
        
        # Plot slices
        save_path = os.path.join(output_dir, f"prediction_{i+1}.png")
        plot_sample_slices(
            image=input_sample,
            mask=target_mask,
            prediction=pred_sample,
            slice_indices=slice_indices,
            save_path=save_path,
        )
        print(f"Saved prediction visualization for sample {i+1} to {save_path}")

def visualize_mri_modalities(data_loader, num_samples=3, output_dir="visualizations/mris"):
    """
    Visualize the 4 different MRI modalities (FLAIR, T1, T1CE, T2) for sample patients.
    Args:
        data_loader: Data loader for the dataset
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples from the data loader
    samples = []
    for inputs, targets in data_loader:
        # Add samples to the list
        for i in range(inputs.shape[0]):
            samples.append((inputs[i].numpy(), targets[i].numpy()))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    # For each modality, create a separate visualization
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
    
    # First, create individual modality visualizations
    for mod_idx, mod_name in enumerate(modality_names):
        # Create a directory for each modality
        mod_dir = os.path.join(output_dir, mod_name.lower())
        os.makedirs(mod_dir, exist_ok=True)
        
        for i, (input_sample, _) in enumerate(samples):
            # Convert from (C, D, H, W) to (D, H, W, C)
            input_sample = np.transpose(input_sample, (1, 2, 3, 0))
            depth, height, width, _ = input_sample.shape
            
            # Create figure for this modality
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            
            # Define slice indices for each view
            slice_indices = {
                "axial": [depth // 4, depth // 2, 3 * depth // 4],
                "coronal": [height // 4, height // 2, 3 * height // 4],
                "sagittal": [width // 4, width // 2, 3 * width // 4],
            }
            
            # Plot axial slices (top row)
            for j, slice_pos in enumerate(slice_indices["axial"]):
                axes[0, j].imshow(input_sample[slice_pos, :, :, mod_idx], cmap='gray')
                axes[0, j].set_title(f'Axial Slice {slice_pos}', fontsize=12)
                axes[0, j].axis('off')
            
            # Plot coronal slices (middle row)
            for j, slice_pos in enumerate(slice_indices["coronal"]):
                axes[1, j].imshow(input_sample[:, slice_pos, :, mod_idx], cmap='gray')
                axes[1, j].set_title(f'Coronal Slice {slice_pos}', fontsize=12)
                axes[1, j].axis('off')
            
            # Plot sagittal slices (bottom row)
            for j, slice_pos in enumerate(slice_indices["sagittal"]):
                axes[2, j].imshow(input_sample[:, :, slice_pos, mod_idx], cmap='gray')
                axes[2, j].set_title(f'Sagittal Slice {slice_pos}', fontsize=12)
                axes[2, j].axis('off')
            
            # Add main title
            fig.suptitle(f'{mod_name} MRI - Sample {i+1}', fontsize=16)
            
            plt.tight_layout()
            save_path = os.path.join(mod_dir, f"sample_{i+1}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved {mod_name} visualization for sample {i+1} to {save_path}")
    
    # Now create a comparison view of all modalities for each sample
    for i, (input_sample, _) in enumerate(samples):
        # Convert from (C, D, H, W) to (D, H, W, C)
        input_sample = np.transpose(input_sample, (1, 2, 3, 0))
        depth, height, width, _ = input_sample.shape
        
        # Create a figure comparing all 4 modalities
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        # Choose a middle slice for each view
        axial_slice = depth // 2
        coronal_slice = height // 2
        sagittal_slice = width // 2
        
        for mod_idx, mod_name in enumerate(modality_names):
            # Axial view (middle of the brain)
            axes[mod_idx, 0].imshow(input_sample[axial_slice, :, :, mod_idx], cmap='gray')
            if mod_idx == 0:
                axes[mod_idx, 0].set_title('Axial View', fontsize=14)
            axes[mod_idx, 0].set_ylabel(mod_name, fontsize=14)
            axes[mod_idx, 0].axis('off')
            
            # Coronal view
            axes[mod_idx, 1].imshow(input_sample[:, coronal_slice, :, mod_idx], cmap='gray')
            if mod_idx == 0:
                axes[mod_idx, 1].set_title('Coronal View', fontsize=14)
            axes[mod_idx, 1].axis('off')
            
            # Sagittal view
            axes[mod_idx, 2].imshow(input_sample[:, :, sagittal_slice, mod_idx], cmap='gray')
            if mod_idx == 0:
                axes[mod_idx, 2].set_title('Sagittal View', fontsize=14)
            axes[mod_idx, 2].axis('off')
        
        # Add main title
        fig.suptitle(f'MRI Modality Comparison - Sample {i+1}', fontsize=16)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"modality_comparison_sample_{i+1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved modality comparison for sample {i+1} to {save_path}")

def visualize_single_slice_with_all_modalities(data_loader, num_samples=3, output_dir="visualizations/mris"):
    """
    Visualize a single slice showing all 4 MRI modalities and the segmentation mask side by side.
    Args:
        data_loader: Data loader for the dataset
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples from the data loader
    samples = []
    for inputs, targets in data_loader:
        # Add samples to the list
        for i in range(inputs.shape[0]):
            samples.append((inputs[i].numpy(), targets[i].numpy()))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    # Define modality names for labels
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
    
    # Create color maps for visualization
    cmap_dict = {
        "image": "gray",
        "mask": matplotlib.colormaps.get_cmap("viridis"),
    }
    
    # Visualize each sample with all modalities in a single row + segmentation
    for i, (input_sample, target_sample) in enumerate(samples):
        # Convert from (C, D, H, W) to (D, H, W, C)
        input_sample = np.transpose(input_sample, (1, 2, 3, 0))
        target_sample = np.transpose(target_sample, (1, 2, 3, 0))
        
        # Convert one-hot encoded mask to class indices
        target_mask = np.argmax(target_sample, axis=-1)
        
        # Get dimensions
        depth, height, width, _ = input_sample.shape
        
        # Create a figure with 1 row and 5 columns (4 modalities + segmentation)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # Choose a middle slice for axial view (can be changed to any desired slice)
        axial_slice = depth // 2
        
        # Plot each modality
        for mod_idx, mod_name in enumerate(modality_names):
            axes[mod_idx].imshow(input_sample[axial_slice, :, :, mod_idx], cmap='gray')
            axes[mod_idx].set_title(f'{mod_name}', fontsize=14)
            axes[mod_idx].axis('off')
        
        # Plot the segmentation mask
        im = axes[4].imshow(target_mask[axial_slice, :, :], cmap=cmap_dict["mask"], vmin=0, vmax=3)
        axes[4].set_title('Segmentation Mask', fontsize=14)
        axes[4].axis('off')
        
        # Add a colorbar for the segmentation mask
        divider = make_axes_locatable(axes[4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        
        # Add legend for tumor types
        class_labels = ['Background', 'Whole Tumor (Edema)', 'Enhancing Tumor', 'Tumor Core (Necrosis)']
        class_colors = [cmap_dict["mask"](i/3) for i in range(4)]
        
        # Create custom legend patches
        legend_patches = [plt.Rectangle((0, 0), 1, 1, color=class_colors[i]) for i in range(4)]
        
        # Add legend below the figure
        fig.legend(legend_patches, class_labels, loc='lower center', ncol=4, 
                   bbox_to_anchor=(0.5, -0.1), fontsize=12, frameon=True,
                   facecolor='white', edgecolor='black')
        
        # Add main title
        fig.suptitle(f'MRI Modalities and Segmentation - Axial Slice {axial_slice} - Sample {i+1}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for the legend
        
        # Save the figure
        save_path = os.path.join(output_dir, f"single_slice_all_modalities_sample_{i+1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved single slice visualization for sample {i+1} to {save_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Visualize BraTS2020 dataset and model predictions"
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
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--visualize_data", action="store_true", help="Visualize dataset examples"
    )
    parser.add_argument(
        "--visualize_predictions",
        action="store_true",
        help="Visualize model predictions",
    )
    parser.add_argument(
        "--visualize_mri_modalities",
        action="store_true",
        help="Visualize the 4 different MRI modalities (FLAIR, T1, T1CE, T2)",
    )
    parser.add_argument(
        "--visualize_single_slice",
        action="store_true",
        help="Visualize a single slice with all MRI modalities and the segmentation mask",
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to visualize"
    )
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
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloader
    data_loader = get_dataloader(
        data_path=args.data_path,
        split_file=args.split_file,
        split_type="test",
        batch_size=1,
        patch_size=tuple(args.patch_size),
        augment=False,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
    )
    
    # Visualize dataset examples
    if args.visualize_data:
        print(f"Visualizing {args.num_samples} dataset examples...")
        visualize_dataset_examples(
            data_loader=data_loader,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, "dataset"),
        )
    
    # Visualize model predictions
    if args.visualize_predictions:
        if not os.path.exists(args.checkpoint):
            print(
                f"Warning: Checkpoint {args.checkpoint} not found. Skipping prediction visualization."
            )
        else:
            print(f"Visualizing {args.num_samples} model predictions...")
            # Load model with the specified type
            model = get_model(
                in_channels=4, out_channels=4, device=device, model_type=args.model_type
            )
            
            # Load checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {args.checkpoint}")
            
            # Visualize predictions
            visualize_predictions(
                model=model,
                data_loader=data_loader,
                device=device,
                num_samples=args.num_samples,
                output_dir=os.path.join(args.output_dir, "predictions"),
            )
    
    # Visualize MRI modalities
    if args.visualize_mri_modalities:
        print(f"Visualizing {args.num_samples} samples of MRI modalities...")
        visualize_mri_modalities(
            data_loader=data_loader,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, "mris"),
        )
    
    # Visualize single slice with all modalities and segmentation
    if args.visualize_single_slice:
        print(f"Visualizing {args.num_samples} samples with all modalities in a single slice...")
        visualize_single_slice_with_all_modalities(
            data_loader=data_loader,
            num_samples=args.num_samples,
            output_dir=os.path.join(args.output_dir, "mris"),
        )
    
    print("Visualization completed!")

if __name__ == "__main__":
    main()