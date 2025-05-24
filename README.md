# 3D U-Net for Brain Tumor Segmentation on BraTS2020

This project implements a 3D U-Net model for brain tumor segmentation using the BraTS2020 dataset. The model segments brain tumors into three regions: necrotic/non-enhancing tumor, edema, and enhancing tumor.

## Dataset

The BraTS2020 (Brain Tumor Segmentation 2020) dataset consists of multi-modal MRI scans of brain tumor patients. Each patient has four MRI modalities:
- T1-weighted (T1)
- T1-weighted with gadolinium contrast enhancement (T1ce)
- T2-weighted (T2)
- T2 Fluid Attenuated Inversion Recovery (FLAIR)

The dataset also includes manually segmented tumor regions labeled as:
- Class 1: Necrotic and Non-enhancing tumor
- Class 2: Edema
- Class 4: Enhancing tumor
- Class 0: Everything else

**Data Source**: This project uses the BraTS2020 dataset from:
[Kaggle: BraTS2020 Dataset (Training & Validation)](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation/code?datasetId=751906&sortBy=voteCount)

## Project Structure

```
.
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── data_preparation.py         # Script to prepare the dataset
├── data_generator.py           # Data generator for training/evaluation
├── model.py                    # Model architectures
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── visualize.py                # Visualization script
├── run.py                      # Main script to run the pipeline
├── data/                       # Directory for the raw dataset
├── processed_data/             # Directory for processed data
├── checkpoints/                # Directory for model checkpoints
├── results/                    # Directory for evaluation results
└── visualizations/             # Directory for visualizations
```

## Installation

1. Clone this repository
2. Create and activate a conda environment:
```bash
conda create -n cosc428 python=3.8
conda activate cosc428
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Project

Always activate the conda environment before running any scripts:
```bash
conda activate cosc428
```

### Using the Run Script (Recommended)

The easiest way to run the pipeline is to use the provided run script:

```bash
# Run the complete pipeline (prepare data, train, evaluate, visualize)
python run.py all

# Prepare the data
python run.py prepare --data_path "path/to/BraTS2020_data"

# Train the model with custom settings
python run.py train --batch_size 4 --num_epochs 50 --model_type unet

# Evaluate the model
python run.py evaluate --model_type unet

# Visualize the results
python run.py visualize --num_samples 5 --visualize_data --visualize_predictions
```

### Running Individual Scripts

#### 1. Data Preparation

```bash
# Basic usage
python data_preparation.py

# Custom data path
python data_preparation.py --data_path "path/to/BraTS2020_data"
```

This script:
- Gets a list of all patient folders in the dataset
- Splits the patients into training (70%), validation (20%), and test (10%) sets
- Saves the split information to `dataset_splits.csv`

#### 2. Model Training

```bash
# Train with default parameters
python train.py

# Train with custom parameters
python train.py --patch_size 128 128 128 --batch_size 2 --num_epochs 100 --learning_rate 0.0001 --model_type attention_residual_unet --cache_size 5
```

Training parameters:
- `--data_path`: Path to BraTS2020 dataset
- `--patch_size`: Size of 3D patches to extract (default: 64 64 64)
- `--batch_size`: Batch size (default: 1)
- `--num_epochs`: Maximum number of epochs (default: 10)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--model_type`: Model architecture (options: unet, attention_residual_unet, depthwise_unet)
- `--cache_size`: Number of patients to cache in memory (default: 0)

#### 3. Model Evaluation

```bash
# Evaluate with default settings
python evaluate.py

# Evaluate with custom settings
python evaluate.py --checkpoint checkpoints/unet_best_model.pth --patch_size 128 128 128 --model_type unet
```

Evaluation parameters:
- `--data_path`: Path to BraTS2020 dataset
- `--checkpoint`: Path to model checkpoint (default: checkpoints/unet_best_model.pth)
- `--patch_size`: Size of 3D patches (default: 64 64 64)
- `--model_type`: Model architecture (default: unet)

The evaluation calculates Dice scores, precision, and recall for:
- No tumor (background)
- Whole tumor (all tumor classes)
- Tumor core (necrotic/non-enhancing + enhancing)
- Enhancing tumor

#### 4. Visualization

```bash
# Visualize only dataset examples
python visualize.py --visualize_data --num_samples 3

# Visualize model predictions
python visualize.py --visualize_predictions --num_samples 5 --checkpoint checkpoints/unet_best_model.pth

# Visualize all modalities
python visualize.py --visualize_modalities
```

Visualization parameters:
- `--data_path`: Path to BraTS2020 dataset
- `--checkpoint`: Path to model checkpoint
- `--num_samples`: Number of samples to visualize (default: 3)
- `--visualize_data`: Show dataset examples
- `--visualize_predictions`: Show model predictions
- `--visualize_modalities`: Show all MRI modalities

## Model Architecture

This project implements three model architectures:
1. **Standard 3D U-Net**: Basic U-Net with 3D convolutions
2. **Attention Residual U-Net**: U-Net with attention gates and residual connections
3. **Depthwise U-Net**: U-Net using depthwise separable convolutions for efficiency

All models are trained using a combination of Dice loss and cross-entropy loss.

## Example Results

Segmentation results are saved in the `visualizations` directory, showing:
- Original MRI scans (FLAIR, T1, T1ce, T2)
- Ground truth segmentation masks
- Predicted segmentation masks

The model's performance is evaluated on the test set and results are saved in the `results` directory.

## License

This project is licensed under the MIT License. 