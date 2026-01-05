# Running Feature Importance Analysis

This guide shows how to run feature importance analysis using the provided shell scripts.

## Quick Start

The repository includes two shell scripts that automatically configure GPU usage and run the analysis:

### 1. Single Model Analysis

```bash
# Run with default settings (ImprovedCNN1D, data in ./data)
./run_analyze_features.sh

# Specify a different model
./run_analyze_features.sh --model ResNet1D

# Use a pretrained model
./run_analyze_features.sh --model InceptionTime --pretrained models/inception_best.pth

# Skip temporal analysis for faster results
./run_analyze_features.sh --model ROCKET --no-temporal
```

### 2. K-Fold Cross-Validation Analysis (Recommended)

```bash
# Run with default settings (5 folds, ImprovedCNN1D)
./run_analyze_features_kfold.sh

# Use a different model
./run_analyze_features_kfold.sh --model InceptionTime

# Use fewer folds for faster analysis
./run_analyze_features_kfold.sh --n-folds 3

# Don't train models if missing (faster, but requires pretrained models)
./run_analyze_features_kfold.sh --no-train
```

## GPU Configuration

Both scripts are preconfigured to use **GPUs 4, 5, and 7**:

```bash
export CUDA_VISIBLE_DEVICES=4,5,7
```

To change which GPUs are used, edit the scripts and modify the `CUDA_VISIBLE_DEVICES` line.

## Command-Line Options

### `run_analyze_features.sh` Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | choice | ImprovedCNN1D | Model to analyze: ImprovedCNN1D, ResNet1D, InceptionTime, ROCKET |
| `--data-root` | string | data | Directory containing POSITIVE/ and CONTROL/ subdirectories |
| `--pretrained` | string | None | Path to pretrained model weights (.pth file) |
| `--save-dir` | string | feature_analysis | Directory to save results |
| `--n-repeats` | int | 10 | Number of permutation repeats per feature |
| `--no-temporal` | flag | False | Skip temporal importance analysis (faster) |
| `--batch-size` | int | 32 | Batch size for data loading |
| `--epochs` | int | 30 | Training epochs if no pretrained model |
| `--seed` | int | 42 | Random seed for reproducibility |

### `run_analyze_features_kfold.sh` Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | choice | ImprovedCNN1D | Model to analyze: ImprovedCNN1D, ResNet1D, InceptionTime, ROCKET |
| `--data-root` | string | data | Directory containing POSITIVE/ and CONTROL/ subdirectories |
| `--save-dir` | string | feature_analysis_kfold | Directory to save results and models |
| `--n-folds` | int | 5 | Number of cross-validation folds |
| `--n-repeats` | int | 5 | Permutation repeats per feature (per fold) |
| `--no-train` | flag | False | Error if models missing instead of training |
| `--batch-size` | int | 32 | Batch size for data loading |
| `--epochs` | int | 50 | Training epochs if models need training |
| `--seed` | int | 42 | Random seed for reproducibility |

## Usage Examples

### Example 1: Quick Analysis with Pretrained Model

```bash
# Analyze a pretrained ResNet1D model
./run_analyze_features.sh \
    --model ResNet1D \
    --pretrained models/resnet_best.pth \
    --n-repeats 20 \
    --save-dir results/resnet_analysis
```

### Example 2: Fast Analysis (Skip Temporal)

```bash
# Quick permutation importance only
./run_analyze_features.sh \
    --model ROCKET \
    --no-temporal \
    --n-repeats 5
```

### Example 3: K-Fold Analysis with Custom Settings

```bash
# 3-fold analysis with more permutations
./run_analyze_features_kfold.sh \
    --model InceptionTime \
    --n-folds 3 \
    --n-repeats 10 \
    --epochs 100 \
    --save-dir results/inception_kfold
```

### Example 4: Analyze All Models

```bash
# Create a script to analyze all models
for model in ImprovedCNN1D ResNet1D InceptionTime ROCKET; do
    echo "Analyzing $model..."
    ./run_analyze_features_kfold.sh \
        --model $model \
        --save-dir results/${model}_kfold
done
```

### Example 5: Use Different Data Location

```bash
# Analyze data from a different directory
./run_analyze_features.sh \
    --data-root /path/to/other/data \
    --model ImprovedCNN1D
```

## Output Files

### Single Model Analysis (`feature_analysis/`)

- `{model}_permutation_importance.png` - Bar chart of top features
- `{model}_permutation_importance.csv` - Full results with all features
- `{model}_temporal_importance.png` - Heatmap of feature importance over time
- `{model}_temporal_importance_summary.csv` - Temporal importance statistics
- `{model}_temporal_importance.npy` - Full temporal importance matrix
- `{model}_trained.pth` - Trained model (if trained from scratch)

### K-Fold Analysis (`feature_analysis_kfold/`)

- `{model}_kfold_importance.png` - Aggregated importance with error bars
- `{model}_kfold_importance.csv` - Mean importance across folds
- `{model}_per_fold_importance.csv` - Importance for each fold separately
- `{model}_stability.csv` - Feature stability scores
- `{model}_fold{i}.pth` - Trained model for each fold

## Running Directly (Without Shell Scripts)

You can also run the Python scripts directly:

```bash
cd examples

# Set GPUs manually
export CUDA_VISIBLE_DEVICES=4,5,7

# Run Python script
python analyze_features.py --model ResNet1D --n-repeats 20

# Or k-fold version
python analyze_features_kfold.py --model InceptionTime --n-folds 5
```

## Troubleshooting

### Script Not Executable

```bash
chmod +x run_analyze_features.sh run_analyze_features_kfold.sh
```

### CUDA Out of Memory

Reduce batch size:
```bash
./run_analyze_features.sh --batch-size 16
```

Or use fewer permutation repeats:
```bash
./run_analyze_features.sh --n-repeats 5
```

### Models Not Found (K-Fold)

Either train them:
```bash
./run_analyze_features_kfold.sh  # Will auto-train
```

Or run training first:
```bash
cd examples
python train_kfold.py
```

### Slow Analysis

Skip temporal importance:
```bash
./run_analyze_features.sh --no-temporal
```

Use fewer folds:
```bash
./run_analyze_features_kfold.sh --n-folds 3
```

Use fewer permutation repeats:
```bash
./run_analyze_features_kfold.sh --n-repeats 3
```

## Advanced: Parallel GPU Execution

To run multiple models in parallel on different GPUs:

```bash
# Run different models on different GPUs
CUDA_VISIBLE_DEVICES=4 python examples/analyze_features.py --model ImprovedCNN1D &
CUDA_VISIBLE_DEVICES=5 python examples/analyze_features.py --model ResNet1D &
CUDA_VISIBLE_DEVICES=7 python examples/analyze_features.py --model InceptionTime &
wait
```

## Getting Help

```bash
# View all available options
./run_analyze_features.sh --help
./run_analyze_features_kfold.sh --help
```

Or consult the full guide:
- [FEATURE_ANALYSIS_GUIDE.md](FEATURE_ANALYSIS_GUIDE.md)
