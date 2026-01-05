# Shell Scripts for Feature Importance Analysis

## Overview

You now have **3 shell scripts** to run feature importance analysis with automatic GPU configuration:

1. **`run_analyze_features.sh`** - Single model analysis
2. **`run_analyze_features_kfold.sh`** - K-fold cross-validation analysis
3. **`analyze.sh`** - Convenient wrapper with presets

All scripts are preconfigured to use **GPUs 4, 5, and 7**.

## Quick Start

### Simplest Usage (Recommended)

```bash
# Single model analysis with defaults
./analyze.sh

# K-fold analysis (more robust)
./analyze.sh --mode kfold --model InceptionTime

# Fast mode (fewer permutations, no temporal analysis)
./analyze.sh --fast

# Quick k-fold (3 folds instead of 5)
./analyze.sh --mode kfold --quick
```

### Direct Script Usage

```bash
# Single model analysis
./run_analyze_features.sh --model ResNet1D --n-repeats 20

# K-fold analysis
./run_analyze_features_kfold.sh --model InceptionTime --n-folds 5
```

## GPU Configuration

All scripts automatically set:
```bash
export CUDA_VISIBLE_DEVICES=4,5,7
```

### To Change GPUs

Edit the scripts and modify this line:
```bash
# For example, to use GPUs 0, 1, 2:
export CUDA_VISIBLE_DEVICES=0,1,2

# To use only GPU 4:
export CUDA_VISIBLE_DEVICES=4

# To use all GPUs:
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## Script Comparison

| Feature | analyze.sh | run_analyze_features.sh | run_analyze_features_kfold.sh |
|---------|-----------|------------------------|------------------------------|
| **Ease of use** | ⭐⭐⭐⭐⭐ Easiest | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Easy |
| **Single model** | ✅ | ✅ | ❌ |
| **K-fold** | ✅ | ❌ | ✅ |
| **Presets** | ✅ --fast, --quick | ❌ | ❌ |
| **GPU config** | ✅ Auto | ✅ Auto | ✅ Auto |

**Recommendation:** Use `analyze.sh` for most cases, it's the most user-friendly!

## Common Use Cases

### 1. Quick Feature Analysis

```bash
./analyze.sh --fast --model ROCKET
```

**What it does:**
- Uses ROCKET model (fastest)
- Fewer permutation repeats (5 instead of 10)
- Skips temporal analysis
- Results in ~5-10 minutes

### 2. Robust K-Fold Analysis

```bash
./analyze.sh --mode kfold --model InceptionTime
```

**What it does:**
- Trains 5 models (one per fold)
- Computes feature importance for each fold
- Aggregates results with confidence intervals
- Results in ~30-60 minutes (depending on data size)

### 3. Analyze with Pretrained Models

```bash
./analyze.sh --pretrained models/my_model.pth --model ImprovedCNN1D
```

**What it does:**
- Loads your pretrained model
- Computes feature importance
- Much faster (no training needed)

### 4. Analyze All Models in Parallel

Create a script `run_all.sh`:
```bash
#!/bin/bash

# Run different models in parallel on different GPUs
CUDA_VISIBLE_DEVICES=4 ./run_analyze_features.sh --model ImprovedCNN1D --save-dir results/cnn &
CUDA_VISIBLE_DEVICES=5 ./run_analyze_features.sh --model ResNet1D --save-dir results/resnet &
CUDA_VISIBLE_DEVICES=7 ./run_analyze_features.sh --model InceptionTime --save-dir results/inception &

wait
echo "All analyses complete!"
```

Then run:
```bash
chmod +x run_all.sh
./run_all.sh
```

### 5. Custom Data Location

```bash
./analyze.sh --data-root /mnt/server/data --save-dir /mnt/server/results
```

## Understanding the Output

All scripts create a directory with results:

### Single Model (`feature_analysis/`)
```
feature_analysis/
├── ImprovedCNN1D_permutation_importance.png    # Top 20 features (bar chart)
├── ImprovedCNN1D_permutation_importance.csv    # Full results
├── ImprovedCNN1D_temporal_importance.png       # Heatmap over time
├── ImprovedCNN1D_temporal_importance_summary.csv
└── ImprovedCNN1D_trained.pth                   # Model weights (if trained)
```

### K-Fold (`feature_analysis_kfold/`)
```
feature_analysis_kfold/
├── ImprovedCNN1D_kfold_importance.png          # Aggregated with error bars
├── ImprovedCNN1D_kfold_importance.csv          # Mean ± std across folds
├── ImprovedCNN1D_per_fold_importance.csv       # Individual fold results
├── ImprovedCNN1D_stability.csv                 # Stability scores
└── ImprovedCNN1D_fold{0-4}.pth                # Model for each fold
```

## Command-Line Arguments

### All Available Options

Run with `--help` to see all options:
```bash
./analyze.sh --help
./run_analyze_features.sh --help  # (via Python script)
./run_analyze_features_kfold.sh --help  # (via Python script)
```

### Most Useful Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Choose model | `--model InceptionTime` |
| `--data-root` | Data directory | `--data-root /path/to/data` |
| `--save-dir` | Output directory | `--save-dir results/exp1` |
| `--pretrained` | Load model | `--pretrained models/best.pth` |
| `--n-repeats` | Permutations | `--n-repeats 20` |
| `--no-temporal` | Skip temporal | `--no-temporal` |
| `--n-folds` | K-fold count | `--n-folds 3` (kfold only) |
| `--fast` | Fast preset | `--fast` (analyze.sh only) |
| `--quick` | Quick k-fold | `--quick` (analyze.sh only) |

## Troubleshooting

### Permission Denied

```bash
chmod +x analyze.sh run_analyze_features.sh run_analyze_features_kfold.sh
```

### Out of Memory

Use smaller batch size:
```bash
./analyze.sh --batch-size 16
```

Or fewer repeats:
```bash
./analyze.sh --fast  # Automatically uses n-repeats=5
```

### CUDA Not Available

Check if PyTorch can see your GPUs:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

If False, your GPU drivers or PyTorch installation may need attention.

### Scripts Can't Find Data

Specify the full path:
```bash
./analyze.sh --data-root /full/path/to/data
```

## Advanced Usage

### Running on a Specific GPU

```bash
CUDA_VISIBLE_DEVICES=4 ./run_analyze_features.sh --model ROCKET
```

### Background Execution

```bash
# Run in background and save output to log
nohup ./analyze.sh --mode kfold > analysis.log 2>&1 &

# Check progress
tail -f analysis.log
```

### Email Notification When Done

```bash
./analyze.sh --mode kfold && echo "Analysis complete!" | mail -s "Job Done" user@email.com
```

### Run with Different Seeds

```bash
for seed in 42 123 456; do
    ./analyze.sh --seed $seed --save-dir results/seed_$seed
done
```

## Performance Tips

1. **Use `--fast` for initial exploration**
   - Quick results to identify promising models
   - Then do full k-fold analysis on best model

2. **Use pretrained models when possible**
   - Much faster than training from scratch
   - First train with `train_kfold.py`, then analyze

3. **Skip temporal analysis if not needed**
   - Add `--no-temporal` to save time
   - Permutation importance is usually sufficient

4. **Use fewer folds for faster results**
   - 3 folds instead of 5: `--quick` or `--n-folds 3`
   - Still provides good confidence intervals

5. **Run models in parallel**
   - Use different GPUs for different models
   - See "Analyze All Models in Parallel" example above

## Files Summary

| File | Purpose |
|------|---------|
| `analyze.sh` | **Main wrapper** - easiest to use, has presets |
| `run_analyze_features.sh` | Direct wrapper for single model analysis |
| `run_analyze_features_kfold.sh` | Direct wrapper for k-fold analysis |
| `examples/analyze_features.py` | Python script for single model |
| `examples/analyze_features_kfold.py` | Python script for k-fold |
| `RUN_ANALYSIS.md` | Detailed usage guide |
| `FEATURE_ANALYSIS_GUIDE.md` | Conceptual guide to feature importance |

## Next Steps

1. **Run a quick test:**
   ```bash
   ./analyze.sh --fast
   ```

2. **Check the results:**
   ```bash
   ls -la feature_analysis/
   ```

3. **For publication-quality results:**
   ```bash
   ./analyze.sh --mode kfold --model InceptionTime
   ```

4. **Analyze the CSV files** to identify important features

5. **Validate findings** with domain knowledge

## Questions?

- See [RUN_ANALYSIS.md](RUN_ANALYSIS.md) for detailed examples
- See [FEATURE_ANALYSIS_GUIDE.md](FEATURE_ANALYSIS_GUIDE.md) for methodology
- Check script help: `./analyze.sh --help`
