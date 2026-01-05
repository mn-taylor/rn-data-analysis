# Feature Importance Analysis Guide

This guide explains how to analyze which features are most important for your time series classification models.

## Overview

The feature importance module provides three main methods:

1. **Permutation Importance** - Model-agnostic method that works with all 5 models
2. **Integrated Gradients** - Gradient-based attribution for neural networks
3. **Temporal Importance** - Shows which features are important at which time points

## Quick Start

### Option 1: Analyze a Single Model

Use this if you want to quickly analyze feature importance for one model:

```bash
cd examples
python analyze_features.py
```

This will:
- Train a model (or load a pretrained one if available)
- Compute permutation importance for all features
- Compute temporal importance (feature importance over time)
- Generate visualizations and save results to `feature_analysis/`

**Configuration** (edit in `analyze_features.py`):
```python
model_name = "ImprovedCNN1D"  # Choose: ImprovedCNN1D, ResNet1D, InceptionTime, ROCKET
use_pretrained = False         # Set True if you have a saved model
pretrained_path = "models/best_model.pth"
n_repeats = 10                 # Number of permutations per feature
compute_temporal = True        # Whether to compute temporal importance
```

### Option 2: K-Fold Cross-Validation Analysis (Recommended)

Use this for more robust feature importance estimates across multiple folds:

```bash
cd examples
python analyze_features_kfold.py
```

This will:
- Run feature importance analysis across 5 folds
- Aggregate results to identify consistently important features
- Compute stability scores (how consistent is each feature's importance?)
- Save results to `feature_analysis_kfold/`

**Configuration** (edit in `analyze_features_kfold.py`):
```python
model_name = "ImprovedCNN1D"
n_folds = 5
n_repeats = 5                  # Fewer repeats per fold to save time
train_if_missing = True        # Train models if not found
```

## Understanding the Results

### Permutation Importance

**What it measures:** Drop in model performance when a feature is randomly shuffled.

**Interpretation:**
- Higher values = more important feature
- If permuting a feature causes accuracy to drop significantly, that feature is important
- Model-agnostic: works with any model

**Output files:**
- `*_permutation_importance.png` - Bar chart of top features
- `*_permutation_importance.csv` - Full results with all features ranked

### Temporal Importance

**What it measures:** Which features are important at which time points.

**Interpretation:**
- Heatmap shows feature importance (C features × T time steps)
- Bright regions = feature is important at that time point
- Helps understand temporal dynamics of your data

**Output files:**
- `*_temporal_importance.png` - Heatmap visualization
- `*_temporal_importance_summary.csv` - Summary statistics
- `*_temporal_importance.npy` - Full matrix for further analysis

### K-Fold Results

**Additional metrics:**
- `importances_mean` - Average importance across folds
- `importances_std` - Standard deviation across folds
- `stability_score` - How consistent is the feature importance? (1 = very stable, 0 = unstable)

**Interpretation:**
- **High mean, low std** → Reliably important feature (most valuable!)
- **High mean, high std** → Important but inconsistent across folds
- **Low mean** → Not important for predictions

## Workflow Example

### 1. Train models with k-fold CV

First, train your models and save checkpoints:

```bash
cd examples
python train_kfold.py  # This saves models per fold
```

### 2. Analyze features across folds

```bash
python analyze_features_kfold.py
```

Modify the script to load your saved models:

```python
model_path = f"models_folds/fold_{fold_idx}/{model_name}_best.pth"
```

### 3. Interpret results

Check the output files:
- `*_kfold_importance.png` - Top 20 features with error bars
- `*_stability.csv` - Features ranked by importance and stability

Focus on features with:
1. High mean importance
2. Low standard deviation
3. High stability score

### 4. Domain validation

Compare the top features with your domain knowledge:
- Do the important features make biological/physical sense?
- Are there any surprising features?
- Can you remove low-importance features to simplify the model?

## Advanced Usage

### Custom Analysis

You can use the functions directly in your own scripts:

```python
from rn_analysis.feature_importance import (
    compute_permutation_importance,
    compute_temporal_importance,
    plot_feature_importance,
    save_importance_results
)

# After training your model...
results = compute_permutation_importance(
    model=model,
    dataloader=test_loader,
    feature_names=feature_cols,
    device="cuda",
    n_repeats=20,           # More repeats = more reliable
    metric="auc"            # or "accuracy"
)

# Visualize
plot_feature_importance(results, top_k=30)

# Save for later
save_importance_results(results, "my_importance_results.csv")
```

### Analyze Specific Samples

To understand which features are important for specific predictions:

```python
from rn_analysis.feature_importance import compute_integrated_gradients

# Get a batch of samples
X, y = next(iter(test_loader))

# Compute attributions for the positive class
attributions = compute_integrated_gradients(
    model=model,
    X=X,
    target_class=1,  # Positive class
    device="cuda"
)

# attributions shape: (B, C, T) - importance of each feature at each time
```

## Tips and Best Practices

1. **Use k-fold analysis** for publication-quality results
   - More robust than single-model analysis
   - Provides confidence intervals
   - Identifies stable features

2. **Start with permutation importance**
   - Fast and interpretable
   - Works with all models
   - Good for initial feature selection

3. **Use temporal importance** to understand dynamics
   - Helps identify critical time windows
   - Can guide feature engineering
   - Useful for explaining model decisions

4. **Validate with domain knowledge**
   - Important features should make sense
   - If results are unexpected, investigate why
   - Consider feature interactions

5. **Feature selection workflow**
   - Identify low-importance features
   - Remove them and retrain
   - Check if performance is maintained
   - Simpler models are easier to interpret and deploy

## Computational Considerations

- **Permutation importance**: Time = n_features × n_repeats × inference_time
  - For 76 features, 10 repeats: ~760 forward passes
  - Use fewer repeats for faster results

- **Temporal importance**: Time = n_samples × integration_steps
  - Default: 50 integration steps
  - Can be slow for large test sets
  - Consider using a subset of data

- **K-fold analysis**: Multiply by number of folds
  - Most time-consuming but most reliable
  - Can run folds in parallel if you have multiple GPUs

## Troubleshooting

**Problem:** "Model not found"
- Set `train_if_missing=True` in k-fold script
- Or train models first using `train_kfold.py`

**Problem:** Out of memory during analysis
- Reduce batch size in dataloader
- Use fewer integration steps for temporal importance
- Analyze one fold at a time

**Problem:** All features have similar importance
- Model might not be well-trained
- Try training longer or with different hyperparameters
- Check if data preprocessing is correct

**Problem:** Results vary a lot across folds
- Normal for small datasets
- Focus on features with high stability scores
- Consider collecting more data

## References

- **Permutation Importance**: Breiman, "Random Forests", 2001
- **Integrated Gradients**: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017

## Questions?

For issues or questions:
1. Check the example scripts in `examples/`
2. Review the module documentation in `rn_analysis/feature_importance.py`
3. Examine the test outputs to understand expected formats
