# RN Data Analysis - Time Series Classification Package

A modular framework for time series classification on medical sensor data.

## Installation

No installation required. Add the parent directory to your Python path:

```python
import sys
sys.path.insert(0, "/path/to/rn-data-analysis")
```

## Package Structure

```
rn_analysis/
├── __init__.py          # Package initialization
├── config.py            # Configuration dataclasses
├── dataloader.py        # Dataset classes and data utilities
├── models.py            # Neural network architectures
├── train.py             # Training and evaluation functions
└── utils.py             # Visualization and helper functions
```

## Quick Start

### 1. Train a CNN model

```python
from rn_analysis.config import DataConfig, TrainingConfig, CNNConfig
from rn_analysis.dataloader import list_csvs_by_class, train_test_split_files, create_dataloaders
from rn_analysis.models import ImprovedCNN1D
from rn_analysis.train import train_model
from rn_analysis.utils import set_seed, get_device
import torch.nn as nn

# Configuration
data_config = DataConfig(root="data", T=512, batch_size=32)
training_config = TrainingConfig(epochs=50, lr=1e-3)
set_seed(42)
device = get_device()

# Load data
all_files = list_csvs_by_class(data_config.root)
train_files, test_files = train_test_split_files(all_files, train_split=0.8, seed=42)
train_loader, test_loader = create_dataloaders(
    train_files, test_files,
    label_map=data_config.label_map,
    T=data_config.T,
    batch_size=data_config.batch_size,
    output_format="channels_first"
)

# Create model
input_channels = next(iter(train_loader))[0].shape[1]
model = ImprovedCNN1D(C=input_channels, dropout=0.3).to(device)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr)
loss_fn = nn.CrossEntropyLoss()
history = train_model(model, train_loader, test_loader, optimizer, loss_fn, device)
```

### 2. K-Fold Cross-Validation

```python
from rn_analysis.train import stratified_kfold_cv
from rn_analysis.models import InceptionTime

results = stratified_kfold_cv(
    model_class=InceptionTime,
    model_kwargs={"input_channels": 76, "num_classes": 2},
    files=all_files,
    label_map={"CONTROL": 0, "POSITIVE": 1},
    n_folds=5,
    epochs=50,
    device="cuda"
)
print(f"Mean AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
```

### 3. Visualize Data

```python
from rn_analysis.utils import visualize_run_cycle_csv

visualize_run_cycle_csv("data/POSITIVE/2511121155_3.csv")
```

## Available Models

1. **ImprovedCNN1D**: Deep CNN with 6 convolutional blocks
2. **TimeSeriesTransformer**: Transformer encoder with positional encoding
3. **ResNet1D**: 1D ResNet adapted from image classification
4. **InceptionTime**: Multi-scale inception modules (SOTA on UCR archive)
5. **RocketClassifier**: Random convolutional kernels (fast, good for small data)

## Configuration

All hyperparameters are managed via dataclasses:

- `DataConfig`: Data loading parameters
- `TrainingConfig`: Training hyperparameters
- `CNNConfig`, `TransformerConfig`, etc.: Model-specific configs
- `ExperimentConfig`: Complete experiment configuration

## Example Scripts

See the `examples/` directory for complete working examples:

- `train_cnn.py`: Train CNN with 80/20 split
- `train_transformer.py`: Train Transformer
- `train_kfold.py`: K-fold CV with model comparison
- `visualize_data.py`: Visualize sensor data

## Data Format

Expected directory structure:

```
data/
├── POSITIVE/
│   ├── run1_cycle1.csv
│   ├── run1_cycle2.csv
│   └── ...
└── CONTROL/
    ├── run2_cycle1.csv
    └── ...
```

Each CSV should have:
- `relative_time_sec`: Time column
- Feature columns: Sensor readings
- Optional: `run_id`, `cycle`, `section`, `patient_id`, etc.

## Key Features

- **Modular design**: Each component has a single responsibility
- **Type-safe configs**: All hyperparameters use dataclasses with type hints
- **Flexible data loading**: Supports both (C, T) and (T, C) formats
- **Multiple models**: 5 SOTA architectures included
- **K-fold CV**: Stratified cross-validation with early stopping
- **Visualization**: Built-in plotting for data and results
- **Reproducible**: Seed setting for deterministic results

## License

MIT License
