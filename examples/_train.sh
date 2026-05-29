#!/bin/bash
# =============================================================================
# _train.sh  —  Create folds (if needed) then train k-fold models
#
# Usage:
#   ./examples/_train.sh
#
# Output structure:
#   results/{model}/{model}_run_{timestamp}/
#     {model}_fold*.pth          — per-fold checkpoints
#     {model}_train_metrics.csv  — per-fold metrics table
#     {model}_train_summary.json
#     {model}_confusion_matrix.png
#
# The timestamp is saved to .last_run_id so that _analyze.sh can
# automatically find the matching run directories.
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Paths — edit these to match your setup
# ---------------------------------------------------------------------------
# CONFIG="examples/configs/analyze_kfold.yaml"
# FOLDS="data/original_data/original/folds_meta.json"
# RESULTS_ROOT="data/original_data/original_results_v2"
# DATA_ROOT="data/original_data/original"

CONFIG="examples/configs/analyze_kfold.yaml"
CUDA_DEVICE="cuda:0"   # e.g. cuda:0, cuda:1, cpu

# CONFIG="examples/configs/analyze_kfold.yaml"
# FOLDS="data/mdd_data_v3/mdd_48h/folds_meta.json"
# RESULTS_ROOT="data/mdd_data_v3/mdd_48h_results"
# DATA_ROOT="data/mdd_data_v3/mdd_48h"
# CUDA_DEVICE="cuda:7"   # e.g. cuda:0, cuda:1, cpu

# ---------------------------------------------------------------------------
# Mode toggle — set USE_FIXED_SPLIT=true to use the pre-defined train/eval
# split, or false for k-fold cross-validation.
# ---------------------------------------------------------------------------
USE_FIXED_SPLIT=true

if [ "$USE_FIXED_SPLIT" = "true" ]; then
    TRAIN_DIR="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data/train"
    EVAL_DIR="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data/eval"
    FOLDS="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data/fixed_split_meta.json"
    RESULTS_ROOT="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data/results"
    DATA_ROOT="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data"
else
    FOLDS="data/mdd_data_v3/mdd_48h/folds_meta.json"
    RESULTS_ROOT="data/mdd_data_v3/mdd_48h_results"
    DATA_ROOT="data/mdd_data_v3/mdd_48h"
fi

# ---------------------------------------------------------------------------
# Models to train — add or remove entries as needed
# ---------------------------------------------------------------------------
MODELS=(
    # "ImprovedCNN1D"
    # "ResNet1D"
    # "InceptionTime"
    # "ROCKET"
    "TimesNet"
    # "TSLANet"
)

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1  # flush Python output immediately (avoids apparent hang)

# Generate a shared timestamp for this training run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Config      : $CONFIG"
echo "Folds meta  : $FOLDS"
echo "Data root   : $DATA_ROOT"
echo "Results root: $RESULTS_ROOT"
echo "Models      : ${MODELS[*]}"
echo "Timestamp   : $TIMESTAMP"
echo "=============================================="

# Step 1: Create fold splits (or fixed-split meta) once, shared across all models
if [ "$USE_FIXED_SPLIT" = "true" ]; then
    echo ""
    echo ">>> Fixed-split mode — using pre-defined train/eval split"
    if [ ! -f "$FOLDS" ]; then
        echo ">>> Creating fixed split meta..."
        python scripts/data_utils/create_fixed_split.py \
            --train-dir "$TRAIN_DIR" \
            --eval-dir  "$EVAL_DIR" \
            --output    "$FOLDS"
    else
        echo ">>> Using existing fixed split meta: $FOLDS"
    fi
else
    if [ ! -f "$FOLDS" ]; then
        echo ""
        echo ">>> Creating fold splits..."
        python scripts/data_utils/create_folds.py --config "$CONFIG" --output "$FOLDS"
    else
        echo ""
        echo ">>> Using existing fold splits: $FOLDS"
    fi
fi

# Step 2: Train each model in its own timestamped run directory
for MODEL in "${MODELS[@]}"; do
    RUN_DIR="$RESULTS_ROOT/$MODEL/${MODEL}_run_${TIMESTAMP}"
    echo ""
    echo "=============================================="
    echo ">>> Training model : $MODEL"
    echo "    Run dir        : $RUN_DIR"
    echo "=============================================="
    python examples/train_kfold.py \
        --config "$CONFIG" \
        --folds-meta "$FOLDS" \
        --model "$MODEL" \
        --run-save-dir "$RUN_DIR" \
        --data-root "$DATA_ROOT" \
        --device "$CUDA_DEVICE"
done

# Save timestamp so _analyze.sh can find these run directories
echo "$TIMESTAMP" > .last_run_id
echo ""
echo "Saved run ID to .last_run_id (timestamp: $TIMESTAMP)"

# Step 3: Print cross-model summary
echo ""
echo "=============================================="
echo "TRAINING SUMMARY"
echo "=============================================="
export _RESULTS_ROOT="$RESULTS_ROOT"
export _TIMESTAMP="$TIMESTAMP"
export _MODELS="${MODELS[*]}"
python3 - <<'PYEOF'
import json, os, sys

results_root = os.environ.get("_RESULTS_ROOT")
timestamp    = os.environ.get("_TIMESTAMP")
models       = [m for m in os.environ.get("_MODELS", "").split() if m]

METRICS = ["auc", "accuracy", "sensitivity", "specificity", "ppv", "brier_score"]
LABELS  = ["AUC", "Accuracy", "Sens", "Spec", "PPV", "Brier"]

rows = []
for model in models:
    path = os.path.join(results_root, model,
                        f"{model}_run_{timestamp}",
                        f"{model}_train_summary.json")
    if not os.path.exists(path):
        print(f"  [WARNING] No summary found for {model}: {path}", file=sys.stderr)
        continue
    with open(path) as f:
        data = json.load(f)
    ms = data["metric_summary"]
    row = {"model": model}
    for key in METRICS:
        m, s = ms[key]["mean"], ms[key]["std"]
        row[key] = f"{m:.4f}±{s:.4f}"
    rows.append(row)

if not rows:
    print("  No results found.")
    sys.exit(0)

col_w = max(len(r["model"]) for r in rows) + 2
print(f"  {'Model':<{col_w}}", end="")
for lbl in LABELS:
    print(f"  {lbl:>15}", end="")
print()
print("  " + "-" * (col_w + len(LABELS) * 17))
for row in rows:
    print(f"  {row['model']:<{col_w}}", end="")
    for key in METRICS:
        print(f"  {row[key]:>15}", end="")
    print()
PYEOF

echo ""
echo "Done. Trained ${#MODELS[@]} model(s): ${MODELS[*]}"
echo "Results in: $RESULTS_ROOT/{model}/{model}_run_${TIMESTAMP}/"
