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
CONFIG="examples/configs/analyze_kfold.yaml"
FOLDS="folds_meta.json"
RESULTS_ROOT="results"
DATA_ROOT="/home/keaneong/rn-data-analysis/data/mdd_data_v3/mdd"

# ---------------------------------------------------------------------------
# Models to train — add or remove entries as needed
# ---------------------------------------------------------------------------
MODELS=(
    "ImprovedCNN1D"
    "ResNet1D"
    "InceptionTime"
    "ROCKET"
)

cd "$(dirname "$0")/.."

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

# Step 1: Create fold splits once (shared across all models)
if [ ! -f "$FOLDS" ]; then
    echo ""
    echo ">>> Creating fold splits..."
    python scripts/data_utils/create_folds.py --config "$CONFIG" --output "$FOLDS"
else
    echo ""
    echo ">>> Using existing fold splits: $FOLDS"
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
        --data-root "$DATA_ROOT"
done

# Save timestamp so _analyze.sh can find these run directories
echo "$TIMESTAMP" > .last_run_id
echo ""
echo "Saved run ID to .last_run_id (timestamp: $TIMESTAMP)"
echo ""
echo "Done. Trained ${#MODELS[@]} model(s): ${MODELS[*]}"
echo "Results in: $RESULTS_ROOT/{model}/{model}_run_${TIMESTAMP}/"
