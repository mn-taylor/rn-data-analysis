#!/bin/bash
# =============================================================================
# _analyze.sh  —  Run k-fold feature importance analysis on trained models
#
# Expects fold splits and trained checkpoints to already exist.
# Run _train.sh first if they don't.
#
# Usage:
#   ./examples/_analyze.sh
#
# To target a specific past training run, set TIMESTAMP below or export it:
#   TIMESTAMP=20260309_143022 ./examples/_analyze.sh
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Paths — edit these to match your setup
# ---------------------------------------------------------------------------
# assume all are run from the project root, so paths are relative to that
# CONFIG="examples/configs/analyze_kfold.yaml"
# FOLDS="data/mdd_data_v3/mdd_48h/folds_meta.json"
# RESULTS_ROOT="data/mdd_data_v3/mdd_48h_results"
# DATA_ROOT="data/mdd_data_v3/mdd_48h"

CONFIG="examples/configs/analyze_kfold.yaml"
CUDA_DEVICE="cuda:0"   # e.g. cuda:0, cuda:1, cpu

# ---------------------------------------------------------------------------
# Mode toggle — must match the setting used in _train.sh for this run.
# ---------------------------------------------------------------------------
USE_FIXED_SPLIT=false

if [ "$USE_FIXED_SPLIT" = "true" ]; then
    TRAIN_DIR="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data/train"
    EVAL_DIR="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data/eval"
    FOLDS="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data/fixed_split_meta.json"
    RESULTS_ROOT="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data/results"
    DATA_ROOT="/home/keaneong/scratch/keane/rn_mdd/may2026_parsed_data"
else
    FOLDS="data/original_data/original/folds_meta.json"
    RESULTS_ROOT="data/original_data/original_results_v2"
    DATA_ROOT="data/original_data/original"
fi

# ---------------------------------------------------------------------------
# Timestamp — identifies which training run's checkpoints to use.
# Leave empty to auto-read from .last_run_id (written by _train.sh).
# Override by setting TIMESTAMP before calling this script:
#   TIMESTAMP=20260309_143022 ./examples/_analyze.sh
# ---------------------------------------------------------------------------
: "${TIMESTAMP:=}"  # default to empty if not set

# ---------------------------------------------------------------------------
# Models to analyse — must match what was trained in _train.sh
# ---------------------------------------------------------------------------
MODELS=(
    "ImprovedCNN1D"
    # "ResNet1D"
    # "InceptionTime"
    # "ROCKET"
    # "TimesNet"
    # "TSLANet"
)

# ---------------------------------------------------------------------------
# NaN strategy — must match the setting used in _train.sh for this run,
# so the dataloader sees the same feature columns as during training.
#   fill_zero   — keep all columns, missing sensor readings → 0 (default)
#   clean_only  — use only the ~56 columns non-NaN in every file
#   drop_sparse — drop columns NaN in > SPARSE_THRESHOLD fraction of files,
#                 fill remaining NaN with 0
# ---------------------------------------------------------------------------
NAN_STRATEGY="fill_zero"
SPARSE_THRESHOLD="0.5"

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1  # flush Python output immediately (avoids apparent hang)

# Resolve timestamp
if [ -n "$TIMESTAMP" ]; then
    echo "Using provided timestamp: $TIMESTAMP"
elif [ -f ".last_run_id" ]; then
    TIMESTAMP=$(cat .last_run_id)
    echo "Using timestamp from .last_run_id: $TIMESTAMP"
else
    echo ""
    echo "ERROR: No timestamp found."
    echo "  Run _train.sh first (it writes .last_run_id),"
    echo "  or set TIMESTAMP before calling this script:"
    echo "    TIMESTAMP=20260309_143022 ./examples/_analyze.sh"
    exit 1
fi

echo "=============================================="
echo "Config      : $CONFIG"
echo "Folds meta  : $FOLDS"
echo "Data root   : $DATA_ROOT"
echo "Results root: $RESULTS_ROOT"
echo "Models      : ${MODELS[*]}"
echo "Timestamp   : $TIMESTAMP"
echo "=============================================="

# Guard: folds meta must exist
if [ ! -f "$FOLDS" ]; then
    echo ""
    echo "ERROR: Folds meta not found at '$FOLDS'."
    echo "       Run ./examples/_train.sh (with the same USE_FIXED_SPLIT setting) first."
    exit 1
fi

# Analyse each model using the matching run directory from training
for MODEL in "${MODELS[@]}"; do
    RUN_DIR="$RESULTS_ROOT/$MODEL/${MODEL}_run_${TIMESTAMP}"

    if [ ! -d "$RUN_DIR" ]; then
        echo ""
        echo "WARNING: Run directory not found for $MODEL: $RUN_DIR"
        echo "         Skipping (run _train.sh for this model first)."
        continue
    fi

    echo ""
    echo "=============================================="
    echo ">>> Analysing model : $MODEL"
    echo "    Run dir         : $RUN_DIR"
    echo "=============================================="
    python examples/analyze_features_kfold.py \
        --config "$CONFIG" \
        --folds-meta "$FOLDS" \
        --model "$MODEL" \
        --run-save-dir "$RUN_DIR" \
        --data-root "$DATA_ROOT" \
        --device "$CUDA_DEVICE" \
        --nan-strategy "$NAN_STRATEGY" \
        --sparse-threshold "$SPARSE_THRESHOLD"
done

echo ""
echo "Done. Analysed ${#MODELS[@]} model(s): ${MODELS[*]}"
echo "Results in: $RESULTS_ROOT/{model}/{model}_run_${TIMESTAMP}/"
