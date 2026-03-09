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
CONFIG="examples/configs/analyze_kfold.yaml"
FOLDS="folds_meta.json"
RESULTS_ROOT="results"
DATA_ROOT="/home/keaneong/rn-data-analysis/data/mdd_data_v3/mdd"

# ---------------------------------------------------------------------------
# Timestamp — identifies which training run's checkpoints to use.
# Leave empty to auto-read from .last_run_id (written by _train.sh).
# Override by setting TIMESTAMP before calling this script:
#   TIMESTAMP=20260309_143022 ./examples/_analyze.sh
# ---------------------------------------------------------------------------
: "${TIMESTAMP:=}"

# ---------------------------------------------------------------------------
# Models to analyse — must match what was trained in _train.sh
# ---------------------------------------------------------------------------
MODELS=(
    "ImprovedCNN1D"
    "ResNet1D"
    "InceptionTime"
    "ROCKET"
)

cd "$(dirname "$0")/.."

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

# Guard: folds must exist
if [ ! -f "$FOLDS" ]; then
    echo ""
    echo "ERROR: Folds meta not found at '$FOLDS'."
    echo "       Run ./examples/_train.sh first."
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
        --data-root "$DATA_ROOT"
done

echo ""
echo "Done. Analysed ${#MODELS[@]} model(s): ${MODELS[*]}"
echo "Results in: $RESULTS_ROOT/{model}/{model}_run_${TIMESTAMP}/"
