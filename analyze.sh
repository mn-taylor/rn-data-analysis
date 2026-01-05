#!/bin/bash

# Convenience wrapper for feature importance analysis
# Provides easy-to-use presets and configurations

# Set GPUs (change this to use different GPUs)
export CUDA_VISIBLE_DEVICES=4,5,7

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/examples"

# Default values
MODE="single"
MODEL="ImprovedCNN1D"
DATA_ROOT="data"
SAVE_DIR=""
EXTRA_ARGS=""

# Usage function
usage() {
    cat << EOF
Usage: ./analyze.sh [OPTIONS]

Easy-to-use wrapper for feature importance analysis.

OPTIONS:
    -m, --mode MODE           Analysis mode: single, kfold (default: single)
    --model MODEL            Model: ImprovedCNN1D, ResNet1D, InceptionTime, ROCKET
    --data-root PATH         Path to data directory (default: data)
    --pretrained PATH        Path to pretrained model
    --save-dir PATH          Directory to save results
    -f, --fast               Fast mode: fewer repeats, no temporal analysis
    -q, --quick              Quick mode: 3 folds, fewer repeats (for kfold)
    --no-train              Don't train models if missing (kfold only)
    -h, --help              Show this help message

EXAMPLES:
    # Single model analysis with default settings
    ./analyze.sh

    # K-fold analysis with InceptionTime
    ./analyze.sh --mode kfold --model InceptionTime

    # Fast single-model analysis
    ./analyze.sh --fast --model ROCKET

    # Quick k-fold with pretrained models
    ./analyze.sh --mode kfold --quick --no-train

    # Custom data location
    ./analyze.sh --data-root /path/to/data --save-dir results/custom

GPUs: Currently using GPUs $CUDA_VISIBLE_DEVICES
      Edit this script to change GPU configuration.

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --pretrained)
            EXTRA_ARGS="$EXTRA_ARGS --pretrained $2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        -f|--fast)
            EXTRA_ARGS="$EXTRA_ARGS --no-temporal --n-repeats 5"
            shift
            ;;
        -q|--quick)
            EXTRA_ARGS="$EXTRA_ARGS --n-folds 3 --n-repeats 3"
            shift
            ;;
        --no-train)
            EXTRA_ARGS="$EXTRA_ARGS --no-train"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "single" && "$MODE" != "kfold" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 'single' or 'kfold'"
    exit 1
fi

# Set default save directory if not specified
if [[ -z "$SAVE_DIR" ]]; then
    if [[ "$MODE" == "kfold" ]]; then
        SAVE_DIR="feature_analysis_kfold"
    else
        SAVE_DIR="feature_analysis"
    fi
fi

# Build command
BASE_ARGS="--model $MODEL --data-root $DATA_ROOT --save-dir $SAVE_DIR"

echo "================================================================"
echo "Feature Importance Analysis"
echo "================================================================"
echo "Mode:      $MODE"
echo "Model:     $MODEL"
echo "Data:      $DATA_ROOT"
echo "Save to:   $SAVE_DIR"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "================================================================"
echo ""

# Run appropriate script
if [[ "$MODE" == "kfold" ]]; then
    python analyze_features_kfold.py $BASE_ARGS $EXTRA_ARGS
else
    python analyze_features.py $BASE_ARGS $EXTRA_ARGS
fi

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "Analysis completed successfully!"
    echo "Results saved to: $SAVE_DIR"
    echo "================================================================"
else
    echo ""
    echo "================================================================"
    echo "Analysis failed with exit code: $exit_code"
    echo "================================================================"
fi

exit $exit_code
