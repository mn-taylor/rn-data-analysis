#!/bin/bash

# Shell script to run feature importance analysis with specific GPUs
# Usage: ./run_analyze_features.sh [options]
# All options are passed to the Python script

# Set which GPUs to use (GPUs 4, 5, 7)
export CUDA_VISIBLE_DEVICES=4,5,7

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python script from the examples directory
cd "$SCRIPT_DIR/examples"

echo "================================================================"
echo "Running Feature Importance Analysis"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"
echo ""

python analyze_features.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "Analysis completed successfully!"
    echo "================================================================"
else
    echo ""
    echo "================================================================"
    echo "Analysis failed with exit code: $exit_code"
    echo "================================================================"
fi

exit $exit_code
