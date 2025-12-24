"""
Example: Visualize time series data from CSV files.

This script demonstrates how to use the visualization utilities to
explore sensor data.
"""

import sys
import os

# Add parent directory to path to import rn_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rn_analysis.dataloader import list_csvs_by_class
from rn_analysis.utils import visualize_run_cycle_csv


def main():
    print("=" * 80)
    print("Visualizing Time Series Data")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load file list
    # -------------------------------------------------------------------------
    root = "data"
    print(f"\nLoading files from: {root}")

    all_files = list_csvs_by_class(root)
    print(f"Total files: {len(all_files)}")

    # Group files by class
    positive_files = [f for f in all_files if "POSITIVE" in f]
    control_files = [f for f in all_files if "CONTROL" in f]

    print(f"  POSITIVE: {len(positive_files)} files")
    print(f"  CONTROL: {len(control_files)} files")

    # -------------------------------------------------------------------------
    # Visualize sample files
    # -------------------------------------------------------------------------
    if len(positive_files) > 0:
        print("\n" + "-" * 80)
        print("Visualizing sample POSITIVE file:")
        print(f"  {positive_files[0]}")
        print("-" * 80)
        try:
            visualize_run_cycle_csv(positive_files[0])
        except Exception as e:
            print(f"Error visualizing file: {e}")

    if len(control_files) > 0:
        print("\n" + "-" * 80)
        print("Visualizing sample CONTROL file:")
        print(f"  {control_files[0]}")
        print("-" * 80)
        try:
            visualize_run_cycle_csv(control_files[0])
        except Exception as e:
            print(f"Error visualizing file: {e}")

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print("\nTo visualize a specific file, modify this script to pass the file path")
    print("to visualize_run_cycle_csv().")


if __name__ == "__main__":
    main()
