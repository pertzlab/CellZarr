# /// script
# requires-python = "<=3.11"
# dependencies = [
#   "ultrack",
#   "ome-zarr",
#   "zarr",
#   "higra",
#   "gurobipy",
#   "natsort",
# ]
# ///
"""
Batch Cell Tracking Script for All FOVs using Ultrack
====================================================

This script automates cell tracking for all available fields of view (FOVs) found in the output directory.
It uses the get_fovs() function from settings.py to retrieve all FOV names and then runs the tracking script
(04_Cell_Tracking_ultrack.py) for each FOV sequentially.

USAGE
-----
Run this script directly:
    python 04Cell_Tracking_ultrack_all_fovs.py

REQUIREMENTS
------------
- The script 04_Cell_Tracking_ultrack.py must be in the same directory and callable via subprocess.
- All dependencies for cell tracking must be installed.
- The configuration/settings.py file must be accessible and contain get_fovs().
"""

import subprocess
from configuration.settings import get_fovs
import sys
import os


def main():
    fovs = get_fovs()
    script_path = os.path.join(os.path.dirname(__file__), "04_Cell_Tracking_ultrack.py")
    for fov in fovs:
        try:
            # Extract the FOV index from the FOV name (e.g., FOV_7 -> 7)
            fov_i = int(fov.split("_")[-1])
        except Exception as e:
            print(f"Skipping {fov}: could not extract FOV index. Error: {e}")
            continue
        print(f"\n--- Processing {fov} (index {fov_i}) ---")
        result = subprocess.run([sys.executable, script_path, str(fov_i)])
        if result.returncode != 0:
            print(f"Tracking failed for {fov} (index {fov_i})")
        else:
            print(f"Tracking completed for {fov} (index {fov_i})")


if __name__ == "__main__":
    main()
