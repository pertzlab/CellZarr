############################################################
# SLURM Batch Script for Cell Tracking with Ultrack
# ------------------------------------------------
# This script submits an array job to the SLURM scheduler to run
# cell tracking in parallel for multiple fields of view (FOVs)
# using the 04_Cell_Tracking_ultrack.py script.
#
# USAGE:
#   1. Make sure this script and the Python script are in the same directory.
#   2. Adjust the job array range ("--array=1-48%2") as needed for your data.
#   3. Submit the job to the cluster with:
#        sbatch 04_Cell_Tracking_slurm.sh
#   4. Each job will process a different FOV (e.g., FOV_1, FOV_2, ...).
#   5. Output and error logs are written to ./logs/.
#
# REQUIREMENTS:
#   - uv must be available.
#   - The Python script and data must be accessible on the compute nodes.
############################################################

#!/bin/bash
#SBATCH --job-name=stp_ult
#SBATCH --array=1-48%2
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=90G
#SBATCH --output=./logs/slurm-%A_%a.out
#SBATCH --error=./logs/slurm-%A_%a.err


cd $BASE_PATH
uv run 04_Cell_Tracking_ultrack.py ${SLURM_ARRAY_TASK_ID}
