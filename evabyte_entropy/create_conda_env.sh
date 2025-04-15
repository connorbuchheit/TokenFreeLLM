#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=01:00:00

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=evabyte_$current_date

# Create the conda environment
module load Miniforge3/24.11.3-fasrc01

source $(conda info --base)/etc/profile.d/conda.sh
conda create -n $env_prefix python=3.11 -y -c anaconda
conda activate $env_prefix

echo "Currently in env $(which python)"
echo "Using pip: $(which pip)"

# Ensure we're using the pip from the conda environment and downgrade pip if needed
python -m pip install --upgrade "pip<24.1"
python -m pip install --requirement evals/requirements.txt
python -m pip install spacy

