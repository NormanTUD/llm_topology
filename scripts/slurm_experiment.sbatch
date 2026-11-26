#!/bin/bash
#SBATCH --job-name=llm-topology
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=llm-topology-%j.log
set -euo pipefail

CONFIG="${1:?Usage: sbatch slurm_experiment.sbatch experiment.yaml}"
python3 app.py --doctor
python3 app.py --config "$CONFIG" --headless --device cuda
