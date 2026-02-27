#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/output_%x.out
#SBATCH --error=logs/error_%x.err
#SBATCH --partition=ndl
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

srun python -u main.py