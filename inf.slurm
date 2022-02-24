#!/bin/bash
#SBATCH --job-name=test
#SBATCH -A IscrC_ML4COVID
#SBATCH --time=4:00:00
#SBATCH -N 1
#SBATCH -p m100_usr_prod
#SBATCH --gres=gpu:1

python unife_inference.py
