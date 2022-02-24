#!/bin/bash
#SBATCH --job-name=main
#SBATCH -A IscrC_ML4COVID
#SBATCH --time=4:00:00
#SBATCH -N 1
#SBATCH -p m100_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=246000

python main.py
