#!/bin/bash
#SBATCH --job-name=ft
#SBATCH -A IscrC_ML4COVID
#SBATCH --time=8:00:00
#SBATCH -N 1
#SBATCH -p m100_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=246000

python main.py
