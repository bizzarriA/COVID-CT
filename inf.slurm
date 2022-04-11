#!/bin/bash
#SBATCH --job-name=test
#SBATCH -A IscrC_ML4COVID
#SBATCH --time=4:00:00
#SBATCH -N 1
#SBATCH -p m100_usr_prod
#SBATCH --mem=100000
#SBATCH --gres=gpu:1

python test_patient.py
