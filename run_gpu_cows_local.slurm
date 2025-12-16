#!/bin/bash
#SBATCH --job-name=gpu_cows_local
#SBATCH --account=personal
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=gpu_cows_local_%j.out
#SBATCH --error=gpu_cows_local_%j.err

module purge
module load shared
module load cuda12.6/toolkit/12.6.2

source ~/opt-tests/gpu-env/bin/activate

cd ~/opt-tests

echo "Running gpu_cows_local on GPU..."
python gpu_cows_local.py
