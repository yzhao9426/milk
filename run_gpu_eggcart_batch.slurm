#!/bin/bash
#SBATCH --job-name=gpu_eggcart_batch
#SBATCH --account=personal
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --output=gpu_eggcart_batch_%j.out
#SBATCH --error=gpu_eggcart_batch_%j.err

module purge
module load shared
module load cuda12.6/toolkit/12.6.2

source ~/opt-tests/gpu-env/bin/activate

cd ~/opt-tests

echo "Running gpu_eggcart_batch on GPU..."
python gpu_eggcart_batch.py
