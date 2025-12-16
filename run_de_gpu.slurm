#!/bin/bash
#SBATCH --job-name=de_gpu_demo
#SBATCH --account=personal
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=de_gpu_%j.out
#SBATCH --error=de_gpu_%j.err

module purge
module load shared
module load cuda12.6/toolkit/12.6.2

source ~/opt-tests/gpu-env/bin/activate

python de_gpu_demo.py
