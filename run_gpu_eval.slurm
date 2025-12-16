#!/bin/bash
#SBATCH --job-name=gpu_eval_test
#SBATCH --account=personal
#SBATCH --partition=a30_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=gpu_eval_%j.out
#SBATCH --error=gpu_eval_%j.err

module purge
module load shared
module load cuda12.6/toolkit/12.6.2

source ~/opt-tests/gpu-env/bin/activate

python gpu_herd_eval.py
