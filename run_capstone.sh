#!/bin/bash
#SBATCH --job-name=capstone-opt
#SBATCH --partition=t4_normal_q
#SBATCH --account=personal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=capstone_%j.out

module purge
module load shared
module load cuda12.6/toolkit/12.6.2

# 進到專案目錄（capstone_optimization.py 在這裡）
cd /home/yzhao9426/opt-tests

# 啟動家目錄底下的 opt-env
source ~/opt-env/bin/activate

# 執行 capstone 主程式
python capstone_optimization.py

