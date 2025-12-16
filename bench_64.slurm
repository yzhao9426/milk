#!/bin/bash
#SBATCH -A personal
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 02:00:00
#SBATCH -o bench_cows_64core_%j.out

# 啟動你的環境
source ~/opt-env/bin/activate

cd ~/opt-tests

python3 bench_cows.py 64
