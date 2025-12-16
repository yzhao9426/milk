#!/bin/bash
#SBATCH -A personal
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 02:00:00
#SBATCH -o bench_cows_128core_%j.out

# 啟動你的虛擬環境
source ~/opt-env/bin/activate

# 進入程式所在目錄
cd ~/opt-tests

# 跑同一個 benchmark 程式，只是 workers 設為 128
python3 bench_cows.py 128
