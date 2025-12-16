#!/bin/bash
#SBATCH --job-name=bench_cows
#SBATCH --account=personal
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 先測 32 cores，之後可以改成 64 / 128
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=bench_cows_%A_%a.out
#SBATCH --error=bench_cows_%A_%a.err
#SBATCH --array=1-3                 # 1: 1萬, 2: 10萬, 3: 100萬 頭牛

module purge
module load python39

# 啟動自己的虛擬環境
source ~/opt-env/bin/activate

# 不要讓 OpenMP 搶 core
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/opt-tests

# 依照 array index 決定牛的數量
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  NCOWS=10000        # 1 萬
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
  NCOWS=100000       # 10 萬
else
  NCOWS=1000000      # 100 萬
fi

WORKERS=$SLURM_CPUS_PER_TASK

echo "Running benchmark with ${NCOWS} cows on ${WORKERS} workers"
python benchmark_cows_parallel.py --ncows $NCOWS --workers $WORKERS
