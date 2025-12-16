#!/bin/bash
#SBATCH --job-name=eggcart_128
#SBATCH --account=personal
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=eggcart_128core_%A_%a.out
#SBATCH --error=eggcart_128core_%A_%a.err
#SBATCH --array=1-3

module purge
module load python39
source ~/opt-env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/opt-tests

# 根據 array ID 選擇不同的任務數量
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  NTASKS=100000      # 10 萬顆蛋
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
  NTASKS=1000000     # 100 萬顆蛋
else
  NTASKS=10000000    # 1000 萬顆蛋
fi

WORKERS=$SLURM_CPUS_PER_TASK
ITERS=50   # 每個任務只做 50 步，讓它變得非常「小」

echo "Running egg-cart test with ${NTASKS} tasks on ${WORKERS} workers, ${ITERS} iters per task"
python eggcart_benchmark.py --ntasks $NTASKS --workers $WORKERS --iters $ITERS
