#!/bin/bash
#SBATCH --job-name=de_slsqp_parallel
#SBATCH --account=personal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # 跟上面 num_processes=4 對應
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=de_slsqp_%j.out
#SBATCH --error=de_slsqp_%j.err

module purge
module load python39

# 如果有 opt-env，就啟動：
source ~/opt-env/bin/activate

# 確保不會跟 OpenMP 搶 CPU
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/opt-tests
python parallel_de_slsqp_example.py
