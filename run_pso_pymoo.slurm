#!/bin/bash
#SBATCH --job-name=pso_pymoo_test
#SBATCH --account=personal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=pso_pymoo_%j.out
#SBATCH --error=pso_pymoo_%j.err

module purge
module load python39

source ~/opt-env/bin/activate

cd ~/opt-tests
python pso_pymoo_test.py
