#!/bin/bash
#SBATCH --job-name=mealpy_ga_test
#SBATCH --account=personal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=mealpy_ga_%j.out
#SBATCH --error=mealpy_ga_%j.err

module purge
module load python39

source ~/opt-env/bin/activate

cd ~/opt-tests
python mealpy_ga_test.py
