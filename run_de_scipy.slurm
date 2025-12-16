#!/bin/bash
#SBATCH --job-name=de_scipy_test
#SBATCH --account=personal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=de_scipy_%j.out
#SBATCH --error=de_scipy_%j.err

module purge
module load python39
source ~/opt-env/bin/activate

cd ~/opt-tests
python de_scipy_test.py
